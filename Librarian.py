import json
from pathlib import Path
from tqdm import tqdm
import shutil
import nest_asyncio
import os
import requests
from colorama import Fore
import numpy as np

from format import Paper, Claim, ListClaims
from Setup import setup

from camel.types import OpenAIBackendRole
from camel.memories import MemoryRecord
from camel.loaders import ChunkrReader
from camel.agents.chat_agent import ChatAgent
from camel.messages.base import BaseMessage

WORKING_DIRECTORY = os.path.abspath('workspace')
Buffer_DIRECTORY = os.path.abspath(f'{WORKING_DIRECTORY}/Buffer')
Extracted_DIRECTORY = os.path.abspath(f'{WORKING_DIRECTORY}/Extracted')
Papers_DIRECTORY = os.path.abspath(f'{WORKING_DIRECTORY}/Papers')
Database_DIRECTORY = os.path.abspath(f'{WORKING_DIRECTORY}/Database')

def update_content(add:str):
    with open('Content.csv','a',encoding='utf-8') as content:
        content.write(f'\n{add}')

def convert_to_text(claims,paper): # checked good
    # Reform into better text
    claims_list = [f"{ind + 1}.{claim.content}".strip() for ind, claim in enumerate(claims)]
    claims_text = '\n'.join(claims_list)
    data = f"""The paper is titled "{paper.title}" by {paper.author}. The claims in the paper are:\n{claims_text}"""
    return data

async def chunkr_read(filepath: str):
    nest_asyncio.apply()
    download_folder = Path(f"{Extracted_DIRECTORY}/{filepath}/")
    download_folder.mkdir(exist_ok=True)
    # Initializing an instance of ChunkrReader
    # This object will be used to submit tasks and manage document processing

    class SubChunkrReader(ChunkrReader):  # So that we do not need to use a protected variable
        def get_chunkr_obj(self):
            return self._chunkr

    chunkr_reader = SubChunkrReader()
    # Submitting a document processing task
    # Replace "local_data/example.pdf" with the path to your target document
    print(Fore.BLUE+ 'Calling chunkr to convert pdf to json')
    task_id_chunkr = await chunkr_reader.submit_task(file_path=f'{Papers_DIRECTORY}/{filepath}')
    # Get complete task data as dictionary
    chunkr_output = await chunkr_reader.get_task_output(task_id=task_id_chunkr)
    task = await chunkr_reader.get_chunkr_obj().get_task(task_id_chunkr)
    # Download all segment images
    print(Fore.WHITE + 'Texts Extracted. Now Downloading Pictures')
    for i, chunk in enumerate(task.output.chunks):
        for j, segment in enumerate(chunk.segments):
            if segment.image:  # Check if segment has an image
                # Download the image
                response = requests.get(segment.image)
                if response.status_code == 200:
                    # Save with descriptive filename
                    picture_name = f"chunk_{i}_segment_{j}_{segment.segment_type}.jpg"
                    pic_path = download_folder / picture_name
                    with open(pic_path, 'wb') as f:
                        f.write(response.content)
                    # Replace the url with the path to the downloaded jpg
                    chunkr_output = chunkr_output.replace(str(segment.image), str(pic_path))
    with open(f'{Extracted_DIRECTORY}/{filepath}/text.json', 'w') as f:
        json.dump(chunkr_output, f)
    json_data = json.loads(chunkr_output)
    chunks = json_data.get('output', {}).get('chunks', [])
    f.close()
    c = 0
    for i in chunks:
        c += 1
        with open(f'{Extracted_DIRECTORY}/{filepath}/{c}.json', 'w') as f:
            json.dump(i, f)

def extract_json_text(json_data):
    """
    递归提取JSON中所有text字段的值

    参数:
        json_data: 可以是字典、列表或其他JSON兼容的数据结构

    返回:
        list: 包含所有text值的列表
    """
    texts = []
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if key == 'text' and value is not None:  # 找到text键且值不为None
                texts.append(value)
            elif key == 'configuration':  # skip config
                continue
            else:  # 递归处理字典的值
                texts.extend(extract_json_text(value))

    elif isinstance(json_data, list):
        for item in json_data:
            texts.extend(extract_json_text(item))

    return ''.join(texts)

def extract_claim(context:str, memory:str):  # Return structured string
    model_backend = setup()
    system_message = \
        f"""Read through the content given below and extract all claims of facts. You should avoid using any pronouns. 
        For example: 'he', 'she', 'it' ,or 'this process', 'that statement'，'the process'...
        you must give explanations or names to all general concepts you mention.
        you must explain all initials if you can."""
    agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Concept Extract Agent",
            content=system_message,
        ),
        model=model_backend,
    )
    agent.memory.write_record(MemoryRecord(message = BaseMessage.make_user_message(
            role_name="User",
            meta_dict=None,
            content=memory),
        role_at_backend=OpenAIBackendRole.USER)
    )
    return agent.step(context,response_format=ListClaims).msgs[0].content

def extract_paper(context:str): # Return structured string
    model_backend = setup()
    system_message = \
        f"""You will be provided with first few section of a paper, 
        you should extract the the title and author of the paper provided."""
    agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Concept Extract Agent",
            content=system_message,
        ),
        model=model_backend,
    )
    return agent.step(context,response_format=Paper).msgs[0].content

async def analyse_paper_pdf(name, autoclean = False):
    # define model
    # extract paper using chunkr
    await chunkr_read(name)
    file_list = os.listdir(f'{Extracted_DIRECTORY}/{name}')
    # Save all in the folder named by the pdf name. split them into each chunk. #############Need Change
    # return the path of the folder
    texts_in_chunks = []
    for f in tqdm(file_list,'loading json files'):
        if f.endswith('.json'): # Take only json files
            with open(f'{Extracted_DIRECTORY}/{name}/{f}', 'r', encoding='utf-8') as json_file:
                texts_in_chunks.append(extract_json_text(json.loads(json_file.read())))
    paper = json.loads(extract_paper(''.join(texts_in_chunks))) # loads all text in to dict

    # Convert to paper obj
    paper = Paper(title=paper['title'], author=paper['author'], published_year=int(paper['published_year']))
    print(f'\nPaper details of {paper.title} by {paper.author} is extracted.\n')
    # Contains authors and title of the paper
    memo = ''
    relations = []
    claims = []
    for i in tqdm(range(len(texts_in_chunks)), 'processing chunks'):
        claims_list = extract_claim(texts_in_chunks[i], memo)
        memo = texts_in_chunks[i]
        claims_list = json.loads(claims_list)
        new_claims = []
        for n in claims_list['listofClaims']: # Converts into obj
            new_claims.append(Claim(content = n['content'])) # claims in the chunk
        claims.extend(new_claims)
    # Save all in json files
    # Save paper
    print('Saving')
    if not os.path.exists(f'{Buffer_DIRECTORY}/{name}'):
        os.mkdir(f'{Buffer_DIRECTORY}/{name}')# Create a folder for storing the paper

    with open(f'{Buffer_DIRECTORY}/{name}/all.txt', 'w', encoding='utf-8') as txf:
        txf.write(convert_to_text(claims,paper))

    with open(f'{Buffer_DIRECTORY}/{name}/paper.json', 'w') as jsf:
        json.dump(paper.model_dump(), jsf)
    # Save claims

    with open(f'{Buffer_DIRECTORY}/{name}/claims.json', 'w') as jsf:
        claims_dict = [claim.model_dump() for claim in claims]
        json.dump(claims_dict, jsf, indent=2)

    # Save relations
    with open(f'{Buffer_DIRECTORY}/{name}/relations.json', 'w') as jsf:
        re_dict = [relation.model_dump() for relation in relations]
        json.dump(re_dict, jsf, indent=2)

    if autoclean:
        shutil.rmtree(f'{Extracted_DIRECTORY}/{name}')

    update_content(name) # update content

def load_info(name): # checked good
    with open(f'{Buffer_DIRECTORY}/{name}/claims.json', 'r', encoding='utf-8') as f:
        claims = json.load(f)
    with open(f'{Buffer_DIRECTORY}/{name}/paper.json', 'r', encoding='utf-8') as f:
        paper = json.load(f)
    # Reform into better text
    claims_list = [f"{ind + 1}.{claim['content']}".strip() for ind, claim in enumerate(claims)]
    claims_text = '\n'.join(claims_list)
    data = f"""The paper is titled "{paper["title"]}" by {paper["author"]}. The claims in the paper are:\n{claims_text}"""
    return data, claims_list

class Librarian(ChatAgent):
    def __init__(self):
        super().__init__(system_message = """You are a helpful assistant to answer question,
         I will give you the Original Query and Retrieved Context,
        answer the Original Query based on the Retrieved Context,
        if you can't answer the question just say I don't know.""")  # inherit first
        from camel.embeddings import OpenAIEmbedding
        from camel.types import EmbeddingModelType
        embedding_instance = OpenAIEmbedding(model_type=EmbeddingModelType.TEXT_EMBEDDING_3_LARGE)
        from camel.storages import QdrantStorage
        storage_instance = QdrantStorage(
            vector_dim=embedding_instance.get_output_dim(),
            path=Database_DIRECTORY,
            collection_name="data",
        )
        from camel.retrievers import VectorRetriever

        self._retriever = VectorRetriever(embedding_model=embedding_instance,
                                             storage=storage_instance)
    def _vectorise(self, fpath):
        self._retriever.process(
            content=fpath,
        )
    def query(self,question, items = 3,similarity_threshold = 0.5):
        retrieved_info = self._retriever.query(query=question, top_k=items, similarity_threshold=similarity_threshold)
        listing = []
        for i in retrieved_info:
            listing.append(i['text'])
        info = '\n'.join(listing)
        response = self.step(f"""
        Question:{question}\nretrieved info(Sorted by similarity from high to low iof multiple lines):{info}
                            """)
        return response.msgs[0].content
    def prepare_data(self, autoclean=False):
        for name in os.listdir(Extracted_DIRECTORY):
            _, cl = load_info(name)
            for i in tqdm(cl,'vectorizing database'):
                self._vectorise(i)
            if autoclean:
                for i in tqdm(os.listdir(f'{Buffer_DIRECTORY}/{name}/'), 'auto-cleaning'):
                    os.unlink(f'{Buffer_DIRECTORY}/{name}/{i}')

def extract_all(autoclean = False): # Extract everything under Papers
    setup()
    # Check if the paper is extracted
    contents = np.loadtxt('Content.csv',dtype=str,delimiter=',',skiprows=1)
    import asyncio
    for i in os.listdir(Papers_DIRECTORY):
        if i not in contents:
            asyncio.run(analyse_paper_pdf(i))
    if autoclean:
        for i in os.listdir(Papers_DIRECTORY):
            os.unlink(i)






