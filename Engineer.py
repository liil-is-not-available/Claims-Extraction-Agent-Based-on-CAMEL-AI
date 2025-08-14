from camel.memories import MemoryRecord
from camel.types import OpenAIBackendRole
from camel.toolkits.human_toolkit import HumanToolkit
from camel.agents.chat_agent import ChatAgent
from camel.messages.base import BaseMessage

from Librarian import Librarian
from Setup import setup

import json
import os
from colorama import Fore

WORKING_DIRECTORY = os.path.abspath('workspace')
Buffer_DIRECTORY = os.path.abspath(f'{WORKING_DIRECTORY}/Buffer')
Extracted_DIRECTORY = os.path.abspath(f'{WORKING_DIRECTORY}/Extracted')
Papers_DIRECTORY = os.path.abspath(f'{WORKING_DIRECTORY}/Papers')
Database_DIRECTORY = os.path.abspath(f'{WORKING_DIRECTORY}/Database')

def ask_query(question:str) -> str:
    """
    Use this tool to query one question.
    This tool match your questions from the database first.
    Then the top 3 results are summarized by an agent
    You will receive summarized answer as the return
    param question: str
    return str:
    """
    finder = Librarian()  # create a Librarian
    message = finder.query(question)
    print(f'A query has been made: {question}')
    print(f'Answers: {message}')
    return message

class Engineer(ChatAgent):
    def __init__(self, memo = '', temperature=0.8, interactable=False):
        super().__init__(system_message=
            """You are a creative engineer, 
            **Capabilities**:You can ask one question each time using the ask_query() tool to gather related information.
            **Requirements**: 
            1.You must use send_message_to_user() tool to inform the user about what you are doing.
            2.You must use ask_query() when you have any progress.You should use the information to further
             develop or improve your idea.
            3. You must act actively on the user's feedback. 
                You mustUnderstand the question first, then make suitable queries and improve your answers
            """,
            model=setup())
        if memo:
            self.memory.write_record(MemoryRecord(message=BaseMessage.make_user_message(
                role_name='user',
                content = memo
            ),
                role_at_backend=OpenAIBackendRole.USER
            ))
        self.model_backend.model_config_dict['temperature'] = temperature
        human = [HumanToolkit().send_message_to_user]
        if interactable:
            human.append(HumanToolkit().ask_human_via_console)
        self.add_tools([*human,ask_query])

    def run(self, task):
        for i in range(50):  # Max conversation 50
            response = self.step(task)
            print(Fore.GREEN + '#' * 50 + ' A very nice splitting ' + '#' * 50)  # 华丽的分割线
            print(Fore.BLUE + 'Here is the response from the Engineer. Any comments?')
            print(Fore.YELLOW + response.msgs[0].content)
            print(Fore.BLUE + 'Press Enter to Next Loop. Will terminate if inputs empty')
            userinput = input(Fore.WHITE + '>>>').strip()
            if not userinput:
                print(Fore.WHITE + 'Terminated')
                break
            else:
                task = userinput

