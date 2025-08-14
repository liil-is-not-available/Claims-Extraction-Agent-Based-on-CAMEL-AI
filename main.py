from Engineer import Engineer
from Librarian import extract_all, Librarian

if __name__=='__main__':
    query_only = True  # Set False if you have new to
    if not query_only:
        extract_all()
        archiver = Librarian()
        archiver.prepare_data()
    eng = Engineer(interactable=True, temperature=1.2)
    eng.run('Is there a way to increase hallucination of ai to enhance creativity?')
