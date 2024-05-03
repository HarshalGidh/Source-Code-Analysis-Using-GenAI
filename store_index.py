from src.helper import repo_ingestion, load_repo, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os

load_dotenv()

# GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embedding()


#storing vector in choramdb
vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')
vectordb.persist()