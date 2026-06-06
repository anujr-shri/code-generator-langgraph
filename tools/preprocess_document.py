from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import get_logger
import os

# --- Configuartion ---
docs_logger = get_logger(__name__)
base_dir = os.path.dirname(os.path.abspath(__file__))
knowledge_path = os.path.join(base_dir, "knowledge_base", "english_python_data.txt")

# --- text loading ---
def load_txt_data():
    loader = TextLoader(knowledge_path, encoding="utf-8")
    document = loader.load()
    docs_logger.info(f"Sucessfully Loaded the data from {knowledge_path}")

    return document

# --- text splitting ---
def text_splitting(chunk_size=750, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    docs_data = load_txt_data()

    split_docs_data = text_splitter.split_documents(docs_data)
    docs_logger.info(f"Sucessfully divide the data into chunks {len(split_docs_data)}")

    return split_docs_data


