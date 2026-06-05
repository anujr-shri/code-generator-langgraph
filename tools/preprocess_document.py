from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
knowledge_path = os.path.join(base_dir, "knowledge_base\english_python_data.txt")

def load_txt_data():
    loader = TextLoader(knowledge_path, encoding="utf-8")
    document = loader.load()

    return document

def text_splitting(chunk_size=100, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    docs_data = load_txt_data()

    split_docs_data = text_splitter.split_documents(docs_data)

    return split_docs_data


