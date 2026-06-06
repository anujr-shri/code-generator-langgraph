from langchain_chroma import Chroma
from langchain_huggingface import (
    HuggingFaceEndpointEmbeddings, 
    HuggingFaceEndpoint, 
    ChatHuggingFace
)
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from tools.preprocess_document import text_splitting
from utils.logger import get_logger
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHAT_MODEL_ID = "google/gemma-4-31B-it"
BATCH_SIZE = 256
MAX_NEW_TOKENS = 512

retriver_logger = get_logger(__name__)

# --- Prompts Setup ---
with open("retriver_prompt.txt", "r") as file:
    answer_template = file.read()

with open("rewrite_query.txt", "r") as file:
    rewrite_prompt = file.read()

# --- Answer Generation Format ---
class CodeGeneration(BaseModel):
    code: str = Field(description="Generated Code")
    flag: bool = Field(description="Set this to true if answer cannot be found or reasonably inferred from the context")
    

# --- Prompts Template ---
rewrite_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI agent whose task is to resolve the user query given previous conversation"),
    ("human", rewrite_prompt) 
])

answer_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI agent"),
    ("human", answer_template)
])



# --- Models Setup ---
chat_endpoint = HuggingFaceEndpoint(
    repo_id=CHAT_MODEL_ID,
    task="text-generation",
    max_new_tokens=MAX_NEW_TOKENS
) # type: ignore

chat_llm = ChatHuggingFace(llm=chat_endpoint)

def create_embedding_model(model_name: str):
    return HuggingFaceEndpointEmbeddings(model=model_name)


embedding_model = create_embedding_model(EMBEDDING_MODEL_NAME)

retriver = Chroma(
    collection_name="code_generator",
    persist_directory="./chroma_code_genrator_db",
    embedding_function=embedding_model
)

# --- Core Functions ---
def store_document(docs_data, batch_size=BATCH_SIZE):
    ids = [f"id{i}" for i in range(len(docs_data))]
    batches = list(range(0, len(docs_data), batch_size))

    retriver_logger.info(f"Divided the {len(docs_data)} chunks into {len(batches)} batches")
    
    for i in range(len(batches)):
        start_idx = batches[i]
        end_idx = min(batches[i] + batch_size, len(docs_data))

        retriver.add_documents(
            documents=docs_data[start_idx: end_idx],
            ids=ids[start_idx: end_idx],
        )
        retriver_logger.info(f"Successfully stored batch number {i + 1} from batches of docs data")

    retriver_logger.info(f"Successfully embedded and stored {len(docs_data)} document chunks.")

def load_data():
    split_data = text_splitting()
    store_document(split_data)

def semantic_search(query: str, top_k: int = 3):
    result = retriver.similarity_search(
        query=query,
        k=top_k
    )
    response = "\n\n".join([docs.page_content for docs in result])
    retriver_logger.info(f"Successfully extracted the knowledge from retriever")

    return response

def rewrite_query(state):
    chain = rewrite_template | chat_llm
    
    response = chain.invoke({
        "chat_history": state["chat_history"],
        "question": state["query"]  
    })

    retriver_logger.info(f"Rewrite The new optimized query is {response.content}")

    return {"optimize_query": response.content, "chat_history": [HumanMessage(response.content)]}

def generate_answer(state):
    chain = answer_prompt_template | chat_llm 

    semantic_serch_result = semantic_search(state["optimize_query"])

    response = chain.invoke({
        "pdf_knowledge": semantic_serch_result,
        "user_input": state["optimize_query"]
    })
    retriver_logger.info("Model Inference is Completed")
    flag = False
    if response.content == "I cannot find the answer for this query":
        flag = True
    return {"code": response.content, "flag": flag}