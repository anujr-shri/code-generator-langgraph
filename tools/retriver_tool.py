from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from tools.preprocess_document import text_splitting
from utils.logger import get_logger
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uuid
import os

load_dotenv()

# Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHAT_MODEL_ID = "gemini-2.5-flash"
FALLBACK_MODEL_ID = "llama-3.1-8b-instant"
TEMPERATURE = 0.3
BATCH_SIZE = 256
API_KEY = os.getenv("GOOGLE_API_KEY")

retriver_logger = get_logger(__name__)

# --- Rewriting Query Schema ---
class RewritingSchema(BaseModel):
    optimize_query: str = Field(description="Rewritten search query")

parser = PydanticOutputParser(pydantic_object=RewritingSchema)

# --- Prompts Setup ---
with open("retriver_prompt.txt", "r") as file:
    answer_template = file.read()

with open("rewrite_query.txt", "r") as file:
    rewrite_prompt = file.read()
    

# --- Prompts Template ---
rewrite_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI agent whose task is to resolve the user query given previous conversation \n [Output Format]: {format_instruction}"),
    ("human", rewrite_prompt) 
])

new_rewrite_template = rewrite_template.partial(
    format_instruction = parser.get_format_instructions()
)

answer_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI agent"),
    ("human", answer_template)
])

# --- Models Setup ---
chat_llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL_ID,
    temperature=TEMPERATURE,
    google_api_key=API_KEY
)

fallback_model = ChatGroq(
    model=FALLBACK_MODEL_ID
)# type: ignore

chat_model = chat_llm.with_fallbacks([fallback_model])

def create_embedding_model(model_name: str):
    """Initializes and returns a HuggingFace embedding model for feature extraction."""
    return HuggingFaceEndpointEmbeddings(model=model_name, task="feature-extraction")

embedding_model = create_embedding_model(EMBEDDING_MODEL_NAME)

retriver = Chroma(
    collection_name="code_generator",
    persist_directory="./chroma_code_genrator_db",
    embedding_function=embedding_model
)

# --- Core Functions ---
def store_document(docs_data, batch_size=BATCH_SIZE):
    """Batches and stores the preprocessed document chunks along with unique IDs into Chroma DB."""
    ids = [str(uuid.uuid4()) for _ in range(len(docs_data))]
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
    """Triggers document text splitting and coordinates storing the chunks into the vector store."""
    split_data = text_splitting()
    store_document(split_data)

def semantic_search(query: str, top_k: int = 3):
    """Performs a similarity search in Chroma DB and joins the top matching page contents into a string."""
    result = retriver.similarity_search(
        query=query,
        k=top_k
    )
    response = "\n\n".join([docs.page_content for docs in result])
    retriver_logger.info(f"Successfully extracted the knowledge from retriever")

    return response

def rewrite_query(state):
    """Optimizes the user query using conversation history and returns a structured, parsed rewrite."""
    chain = new_rewrite_template | chat_model | parser
    
    try:
        response = chain.invoke({
            "chat_history": state["chat_history"],
            "question": state["query"]  
        })
        retriver_logger.info(f"Rewrite The new optimized query is {response.optimize_query}")
        return {"optimize_query": response.optimize_query}
    
    except Exception as e:
        retriver_logger.error(f"Error while Rewriting query {e}")
        return {"optimize_query": state["query"]}

    
def generate_answer(state):
    """Generates code output using retrieved vector knowledge, the optimized query, and previous feedback."""
    chain = answer_prompt_template | chat_model
    previous_attempt = state["feedback"]

    semantic_serch_result = semantic_search(state["optimize_query"])
    try:
        response = chain.invoke({
            "pdf_knowledge": semantic_serch_result,
            "user_input": state["optimize_query"],
            "feedback": previous_attempt
        })

        flag = False
        if response.content == "I cannot find the answer for this query":
            flag = True
        count = state["count"] + 1
        retriver_logger.info("Model Inference is Completed")
        return {"code": response.content, "flag": flag, "count": count}
    
    except Exception as e:
        retriver_logger.error(f"Error while genrating code {e}")
        return {"code": "", "flag": True, "count": state["count"] + 1}