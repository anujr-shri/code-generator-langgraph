from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from utils.logger import get_logger
from dotenv import load_dotenv

# Configuration
CHAT_MODEL_ID = "llama-3.1-8b-instant"
logger_inst = get_logger(__name__)
load_dotenv()

# --- output schema ---
class CheckResponseSchema(BaseModel):
    response: str = Field("Response for user query")
    general_query: Literal["genral_query", "code_query"]

# --- model setup ---
chat_llm = ChatGroq(
    model=CHAT_MODEL_ID
)

# --- prompt template ---


