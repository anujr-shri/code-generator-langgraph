from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from utils.logger import get_logger
from dotenv import load_dotenv
import os

# Configuration
CHAT_MODEL_ID = "gemini-2.5-flash"
FALLBACK_MODEL_ID = "llama-3.1-8b-instant"
TEMPERATURE = 0.3
API_KEY = os.getenv("GOOGLE_API_KEY")
logger_inst = get_logger(__name__)
load_dotenv()

# --- model setup ---
chat_llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL_ID,
    temperature=TEMPERATURE,
    google_api_key=API_KEY
)

fallback_model = ChatGroq(
    model=FALLBACK_MODEL_ID
)

chat_model = chat_llm.with_fallbacks([fallback_model])

# --- prompt template ---
with open("query_type.txt", "r") as file:
    prompt = file.read()

template = PromptTemplate(
    template=prompt,
    input_variables=["query", "chat_history"]
)

template_general = PromptTemplate(
    template="You are AI agent answer the user query: {query} on the basis of past conversation: {chat_history} if needed and response with appropriate answer. Answer Sorry I can not help you if you can not answer the query",
    input_variables=["query", "chat_history"]
)

# --- Core function ---
def know_query_type(state):
    """Classifies the user's query type using an LLM chain based on the input and chat history."""
    chain = template | chat_model
    query = state["query"]
    chat_history = state["chat_history"]
    try:
        response = chain.invoke({
            "query": query,
            "chat_history": chat_history
        })
        logger_inst.info(f"Query type is determined it is {response.content}")
        return {"query_type": response.content, "chat_history": [HumanMessage(query)]}

    except Exception as e:
        logger_inst.error(f"Error while knowing query type {e}")
        return {"query_type": "general", "chat_history": [HumanMessage(query)]}
    

def get_general_response(state):
    """Generates a conversational AI response for general queries using the chat history."""
    chain = template_general | chat_model

    query = state["query"]
    chat_history = state["chat_history"]

    try:
        response = chain.invoke({
        "query": query,
        "chat_history": chat_history
        })
        logger_inst.info(f"Response for genral query is {response.content}")
        return {"response": response.content, "chat_history": [AIMessage(response.content)]}
    
    except Exception as e:
        logger_inst.error(f"Error while genearting genral response {e}")
        return {"response": "Sorry I can not help you", "chat_history": [AIMessage(response.content)]}