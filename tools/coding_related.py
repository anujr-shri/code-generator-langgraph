from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from utils.logger import get_logger
from dotenv import load_dotenv

# Configuration
load_dotenv()
with open("llm_query.txt", "r") as file:
    prompt = file.read()

logger_inst = get_logger(__name__)
model_id = "openai/gpt-oss-20b"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# --- model Setup ---
llm_endpoint = HuggingFaceEndpoint(
    repo_id=model_id,
    task="text-generation"
)# type: ignore

chat_llm = ChatHuggingFace(
    llm=llm_endpoint,
    temperature=0.2
)

fallback_model = ChatGroq(
    model=FALLBACK_MODEL
)

chat_model = chat_llm.with_fallbacks([fallback_model])

# --- template ---
template = PromptTemplate(template=prompt, input_variables=["chat_history", "query_type", "query"])

def direct_llm(state):
    query = state["query"]
    chat_history = state["chat_history"]
    query_type = state["query_type"]

    chain = template | chat_model

    try:
        response = chain.invoke({
            "query": query,
            "chat_history": chat_history,
            "query_type": query_type
        })
        logger_inst.info("Get The response from direact llm node")
        return {"response": response.content, "chat_history": [AIMessage(response.content)]}
    except Exception as e:
        logger_inst.error("An Exception has happend in direact_llm node [Error]: {e}")
        return {"response": "Sorry I can not help you with this query",
                "chat_history": ["Sorry I can not help you with this query"]}
