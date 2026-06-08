from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from utils.logger import get_logger
import os

load_dotenv()

# configuration
logger_inst = get_logger(__name__)
CHAT_MODEL_ID = "llama-3.1-8b-instant"

# --- Prompt Template ---
with open("code_explanation_prompt.txt", "r") as file:
    prompt = file.read()

template = PromptTemplate(
    template=prompt,
    input_variables=["code"]
)

# --- model setup ---
chat_llm = ChatGroq(
    model=CHAT_MODEL_ID
)

def get_explanation(state):
    chain = template | chat_llm

    code = state["code"]
    try:
        response = chain.invoke({
            "code": code
        })
        logger_inst.info(f"Generated Explanation for the code")
        combined = "\n".join([state["code"], str(response.content)])
        return {"explain": response.content, "response": combined, "chat_history": [combined]}
    
    except Exception as e:
        logger_inst.info(f"An Exception has occured while genrating an explnation messgae: {e}")
        return {"explain": ""}

    
    

    

