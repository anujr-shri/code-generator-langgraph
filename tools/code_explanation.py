from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()

logger_inst = get_logger(__name__)
CHAT_MODEL_ID = "llama-3.1-8b-instant"

with open("code_explanation_prompt.txt", "r") as file:
    prompt = file.read()

template = PromptTemplate(
    template=prompt,
    input_variables=["code"]
)

chat_llm = ChatGroq(
    model=CHAT_MODEL_ID
)

def get_explanation(state):
    """Generates a natural language explanation for the code and updates response and chat history."""
    chain = template | chat_llm
    code = state["code"]
    try:
        response = chain.invoke({"code": code})
        logger_inst.info(f"Generated Explanation for the code")
        combined = "\n".join([state["code"], str(response.content)])
        return {"explain": response.content, "response": combined, "chat_history": [AIMessage(combined)]}
    
    except Exception as e:
        logger_inst.info(f"An Exception has occured while genrating an explnation messgae: {e}")
        return {"explain": ""}