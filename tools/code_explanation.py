from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()

logger_inst = get_logger(__name__)
CHAT_MODEL_ID = "llama-3.1-8b-instant"
FALLBACK_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

with open("code_explanation_prompt.txt", "r") as file:
    prompt = file.read()

template = PromptTemplate(
    template=prompt,
    input_variables=["code"]
)

chat_llm = ChatGroq(
    model=CHAT_MODEL_ID
)

llm = HuggingFaceEndpoint(
    repo_id=FALLBACK_MODEL_ID,
    task="text-generation",
    max_new_tokens=512
) # type: ignore

fallback_model = ChatHuggingFace(llm=llm)

chat_model = chat_llm.with_fallbacks([fallback_model])

def get_explanation(state):
    """Generates a natural language explanation for the code and updates response and chat history."""
    chain = template | chat_model
    code = state["code"]
    try:
        response = chain.invoke({"code": code})
        logger_inst.info(f"Generated Explanation for the code")
        combined = "\n".join([state["code"], str(response.content)])
        return {"explain": response.content, "response": combined, "chat_history": [AIMessage(combined)]}
    
    except Exception as e:
        logger_inst.info(f"An Exception has occured while genrating an explnation messgae: {e}")
        return {"explain": ""}
