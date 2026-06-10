from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from utils.logger import get_logger
from dotenv import load_dotenv

# configuration
load_dotenv()
model_id="google/gemma-4-31B-it"
log = get_logger(__name__)

# --- Prompt ---
prompt = """
You are a chat title generator. Output only a short chat title (2-5 words). No punctuation, no explanation, nothing else.
Never output a title that already exists in the used titles list.

Used titles: {chat_history}
User message: {query}

"""

# --- model Setup ---
chat_endpoint = HuggingFaceEndpoint(
    repo_id=model_id,
    task="text-generation",
    max_new_tokens=256
)# type: ignore

model = ChatHuggingFace(
    llm=chat_endpoint
)

# --- Preparing template ---
template = PromptTemplate(
    template=prompt,
    input_variables=["query", "chat_history"]
)

def generate_chat_name(query: str, chat_history: list[str]):
    chain = template | model

    response = chain.invoke({
        "query": query,
        "chat_history": chat_history
    })
    log.info(f"New Chat name is {response.content}")

    return response.content