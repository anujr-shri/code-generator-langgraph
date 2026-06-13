from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from utils.logger import get_logger
from dotenv import load_dotenv


# General Prompt
general_prompt = """You are a helpful AI assistant. Use the conversation history below to maintain context and answer the user's current query accurately and helpfully.

Conversation history:
{chat_history}

User query:
{query}

Instructions:
- Use the conversation history only if it provides relevant context for understanding or answering the current query.
- Answer directly and concisely, using clear language appropriate to the topic (code, explanations, writing, analysis, etc.).
- If the query depends on missing information from the conversation history (e.g., refers to "it," "that," or a previous step that wasn't actually discussed), ask a brief clarifying question instead of guessing.
- If the query is ambiguous, outside your capabilities, or cannot be reasonably answered even with the given context, respond with exactly: "Sorry, I can't help with that"
- Do not fabricate information about prior turns that isn't present in the conversation history..
"""

# Configuration
FALLBACK_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CHAT_MODEL_ID = "llama-3.1-8b-instant"
TEMPERATURE = 0.3
logger_inst = get_logger(__name__)
load_dotenv()

# --- model setup ---

chat_llm = ChatGroq(
    model=CHAT_MODEL_ID
)

fallback_endpoint = HuggingFaceEndpoint(
    repo_id=FALLBACK_MODEL_ID,
    task="text-generation",
    max_new_tokens=512
) # type: ignore

fallback_model = ChatHuggingFace(llm=fallback_endpoint)

chat_model = chat_llm.with_fallbacks([fallback_model])

# --- prompt template ---
with open("query_type.txt", "r") as file:
    prompt = file.read()

template = PromptTemplate(
    template=prompt,
    input_variables=["query", "chat_history"]
)

template_general = PromptTemplate(
    template=general_prompt,
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
