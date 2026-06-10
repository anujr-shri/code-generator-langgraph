from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from tools.retriver_tool import rewrite_query, generate_answer
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from tools.compilation_check import compilation_step, cannot_generate
from tools.code_explanation import get_explanation
from tools.type_of_query import know_query_type, get_general_response


# --- utility function ---
def checker_function(state):
    """Routes to 'cannot_generate' if flag is True, else 'compilation_step'."""
    if state["flag"] == True:
        return "cannot_generate"
    return "compilation_step"

def iterative_function(state):
    """Routes to 'get_explanation' if count > 3 or exec succeeded, else loops back to 'generate_answer'."""
    if state["count"] > 3:
        return "get_explanation"
    if state["exec"] == True:
        return "get_explanation"
    return "generate_answer"

def general_response(state):
    """Routes to 'get_general_response' for general queries, else 'rewrite_query'."""
    if state["query_type"] == "general":
        return "get_general_response"
    return "rewrite_query"


# --- Maintain Persistance ---
checkpointer = InMemorySaver()


# --- Schema For state ---
class CodeGenerationSchema(TypedDict):
    chat_history: Annotated[list[BaseMessage], add_messages]
    query: str
    optimize_query: str
    query_type: str
    response: str
    code: str
    explain: str
    flag: bool
    feedback: Annotated[list[str], add_messages]
    exec: bool
    count: int

# --- add Node in graph ---
graph = StateGraph(CodeGenerationSchema)
graph.add_node(know_query_type)
graph.add_node(get_general_response)
graph.add_node(rewrite_query)
graph.add_node(generate_answer)
graph.add_node(compilation_step)
graph.add_node(cannot_generate)
graph.add_node(get_explanation)


# --- add Edges ---
graph.add_edge(START, "know_query_type")
graph.add_conditional_edges("know_query_type", general_response)
graph.add_edge("get_general_response", END)
graph.add_edge("rewrite_query", "generate_answer")
graph.add_conditional_edges("generate_answer", checker_function)
graph.add_conditional_edges("compilation_step", iterative_function)
graph.add_edge("get_explanation", END)
graph.add_edge("cannot_generate", END)


# --- Compilation ---
workflow = graph.compile(checkpointer=checkpointer)


def response_generator(user_query, thread_id):
    """Invokes the workflow and returns the final response string."""
    CONFIG = {"configurable": {"thread_id": thread_id}}
    
    chunks = workflow.invoke(
        {
            "query": user_query,
            "count": 0,
            "feedback": [],
        }, # type: ignore
        config=CONFIG, # type: ignore
        stream_mode="values"
    ) # type: ignore
    
    return chunks["response"]

def get_all_messages(thread_id):
    """Returns chat history for a thread as a list of role/message dicts."""
    CONFIG: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    history = workflow.get_state(CONFIG)
    response_dict = []
    for msg in history.values["chat_history"]:
        if isinstance(msg, HumanMessage):
            response_dict.append({"role": "user", "message": msg.content})
        else:
            response_dict.append({"role": "assistant", "message": msg.content})
    
    return response_dict
    
