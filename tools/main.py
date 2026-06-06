from langgraph.graph import StateGraph, START, END
from tools.retriver_tool import rewrite_query, generate_answer
from typing import TypedDict, Annotated
from langgraph.cache.memory import InMemoryCache
import operator

# Please Change the prompt for retriver promprt

# --- Maintain Persistance ---

# --- Schema For state ---
class CodeGenerationSchema(TypedDict):
    chat_history: Annotated[list[str], operator.add]
    query: str
    optimize_query: str
    response: str

# --- add Node in graph ---
graph = StateGraph(CodeGenerationSchema)
graph.add_node(rewrite_query)
graph.add_node(generate_answer)


# --- add Edges ---
graph.add_edge(START, "rewrite_query")
graph.add_edge("rewrite_query", "generate_answer")
graph.add_edge("generate_answer", END)

# --- Compilation ---
workflow = graph.compile()

while True:
    query = input("Enter Your Query: ")
    if (query == -1):
        break
    result = workflow.invoke({
        "chat_history":[],
        "query": query
    }) # type: ignore
    print(result["response"])