from langgraph.graph import StateGraph, START, END
from tools.retriver_tool import rewrite_query, generate_answer
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage
from tools.compilation_check import compilation_step, cannot_generate
from tools.code_explanation import get_explanation
import operator

# --- condtional function ---
def checker_function(state):
    if state["flag"] == True:
        return "cannot_generate"
    return "compilation_step"

def iterative_function(state):
    if state["count"] > 3:
        return "get_explanation"
    if state["exec"] == True:
        return "get_explanation"
    return "generate_answer"

# --- Maintain Persistance ---

# --- Schema For state ---
class CodeGenerationSchema(TypedDict):
    chat_history: Annotated[list[BaseMessage], operator.add]
    query: str
    optimize_query: str
    query_type: str
    response: str
    code: str
    explain: str
    flag: bool
    feedback: Annotated[list[str], operator.add]
    exec: bool
    count: int

# --- add Node in graph ---
graph = StateGraph(CodeGenerationSchema)
graph.add_node(rewrite_query)
graph.add_node(generate_answer)
graph.add_node(compilation_step)
graph.add_node(cannot_generate)
graph.add_node(get_explanation)


# --- add Edges ---
graph.add_edge(START, "rewrite_query")
graph.add_edge("rewrite_query", "generate_answer")
graph.add_conditional_edges("generate_answer", checker_function)
graph.add_conditional_edges("compilation_step", iterative_function)
graph.add_edge("get_explanation", END)
graph.add_edge("cannot_generate", END)

# --- Compilation ---
workflow = graph.compile()

while True:
    query = input("Enter The Query: ")
    final_state = workflow.invoke({
        "query": query,
        "count": 0
    })# type: ignore

    print(final_state["code"])
    print(final_state["explain"])