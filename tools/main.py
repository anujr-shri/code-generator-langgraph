from langgraph.graph import StateGraph, START, END
from tools.retriver_tool import rewrite_query, generate_answer
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage
from tools.compilation_check import compilation_step, cannot_generate
import operator

def checker_function(state):
    if state["flag"] == True:
        return "cannot_generate"
    return "compilation_step"
# --- Maintain Persistance ---

# --- Schema For state ---
class CodeGenerationSchema(TypedDict):
    chat_history: Annotated[list[BaseMessage], operator.add]
    query: str
    optimize_query: str
    response: str
    code: str
    flag: Literal[True, False]

# --- add Node in graph ---
graph = StateGraph(CodeGenerationSchema)
graph.add_node(rewrite_query)
graph.add_node(generate_answer)
graph.add_node(compilation_step)
graph.add_node(cannot_generate)


# --- add Edges ---
graph.add_edge(START, "rewrite_query")
graph.add_edge("rewrite_query", "generate_answer")
graph.add_conditional_edges("generate_answer", checker_function)
graph.add_edge("compilation_step", END)
graph.add_edge("cannot_generate", END)

# --- Compilation ---
workflow = graph.compile()

