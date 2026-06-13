import ast
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
FALLBACK_MODEL_ID = "llama-3.1-8b-instant"
CHAT_MODEL_ID = "Qwen/Qwen3-Coder-Next"
logger_inst = get_logger(__name__)

with open("test_call_prompt.txt", "r") as file:
    test_call_txt = file.read()

test_call_template = PromptTemplate(
    template=test_call_txt,
    input_variables=["code"]
)

fallback_model = ChatGroq(
    model=FALLBACK_MODEL_ID
)

llm = HuggingFaceEndpoint(
    repo_id=CHAT_MODEL_ID,
    task="text-generation",
    max_new_tokens=512
)# type: ignore

chat_model = ChatHuggingFace(llm=llm)

chat_llm = chat_model.with_fallbacks([fallback_model])

def generate_test_call(code: str):
    """Generates a test call snippet for the given code using the LLM."""
    try:
        chain = test_call_template | chat_llm
        llm_response = chain.invoke({"code": code})
        logger_inst.info(f"Generated Test Call required to run the code test call : {llm_response.content}")
        return llm_response.content
    except Exception as e:
        logger_inst.error(f"Error while generating test call {e}")
        return ""

def parse_code(code_str: str):
    """Strips the markdown code fence from the generated code string."""
    logger_inst.info(f"Parse the code sucessfully")
    return code_str[10: len(code_str)-4]


def compilation_step(state):
    """Parses, compiles, and executes the generated code; returns feedback and execution status."""
    code = parse_code(state["code"])
    test_call = generate_test_call(code)
    final_code = code + "\n" + test_call # type: ignore
    feedback = []
    executable = False

    try:
        tree = ast.parse(final_code, mode='exec')
        compile_code = compile(tree, filename="<string>", mode="exec")
        namespace = {}
        exec(compile_code, namespace)
        executable = True
        error_msg = "Code Excuted Sucessfully"

    except SyntaxError as e:
        error_msg = f"on line {e.lineno} there is error {e.msg}"

    except NameError as e:
        error_msg = f"[NameError]: {e}"

    except TypeError as e:
        error_msg = f"[TypeError]: {e}"

    except Exception as e:
        error_msg = f"[RuntimeError]: {e}"
    
    feedback.append(error_msg)
    logger_inst.info(f"Excuted the code and output feedback is {feedback}")
    return {"feedback": feedback, "exec": executable}


def cannot_generate(state):
    """Returns a fallback message when the query is outside the tool's scope."""
    return {
        "response": "Sorry I can Not help you with query this tool can only generate the basic python code"
    }
