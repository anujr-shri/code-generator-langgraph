import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import re
import uuid
from tools.main import response_generator, get_all_messages
from frontend_utils import generate_chat_name


st.set_page_config(
    page_title="Code Generator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Sidebar thread buttons */
    .stButton > button {
        width: 100%;
        text-align: left;
        background: transparent;
        border: 1px solid #333;
        border-radius: 6px;
        color: #ccc;
        padding: 6px 10px;
        margin-bottom: 4px;
        font-size: 0.85rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stButton > button:hover {
        background: #1e1e2e;
        border-color: #555;
        color: #fff;
    }
    /* New chat button stands out */
    div[data-testid="stSidebar"] div:first-child .stButton > button {
        background: #2563eb;
        border-color: #2563eb;
        color: white;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] div:first-child .stButton > button:hover {
        background: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "messages": [],
        "thread_id": str(uuid.uuid4()),
        "chat_thread": [],
        "one_time": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def render_message(content: str):
    """
    Splits a response on fenced code blocks and renders each part correctly.
    Plain text → st.markdown | Code blocks → st.code with language
    """
    pattern = r"```(\w+)?\n([\s\S]*?)```"
    parts = re.split(pattern, content)

    i = 0
    while i < len(parts):
        if i % 3 == 0:
            text = parts[i].strip()
            if text:
                st.markdown(text)
        elif i % 3 == 1:
            lang = parts[i] or "python"
        elif i % 3 == 2:
            st.code(parts[i], language=lang)
        i += 1



with st.sidebar:
    st.title("💬 Chats")
    st.divider()

    if st.button("＋  New Chat", key="new_chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.one_time = True
        st.rerun()

    st.divider()

    past_threads = [t for t in reversed(st.session_state.chat_thread)
                    if t != st.session_state.thread_id]

    if past_threads:
        st.caption("Recent")
        for thread in past_threads:
            label = thread if len(thread) < 35 else thread[:32] + "…"
            if st.button(label, key=f"thread_{thread}"):
                st.session_state.thread_id = thread
                st.session_state.messages = get_all_messages(thread)
    else:
        st.caption("No previous chats yet.")

    # Register current thread
    if st.session_state.thread_id not in st.session_state.chat_thread:
        st.session_state.chat_thread.append(st.session_state.thread_id)



st.title("⚡ Code Generator")
st.caption("Powered by LangGraph · Self-correcting agentic pipeline")
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_message(msg["message"])

user_query = st.chat_input("Ask me to write, fix, or explain code…")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "message": user_query})

    if st.session_state.one_time:
        new_name = generate_chat_name(user_query, st.session_state.chat_thread)
        idx = st.session_state.chat_thread.index(st.session_state.thread_id)
        st.session_state.thread_id = new_name
        st.session_state.chat_thread[idx] = new_name
        st.session_state.one_time = False

    # Generate response
    thread_id = st.session_state.thread_id
    full_response = ""

    with st.chat_message("assistant"):
        with st.spinner("Generating…"):
            try:
                full_response = response_generator(user_query, thread_id)
                render_message(full_response)
            except Exception as e:
                full_response = "Sorry, I encountered an error processing that request."
                st.error(f"Error: {e}")
                st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "message": full_response})

