# Updated app.py with file upload and flowing chat context

import streamlit as st
from support_agent import run_customer_support
import os
from dotenv import load_dotenv
import tempfile
# Initialize default state
if "query" not in st.session_state:
    st.session_state["query"] = ""

if "clear_query" not in st.session_state:
    st.session_state["clear_query"] = False

# Safely clear query if flagged
if st.session_state.clear_query:
    st.session_state["query"] = ""
    st.session_state["clear_query"] = False

load_dotenv()

st.set_page_config(page_title="Multilingual AI Support", layout="wide")

# Custom CSS for light/dark adaptive UI
st.markdown("""
<style>
    .chat-message {
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e8f0fe;
        color: black;
        border-left: 4px solid #4285f4;
    }
    .chat-message.agent {
        background-color: #f1f3f4;
        color: black;
        border-left: 4px solid #34a853;
    }
    @media (prefers-color-scheme: dark) {
        .chat-message.user {
            background-color: #2b2b2b;
            color: white;
        }
        .chat-message.agent {
            background-color: #1e1e1e;
            color: white;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🌍 Multilingual AI Support</h1>", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    display_language = st.selectbox(
        "Force response language:",
        ["Auto (detect)", "English", "Spanish", "French", "German", "Bengali", "Italian", "Portuguese"]
    )
    language_mapping = {
        "Auto (detect)": None,
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Bengali": "bn",
        "Italian": "it",
        "Portuguese": "pt"
    }
    st.markdown("---")
    st.markdown("Upload a file (PDF, TXT)")
    uploaded_file = st.file_uploader("Attach a file", type=["pdf", "txt"])

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "agent"
    with st.container():
        st.markdown(f"<div class='chat-message {role}'><strong>{role.capitalize()}:</strong><br>{msg['content']}</div>", unsafe_allow_html=True)

# Message input
query = st.text_area("Your message:", height=100, key="query")

# Send button logic
if st.button("Send"):
    if not query.strip():
        st.warning("Please enter a message.")
    else:
        # Store file temporarily if uploaded
        file_path = None
        if uploaded_file:
            suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

        # Append user message
        st.session_state.messages.append({"role": "user", "content": query})

        # Prepare chat history in model format
        chat_history = []
        for i in range(0, len(st.session_state.messages) - 1, 2):
            if i + 1 < len(st.session_state.messages):
                chat_history.append({
                    "user": st.session_state.messages[i]["content"],
                    "agent": st.session_state.messages[i+1]["content"]
                })

        # Run support function
        with st.spinner("Thinking..."):
            result = run_customer_support(
                query=query,
                force_language=language_mapping[display_language],
                chat_history=chat_history,
                file_path=file_path
            )

        # Append assistant message
        st.session_state.messages.append({
            "role": "agent",
            "content": result["response"]
        })
        st.session_state["clear_query"] = True
        st.rerun()

       


# Clear chat button
if st.sidebar.button("Start New Conversation"):
    st.session_state.clear()
    st.rerun()
