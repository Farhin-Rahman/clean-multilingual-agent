import streamlit as st
from support_agent import run_customer_support
import os
from dotenv import load_dotenv

load_dotenv()

# Page setup with improved styling
st.set_page_config(page_title="Multilingual AI Support", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f2f6;
        border-left: 5px solid #4c8bf5;
    }
    .chat-message.assistant {
        background-color: #f8f9fa;
        border-left: 5px solid #34a853;
    }
    .chat-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        background-color: #4c8bf5;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #4c8bf5;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>🌍 Multilingual AI Support Center</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 2rem;'>Get help in your preferred language. Our AI understands and responds in multiple languages.</p>", unsafe_allow_html=True)

# Settings in sidebar
with st.sidebar:
    st.title("Settings")
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
    
    with st.expander("About this app"):
        st.markdown("""
        This support agent can:
        - Automatically detect your language
        - Categorize your issue
        - Analyze sentiment
        - Provide helpful responses in your language
        - Handle follow-up questions
        """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Display chat history
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.container():
            st.markdown(f"<div class='chat-message user'><div class='chat-header'>You:</div>{message['content']}</div>", unsafe_allow_html=True)
            if "image" in message:
                st.image(message["image"], caption="Attached image")
    else:
        with st.container():
            st.markdown(f"<div class='chat-message assistant'><div class='chat-header'>Support Agent:</div>{message['content']}</div>", unsafe_allow_html=True)
            if i > 0 and message.get("analysis"):
                with st.expander("View Analysis"):
                    st.write(f"**Category:** {message['analysis']['category']}")
                    st.write(f"**Sentiment:** {message['analysis']['sentiment']}")
                    st.write(f"**Language:** {message['analysis']['language']}")

# File upload
uploaded_file = st.file_uploader("📎 Attach a file (optional)", type=["png", "jpg", "jpeg", "pdf", "txt"])

# Message input
query = st.text_area("Type your message:", height=100)

# Send button
if st.button("Send Message"):
    if query:
        # Add user message to chat history
        user_message = {"role": "user", "content": query}
        if uploaded_file:
            # If it's an image, store it for display
            if uploaded_file.type.startswith('image'):
                user_message["image"] = uploaded_file
                
            # Add file info to the query
            query += f"\n[Attached file: {uploaded_file.name}]"
        
        st.session_state.messages.append(user_message)
        
        # Get selected language
        selected_lang = language_mapping[display_language]
        
        # Process with support agent
        with st.spinner("Processing your query..."):
            try:
                result = run_customer_support(query, force_language=selected_lang)
                
                # Add AI response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["response"],
                    "analysis": {
                        "category": result["category"],
                        "sentiment": result["sentiment"],
                        "language": result["original_language"]
                    }
                })
                
                # Force UI refresh
                st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a message before sending.")

# Clear chat button
if st.sidebar.button("Start New Conversation"):
    st.session_state.messages = []
    st.rerun()
