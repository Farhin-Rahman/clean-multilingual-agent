import streamlit as st
from support_agent import run_customer_support
from dotenv import load_dotenv
import os
import base64
from gtts import gTTS
import tempfile
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel

# Load environment variables
load_dotenv()
st.set_page_config(page_title="Multilingual AI Support", layout="wide")

# Whisper model loading (only once)
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", compute_type="auto")

def transcribe_audio(audio_bytes, sample_rate):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp_path = tmp.name

    model = load_whisper_model()
    segments, _ = model.transcribe(tmp_path)
    transcription = "".join([segment.text for segment in segments])
    return transcription.strip()
def generate_tts_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        tmp.seek(0)
        audio_bytes = tmp.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f"""
        <audio controls>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """


# Session state
if "query" not in st.session_state:
    st.session_state.query = ""
if "clear_query" not in st.session_state:
    st.session_state.clear_query = False
if st.session_state.clear_query:
    st.session_state.query = ""
    st.session_state.clear_query = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    display_language = st.selectbox("Force response language:", [
        "Auto (detect)", "English", "Spanish", "French", "German", "Bengali", "Italian", "Portuguese"
    ])
    uploaded_file = st.file_uploader("Upload a file (PDF, TXT)", type=["pdf", "txt"])
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
    if st.button("Start New Conversation"):
        st.session_state.clear()
        st.rerun()

# Header
st.markdown("<h1 style='text-align:center;'>🌍 Multilingual AI Support</h1>", unsafe_allow_html=True)

# Chat display
for msg in st.session_state.messages:
    role = msg["role"]
    st.markdown(f"<div class='chat-message {role}'><strong>{role.capitalize()}:</strong><br>{msg['content']}</div>", unsafe_allow_html=True)
    if role == "agent":
        st.markdown(generate_tts_audio(msg["content"]), unsafe_allow_html=True)

# 🎤 Live mic input
st.markdown("### 🎙️ Speak your message")
audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="⏹ Stop", just_once=True)

if audio:
    st.audio(audio['bytes'], format="audio/wav")
    transcribed = transcribe_audio(audio['bytes'], audio['sample_rate'])
    st.session_state.query = transcribed
    st.success("Transcription ready!")

# Text input
query = st.text_area("Your message:", value=st.session_state.query, height=100, key="query", placeholder="Ask me anything...")

# Submit
if st.button("Send"):
    if query.strip():
        st.session_state.messages.append({"role": "user", "content": query})
        file_path = None
        if uploaded_file:
            suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

        chat_history = []
        for i in range(0, len(st.session_state.messages) - 1, 2):
            if i + 1 < len(st.session_state.messages):
                chat_history.append({
                    "user": st.session_state.messages[i]["content"],
                    "agent": st.session_state.messages[i + 1]["content"]
                })

        with st.spinner("Thinking..."):
            result = run_customer_support(query, language_mapping[display_language], chat_history, file_path)
        st.session_state.messages.append({"role": "agent", "content": result["response"]})
        st.session_state.clear_query = True
        st.rerun()