import streamlit as st
from support_agent import run_customer_support
import os
from dotenv import load_dotenv
import tempfile
from gtts import gTTS
import base64
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import av

# Load Whisper model once
@st.cache_resource
def load_whisper_model():
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return processor, model

# Whisper processor for live mic
class WhisperAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.processor, self.model = load_whisper_model()
        self.buffer = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.buffer.append(audio)
        return frame

    def get_transcription(self):
        if not self.buffer:
            return ""
        audio_data = np.concatenate(self.buffer).astype(np.float32)
        input_features = self.processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = self.model.generate(input_features)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

# TTS helper
def generate_tts_audio(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        with open(tmp.name, "rb") as f:
            audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f"""
    <audio controls>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

# Load env
load_dotenv()

# Session setup
if "query" not in st.session_state:
    st.session_state["query"] = ""

if "clear_query" not in st.session_state:
    st.session_state["clear_query"] = False

if st.session_state.clear_query:
    st.session_state["query"] = ""
    st.session_state["clear_query"] = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# UI setup
st.set_page_config(page_title="Multilingual AI Support", layout="wide")

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

# Sidebar
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

# Show chat history
for msg in st.session_state.messages:
    role = msg["role"]
    with st.container():
        st.markdown(f"<div class='chat-message {role}'><strong>{role.capitalize()}:</strong><br>{msg['content']}</div>", unsafe_allow_html=True)
        if role == "agent":
            st.markdown(generate_tts_audio(msg["content"]), unsafe_allow_html=True)

# Live mic transcription
st.markdown("🎤 Speak to transcribe:")
ctx = webrtc_streamer(
    key="live-stt",
    mode="SENDRECV",
    audio_receiver_size=1024,
    client_settings=ClientSettings(media_stream_constraints={"audio": True, "video": False}),
    async_processing=True,
)

if ctx and ctx.state.playing:
    if "stt_processor" not in st.session_state:
        st.session_state["stt_processor"] = WhisperAudioProcessor()

    if st.button("🎙️ Transcribe Now"):
        text = st.session_state["stt_processor"].get_transcription()
        if text:
            st.session_state["query"] = text
            st.success("Transcription complete! You can now send it.")
        else:
            st.warning("Couldn't detect speech.")

# Text input box
query = st.text_area("Your message:", height=100, key="query", placeholder="Ask me anything...")

# Send button
if st.button("Send"):
    if not query.strip():
        st.warning("Please enter a message.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        chat_history = []
        for i in range(0, len(st.session_state.messages) - 1, 2):
            if i + 1 < len(st.session_state.messages):
                chat_history.append({
                    "user": st.session_state.messages[i]["content"],
                    "agent": st.session_state.messages[i + 1]["content"]
                })

        with st.spinner("Thinking..."):
            result = run_customer_support(
                query=query,
                force_language=language_mapping[display_language],
                chat_history=chat_history
            )

        st.session_state.messages.append({"role": "agent", "content": result["response"]})
        st.session_state["clear_query"] = True
        st.rerun()

# Start new conversation
if st.sidebar.button("Start New Conversation"):
    st.session_state.clear()
    st.rerun()
