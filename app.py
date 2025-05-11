# Updated app.py with cloud-compatible voice input using streamlit-audio-recorder

import streamlit as st
from support_agent import run_customer_support
import os
from dotenv import load_dotenv
import tempfile
from gtts import gTTS
import base64
from streamlit_audio_recorder import audio_recorder
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

@st.cache_resource
def load_whisper_model():
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return processor, model

def transcribe_audio_bytes(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    waveform, sample_rate = torchaudio.load(tmp_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    processor, model = load_whisper_model()
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(inputs)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

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

# Init
load_dotenv()
st.set_page_config(page_title="Multilingual AI Support", layout="wide")
if "query" not in st.session_state:
    st.session_state.query = ""
if "clear_query" not in st.session_state:
    st.session_state.clear_query = False
if st.session_state.clear_query:
    st.session_state.query = ""
    st.session_state.clear_query = False

# UI
st.title("🌍 Multilingual AI Support")

with st.sidebar:
    st.header("Settings")
    display_language = st.selectbox("Force response language:", ["Auto (detect)", "English", "Spanish", "French", "German", "Bengali", "Italian", "Portuguese"])
    uploaded_file = st.file_uploader("Upload a file (PDF, TXT)", type=["pdf", "txt"])
    language_mapping = {"Auto (detect)": None, "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Bengali": "bn", "Italian": "it", "Portuguese": "pt"}

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = msg["role"]
    st.markdown(f"<div class='chat-message {role}'><strong>{role.capitalize()}:</strong><br>{msg['content']}</div>", unsafe_allow_html=True)
    if role == "agent":
        st.markdown(generate_tts_audio(msg["content"]), unsafe_allow_html=True)

# 🎤 Record voice
st.markdown("### 🎙️ Speak your question below")
audio = audio_recorder(text="Click to record", pause_threshold=3.0, sample_rate=16000)
if audio:
    transcribed = transcribe_audio_bytes(audio)
    st.session_state.query = transcribed
    st.success("Transcription ready! You can now edit or send it.")

# Input text
query = st.text_area("Your message:", value=st.session_state.query, height=100, key="query", placeholder="Ask me anything...")

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
                chat_history.append({"user": st.session_state.messages[i]["content"], "agent": st.session_state.messages[i+1]["content"]})
        with st.spinner("Thinking..."):
            result = run_customer_support(query, language_mapping[display_language], chat_history, file_path)
        st.session_state.messages.append({"role": "agent", "content": result["response"]})
        st.session_state.clear_query = True
        st.rerun()

if st.sidebar.button("Start New Conversation"):
    st.session_state.clear()
    st.rerun()
