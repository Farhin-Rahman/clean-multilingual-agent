from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Check the key
api_key = os.getenv("GROQ_API_KEY")

if api_key:
    print("✅ GROQ_API_KEY loaded successfully!")
    print(f"Value: {api_key[:6]}...{api_key[-4:]}")  # Obfuscate for safety
else:
    print("❌ GROQ_API_KEY not found. Check your .env file and path.")
