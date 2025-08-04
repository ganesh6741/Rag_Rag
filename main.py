import streamlit as st
from pathlib import Path
import os

# ========== CONFIG ==========
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# API Keys (securely pulled from secrets)
PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# ========== UI ==========
st.title("GRAG Explorer 🔍")
st.markdown("Query research papers with powerful RAG techniques.")

# ========== USER INPUT ==========
query = st.text_input("🔎 Enter your research query")

if query:
    with st.spinner("Embedding and retrieving…"):
        try:
            from Scripts.generate_response_perplexity import generate_perplexity_response
            response = generate_perplexity_response(
                query=query,
                perplexity_key=PERPLEXITY_API_KEY,
                openai_key=OPENAI_API_KEY
            )
            st.markdown("### 🧠 Generated Response:")
            st.write(response)

        except Exception as e:
            st.error(f"Something went wrong: {e}")