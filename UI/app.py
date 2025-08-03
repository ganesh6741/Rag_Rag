import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Scripts'))

from generate_response_perplexity import generate_response_from_query as generate_perplexity
import streamlit as st
from Scripts.generate_response_perplexity import generate_response_from_query as generate_perplexity
# from Scripts.generate_response_openai import generate_response_from_query as generate_openai

# === Title and Description ===
st.set_page_config(page_title="GRAG Explorer", layout="wide")
st.title("🔎 GRAG Explorer")
st.markdown("Explore RAG-powered answers using LLMs and embedded research chunks.")

# === Query Input ===
query = st.text_input("Enter your question:", "")

# === Model Selection ===
model_choice = st.selectbox("Choose LLM:", ["Perplexity", "OpenAI"])

# === Response Trigger ===
if st.button("Generate Answer") and query:
    with st.spinner("🔍 Thinking..."):
        if model_choice == "Perplexity":
            response = generate_perplexity(query)
        # else:
        #     response = generate_openai(query)
        # st.markdown("## 💬 Answer")
        # st.write(response)

# === Footer ===
st.markdown("---")
st.caption("Powered by FAISS indexing and GRAG architecture.")