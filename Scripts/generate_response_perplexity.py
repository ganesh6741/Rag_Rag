import os
import requests
from query_faiss_index import search_faiss
from sentence_transformers import SentenceTransformer

# 🌐 Set your Perplexity API key
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# 🔧 Initialize embedding model (valid Hugging Face model)
model = SentenceTransformer("all-MiniLM-L6-v2")  # or replace with any supported model

def generate_response_from_query(query_text, top_k=5):
    # 🔍 Retrieve relevant chunks from FAISS
    retrieved_chunks = search_faiss(query_text, model, top_k=top_k)

    # 🧠 Create context from retrieved chunks
    context = "\n".join([chunk['chunk'] for chunk in retrieved_chunks])
    full_prompt = f"""You are an assistant explaining RAG concepts to beginners using retrieved research content.

Context:
{context}

Question:
{query_text}

Answer:"""

    # 📦 API payload for Perplexity
    payload = {
        "model": "sonar-pro",  # Use sonar-reasoning or sonar-deep-research if preferred
        "messages": [
            {"role": "user", "content": full_prompt}
        ]
    }

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    # 📡 Send request to Perplexity
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload
    )

    # 📨 Return response or error
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"⚠️ Error: {response.status_code} — {response.text}"

# 🧪 Run standalone test
if __name__ == "__main__":
    query = "How does a retriever in RAG differ from a generator?"
    reply = generate_response_from_query(query)
    print("\n💬 Perplexity Response:\n", reply)

# import os
# import requests
# from query_faiss_index import search_faiss
# from sentence_transformers import SentenceTransformer

# # Set your Perplexity API key
# PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# # Initialize your embedding model once (outside function for efficiency)
# # model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('sonar-pro')


# def generate_response_from_query(query_text, top_k=5):
#     # Pass model explicitly
#     retrieved_chunks = search_faiss(query_text, model, top_k=top_k)

#     context = "\n".join([chunk['chunk'] for chunk in retrieved_chunks])
#     full_prompt = f"""You are an assistant explaining RAG concepts to beginners using retrieved research content.

# Context:
# {context}

# Question:
# {query_text}

# Answer:"""

#     headers = {
#         "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": "mistral-7b-instruct",
#         "messages": [
#             {"role": "user", "content": full_prompt}
#         ]
#     }

#     response = requests.post(
#         "https://api.perplexity.ai/chat/completions",
#         headers=headers,
#         json=payload
#     )

#     if response.status_code == 200:
#         return response.json()["choices"][0]["message"]["content"]
#     else:
#         return f"⚠️ Error: {response.status_code} — {response.text}"

# # === Run test
# if __name__ == "__main__":
#     query = "How does a retriever in RAG differ from a generator?"
#     reply = generate_response_from_query(query)
#     print("\n💬 Perplexity Response:\n", reply)


