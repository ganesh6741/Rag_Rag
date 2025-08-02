import os
import openai
from query_faiss_index import search_faiss

# Set your API key securely
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response_from_query(query_text, top_k=5):
    retrieved_chunks = search_faiss(query_text, top_k)

    # Concatenate the retrieved chunks as context
    context = "\n".join([chunk['chunk'] for chunk in retrieved_chunks])

    prompt = f"""You are an assistant helping answer academic questions based on retrieved content.
Use the information below to answer the question.
Context:
{context}

Question:
{query_text}

Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if available
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )

    return response["choices"][0]["message"]["content"]

# === Run a sample generation ===
if __name__ == "__main__":
    query = "Explain the concept of contrastive learning."
    answer = generate_response_from_query(query)
    print("\n💬 Generated Answer:\n", answer)