import faiss
import json
from sentence_transformers import SentenceTransformer

# === Load FAISS index ===
index_path = "D:/Rag Rag/Indexes/index.faiss"
index = faiss.read_index(index_path)

# === Load metadata from JSON ===
metadata_json_path = "D:/Rag Rag/Indexes/metadata_map.json"
with open(metadata_json_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === Load embedding model ===
# Modular import – assumes Models/model_loader.py exists
# from Models.model_loader import load_embedding_model
# model = load_embedding_model()
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Define search function ===
def search_faiss(query_text, top_k=5):
    query_vector = model.encode([query_text])
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i in range(len(indices[0])):
        idx = str(indices[0][i])  # JSON keys are strings
        result = {
            "score": distances[0][i],
            "chunk": metadata[idx]["chunk"],   # ✅ Corrected key
            "source": metadata[idx]["source"]
        }
        results.append(result)
    return results

# === Run a sample query ===
if __name__ == "__main__":
    query = "Explain contrastive learning"
    results = search_faiss(query)

    for i, res in enumerate(results):
        print(f"\n🔍 Result {i+1}")
        print(f"Score: {res['score']:.4f}")
        print(f"Source: {res['source']}")
        print(f"Chunk: {res['chunk']}")

# import faiss
# import pickle
# from sentence_transformers import SentenceTransformer

# # === Load FAISS index ===
# index_path = "D:/Rag Rag/Indexes/index.faiss"
# metadata_path = "D:/Rag Rag/Indexes/metadata.pkl"

# index = faiss.read_index(index_path)

# with open(metadata_path, "rb") as f:
#     metadata = pickle.load(f)

# # === Load embedding model ===
# # model = SentenceTransformer("all-MiniLM-L6-v2")  # Or keep modular via model_loader.py
# from Models.model_loader import load_embedding_model
# model = load_embedding_model()
# # === Define search function ===
# def search_faiss(query_text, top_k=5):
#     query_vector = model.encode([query_text])
#     distances, indices = index.search(query_vector, top_k)

#     results = []
#     for i in range(len(indices[0])):
#         idx = indices[0][i]
#         result = {
#             "score": distances[0][i],
#             "chunk": metadata[idx]["text"],
#             "source": metadata[idx]["source"]
#         }
#         results.append(result)
#     return results

# # === Run a sample query ===
# if __name__ == "__main__":
#     query = "Explain contrastive learning"
#     results = search_faiss(query)

#     for i, res in enumerate(results):
#         print(f"\n🔍 Result {i+1}")
#         print(f"Score: {res['score']:.4f}")
#         print(f"Source: {res['source']}")
#         print(f"Chunk: {res['chunk']}")