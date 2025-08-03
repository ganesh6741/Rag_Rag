import os
import faiss
import json
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(BASE_DIR, "Indexes", "index.faiss")
METADATA_JSON_PATH = os.path.join(BASE_DIR, "Indexes", "metadata_map.json")

print(f"Loading index from: {INDEX_PATH}")
print(f"Loading metadata from: {METADATA_JSON_PATH}")

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Check if the file is present or generate it dynamically.")

index = faiss.read_index(INDEX_PATH)

if not os.path.exists(METADATA_JSON_PATH):
    raise FileNotFoundError(f"Metadata JSON not found at {METADATA_JSON_PATH}. Check if the file exists.")

with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
    metadata_map = json.load(f)

def search_faiss(query, model, index=index, metadata_map=metadata_map, top_k=5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        metadata = metadata_map.get(str(idx), {})
        results.append({
            "chunk": metadata.get("chunk", ""),
            "score": round(float(score), 4),
            "source": metadata.get("source", "")
        })

    return results

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