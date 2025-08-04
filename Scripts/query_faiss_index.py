import os
import faiss
import json
import numpy as np


# Use absolute Windows paths where your files are located
INDEX_PATH = r"D:\Rag Rag\Indexes\index.faiss"
METADATA_JSON_PATH = r"D:\Rag Rag\Indexes\metadata_map.json"


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



# import os
# import faiss
# import json
# import numpy as np

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# INDEX_PATH = os.path.join(BASE_DIR, "Indexes", "index.faiss")
# METADATA_JSON_PATH = os.path.join(BASE_DIR, "Indexes", "metadata_map.json")

# print(f"Loading index from: {INDEX_PATH}")
# print(f"Loading metadata from: {METADATA_JSON_PATH}")

# if not os.path.exists(INDEX_PATH):
#     raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Check if the file is present or generate it dynamically.")

# index = faiss.read_index(INDEX_PATH)

# if not os.path.exists(METADATA_JSON_PATH):
#     raise FileNotFoundError(f"Metadata JSON not found at {METADATA_JSON_PATH}. Check if the file exists.")

# with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
#     metadata_map = json.load(f)

# def search_faiss(query, model, index=index, metadata_map=metadata_map, top_k=5):
#     query_embedding = model.encode([query])
#     query_embedding = np.array(query_embedding).astype('float32')
    
#     distances, indices = index.search(query_embedding, top_k)

#     results = []
#     for score, idx in zip(distances[0], indices[0]):
#         if idx == -1:
#             continue
#         metadata = metadata_map.get(str(idx), {})
#         results.append({
#             "chunk": metadata.get("chunk", ""),
#             "score": round(float(score), 4),
#             "source": metadata.get("source", "")
#         })

#     return results