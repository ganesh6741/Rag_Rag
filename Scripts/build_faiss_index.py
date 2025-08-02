import os
import faiss
import json
import numpy as np
from embed_chunks import embed_chunks

def build_index():
    output_dir = "D:/Rag Rag/Output"
    all_chunks = []
    metadata_map = {}

    for folder_name in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder_name)
        chunk_file = os.path.join(folder_path, "chunks.txt")
        meta_file = os.path.join(folder_path, "metadata.json")

        if os.path.exists(chunk_file):
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = [line.strip() for line in f if line.strip()]
                all_chunks.extend(chunks)

                for i, chunk in enumerate(chunks):
                    metadata_map[len(all_chunks)-len(chunks)+i] = {
                        "chunk": chunk,
                        "source": folder_name
                    }

    # Embed and index
    embeddings = embed_chunks(all_chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    # Save
    os.makedirs("D:/Rag Rag/Indexes", exist_ok=True)
    
    faiss.write_index(index, "D:/Rag Rag/Indexes/index.faiss")
    with open("D:/Rag Rag/Indexes/metadata_map.json", "w", encoding="utf-8") as f:
        json.dump(metadata_map, f, indent=2)

    print(f"✅ Indexed {len(all_chunks)} chunks.")

if __name__ == "__main__":
    build_index()