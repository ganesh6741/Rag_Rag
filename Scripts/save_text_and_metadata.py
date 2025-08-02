import os
import json

def save_outputs(chunks, metadata, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "chunks.txt"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c + "\n")

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)