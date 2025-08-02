import os
from extract_text_chunks import extract_chunks_from_pdf
from save_text_and_metadata import save_outputs

def process_pdf(pdf_path, output_base="D:/Rag Rag/Output"):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(output_base, filename)

    chunks = extract_chunks_from_pdf(pdf_path)
    metadata = {
        "title": filename,
        "source_pdf": pdf_path,
        "num_chunks": len(chunks)
    }

    save_outputs(chunks, metadata, output_dir)
    print(f"✅ Processed: {filename} → {output_dir}")

# Example usage
if __name__ == "__main__":
    sample_pdf_path = "D:/Rag Rag/Pdf Data/paper1.pdf"
    process_pdf(sample_pdf_path)