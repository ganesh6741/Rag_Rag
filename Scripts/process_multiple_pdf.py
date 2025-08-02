import os
from process_single_pdf import process_pdf

def batch_process_pdfs(input_dir="D:/Rag Rag/Pdf Data", output_base="D:/Rag Rag/Output"):
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDF files found in input directory.")
        return

    print(f"🚀 Found {len(pdf_files)} PDF(s). Starting batch processing...\n")
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        process_pdf(pdf_path, output_base)

if __name__ == "__main__":
    batch_process_pdfs()