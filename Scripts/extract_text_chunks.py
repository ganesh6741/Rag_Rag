import fitz  # PyMuPDF

def extract_chunks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            chunks.extend([chunk.strip() for chunk in text.split('\n') if chunk.strip()])
    return chunks