from pypdf import PdfReader


# =========================
# 1. Extract text from PDF
# =========================
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================
# 2. Chunking function
# =========================
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap  # overlap helps preserve context

    return chunks


# =========================
# 3. Test the pipeline
# =========================
if __name__ == "__main__":

    # IMPORTANT: adjust path based on your structure
    pdf_path = r"data/Hands On Machine Learning with Scikit Learn and TensorFlow.pdf"

    print(" Extracting text...")
    text = extract_text(pdf_path)

    print(" Chunking text...")
    chunks = chunk_text(text)

    print("\n Total chunks:", len(chunks))

    print("\n Sample chunk:\n")
    print(chunks[0])