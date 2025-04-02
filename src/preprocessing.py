import fitz
import re
import os
import json
from chunking import chunk_by_tokens

# generate text from pdfs with PyMuPDF
def get_pdf_text(file_path):
    pdf = fitz.open(file_path)
    text = ""
    for page in pdf:
        page_text = page.get_text("text")
        text += page_text + "\n"
    return text

# clean text
def clean_text(text):
    text = re.sub(r'\n+', '\n', text).strip()
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'DS\s*4300', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*[-–•*]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^(Figure from:|Reference:).*\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'(ACID Properties\s*){2,}', 'ACID Properties\n', text, flags=re.IGNORECASE)
    text = re.sub(r'^[\?]+\s*$', '', text, flags=re.MULTILINE)
    return text

# generate and clean text from pdfs
def process_pdf(file_path):
    raw_text = get_pdf_text(file_path)
    cleaned_text = clean_text(raw_text)
    return cleaned_text

# generate, clean, and chunk text from pdfs appropriately
def process_folder(folder_path, chunk_size=200, overlap=50, output_json="slides_metadata.json"):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            cleaned_text = process_pdf(file_path)
            module_name = os.path.splitext(filename)[0]
            chunks = chunk_by_tokens(cleaned_text, chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "slide_number": i + 1,
                    "text": chunk,
                    "module": module_name,
                    "source": filename
                })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4)

    return {chunk["source"]: chunk["text"] for chunk in all_chunks}  # to support .items() in main

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    folder_path = os.path.join(PROJECT_ROOT, "data")
    raw_data = process_folder(folder_path)
    print(f"Saved {len(raw_data)} chunks to slides_metadata.json")