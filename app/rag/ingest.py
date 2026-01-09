from __future__ import annotations
from pathlib import Path
from PyPDF2 import PdfReader

def save_uploaded_pdf(uploaded_file, save_dir: str = "data/uploads") -> str:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(save_dir) / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)

def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    reader = PdfReader(pdf_path)
    pages: list[dict] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": idx, "text": text})
    return pages
