
import fitz
import os
import re
from typing import List, Tuple

def extract_text_from_pdf(path: str) -> str:
    """
    Extrai todo o texto de um PDF usando PyMuPDF (fitz).
    """
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    doc.close()
    return "\n".join(texts)

def clean_text(text: str) -> str:
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Quebra o texto em pedaços com sobreposição.
    """
    cleaned = clean_text(text)
    start = 0
    n = len(cleaned)
    chunks = []
    while start < n:
        end = start + chunk_size
        chunk = cleaned[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)  
    return chunks

def iter_pdfs(folder: str):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                yield os.path.join(root, f)
