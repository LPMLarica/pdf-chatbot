
import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from utils import extract_text_from_pdf, chunk_text, iter_pdfs
from tqdm import tqdm

MODELS_DIR = "models"
INDEX_FILE = os.path.join(MODELS_DIR, "faiss.index")
META_FILE = os.path.join(MODELS_DIR, "metadata.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  

def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)

def index_pdfs(pdf_folder: str, embed_model_name: str = EMBED_MODEL_NAME):
    ensure_models_dir()
    model = SentenceTransformer(embed_model_name)

    texts = []     
    metas = []    

    for path in iter_pdfs(pdf_folder):
        raw = extract_text_from_pdf(path)
        chunks = chunk_text(raw, chunk_size=1000, overlap=200)
        for i, c in enumerate(chunks):
            texts.append(c)
            metas.append({
                "source": path,
                "chunk_id": i,
                "length": len(c)
            })

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]

    texts_file = os.path.join(MODELS_DIR, "texts.json")
    with open(texts_file, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
        
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"Index criado: {INDEX_FILE}, {len(texts)} vetores, metadata: {META_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-folder", default="data", help="pasta com PDFs")
    parser.add_argument("--model", default=EMBED_MODEL_NAME)
    args = parser.parse_args()
    index_pdfs(args.pdf_folder, args.model)
