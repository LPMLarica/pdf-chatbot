
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODELS_DIR = "models"
INDEX_FILE = os.path.join(MODELS_DIR, "faiss.index")
META_FILE = os.path.join(MODELS_DIR, "metadata.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, embed_model_name=EMBED_MODEL_NAME, top_k=5):
        self.model = SentenceTransformer(embed_model_name)
        self.top_k = top_k
        self.index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            self.metas = json.load(f)

    def query(self, question: str, top_k: int = None):
        if top_k is None:
            top_k = self.top_k
        q_emb = self.model.encode([question], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            meta = self.metas[idx]
            results.append({
                "score": float(dist),
                "text": None, 
                "meta": meta,
                "idx": int(idx)
            })
        return results

    def get_texts_for_indices(self, indices):
        texts_file = os.path.join(MODELS_DIR, "texts.json")
        if not os.path.exists(texts_file):
            raise FileNotFoundError("texts.json não encontrado — durante indexação salve os chunks em models/texts.json")
        with open(texts_file, "r", encoding="utf-8") as f:
            texts = json.load(f)
        return [texts[i] for i in indices]

if __name__ == "__main__":
    r = Retriever()
    q = "O que é o objetivo do projeto?"
    print(r.query(q))
