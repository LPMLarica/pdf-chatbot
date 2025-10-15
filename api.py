from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever import Retriever
from typing import List
import os
import json
import dotenv
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
USE_OPENAI = bool(OPENAI_API_KEY)  

app = FastAPI()
retriever = Retriever()
with open(os.path.join("models", "texts.json"), "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)

class ChatRequest(BaseModel):
    question: str
    top_k: int = 4
    use_openai: bool = None  

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

# ===== geração de resposta  =====
def build_context_text(retrieved_chunks: list):
    # coloca os chunks em sequência como contexto
    parts = []
    for c in retrieved_chunks:
        parts.append(f"Source: {c['meta']['source']} (chunk {c['meta']['chunk_id']})\n{c['text']}\n---")
    return "\n".join(parts)

from openai import OpenAI  

def answer_with_openai(question: str, context: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""Você é um assistente que responde perguntas com base no contexto abaixo. Use apenas as informações presentes no contexto.

Contexto:
{context}

Pergunta:
{question}

Responda de forma objetiva e cite a(s) fonte(s) (arquivo e chunk) no final."""
    
    resp = client.chat.completions.create(
        model="gpt-4o", #Modelo da api 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def answer_with_transformers(question: str, context: str) -> str:
    try:
        from transformers import pipeline
        gen = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=-1  
        )
        prompt = f"""Answer the question based on the context below.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        out = gen(prompt, max_length=150, min_length=30)
        return out[0]["generated_text"].strip()
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    use_openai = req.use_openai if req.use_openai is not None else USE_OPENAI
    results = retriever.query(req.question, top_k=req.top_k)
    indices = [r["idx"] for r in results]
    texts = retriever.get_texts_for_indices(indices)
    for r, t in zip(results, texts):
        r["text"] = t

    context = build_context_text(results)
    try:
        if use_openai:
            if not OPENAI_API_KEY:
                raise HTTPException(status_code=400, detail="OPENAI_API_KEY não definido no servidor.")
            answer = answer_with_openai(req.question, context)
        else:
            answer = answer_with_transformers(req.question, context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro gerando resposta: {e}")

    return ChatResponse(answer=answer, sources=[{"meta": r["meta"], "score": r["score"]} for r in results])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
