
import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PDF QA Chatbot", layout="centered")

st.title("Chatbot (PDF) — Perguntas a partir de documentos")

question = st.text_area("Pergunta", height=120)
top_k = st.slider("Número de trechos a considerar (top_k)", min_value=1, max_value=10, value=4)

use_openai = st.checkbox("Usar OpenAI (se disponível)", value=True)

if st.button("Perguntar"):
    if not question.strip():
        st.warning("Escreva sua pergunta antes.")
    else:
        with st.spinner("Buscando resposta..."):
            payload = {"question": question, "top_k": top_k, "use_openai": use_openai}
            resp = requests.post(f"{API_URL}/chat", json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                st.subheader("Resposta")
                st.write(data["answer"])
                st.subheader("Fontes (recuperadas)")
                for s in data["sources"]:
                    meta = s["meta"]
                    st.write(f"- {meta['source']} (chunk {meta['chunk_id']}) — score {s['score']:.4f}")
            else:
                st.error(f"Erro: {resp.status_code} - {resp.text}")
