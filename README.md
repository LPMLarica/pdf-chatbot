# 🤖 Chatbot Inteligente com PDFs  

> Sistema de perguntas e respostas baseado em documentos PDF, utilizando **Processamento de Linguagem Natural (NLP)** e **Machine Learning**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-API%20Backend-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/Streamlit-Frontend-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20DB-orange?logo=facebook" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## 🌟 Visão Geral

Este projeto implementa um **chatbot de perguntas e respostas** que utiliza documentos **PDF como base de conhecimento**.  
Ele combina **técnicas de NLP** para analisar o conteúdo dos PDFs com **modelos de embeddings semânticos** para encontrar trechos relevantes.

O sistema pode ser utilizado para:
- 📚 Consultar relatórios, apostilas ou políticas internas;
- 🧠 Extrair informações de forma inteligente sem ler todo o PDF;
- ☁️ Implantar como serviço web interativo para uso interno em empresas.

---

## 🧩 Arquitetura do Projeto

📁 pdf-chatbot/
├─ data/ → PDFs para indexar
├─ models/ → Embeddings e índices FAISS
├─ indexer.py → Cria o banco vetorial a partir dos PDFs
├─ retriever.py → Busca semântica nos documentos
├─ api.py → API FastAPI para respostas automáticas
├─ streamlit_app.py → Interface visual de chat
├─ utils.py → Funções auxiliares (extração, chunking, limpeza)
├─ requirements.txt → Dependências do projeto
└─ README.md → Este arquivo

yaml
Copiar código

---

## ⚙️ Tecnologias Utilizadas

| Componente | Função | Tecnologia |
|-------------|--------|-------------|
| 🧠 **Embeddings** | Geração de vetores semânticos | `sentence-transformers (MiniLM-L6-v2)` |
| 🔎 **Busca vetorial** | Similaridade entre trechos e perguntas | `FAISS` |
| 💬 **Modelo de linguagem** | Geração de resposta final | `OpenAI GPT` ou `Transformers local` |
| 🌐 **Backend** | API REST para consultas | `FastAPI` |
| 🖥️ **Frontend** | Interface de chat interativa | `Streamlit` |
| 📄 **Extração PDF** | Leitura e limpeza de conteúdo | `PyMuPDF (fitz)` |

---

## 🚀 Como Executar o Projeto

### 1️⃣ Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/pdf-chatbot.git
cd pdf-chatbot
2️⃣ Criar e Ativar Ambiente Virtual
bash
Copiar código
python -m venv .venv
# source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows
3️⃣ Instalar Dependências
bash
Copiar código
pip install -r requirements.txt
4️⃣ Adicionar os PDFs
Coloque todos os seus arquivos PDF dentro da pasta:

kotlin
Copiar código
data/
Exemplo:

kotlin
Copiar código
data/
├─ Manual_Interno.pdf
├─ Relatorio_2024.pdf
└─ Politica_de_Seguranca.pdf
5️⃣ Indexar os PDFs
Execute o indexador para gerar os embeddings e criar o banco FAISS:

bash
Copiar código
python indexer.py --pdf-folder data
Isso irá criar:

pgsql
Copiar código
models/
├─ faiss.index
├─ metadata.json
└─ texts.json
6️⃣ (Opcional) Configurar a Chave da OpenAI
Se quiser usar o modelo GPT para respostas mais precisas, crie um arquivo .env na raiz do projeto:

ini
Copiar código
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
Caso não configure, o sistema usará modelos locais do transformers.

7️⃣ Executar a API Backend
bash
Copiar código
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Acesse no navegador:

bash
Copiar código
http://localhost:8000/docs
✨ Lá você pode testar o endpoint /chat.

8️⃣ Executar o Frontend (Streamlit)
Em outro terminal, rode:

bash
Copiar código
streamlit run streamlit_app.py
Acesse:

arduino
Copiar código
http://localhost:8501/
💬 Exemplo de Uso
Pergunta:
“Qual é o objetivo principal do projeto descrito no documento?”

Resposta esperada:
O projeto visa desenvolver um sistema de chatbot capaz de responder perguntas com base em informações extraídas de documentos PDF, utilizando técnicas de Processamento de Linguagem Natural e Machine Learning.

Fontes:
📄 Manual_Interno.pdf (chunk 2)

📄 Relatorio_2024.pdf (chunk 1)

🔁 Fluxo de Funcionamento
<p align="center"> <img src="https://mermaid.ink/img/pako:eNqNkc1OwzAQRX8F2RYHg8V2QFkR5wiRIEkbRpZVqRuUVm6RMRP-fuXKS5rHbX_3vfs8lMImh6EczxzN3Nhw8s7UCxQTBZ6FK1T0KUBSRV6jqLgZLBaKUXzngy1xDZ4e5tBhkoeFdxK1VEyJCBu4vQ4cY4nDS7VikDE3Z6s7qCghAGFpFwlvPQGgHyOlEPUoN0ZZfvmQ-JTx1MzvHKmugkyCXqDoyVXzIEJwCuQkqvUsKqv2MSVqNKq8mkoXLxV4_gzL4K5iA0rztRpMqPGW9qapPLzyae65rAtHwHcE6p9WUv9UeV_dC6bxvHgJne-DNULsIV" width="700px" alt="Diagrama do fluxo do chatbot"> </p>
🔹 Etapas do fluxo:

O usuário envia uma pergunta.

O sistema busca trechos similares nos PDFs via embeddings.

Os trechos mais relevantes são combinados em um contexto.

O modelo de linguagem gera uma resposta fundamentada.

🧠 Estrutura Lógica Simplificada
python
Copiar código
# pipeline resumido

pdf_texts = extract_text_from_pdf("Relatorio.pdf")
chunks = chunk_text(pdf_texts)
embeddings = model.encode(chunks)
index.add(embeddings)

# consulta
question = "O que é o objetivo do projeto?"
query_vec = model.encode([question])
result = index.search(query_vec, top_k=5)

# resposta (via OpenAI ou local)
answer = generate_answer(context, question)
🧰 Personalização
Você pode:

🔄 Trocar FAISS por Qdrant ou Pinecone;

💬 Substituir o modelo local por OpenAI Embeddings;

🧾 Incluir páginas e títulos nos metadados;

🔐 Adicionar autenticação JWT na API;

📈 Monitorar logs e métricas no Streamlit.

🧑‍💻 Contribuindo
Pull requests são bem-vindos!
Se quiser contribuir:

Faça um fork 🍴

Crie uma branch (git checkout -b feature/nome)

Faça suas alterações ✨

Envie um PR 🚀

🪪 Licença
Distribuído sob licença MIT.
Sinta-se à vontade para usar, modificar e compartilhar.

<p align="center"> Feito com 💙 por <b>Larissa Campos</b> — Projeto Chatbot PDF 💬 </p> ```