# Document_AI_Assistant
# 📄 Document AI Assistant

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload documents and ask questions about them — with answers grounded strictly in your document content, not hallucinated.

Built entirely from scratch in Python — **no LangChain, no LlamaIndex** — to keep the pipeline transparent and easy to learn from. Every stage (document loading, chunking, embedding, vector search, generation) is a plain, readable module.

🔗 **Live demo:** [askyourdocx.streamlit.app](#) <!-- replace with your actual deployed URL -->

---

## ✨ Features

- 📁 Upload multiple documents at once — PDF, DOCX, TXT, CSV
- ✂️ Smart text chunking with overlap, so context isn't lost at chunk boundaries
- 🧠 Local, free embeddings via `sentence-transformers` (no API key required)
- ⚡ Fast, exact vector similarity search using **FAISS**
- 🤖 Grounded answer generation via the **Groq API** (free tier, blazing-fast inference)
- 🔍 Transparent sourcing — every answer shows which document and chunk it came from, with a relevance score
- 💬 Persistent chat history within a session
- 🖥️ Clean Streamlit interface, no separate backend required

---

## 🏗️ Architecture

```
Upload → Document Loader → Text Chunker → Embedder → FAISS Vector Store
                                                              │
User question ──────────────► Embedder ─────────────────────┤
                                                              ▼
                                                    Top-k similar chunks
                                                              │
                                                              ▼
                                              Groq LLM (context + question)
                                                              │
                                                              ▼
                                                   Grounded answer + sources
```

| Stage | Module | What it does |
|---|---|---|
| 1 | `loaders.py` | Extracts raw text from PDF, DOCX, TXT, CSV |
| 2 | `chunker.py` | Splits text into overlapping chunks with metadata |
| 3 | `embedder.py` | Converts chunks/queries into vectors (`all-MiniLM-L6-v2`) |
| 4 | `vector_store.py` | FAISS index wrapper — add, search, save, load |
| 5 | `rag_pipeline.py` | End-to-end ingestion + retrieval + Groq generation |
| 6 | `app.py` | Streamlit UI — upload, chat, view sources |

---

## 🛠️ Tech stack

- **Language:** Python
- **Embeddings:** [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`, 384-dim, runs locally)
- **Vector database:** [FAISS](https://github.com/facebookresearch/faiss) (`IndexFlatIP` — exact cosine similarity search)
- **LLM:** [Groq API](https://console.groq.com/) (`llama-3.3-70b-versatile`)
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Document parsing:** `pypdf`, `python-docx`, `pandas`

---

## 🚀 Getting started

### 1. Clone the repo
```bash
git clone https://github.com/Pankaj2002Chaudhary/Document_AI_Assistant.git
cd Document_AI_Assistant/document_ai_assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) and generate an API key — no credit card required.

### 4. Configure your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_actual_key_here
```

### 5. Run the app
```bash
python -m streamlit run app.py
```

Open the browser tab at `http://localhost:8501`, upload a document, click **Ingest documents**, and start asking questions.

---

## 📂 Project structure

```
document_ai_assistant/
├── loaders.py          # Step 1: multi-format document loading
├── chunker.py           # Step 2: overlapping text chunking
├── embedder.py           # Step 3: local embedding generation
├── vector_store.py        # Step 4: FAISS vector database wrapper
├── rag_pipeline.py         # Step 5: retrieval + Groq generation
├── app.py                  # Step 6: Streamlit frontend
├── requirements.txt
├── .gitignore
└── .env                     # not committed — holds your API key
```

---

## ⚙️ How it works

1. **Ingestion** — When you upload a document, it's loaded, cleaned, split into ~800-character overlapping chunks, embedded into 384-dimensional vectors, and added to an in-memory FAISS index.
2. **Retrieval** — Your question is embedded the same way, and FAISS finds the most similar chunks using cosine similarity (via normalized inner product).
3. **Generation** — The retrieved chunks are inserted into a prompt that instructs the LLM to answer *only* from that context, and to say so explicitly if the answer isn't there. This is sent to Groq's hosted Llama 3.3 model.
4. **Transparency** — The answer is shown alongside the exact source chunks used and their relevance scores, so you can verify the answer isn't hallucinated.

---

## 🌐 Deployment

This app is deployed on [Streamlit Community Cloud](https://share.streamlit.io) (free tier). To deploy your own copy:

1. Push your repo to GitHub (make sure `.env` is in `.gitignore` and never committed).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → connect your repo.
3. Set the main file path to `document_ai_assistant/app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
5. Deploy.

---


## 🙏 Acknowledgements

Built as a learning project to understand RAG pipelines from first principles — no orchestration frameworks, just the raw mechanics of retrieval-augmented generation.