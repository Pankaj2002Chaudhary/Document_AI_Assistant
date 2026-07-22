"""
Step 5: RAG Pipeline
Combines document loading, chunking, embedding, FAISS storage, and Groq
generation into one end-to-end pipeline.
"""

import os
import requests
from dotenv import load_dotenv

from loaders import load_document
from chunker import chunk_document
from embedder import embed_texts, embed_query
from vector_store import VectorStore

load_dotenv()  # reads .env into environment variables

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
import streamlit as st

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("GROQ_API_KEY")

GROQ_API_KEY = get_api_key()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # check console.groq.com/docs/models for current options


class RAGPipeline:
    def __init__(self, embedding_dim: int = 384):
        self.store = VectorStore(dimension=embedding_dim)

    # ---------- Ingestion ----------

    def add_document(self, filepath: str, chunk_size: int = 800, chunk_overlap: int = 150):
        source_name = os.path.basename(filepath)

        raw_text = load_document(filepath)
        chunks = chunk_document(raw_text, source_name, chunk_size, chunk_overlap)

        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)

        self.store.add(vectors, chunks)
        print(f"Ingested '{source_name}': {len(chunks)} chunks added.")

    # ---------- Retrieval ----------

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = embed_query(query)
        return self.store.search(query_vector, top_k=top_k)

    # ---------- Generation ----------

    def _build_prompt(self, query: str, retrieved_chunks: list[dict]) -> str:
        context = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
        )
        return f"""Answer the question using ONLY the context below.
If the answer is not contained in the context, say "I don't have enough information in the documents to answer that."

Context:
{context}

Question: {query}

Answer:"""

    def _call_groq(self, prompt: str) -> str:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers questions strictly based on provided document context."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    # ---------- End-to-end ----------

    def ask(self, query: str, top_k: int = 5) -> dict:
        retrieved_chunks = self.retrieve(query, top_k=top_k)

        if not retrieved_chunks:
            return {
                "answer": "No documents have been ingested yet, so I have nothing to search.",
                "sources": [],
            }

        prompt = self._build_prompt(query, retrieved_chunks)
        answer = self._call_groq(prompt)

        return {
            "answer": answer,
            "sources": [
                {"source": c["source"], "score": c["score"], "text_preview": c["text"][:150]}
                for c in retrieved_chunks
            ],
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()

    pipeline.add_document("docs/sample_company_report.pdf")

    result = pipeline.ask("What was the company's revenue performance?")

    print("\n--- ANSWER ---")
    print(result["answer"])

    print("\n--- SOURCES USED ---")
    for s in result["sources"]:
        print(f"  {s['source']} (score={s['score']:.4f}): {s['text_preview']}...")