"""
Step 6: Streamlit Frontend
File upload -> ingest into RAG pipeline -> ask questions -> see grounded answers + sources.
"""

import os
import tempfile
import streamlit as st

from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Document AI Assistant", page_icon="📄", layout="wide")

# ---------- Session state setup ----------
# session_state persists across Streamlit reruns (unlike normal variables)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)


# ---------- Sidebar: document upload ----------

with st.sidebar:
    st.header("📁 Upload documents")
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, TXT, CSV",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
    )

    if st.button("Ingest documents", disabled=not uploaded_files):
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.ingested_files:
                st.info(f"Skipping '{uploaded_file.name}' — already ingested.")
                continue

            # Save to a temp file so our loaders (which expect a filepath) can read it
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                try:
                    st.session_state.pipeline.add_document(tmp_path)
                    # Fix source name in metadata so it shows the real filename, not the temp path
                    for chunk in st.session_state.pipeline.store.metadata:
                        if chunk["source"] == os.path.basename(tmp_path):
                            chunk["source"] = uploaded_file.name
                    st.session_state.ingested_files.append(uploaded_file.name)
                    st.success(f"Ingested '{uploaded_file.name}'")
                except Exception as e:
                    st.error(f"Failed to ingest '{uploaded_file.name}': {e}")
                finally:
                    os.remove(tmp_path)

    st.divider()
    st.subheader("Ingested documents")
    if st.session_state.ingested_files:
        for fname in st.session_state.ingested_files:
            st.write(f"✅ {fname}")
    else:
        st.write("No documents yet.")


# ---------- Main area: chat ----------

st.title("📄 Document AI Assistant")
st.caption("Ask questions about your uploaded documents. Answers are grounded strictly in their content.")

# Render past Q&A
for question, answer, sources in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            with st.expander("View sources"):
                for s in sources:
                    st.markdown(f"**{s['source']}** (relevance: {s['score']:.2f})")
                    st.caption(s["text_preview"] + "...")

# New question input
query = st.chat_input("Ask a question about your documents...")

if query:
    if not st.session_state.ingested_files:
        st.warning("Please upload and ingest at least one document first.")
    else:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    result = st.session_state.pipeline.ask(query)
                    st.write(result["answer"])
                    if result["sources"]:
                        with st.expander("View sources"):
                            for s in result["sources"]:
                                st.markdown(f"**{s['source']}** (relevance: {s['score']:.2f})")
                                st.caption(s["text_preview"] + "...")

                    st.session_state.chat_history.append((query, result["answer"], result["sources"]))
                except Exception as e:
                    st.error(f"Error generating answer: {e}")