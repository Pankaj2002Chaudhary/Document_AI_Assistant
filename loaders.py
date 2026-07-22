"""
Step 1: Document Loader
Reads a file from disk and returns its raw text content, regardless of format.
"""

import os
from pypdf import PdfReader
from docx import Document
import pandas as pd


def load_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def load_docx(filepath: str) -> str:
    doc = Document(filepath)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def load_csv(filepath: str) -> str:
    df = pd.read_csv(filepath)
    # Turn each row into a readable line: "col1: val1, col2: val2, ..."
    lines = []
    for _, row in df.iterrows():
        line = ", ".join(f"{col}: {row[col]}" for col in df.columns)
        lines.append(line)
    return "\n".join(lines)


# Dispatch table: maps file extension -> loader function
LOADERS = {
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".csv": load_csv,
}


def load_document(filepath: str) -> str:
    """
    Main entry point. Detects file type from extension and routes
    to the correct loader function.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext not in LOADERS:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {list(LOADERS.keys())}"
        )

    loader_fn = LOADERS[ext]
    text = loader_fn(filepath)

    if not text.strip():
        raise ValueError(f"No text could be extracted from {filepath}")

    return text


if __name__ == "__main__":
    # Quick manual test — replace with a real file path on your machine
    sample_path = "docs/sample_employees.csv"
    if os.path.exists(sample_path):
        content = load_document(sample_path)
        print(f"Loaded {len(content)} characters")
        print(content[:300])
    else:
        print("Put a sample.txt next to this file to test.")