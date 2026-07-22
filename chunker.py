"""
Step 2: Text Chunker
Splits a long string of text into overlapping chunks suitable for embedding.
"""
from loaders import load_document

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """
    Splits text into overlapping chunks.

    chunk_size: max number of characters per chunk
    chunk_overlap: number of characters shared between consecutive chunks
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Normalize whitespace a bit so we don't chunk on messy formatting
    text = " ".join(text.split())

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        # Try to avoid cutting a chunk mid-word: back up to the last space
        if end < text_length:
            last_space = chunk.rfind(" ")
            if last_space != -1:
                end = start + last_space
                chunk = text[start:end]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        # Move the window forward, keeping the overlap
        start = end - chunk_overlap

    return chunks


def chunk_document(text: str, source_name: str, chunk_size: int = 600, chunk_overlap: int = 100) -> list[dict]:
    """
    Wraps chunk_text() and attaches metadata to every chunk.
    This metadata is what we'll later store alongside vectors in FAISS,
    so we can trace an answer back to its source document and chunk position.
    """
    raw_chunks = chunk_text(text, chunk_size, chunk_overlap)

    return [
        {
            "id": f"{source_name}_chunk_{i}",
            "text": chunk,
            "source": source_name,
            "chunk_index": i,
        }
        for i, chunk in enumerate(raw_chunks)
    ]


if __name__ == "__main__":
    # sample = "This is a long piece of text. " * 100
    result=chunk_document(load_document("docs/sample.txt"), source_name="docs/sample.txt")
    # result = chunk_document(sample, source_name="test_doc.txt")
    print(f"Created {len(result)} chunks")
    print(result[0])
    print("---")
    print(result[1])