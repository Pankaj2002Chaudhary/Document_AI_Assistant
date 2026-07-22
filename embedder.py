"""
Step 3: Embedder
Converts text chunks into numeric vectors using a local embedding model.
No API key needed — runs entirely on your machine.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Loaded once and reused. Loading a model is slow (~1-3 sec); embedding is fast.
_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    """
    Lazily loads the embedding model once, then reuses it.
    Avoids reloading the model on every function call.
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {_MODEL_NAME} (first run downloads it)...")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Converts a list of text strings into a 2D numpy array of vectors.
    Shape: (num_texts, embedding_dim) -> e.g. (50, 384)
    """
    if not texts:
        return np.array([])

    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # important: makes cosine similarity == dot product
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single query string (e.g. user's question).
    Returns a 1D vector, shape: (embedding_dim,)
    """
    model = get_model()
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding[0]


if __name__ == "__main__":
    sample_chunks = [
        "The company reported record revenue in Q3.",
        "Machine learning models require large datasets.",
        "Quarterly earnings exceeded analyst expectations.",
    ]

    vectors = embed_texts(sample_chunks)
    print(f"Embedded {len(sample_chunks)} chunks into shape: {vectors.shape}")

    query_vec = embed_query("How did the company perform financially?")
    print(f"Query vector shape: {query_vec.shape}")

    # Sanity check: cosine similarity via dot product (vectors are normalized)
    similarities = vectors @ query_vec
    print("\nSimilarity of query to each chunk:")
    for text, score in zip(sample_chunks, similarities):
        print(f"  {score:.4f}  ->  {text}")