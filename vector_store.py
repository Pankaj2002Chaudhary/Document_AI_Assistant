"""
Step 4: Vector Store
Wraps a FAISS index + a metadata list so we can:
  - add chunks (vector + text + source) to the index
  - search for the top-k most similar chunks to a query vector
  - persist everything to disk and reload it later
"""

import os
import pickle
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int = 384):
        """
        dimension must match your embedding model's output size.
        all-MiniLM-L6-v2 -> 384
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # inner product = cosine (since vectors are normalized)
        self.metadata = []  # parallel list: metadata[i] describes vector at index i

    def add(self, vectors: np.ndarray, chunk_metadata: list[dict]):
        """
        vectors: shape (n, dimension) — from embedder.embed_texts()
        chunk_metadata: list of n dicts, each describing one chunk
                        (e.g. {"id":..., "text":..., "source":...})
        """
        if len(vectors) != len(chunk_metadata):
            raise ValueError("vectors and chunk_metadata must have the same length")

        if len(vectors) == 0:
            return

        vectors = vectors.astype("float32")  # FAISS requires float32
        self.index.add(vectors)
        self.metadata.extend(chunk_metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Returns the top_k most similar chunks to query_vector,
        each with its similarity score attached.
        """
        if self.index.ntotal == 0:
            return []

        query_vector = query_vector.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 if fewer than top_k results exist
                continue
            chunk = self.metadata[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def save(self, folder_path: str):
        """Persists the FAISS index and metadata to disk."""
        os.makedirs(folder_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
        with open(os.path.join(folder_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, folder_path: str):
        """Loads a previously saved FAISS index and metadata."""
        index_path = os.path.join(folder_path, "index.faiss")
        meta_path = os.path.join(folder_path, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"No saved vector store found at {folder_path}")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def __len__(self):
        return self.index.ntotal


if __name__ == "__main__":
    from embedder import embed_texts, embed_query

    sample_chunks = [
        {"id": "doc1_0", "text": "The company reported record revenue in Q3.", "source": "doc1.txt"},
        {"id": "doc1_1", "text": "Machine learning models require large datasets.", "source": "doc1.txt"},
        {"id": "doc1_2", "text": "Quarterly earnings exceeded analyst expectations.", "source": "doc1.txt"},
    ]
    texts = [c["text"] for c in sample_chunks]

    vectors = embed_texts(texts)

    store = VectorStore(dimension=vectors.shape[1])
    store.add(vectors, sample_chunks)
    print(f"Store now has {len(store)} vectors")

    query = "How did the company perform financially?"
    query_vec = embed_query(query)
    results = store.search(query_vec, top_k=2)

    print(f"\nTop results for: '{query}'")
    for r in results:
        print(f"  score={r['score']:.4f}  source={r['source']}  text={r['text']}")

    # Test persistence
    store.save("test_store")
    print("\nSaved to disk. Reloading into a new store...")

    new_store = VectorStore(dimension=vectors.shape[1])
    new_store.load("test_store")
    print(f"Reloaded store has {len(new_store)} vectors")