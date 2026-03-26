"""In-memory vector similarity search store."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional


class VectorStore:
    """Lightweight in-memory vector store using numpy for similarity search.

    Stores document embeddings and supports cosine similarity retrieval.

    Args:
        dimension: The embedding vector dimension (auto-detected on first add).
    """

    def __init__(self, dimension: Optional[int] = None):
        self._dimension = dimension
        self._vectors: List[np.ndarray] = []
        self._documents: List[dict] = []

    def add(self, embedding: np.ndarray, document: dict) -> int:
        """Add a vector and its associated document to the store.

        Args:
            embedding: The vector embedding (numpy array).
            document: Dict with at least 'text' key, plus optional metadata.

        Returns:
            The index of the added document.
        """
        if self._dimension is None:
            self._dimension = len(embedding)
        
        assert len(embedding) == self._dimension, (
            f"Expected dimension {self._dimension}, got {len(embedding)}"
        )
        
        idx = len(self._vectors)
        self._vectors.append(embedding / np.linalg.norm(embedding))
        self._documents.append(document)
        return idx

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[dict, float]]:
        """Find the top-k most similar documents to the query.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.

        Returns:
            List of (document_dict, similarity_score) tuples, sorted by score descending.
        """
        if not self._vectors:
            return []

        query_norm = query_embedding / np.linalg.norm(query_embedding)
        matrix = np.stack(self._vectors)
        similarities = matrix @ query_norm

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self._documents[idx], float(similarities[idx])))
        return results

    def clear(self) -> None:
        """Remove all vectors and documents."""
        self._vectors.clear()
        self._documents.clear()

    @property
    def size(self) -> int:
        """Number of stored vectors."""
        return len(self._vectors)

    @property
    def dimension(self) -> Optional[int]:
        """Embedding dimension."""
        return self._dimension

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"VectorStore(size={self.size}, dim={self._dimension})"
