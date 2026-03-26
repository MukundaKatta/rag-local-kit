"""Local embedding generation via Ollama."""

from __future__ import annotations

import json
from typing import List

import requests
import numpy as np


class OllamaEmbeddings:
    """Generate text embeddings using a local Ollama instance.

    Args:
        model: The Ollama model to use for embeddings.
        base_url: The Ollama API base URL.
    """

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dimension = None

    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text string."""
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        embedding = np.array(data["embedding"], dtype=np.float32)
        self._dimension = len(embedding)
        return embedding

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        embeddings = []
        total = len(texts)
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Embedding {i + 1}/{total}...")
            emb = self.embed_text(text)
            embeddings.append(emb)
        if show_progress:
            print(f"  Done. Embedded {total} texts.")
        return embeddings

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (available after first call)."""
        if self._dimension is None:
            # Generate a test embedding to discover dimension
            test = self.embed_text("test")
            self._dimension = len(test)
        return self._dimension

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False
