"""Main RAG pipeline orchestrator."""

from __future__ import annotations

import requests
from typing import List, Optional

from .chunker import Chunker, ChunkStrategy, Chunk
from .embeddings import OllamaEmbeddings
from .vectorstore import VectorStore


class RAGPipeline:
    """End-to-end RAG pipeline: ingest, embed, store, query.

    Args:
        model: Ollama model for text generation.
        embed_model: Ollama model for embeddings.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.
        top_k: Number of chunks to retrieve per query.
        ollama_url: Ollama API base URL.
    """

    def __init__(
        self,
        model: str = "llama3",
        embed_model: str = "nomic-embed-text",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 3,
        ollama_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.top_k = top_k
        self.ollama_url = ollama_url.rstrip("/")
        self.chunker = Chunker(
            strategy=ChunkStrategy.FIXED,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_url)
        self.store = VectorStore()
        self._chunks: List[Chunk] = []

    def ingest(self, path: str, extensions: list = None) -> int:
        """Ingest documents from a file or directory.

        Args:
            path: Path to a file or directory.
            extensions: File extensions to include (for directories).

        Returns:
            Number of chunks ingested.
        """
        from pathlib import Path as P
        p = P(path)

        if p.is_file():
            chunks = self.chunker.chunk_file(str(p))
        elif p.is_dir():
            chunks = self.chunker.chunk_directory(str(p), extensions=extensions)
        else:
            raise ValueError(f"Path does not exist: {path}")

        if not chunks:
            print("Warning: No chunks were created from the input.")
            return 0

        print(f"Created {len(chunks)} chunks. Generating embeddings...")
        embeddings = self.embeddings.embed_batch([c.text for c in chunks])

        for chunk, emb in zip(chunks, embeddings):
            self.store.add(emb, {"text": chunk.text, "source": chunk.source, "index": chunk.index})
            self._chunks.append(chunk)

        print(f"Ingested {len(chunks)} chunks into vector store. Total: {self.store.size}")
        return len(chunks)

    def query(self, question: str, top_k: int = None) -> str:
        """Query the knowledge base and generate an answer.

        Args:
            question: The question to answer.
            top_k: Override the default number of chunks to retrieve.

        Returns:
            The generated answer string.
        """
        k = top_k or self.top_k
        query_emb = self.embeddings.embed_text(question)
        results = self.store.search(query_emb, top_k=k)

        if not results:
            return "No relevant documents found. Please ingest some documents first."

        context_parts = []
        for doc, score in results:
            context_parts.append(f"[Source: {doc.get('source', 'unknown')}, Score: {score:.3f}]\n{doc['text']}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        """Generate text using the Ollama API."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.ConnectionError:
            return "Error: Cannot connect to Ollama. Is it running? (ollama serve)"
        except Exception as e:
            return f"Error generating response: {e}"

    @property
    def num_chunks(self) -> int:
        """Number of chunks in the store."""
        return self.store.size

    def __repr__(self):
        return f"RAGPipeline(model='{self.model}', chunks={self.store.size})"
