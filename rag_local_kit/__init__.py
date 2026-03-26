"""rag-local-kit: Simple local RAG toolkit powered by Ollama."""

from .pipeline import RAGPipeline
from .chunker import Chunker, ChunkStrategy
from .embeddings import OllamaEmbeddings
from .vectorstore import VectorStore

__version__ = "0.1.0"
__all__ = ["RAGPipeline", "Chunker", "ChunkStrategy", "OllamaEmbeddings", "VectorStore"]
