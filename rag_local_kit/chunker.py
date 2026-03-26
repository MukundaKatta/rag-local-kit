"""Document chunking strategies for RAG ingestion."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import List, Optional


class ChunkStrategy(Enum):
    """Available chunking strategies."""
    FIXED = "fixed"
    SENTENCE = "sentence"
    SLIDING_WINDOW = "sliding_window"


class Chunk:
    """A chunk of text from a document."""

    def __init__(self, text: str, source: str = "", index: int = 0, metadata: dict = None):
        self.text = text
        self.source = source
        self.index = index
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Chunk(source='{self.source}', index={self.index}, len={len(self.text)})"


class Chunker:
    """Splits documents into chunks using configurable strategies.

    Args:
        strategy: The chunking strategy to use.
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.
    """

    def __init__(self, strategy=ChunkStrategy.FIXED, chunk_size=512, chunk_overlap=50):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str = "") -> List[Chunk]:
        """Split a text string into chunks."""
        if self.strategy == ChunkStrategy.FIXED:
            return self._fixed_chunks(text, source)
        elif self.strategy == ChunkStrategy.SENTENCE:
            return self._sentence_chunks(text, source)
        elif self.strategy == ChunkStrategy.SLIDING_WINDOW:
            return self._sliding_window_chunks(text, source)
        return [Chunk(text=text, source=source, index=0)]

    def chunk_file(self, filepath: str) -> List[Chunk]:
        """Load and chunk a single file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.chunk_text(text, source=str(path.name))

    def chunk_directory(self, dirpath: str, extensions=None) -> List[Chunk]:
        """Load and chunk all matching files in a directory."""
        if extensions is None:
            extensions = [".txt", ".md", ".py", ".json"]
        
        path = Path(dirpath)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dirpath}")

        all_chunks = []
        for ext in extensions:
            for file in sorted(path.rglob(f"*{ext}")):
                try:
                    chunks = self.chunk_file(str(file))
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Warning: Could not process {file}: {e}")
        return all_chunks

    def _fixed_chunks(self, text: str, source: str) -> List[Chunk]:
        """Split text into fixed-size chunks."""
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, source=source, index=idx))
                idx += 1
            start = end - self.chunk_overlap if self.chunk_overlap else end
        return chunks

    def _sentence_chunks(self, text: str, source: str) -> List[Chunk]:
        """Split text at sentence boundaries, grouping into target chunk size."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_len = 0
        idx = 0

        for sentence in sentences:
            if current_len + len(sentence) > self.chunk_size and current:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append(Chunk(text=chunk_text, source=source, index=idx))
                    idx += 1
                current = []
                current_len = 0
            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, source=source, index=idx))
        return chunks

    def _sliding_window_chunks(self, text: str, source: str) -> List[Chunk]:
        """Create overlapping sliding window chunks."""
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        idx = 0
        for start in range(0, len(text), step):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, source=source, index=idx))
                idx += 1
            if end >= len(text):
                break
        return chunks
