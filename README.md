# rag-local-kit

A simple, self-contained RAG (Retrieval-Augmented Generation) toolkit that runs entirely on your local machine. Ingest documents, build vector indexes, and ask questions -- all powered by local LLMs through Ollama.

No cloud APIs. No API keys. No data leaves your machine.

## Features

- **Local-first** - Everything runs on your machine using Ollama for embeddings and generation
- **Simple document ingestion** - Load .txt, .md, and .pdf files with one command
- **Chunking strategies** - Fixed-size, sentence-based, or sliding window chunking
- **In-memory vector store** - Lightweight numpy-based vector similarity search
- **Conversational QA** - Ask questions about your documents with context-aware responses
- **Model-agnostic** - Works with any model Ollama supports (Llama, Mistral, Gemma, etc.)

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running
- A local model pulled (e.g., `ollama pull llama3`)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from rag_local_kit import RAGPipeline

# Create a pipeline with your preferred model
rag = RAGPipeline(model="llama3")

# Ingest documents
rag.ingest("./my_documents/")

# Ask questions
answer = rag.query("What are the key findings in the report?")
print(answer)
```

## CLI Usage

```bash
# Ingest a folder of documents
python -m rag_local_kit ingest ./docs/

# Query your knowledge base
python -m rag_local_kit query "Summarize the main points"

# Interactive chat mode
python -m rag_local_kit chat
```

## Project Structure

```
rag_local_kit/
  __init__.py       - Package exports
  chunker.py        - Document chunking strategies
  embeddings.py     - Local embedding generation via Ollama
  vectorstore.py    - In-memory vector similarity search
  pipeline.py       - Main RAG pipeline orchestrator
  cli.py            - Command-line interface
examples/
  demo_basic.py     - Basic ingestion and query demo
  demo_chat.py      - Interactive chat demo
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"llama3"` | Ollama model for generation |
| `embed_model` | `"nomic-embed-text"` | Ollama model for embeddings |
| `chunk_size` | `512` | Characters per chunk |
| `chunk_overlap` | `50` | Overlap between chunks |
| `top_k` | `3` | Number of chunks to retrieve |

## How It Works

1. **Ingest** - Documents are loaded and split into chunks using configurable strategies
2. **Embed** - Each chunk is converted to a vector embedding using Ollama
3. **Store** - Embeddings are stored in a lightweight in-memory vector store
4. **Query** - User questions are embedded and matched against stored chunks
5. **Generate** - Retrieved chunks are passed as context to the LLM for answer generation

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.
