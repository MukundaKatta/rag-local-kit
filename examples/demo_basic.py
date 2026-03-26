"""Basic demo of rag-local-kit: ingest and query documents."""

from rag_local_kit import RAGPipeline


def main():
    # Create a RAG pipeline with default settings
    # Requires Ollama running with llama3 and nomic-embed-text models
    rag = RAGPipeline(
        model="llama3",
        embed_model="nomic-embed-text",
        chunk_size=512,
        top_k=3,
    )

    print("=" * 60)
    print("rag-local-kit Basic Demo")
    print("=" * 60)

    # Check if Ollama is available
    if not rag.embeddings.is_available():
        print("\nOllama is not running or the embedding model is not available.")
        print("Please start Ollama and pull the required models:")
        print("  ollama serve")
        print("  ollama pull llama3")
        print("  ollama pull nomic-embed-text")
        return

    # Ingest documents from a directory
    print("\nIngesting documents...")
    try:
        count = rag.ingest("./docs/")
        print(f"Ingested {count} chunks.")
    except FileNotFoundError:
        print("No ./docs/ directory found. Creating a sample document...")
        # Create a sample document for demo purposes
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        sample = os.path.join(tmpdir, "sample.txt")
        with open(sample, "w") as f:
            f.write("""Artificial intelligence (AI) is transforming industries worldwide.
Machine learning, a subset of AI, enables systems to learn from data.
Deep learning uses neural networks with many layers to model complex patterns.
Natural language processing (NLP) allows machines to understand human language.
Computer vision enables machines to interpret and analyze visual information.
Reinforcement learning trains agents to make decisions through trial and error.
Transfer learning allows models trained on one task to be applied to another.
RAG (Retrieval-Augmented Generation) combines information retrieval with text generation.""")
        count = rag.ingest(tmpdir)
        print(f"Ingested {count} chunks from sample data.")

    # Ask some questions
    questions = [
        "What is machine learning?",
        "How does RAG work?",
        "What are the main types of AI?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = rag.query(q)
        print(f"A: {answer}")

    print(f"\nTotal chunks in store: {rag.num_chunks}")


if __name__ == "__main__":
    main()
