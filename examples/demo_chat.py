"""Interactive chat demo for rag-local-kit."""

from rag_local_kit import RAGPipeline
import sys


def main():
    print("=" * 60)
    print("rag-local-kit Interactive Chat")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nUsage: python demo_chat.py <path_to_documents>")
        print("Example: python demo_chat.py ./my_docs/")
        return

    doc_path = sys.argv[1]

    rag = RAGPipeline(
        model="llama3",
        embed_model="nomic-embed-text",
        chunk_size=512,
        top_k=3,
    )

    # Check Ollama availability
    if not rag.embeddings.is_available():
        print("\nOllama is not running or models are not available.")
        print("Run: ollama serve && ollama pull llama3 && ollama pull nomic-embed-text")
        return

    # Ingest documents
    print(f"\nIngesting documents from: {doc_path}")
    count = rag.ingest(doc_path)
    if count == 0:
        print("No documents were ingested. Check the path and try again.")
        return

    print(f"\nReady! {count} chunks loaded.")
    print("Ask questions about your documents. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not question:
                continue

            answer = rag.query(question)
            print(f"\nAssistant: {answer}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
