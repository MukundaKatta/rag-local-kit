"""Command-line interface for rag-local-kit."""

import argparse
import sys

from .pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="rag-local-kit: Local RAG pipeline powered by Ollama"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the vector store")
    ingest_parser.add_argument("path", help="File or directory path to ingest")
    ingest_parser.add_argument("--model", default="nomic-embed-text", help="Embedding model")
    ingest_parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in chars")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--model", default="llama3", help="Generation model")
    query_parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with your documents")
    chat_parser.add_argument("path", help="File or directory to chat about")
    chat_parser.add_argument("--model", default="llama3", help="Generation model")

    args = parser.parse_args()

    if args.command == "ingest":
        rag = RAGPipeline(embed_model=args.model, chunk_size=args.chunk_size)
        count = rag.ingest(args.path)
        print(f"Successfully ingested {count} chunks.")

    elif args.command == "query":
        rag = RAGPipeline(model=args.model, top_k=args.top_k)
        print("Note: You need to ingest documents first in the same session.")
        answer = rag.query(args.question)
        print(f"\nAnswer: {answer}")

    elif args.command == "chat":
        rag = RAGPipeline(model=args.model)
        print(f"Ingesting documents from: {args.path}")
        count = rag.ingest(args.path)
        print(f"Ready! {count} chunks loaded. Type 'quit' to exit.\n")

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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
