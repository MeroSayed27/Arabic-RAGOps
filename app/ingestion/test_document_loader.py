from pathlib import Path

from app.ingestion.document_loader import DocumentLoader


def main() -> None:
    sample_file = Path("data/sample/rag_basics.md")

    loader = DocumentLoader()
    document = loader.load_file(sample_file)

    print("=" * 60)
    print("Arabic-RAGOps — Document Ingestion Test")
    print("=" * 60)
    print(f"Document ID: {document.document_id}")
    print(f"Title: {document.title}")
    print(f"Source: {document.source}")
    print(f"Language: {document.language}")
    print(f"Characters: {len(document.text)}")
    print("-" * 60)
    print("First 300 characters:")
    print(document.text[:300])
    print("=" * 60)


if __name__ == "__main__":
    main()