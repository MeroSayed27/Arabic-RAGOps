from pathlib import Path

from app.chunking.text_chunker import TextChunker
from app.ingestion.document_loader import DocumentLoader


def main() -> None:
    source_file = Path("data/sample/rag_basics.md")

    loader = DocumentLoader()
    document = loader.load_file(source_file)

    chunker = TextChunker(
        chunk_size=300,
        overlap=60,
    )

    chunks = chunker.split(
        document_id=document.document_id,
        text=document.text,
    )

    print("=" * 60)
    print("Arabic-RAGOps — Ingestion → Chunking Pipeline")
    print("=" * 60)
    print(f"Source: {document.source}")
    print(f"Language: {document.language}")
    print(f"Document characters: {len(document.text)}")
    print(f"Chunk size: {chunker.chunk_size}")
    print(f"Overlap: {chunker.overlap}")
    print(f"Total chunks: {len(chunks)}")
    print("-" * 60)

    for chunk in chunks:
        print(f"{chunk.chunk_id}")
        print(f"Range: {chunk.start}:{chunk.end}")
        print(chunk.text)
        print("-" * 60)


if __name__ == "__main__":
    main()