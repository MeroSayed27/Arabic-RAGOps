from app.chunking.text_chunker import TextChunker


def main() -> None:
    text = (
        "RAG combines retrieval with generation. "
        "A retrieval system finds relevant evidence. "
        "The language model then uses that evidence to answer the question. "
        "Chunking helps divide larger documents into searchable pieces. "
        "Arabic-RAGOps evaluates Arabic, English, and mixed-language queries."
    )

    chunker = TextChunker(chunk_size=120, overlap=30)
    chunks = chunker.split("demo_document", text)

    print("=" * 60)
    print("Arabic-RAGOps — Chunking Test")
    print("=" * 60)
    print(f"Chunk size: {chunker.chunk_size}")
    print(f"Overlap: {chunker.overlap}")
    print(f"Total chunks: {len(chunks)}")
    print("-" * 60)

    for chunk in chunks:
        print(f"ID: {chunk.chunk_id}")
        print(f"Range: {chunk.start}:{chunk.end}")
        print(f"Text: {chunk.text}")
        print("-" * 60)


if __name__ == "__main__":
    main() 