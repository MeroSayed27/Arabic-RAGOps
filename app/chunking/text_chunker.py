from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """A searchable piece of a document."""

    chunk_id: str
    document_id: str
    text: str
    start: int
    end: int


class TextChunker:
    """Split documents into overlapping character-based chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document_id: str, text: str) -> list[TextChunk]:
        """Split text into overlapping chunks."""
        cleaned_text = text.strip()

        if not cleaned_text:
            return []

        chunks: list[TextChunk] = []
        step = self.chunk_size - self.overlap
        start = 0
        index = 0

        while start < len(cleaned_text):
            end = min(start + self.chunk_size, len(cleaned_text))
            chunk_text = cleaned_text[start:end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document_id}_chunk_{index:03d}",
                        document_id=document_id,
                        text=chunk_text,
                        start=start,
                        end=end,
                    )
                )

            if end >= len(cleaned_text):
                break

            start += step
            index += 1

        return chunks