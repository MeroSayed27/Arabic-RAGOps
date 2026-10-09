from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass(frozen=True)
class Document:
    """A normalized document ready for later RAG processing."""

    document_id: str
    source: str
    title: str
    text: str
    language: str


class DocumentLoader:
    """Load safe local text-based documents into a common structure."""

    def load_file(self, file_path: str | Path) -> Document:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported types: {supported}"
            )

        text = path.read_text(encoding="utf-8").strip()

        if not text:
            raise ValueError(f"Document is empty: {path}")

        return Document(
            document_id=path.stem,
            source=str(path),
            title=path.stem.replace("_", " ").replace("-", " ").title(),
            text=text,
            language=self._detect_language(text),
        )

    @staticmethod
    def _detect_language(text: str) -> str:
        """Simple baseline language detection for Arabic/English text."""

        arabic_characters = sum(
            "\u0600" <= character <= "\u06FF"
            for character in text
        )

        latin_characters = sum(
            ("a" <= character.lower() <= "z")
            for character in text
        )

        if arabic_characters > 0 and latin_characters > 0:
            return "mixed"

        if arabic_characters > 0:
            return "ar"

        if latin_characters > 0:
            return "en"

        return "unknown"