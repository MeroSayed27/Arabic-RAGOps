"""
Arabic-RAGOps
Multilingual AI Knowledge & Evaluation Assistant

Initial application entry point.

This file is intentionally small at the center of the system.
As the project grows, ingestion, retrieval, generation, and evaluation
will move into their own modules under the app/ package.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class ProjectInfo:
    """Basic project metadata."""

    name: str = "Arabic-RAGOps"
    description: str = (
        "Multilingual AI Knowledge & Evaluation Assistant"
    )
    version: str = "0.1.0"


def build_project_info() -> ProjectInfo:
    """Return the current project metadata."""
    return ProjectInfo()


def run_health_check() -> dict[str, str]:
    """
    Run a minimal application health check.

    This will become more useful later when we connect:
    - configuration
    - document ingestion
    - embeddings
    - vector storage
    - retrieval
    - reranking
    - generation
    - evaluation
    """
    now = datetime.now(UTC).isoformat()

    return {
        "status": "ok",
        "project": "Arabic-RAGOps",
        "version": "0.1.0",
        "timestamp_utc": now,
    }


def main() -> None:
    """Application entry point."""

    info = build_project_info()
    health = run_health_check()

    print("=" * 60)
    print(info.name)
    print(info.description)
    print(f"Version: {info.version}")
    print("=" * 60)
    print(f"Status: {health['status']}")
    print(f"UTC Time: {health['timestamp_utc']}")
    print("=" * 60)
    print("Arabic-RAGOps is ready for development.")
    print("")


if __name__ == "__main__":
    main()