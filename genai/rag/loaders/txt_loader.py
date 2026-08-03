"""Plain-text document loader."""

from pathlib import Path


def load_txt(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [{"text": text, "metadata": {"source": str(path), "doc_type": "txt"}}]
