"""PDF document loader using pypdf, one chunk per page."""

from pathlib import Path


def load_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                {
                    "text": text,
                    "metadata": {"source": str(path), "doc_type": "pdf", "page": page_num + 1},
                }
            )
    return docs
