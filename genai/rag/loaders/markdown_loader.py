"""Markdown document loader. Splits on top-level (`#`/`##`) headings so each
section carries its own heading in metadata for better citation quality."""

import re
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def load_markdown(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    headings = list(_HEADING_RE.finditer(text))

    if not headings:
        return [{"text": text, "metadata": {"source": str(path), "doc_type": "markdown"}}]

    sections = []
    for i, match in enumerate(headings):
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append(
                {
                    "text": section_text,
                    "metadata": {
                        "source": str(path),
                        "doc_type": "markdown",
                        "heading": match.group(2).strip(),
                    },
                }
            )
    return sections
