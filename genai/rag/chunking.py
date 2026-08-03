"""Recursive/token-aware text splitter for RAG indexing.

A lightweight, dependency-free recursive splitter (mirrors the strategy of
LangChain's `RecursiveCharacterTextSplitter` without requiring it): tries to
split on paragraph boundaries first, then sentences, then hard character
windows, always respecting `CHUNK_SIZE`/`CHUNK_OVERLAP`.
"""

from __future__ import annotations

import re

from genai.config.genai_settings import genai_settings

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_on(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    return text.split(separator)


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, *rest = separators
    parts = _split_on(text, sep)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_recursive_split(part, chunk_size, rest))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split `text` into overlapping chunks of at most `chunk_size` chars."""
    chunk_size = chunk_size or genai_settings.CHUNK_SIZE
    overlap = overlap if overlap is not None else genai_settings.CHUNK_OVERLAP
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    raw_chunks = _recursive_split(text, chunk_size, _SEPARATORS)
    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
            continue
        prev_tail = raw_chunks[i - 1][-overlap:]
        overlapped.append((prev_tail + " " + chunk).strip())
    return overlapped


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk a list of `{"text": ..., "metadata": {...}}` documents,
    propagating and enriching metadata with a `chunk_index`."""
    chunked = []
    for doc in documents:
        pieces = chunk_text(doc["text"])
        for idx, piece in enumerate(pieces):
            chunked.append(
                {
                    "text": piece,
                    "metadata": {**doc.get("metadata", {}), "chunk_index": idx},
                }
            )
    return chunked
