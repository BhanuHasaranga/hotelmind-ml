"""Retrieval -> context compression -> prompt -> LLM -> answer, with
retrieval citations attached to the response (not just prose).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from genai.llm.base import LLMProvider
from genai.prompts.loader import load_prompt
from genai.rag.memory import SessionMemory, get_memory
from genai.rag.retriever import HybridRetriever, MetadataFilter, compress_context
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Citation:
    source: str
    doc_type: str | None = None
    score: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class QAResult:
    answer: str
    citations: list[Citation]
    used_llm: bool


class NoContextError(RuntimeError):
    """Raised when the vector store is empty (not yet indexed)."""


def _build_citations(chunks) -> list[Citation]:
    citations = []
    for chunk in chunks:
        citations.append(
            Citation(
                source=chunk.metadata.get("source", "unknown"),
                doc_type=chunk.metadata.get("doc_type"),
                score=round(chunk.score, 4),
                extra={k: v for k, v in chunk.metadata.items() if k not in ("source", "doc_type", "text")},
            )
        )
    return citations


def _fallback_answer(query: str, context: str) -> str:
    if not context:
        return (
            "I don't have an LLM provider available right now, and no relevant "
            "indexed context was found for this question."
        )
    return (
        "No LLM provider is available/configured, so here is the most relevant "
        f"retrieved context for your question ('{query}'):\n\n{context[:1000]}"
    )


def answer_question(
    query: str,
    retriever: HybridRetriever,
    llm_provider: LLMProvider | None,
    persona: str = "hotel_analyst",
    session_id: str | None = None,
    metadata_filter: MetadataFilter | None = None,
    memory: SessionMemory | None = None,
) -> QAResult:
    if retriever.store.size == 0:
        raise NoContextError("RAG index is empty; run POST /rag/index first.")

    chunks = retriever.retrieve(query, metadata_filter=metadata_filter)
    context = compress_context(chunks)
    citations = _build_citations(chunks)

    memory = memory or get_memory()
    history_context = memory.as_prompt_context(session_id) if session_id else ""

    system_prompt = load_prompt(persona).text
    prompt_parts = [f"Context:\n{context}" if context else "Context: (no relevant context retrieved)"]
    if history_context:
        prompt_parts.append(f"Conversation so far:\n{history_context}")
    prompt_parts.append(f"Question: {query}")
    prompt = "\n\n".join(prompt_parts)

    if llm_provider is None or not llm_provider.is_available():
        answer = _fallback_answer(query, context)
        used_llm = False
    else:
        try:
            result = llm_provider.generate(prompt, system=system_prompt)
            answer = result.text
            used_llm = True
        except Exception as exc:
            logger.warning("LLM generation failed in qa_chain, falling back: %s", exc)
            answer = _fallback_answer(query, context)
            used_llm = False

    if session_id:
        memory.add_turn(session_id, "user", query)
        memory.add_turn(session_id, "assistant", answer)

    return QAResult(answer=answer, citations=citations, used_llm=used_llm)


def stream_answer(
    query: str,
    retriever: HybridRetriever,
    llm_provider: LLMProvider | None,
    persona: str = "hotel_analyst",
    metadata_filter: MetadataFilter | None = None,
) -> Iterator[str]:
    """Streaming variant for SSE. Yields text chunks; if no LLM is
    available, yields the fallback answer as a single chunk."""
    if retriever.store.size == 0:
        raise NoContextError("RAG index is empty; run POST /rag/index first.")

    chunks = retriever.retrieve(query, metadata_filter=metadata_filter)
    context = compress_context(chunks)
    system_prompt = load_prompt(persona).text
    prompt = f"Context:\n{context}\n\nQuestion: {query}"

    if llm_provider is None or not llm_provider.is_available():
        yield _fallback_answer(query, context)
        return

    try:
        yield from llm_provider.stream(prompt, system=system_prompt)
    except Exception as exc:
        logger.warning("LLM streaming failed in qa_chain, falling back: %s", exc)
        yield _fallback_answer(query, context)
