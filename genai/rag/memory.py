"""Per-session conversation memory with TTL expiration and context-window
trimming, gated by `ENABLE_MEMORY`. In-process (dict-backed); adequate for a
single-process FastAPI deployment as documented for the rest of this repo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from genai.config.genai_settings import genai_settings


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class SessionMemory:
    def __init__(self, max_turns: int | None = None, ttl_seconds: int | None = None) -> None:
        self.max_turns = max_turns or genai_settings.MEMORY_MAX_TURNS
        self.ttl_seconds = ttl_seconds or genai_settings.MEMORY_TTL_SECONDS
        self._sessions: dict[str, list[Turn]] = {}

    def _prune_expired(self, session_id: str) -> None:
        turns = self._sessions.get(session_id, [])
        cutoff = time.time() - self.ttl_seconds
        self._sessions[session_id] = [t for t in turns if t.timestamp >= cutoff]

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        if not genai_settings.ENABLE_MEMORY:
            return
        self._prune_expired(session_id)
        turns = self._sessions.setdefault(session_id, [])
        turns.append(Turn(role=role, content=content))
        if len(turns) > self.max_turns:
            self._sessions[session_id] = turns[-self.max_turns :]

    def get_history(self, session_id: str) -> list[Turn]:
        if not genai_settings.ENABLE_MEMORY:
            return []
        self._prune_expired(session_id)
        return list(self._sessions.get(session_id, []))

    def as_prompt_context(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return ""
        lines = [f"{turn.role}: {turn.content}" for turn in history]
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


_default_memory: SessionMemory | None = None


def get_memory() -> SessionMemory:
    global _default_memory
    if _default_memory is None:
        _default_memory = SessionMemory()
    return _default_memory
