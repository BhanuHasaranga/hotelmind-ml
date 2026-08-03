"""Loads versioned prompt templates from `genai/prompts/system/*.md`.

Every prompt file starts with a version-header comment:
`<!-- prompt: <name> | version: <semver> -->`. Prompts are never inlined in
Python — always loaded from these files.
"""

import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "system"

_HEADER_RE = re.compile(r"<!--\s*prompt:\s*(?P<name>[\w\-]+)\s*\|\s*version:\s*(?P<version>[\w.]+)\s*-->")


@dataclass
class Prompt:
    name: str
    version: str
    text: str


class PromptNotFoundError(FileNotFoundError):
    pass


def _parse(raw: str, filename: str) -> Prompt:
    match = _HEADER_RE.search(raw.splitlines()[0] if raw.splitlines() else "")
    if not match:
        raise ValueError(f"Prompt file {filename} is missing the version header comment")
    body = raw[match.end() :].strip()
    return Prompt(name=match.group("name"), version=match.group("version"), text=body)


def load_prompt(name: str) -> Prompt:
    """Load a system prompt by filename stem, e.g. `load_prompt('hotel_analyst')`."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise PromptNotFoundError(f"No prompt file found for {name!r} at {path}")
    raw = path.read_text(encoding="utf-8")
    return _parse(raw, path.name)


def list_prompts() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))
