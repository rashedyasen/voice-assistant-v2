from dataclasses import dataclass, field
from typing import Dict, List

from src.va.orchestrator.turn_context import TurnContext


@dataclass(frozen=True)
class GenerationTask:
    """Instructions for the LLM Worker."""

    messages: List[Dict[str, str]]  # Full conversation context [{"role": "user", ...}]
    ctx: TurnContext
    stop_tokens: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneratedToken:
    token: str | None
    ctx: TurnContext

