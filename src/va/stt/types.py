from dataclasses import dataclass
from enum import Enum

from src.va.orchestrator.turn_context import TurnContext


class TranscriptionType(Enum):
    PARTIAL = "partial"  # Unfinished sentence (changing)
    FINAL = "final"  # Finished sentence (committed)


@dataclass
class TranscriptionMsg:
    """The output payload for your Intent LLM."""

    text: str
    type: TranscriptionType
    timestamp: float
    ctx: TurnContext
