from dataclasses import dataclass

from src.va.orchestrator.turn_context import TurnContext


@dataclass(frozen=True)
class TTSAudio:
    pcm: bytes | None  # int frames for playback
    sample_rate: int  # e.g., 22050 (Piper usually)
    ctx: TurnContext
