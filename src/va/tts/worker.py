from multiprocessing.queues import Queue

from src.va.config.va_config import VAConfig
from src.va.ipc.events import Event
from src.va.response.types import GeneratedToken

from .pipeline import TTSPipeline
from .types import TTSAudio


def run_tts_process(
    text_queue: Queue[GeneratedToken],
    playback_queue: Queue[TTSAudio],
    event_queue: Queue[Event],
    cfg: VAConfig,
):
    """
    TTS Process Entry Point.
    """
    worker = TTSPipeline(text_queue, playback_queue, event_queue, cfg)
    try:
        worker.run()
    except KeyboardInterrupt:
        pass
