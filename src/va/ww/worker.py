from multiprocessing.queues import Queue

from src.va.audio.types import AudioFrame
from src.va.config.va_config import VAConfig
from src.va.ipc.events import Event
from src.va.ww.wakeword_engine import PorcupineWorker


def run_porcupine_worker(
    audio_queue: Queue[AudioFrame],
    event_queue: Queue[Event],
    access_key: str,
    cfg: VAConfig,
):
    """
    Wake Word Process Entry Point.
    """
    worker = PorcupineWorker(audio_queue, event_queue, access_key, cfg.keyword_paths)
    try:
        worker.run()
    except KeyboardInterrupt:
        pass
    finally:
        worker.cleanup()
