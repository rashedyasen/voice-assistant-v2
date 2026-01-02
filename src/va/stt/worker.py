from multiprocessing.queues import Queue

from src.va.audio.types import AudioFrame
from src.va.config.va_config import VAConfig
from src.va.ipc.events import Event

from .pipeline import SpeechPipeline
from .types import TranscriptionMsg


def run_speech_worker(
    audio_queue: Queue[AudioFrame],
    text_queue: Queue[TranscriptionMsg],
    event_queue: Queue[Event],
    config: VAConfig,
):
    """
    Speech process Entry Point
    """
    pipeline = SpeechPipeline(audio_queue, text_queue, event_queue, config)

    try:
        pipeline.run()
    except Exception as e:
        print(f"[Child Process Error] {e}")
