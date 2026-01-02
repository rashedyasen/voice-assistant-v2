from multiprocessing.queues import Queue

from src.va.intent.intent_engine import IntentEngine
from src.va.ipc.events import Event
from src.va.stt.types import TranscriptionMsg


def run_intent_worker(
    text_queue: Queue[TranscriptionMsg],
    event_queue: Queue[Event],
    model_name: str = "qwen3:0.6b",
):
    """
    Intent Process Entry Point.
    """
    worker = IntentEngine(text_queue, event_queue, model_name)

    try:
        worker.run()
    except KeyboardInterrupt:
        print("[Intent] Stopping worker...")
    except Exception as e:
        print(f"[Intent] Critical Process Error: {e}")
