from multiprocessing.queues import Queue

from src.va.ipc.events import Event
from src.va.response.pipeline import ResponseWorker

from .types import GeneratedToken, GenerationTask


def run_response_worker(
    prompt_queue: Queue[GenerationTask],
    tts_text_queue: Queue[GeneratedToken],
    event_queue: Queue[Event],
    model_name: str = "qwen3:0.6b",
):
    """
    Response LLM Process Entry Point.
    """
    worker = ResponseWorker(prompt_queue, tts_text_queue, event_queue, model_name)
    try:
        worker.run()
    except KeyboardInterrupt:
        pass
