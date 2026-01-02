from multiprocessing.queues import Queue

import pvporcupine

from src.va.audio.types import AudioFrame
from src.va.ipc.events import Event, WakeEvent


class PorcupineWorker:
    def __init__(
        self,
        audio_queue: Queue[AudioFrame],
        event_queue: Queue[Event],
        access_key: str,
        keyword_paths: list[str],
    ):
        self.audio_queue = audio_queue
        self.event_queue = event_queue

        self.keyword_paths = keyword_paths

        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key, keyword_paths=self.keyword_paths
            )
            print("[Porcupine] Initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Porcupine: {e}")

    def run(self):
        """Blocking loop to process audio frames."""
        print("[Porcupine] Worker running...")

        while True:
            frame: AudioFrame = self.audio_queue.get()

            try:
                keyword_index = self.porcupine.process(frame.intpcm)

                if keyword_index >= 0:
                    print(f"[Porcupine] Wake Word Detected")
                    self.event_queue.put(WakeEvent())

            except Exception as e:
                print(f"[Porcupine] Error processing frame: {e}")

    def cleanup(self):
        if self.porcupine:
            self.porcupine.delete()
