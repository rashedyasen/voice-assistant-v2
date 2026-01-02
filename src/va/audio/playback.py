from multiprocessing.queues import Queue

import numpy as np
import sounddevice as sd

from src.va.ipc.events import Event, PlayBackEvent
from src.va.tts.types import TTSAudio


def playback_thread_func(
    playback_q: Queue[TTSAudio], event_q: Queue[Event], sample_rate=22050
):
    """
    Playback thread for Speaking
    """
    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16")

    stream.start()
    print("[Playback] Thread Started.")

    while True:
        try:
            item = playback_q.get()

            if item is None:
                break

            if isinstance(item, TTSAudio):
                if item.ctx.cancelled.is_set():
                    continue
                if item.pcm is None:
                    event_q.put(PlayBackEvent())
                    print("[PLAY] Speaking done!")
                else:
                    audio_data = np.frombuffer(item.pcm, dtype=np.int16)
                    stream.write(audio_data)

        except Exception as e:
            print(f"[Playback] Error: {e}")

    stream.stop()
    stream.close()
