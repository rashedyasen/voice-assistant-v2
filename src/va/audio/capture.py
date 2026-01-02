import time
from typing import Iterator

import numpy as np
import sounddevice as sd

from src.va.audio.types import AudioFrame


class AudioInput:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,
        device: int | None = None,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.device = device

        self._stream = sd.InputStream(
            samplerate=sample_rate,
            blocksize=frame_size,
            channels=1,
            dtype="int16",
            device=device,
        )

    def frames(self) -> Iterator[AudioFrame]:
        frame_duration = self.frame_size / self.sample_rate
        timestamp = time.monotonic()

        with self._stream:
            while True:
                raw_pcm, status = self._stream.read(self.frame_size)
                if status:
                    raise RuntimeError(f"Audio input overflow: {status}")

                assert raw_pcm.shape[1] == 1
                int_pcm = raw_pcm.flatten()
                float_pcm = int_pcm.astype(np.float32) / 32768.0

                yield AudioFrame(
                    pcm=float_pcm,
                    intpcm=int_pcm,
                    sample_rate=self.sample_rate,
                    timestamp=timestamp,
                )

                timestamp += frame_duration
