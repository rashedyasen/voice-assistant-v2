from collections import deque
from typing import List

from src.va.audio.types import AudioFrame


class RingBuffer:
    def __init__(self, seconds: float, sample_rate: int, frame_size: int):
        self.max_frames = int((seconds * sample_rate) / frame_size)
        self._buffer: deque[AudioFrame] = deque(maxlen=self.max_frames)

    def push(self, frame: AudioFrame) -> None:
        self._buffer.append(frame)

    def dump(self) -> List[AudioFrame]:
        """Return a copy of buffered frames (oldest → newest)."""
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def has_data(self) -> bool:
        return bool(self._buffer)
