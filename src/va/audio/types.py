from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioFrame:
    pcm: np.ndarray    # float for moonshine & vad
    sample_rate: int   # 16 KHz
    timestamp: float   # monotonic time
    intpcm: np.ndarray # int for porcupine
