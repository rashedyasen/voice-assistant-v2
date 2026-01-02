# Thanks to Piper official Github repo for phonemizer configurations
import json
from pathlib import Path

import numpy as np
from piper.phoneme_ids import phonemes_to_ids
from piper.phonemize_espeak import EspeakPhonemizer


class PhonemizerEngine:
    def __init__(self, config_path: str, espeak_data_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.phoneme_id_map = self.config["phoneme_id_map"]
        self.voice_name = self.config["espeak"]["voice"]

        self.phonemizer = EspeakPhonemizer(Path(espeak_data_path))

    def text_to_ids(self, text: str) -> np.ndarray:
        """Converts text string to numpy array of IDs."""

        # 1. Text -> Phonemes
        sentence_phonemes = self.phonemizer.phonemize(self.voice_name, text)

        # Flatten list of lists
        phonemes = []
        for sent in sentence_phonemes:
            phonemes.extend(sent)

        # 2. Phonemes -> IDs
        ids = phonemes_to_ids(phonemes, self.phoneme_id_map)

        # 3. Format for ONNX (Add Batch Dimension)
        return np.array(ids, dtype=np.int64)[None, :]
