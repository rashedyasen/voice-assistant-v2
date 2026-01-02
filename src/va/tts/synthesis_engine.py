# Thanks to Piper Github Repo for onnx configuration and scales.
import json

import numpy as np
import onnxruntime as ort


class PiperEngine:
    def __init__(self, model_path: str, config_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.sample_rate = config["audio"]["sample_rate"]

        # Pre-calculate scales (Optimization)
        self.scales = np.array(
            [
                config.get("noise_scale", 0.667),
                config.get("length_scale", 1.0),
                config.get("noise_w_scale", 0.8),
            ],
            dtype=np.float32,
        )
        self.multi_speaker = config.get("num_speakers", 1) > 1

    def synthesize(self, phoneme_ids: np.ndarray) -> np.ndarray:
        """Runs the ONNX inference."""

        # Prepare Inputs
        phoneme_lengths = np.array([phoneme_ids.shape[1]], dtype=np.int64)

        inputs = {
            "input": phoneme_ids,
            "input_lengths": phoneme_lengths,
            "scales": self.scales,
        }

        if self.multi_speaker:
            inputs["sid"] = np.array([0], dtype=np.int64)  # Default speaker 0

        # Run Inference
        audio = self.session.run(None, inputs)[0].squeeze()

        return audio
