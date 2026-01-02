# Thanks to Silero official Github repo for inference configurations
import numpy as np
import onnxruntime as ort


class SileroVAD:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.threshold = threshold

        # ONNX IO Names
        self.in_audio = self.session.get_inputs()[0].name
        self.in_state = self.session.get_inputs()[1].name
        self.in_sr = self.session.get_inputs()[2].name
        self.out_prob = self.session.get_outputs()[0].name
        self.out_state = self.session.get_outputs()[1].name

        # Internal State
        self.reset_state()
        self.sr = np.array([16000], dtype=np.int64)

    def reset_state(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, frame: np.ndarray) -> bool:
        # Expects frame shape: (512,) float32
        prob, self.state = self.session.run(
            [self.out_prob, self.out_state],
            {
                self.in_audio: frame[None, :],  # Add batch dim -> (1, 512)
                self.in_state: self.state,
                self.in_sr: self.sr,
            },
        )
        return prob[0][0] > self.threshold
