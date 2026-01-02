# Thanks to official Github Repo of moonshine - live caption implementation
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class MoonshineSTT:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        tokenizer_path: str,
        sample_rate: int = 16000,
        token_rate: int = 50,
    ):
        self.encoder = ort.InferenceSession(
            encoder_path, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            decoder_path, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # ---- Model constants (must match ONNX export) ----
        self.num_layers = 8
        self.num_key_value_heads = 8
        self.head_dim = 52

        self.start_token = 1
        self.eos_token = 2

        self.sample_rate = sample_rate
        self.token_rate = token_rate

        # Cache encoder input names
        self.encoder_input_names = [i.name for i in self.encoder.get_inputs()]

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Full offline transcription.
        Faithfully mirrors the reference Moonshine inference script.
        """

        # -------------------------
        # 1. Encoder
        # -------------------------
        audio = audio.astype(np.float32)
        audio_in = audio[None, :]  # (1, T)

        enc_inputs = {"input_values": audio_in}

        if "attention_mask" in self.encoder_input_names:
            enc_inputs["attention_mask"] = np.ones_like(audio_in, dtype=np.int64)

        encoder_hidden_states = self.encoder.run(None, enc_inputs)[0]

        # -------------------------
        # 2. Decoder setup
        # -------------------------
        past = self._init_past()

        tokens = [self.start_token]
        input_ids = np.array([[self.start_token]], dtype=np.int64)

        max_len = int((len(audio) / self.sample_rate) * self.token_rate)

        use_cache = False

        # -------------------------
        # 3. Autoregressive decode
        # -------------------------
        for _ in range(max_len):
            dec_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "use_cache_branch": np.array([use_cache], dtype=np.bool_),
                **past,
            }

            logits, *present = self.decoder.run(None, dec_inputs)

            next_token = int(np.argmax(logits[0, -1]))

            tokens.append(next_token)

            if next_token == self.eos_token:
                break

            input_ids = np.array([[next_token]], dtype=np.int64)

            # ---- Update cache (CRITICAL) ----
            for k, v in zip(past.keys(), present):
                if (not use_cache) or ("decoder" in k):
                    past[k] = v

            use_cache = True

        # -------------------------
        # 4. Decode tokens → text
        # -------------------------
        return self.tokenizer.decode(tokens)

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _init_past(self):
        """
        Create empty KV cache with EXACT shape expected by ONNX.
        """
        past = {}

        shape = (0, self.num_key_value_heads, 1, self.head_dim)

        for i in range(self.num_layers):
            past[f"past_key_values.{i}.decoder.key"] = np.zeros(shape, np.float32)
            past[f"past_key_values.{i}.decoder.value"] = np.zeros(shape, np.float32)
            past[f"past_key_values.{i}.encoder.key"] = np.zeros(shape, np.float32)
            past[f"past_key_values.{i}.encoder.value"] = np.zeros(shape, np.float32)

        return past
