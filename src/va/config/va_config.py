from dataclasses import dataclass


@dataclass(frozen=True)
class VAConfig:
    keyword_paths: list[str]
    piper_path: str = ""
    phoneme_config_path: str = ""
    espeak_path: str = ""
    moonshine_enc_path: str = ""
    moonshine_dec_path: str = ""
    tokenizer_path: str = ""
    silero_path: str = ""
    intent_llm: str = ""
    simple_llm: str = ""
    complex_llm: str = ""


def default_config() -> VAConfig:
    return VAConfig(
        keyword_paths=["src/va/data/astra.ppn"],
        piper_path="src/va/data/piper.onnx",
        phoneme_config_path="src/va/data/piper.json",
        espeak_path="src/va/data/espeak-ng-data",
        moonshine_enc_path="src/va/data/moonshine_enc.onnx",
        moonshine_dec_path="src/va/data/moonshine_dec.onnx",
        tokenizer_path="src/va/data/tokenizer.json",
        silero_path="src/va/data/silero_vad.onnx",
        intent_llm="qwen3-0.6b",
        simple_llm="qwen3-0.6b",
        complex_llm="llama3.1:8b",
    )
