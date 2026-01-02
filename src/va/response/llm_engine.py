from typing import Dict, Iterator, List

import ollama


class LLMEngine:
    def __init__(self, model_name: str, base_url: str | None = None):
        self.model_name = model_name
        self.client = ollama.Client(host=base_url) if base_url else ollama

    def generate_stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """
        Yields text chunks from the LLM.
        """
        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                think=False,
                options={
                    "temperature": 0.7,
                    "num_ctx": 2048,
                },
            )

            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content

        except Exception as e:
            print(f"[LLM Engine] Inference Error: {e}")
            yield "I'm sorry, I'm having trouble thinking right now."
