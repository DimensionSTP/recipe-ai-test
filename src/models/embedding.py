import numpy as np

from vllm import LLM


class VllmEmbedding:
    def __init__(
        self,
        model_id: str,
        num_gpus: int,
        seed: int,
        instruction: str,
    ) -> None:
        self.llm = LLM(
            model=model_id,
            task="embed",
            tensor_parallel_size=num_gpus,
            seed=seed,
            trust_remote_code=True,
        )

        self.instruction = instruction

    def __call__(
        self,
        query: str,
    ) -> np.ndarray:
        embedding = self.embed(query=query)
        return embedding

    def embed(
        self,
        query: str,
    ) -> np.ndarray:
        input_text = self.get_detailed_instruct(query=query)
        output = self.llm.embed(input_text)
        embedding = output.outputs.embedding
        return embedding.cpu().numpy(dtype=np.float32)

    def get_detailed_instruct(
        self,
        query: str,
    ) -> str:
        return f"Instruct: {self.instruction}\nQuery:{query}"
