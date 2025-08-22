from typing import Dict, Any

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


class VllmGenerator:
    def __init__(
        self,
        model_id: str,
        num_gpus: int,
        seed: int,
        role_column_name: str,
        content_column_name: str,
        instruction: str,
        max_new_tokens: int,
        do_sample: bool,
        generation_config: Dict[str, Any],
    ) -> None:
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=num_gpus,
            seed=seed,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            is_enable_thinking=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.instruction = instruction

        if do_sample:
            self.generation_config = generation_config
        else:
            self.generation_config = {
                "temperature": 0,
                "top_p": 1,
            }

        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
            stop_token_ids=[self.tokenizer.eos_token_id],
            **self.generation_config,
        )

    def __call__(
        self,
        recommendation: str,
    ) -> str:
        prompt = self.get_prompt(recommendation=recommendation)
        generation = self.generate(prompt=prompt)
        return generation

    def generate(
        self,
        prompt: str,
    ) -> str:
        output = self.llm.generate(
            prompts=[prompt],
            sampling_params=self.sampling_params,
        )
        generation = output.outputs[0].text.strip()
        return generation

    def get_prompt(
        self,
        recommendation: str,
    ) -> str:
        conversation = [
            {
                self.role_column_name: "system",
                self.content_column_name: self.instruction,
            },
            {
                self.role_column_name: "user",
                self.content_column_name: recommendation,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
