from typing import Dict, List, Tuple

import math

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


class VllmReranker:
    def __init__(
        self,
        model_id: str,
        num_gpus: int,
        seed: int,
        instruction: str,
        max_length: int,
    ) -> None:
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=num_gpus,
            seed=seed,
            trust_remote_code=True,
            max_model_len=10000,
            enable_prefix_caching=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix,
            add_special_tokens=False,
        )

        self.true_token = self.tokenizer(
            "yes",
            add_special_tokens=False,
        ).input_ids[0]
        self.false_token = self.tokenizer(
            "no",
            add_special_tokens=False,
        ).input_ids[0]
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[
                self.true_token,
                self.false_token,
            ],
        )

        self.instruction = instruction
        self.max_length = max_length

    def __call__(
        self,
        query: str,
        candidates: List[str],
    ) -> List[float]:
        score = self.get_score(
            query=query,
            candidates=candidates,
        )
        return score

    def get_score(
        self,
        query: str,
        candidates: List[str],
    ) -> List[float]:
        pairs = list(zip([query] * len(candidates), candidates))
        messages = self.process_inputs(pairs=pairs)
        outputs = self.llm.generate(
            messages,
            self.sampling_params,
            use_tqdm=False,
        )
        scores = []
        for i in range(len(outputs)):
            final_logits = outputs[i].outputs[0].logprobs[-1]
            if self.true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[self.true_token].logprob
            if self.false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[self.false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)
        return scores

    def format_instruction(
        self,
        query: str,
        doc: str,
    ) -> List[Dict[str, str]]:
        text = [
            {
                "role": "system",
                "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".',
            },
            {
                "role": "user",
                "content": f"<Instruct>: {self.instruction}\n\n<Query>: {query}\n\n<Document>: {doc}",
            },
        ]
        return text

    def process_inputs(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[TokensPrompt]:
        messages = [
            self.format_instruction(
                query=query,
                doc=doc,
            )
            for query, doc in pairs
        ]
        messages = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        messages = [
            message[: self.max_length] + self.suffix_tokens for message in messages
        ]
        messages = [TokensPrompt(prompt_token_ids=message) for message in messages]
        return messages
