# NOTES
# The plumbing that runs the AI model on the cluster's graphics cards (GPUs).
# Everything else produces a prompt. This is what turns a prompt into the
# model's answer.
#
# Two runners:
#   VLLMModelClient is the fast one. Built for running lots of prompts and
#   splitting one huge model across several GPUs. We use this for the real
#   58,000-paper run.
#   TransformersModelClient is the simple one. Fine for the pilot or debugging on
#   one GPU.
#
# You don't need the internals unless you're changing how the model runs. The
# key knob is tensor_parallel_size, which is how many GPUs to spread the
# 70-billion-parameter model across. It's too big to fit on one.
#
# temperature = 0 means the model answers as flatly and repeatably as possible.
# We want faithful extraction, not creative writing, so creativity is off.
"""Run the model on the GPUs. Two backends: vLLM (fast) and transformers (simple)."""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0  # deterministic extraction, not creative writing


@dataclass
class VLLMModelClient:
    """The fast runner. tensor_parallel_size should match the number of GPUs the
    job requests, e.g. 4 for a 70B model split across four A100-80GB cards.
    """

    model_id: str
    tensor_parallel_size: int = 1
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    dtype: str = "bfloat16"
    _llm: object = field(default=None, init=False, repr=False)

    def _load(self):
        # Load the model once, the first time it's needed. It's about 140GB, so
        # we don't want to load it twice. The import is inside the function on
        # purpose. It's a heavy GPU-only library we only touch on a GPU node.
        if self._llm is None:
            from vllm import LLM

            self._llm = LLM(
                model=self.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=self.dtype,
            )
        return self._llm

    def generate(self, prompt: str) -> str:
        # One prompt in, one answer out.
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        # Many prompts at once. Much more efficient on a GPU than one at a time.
        from vllm import SamplingParams

        llm = self._load()
        params = SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens)
        outputs = llm.generate(prompts, params)
        # vLLM can hand results back out of order, so sort by request id.
        outputs = sorted(outputs, key=lambda o: o.request_id)
        return [o.outputs[0].text for o in outputs]


@dataclass
class TransformersModelClient:
    """The simple runner. Loads the model once and splits it across whatever GPUs
    are visible. Fine for the pilot. For the full run prefer the vLLM one.
    """

    model_id: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    _pipeline: object = field(default=None, init=False, repr=False)

    def _load(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                device_map="auto",  # spread across whatever GPUs are visible
                torch_dtype=torch.bfloat16,
            )
        return self._pipeline

    def generate(self, prompt: str) -> str:
        pipe = self._load()
        do_sample = self.temperature > 0
        result = pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            return_full_text=False,
        )
        return result[0]["generated_text"]


def build_client(model_id: str, backend: str = "vllm", **kwargs):
    # Pick a runner by name. Defaults to the fast one.
    if backend == "vllm":
        return VLLMModelClient(model_id=model_id, **kwargs)
    if backend == "transformers":
        return TransformersModelClient(model_id=model_id, **kwargs)
    raise ValueError(f"Unknown backend {backend!r}, expected 'vllm' or 'transformers'")
