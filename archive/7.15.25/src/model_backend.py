"""Model backends for running inference locally on Rivanna, where GPUs are the
point (no hosted API, no rate limits). They all implement the same small
ModelClient interface from src/extract_demographics.py (a .generate(prompt)
method that returns a string), so any of them slots straight into
extract_demographics()/run_pilot.py/run_pipeline.py.

Use vLLM for anything past the 46-article pilot: it batches requests and uses
paged attention, which starts to matter once you're processing the whole corpus.
The transformers backend is a simpler fallback for small allocations or debugging
on a single GPU. The echo backend loads no model at all; it's a GPU-free dry-run
for checking the wiring before you request a GPU allocation (see EchoModelClient).

Note: none of the GPU backends have run in this sandbox (no GPU, no weights, no
network to huggingface.co), so see docs/RUNNING_ON_RIVANNA.md for how to validate
them on a real Rivanna node before trusting the output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0  # we want deterministic extraction, not creativity


@dataclass
class VLLMModelClient:
    """Batched local inference through vLLM's offline LLM API.

    Set tensor_parallel_size to the number of GPUs you asked for in the SLURM job
    (see slurm/run_pipeline.slurm), e.g. 4 for a 70B model split across four
    A100-80GBs.
    """

    model_id: str
    tensor_parallel_size: int = 1
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    dtype: str = "bfloat16"
    _llm: object = field(default=None, init=False, repr=False)

    def _load(self):
        if self._llm is None:
            from vllm import LLM  # imported here since it's heavy and GPU-only

            self._llm = LLM(
                model=self.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=self.dtype,
            )
        return self._llm

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        from vllm import SamplingParams

        llm = self._load()
        params = SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens)
        outputs = llm.generate(prompts, params)
        # vLLM doesn't guarantee the outputs come back in the order we sent them,
        # so sort by the request index it attaches.
        outputs = sorted(outputs, key=lambda o: o.request_id)
        return [o.outputs[0].text for o in outputs]


@dataclass
class TransformersModelClient:
    """Simpler single-process fallback using Hugging Face transformers.

    Loads the model once per process with device_map="auto" (it spreads across
    whatever GPUs it can see). Fine for the 46-article pilot; for the full run
    you want vLLM's batching instead.
    """

    model_id: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    _pipeline: object = field(default=None, init=False, repr=False)

    def _load(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline  # imported here since it's heavy

            self._pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                device_map="auto",
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


_DOI_IN_PROMPT = re.compile(r"Article DOI:\s*(\S+)")


@dataclass
class EchoModelClient:
    """GPU-free dry-run backend: no model, no network.

    It returns a schema-valid "nothing reported" answer for whatever DOI is in the
    prompt (extract_demographics.EXTRACTION_PROMPT_TEMPLATE writes an
    "Article DOI: <doi>" line). That's enough to run the whole path (read the CSV,
    read the XML, build the prompt, parse and validate, write the JSON) on the
    login node before you request GPUs. It's a wiring check, not a real
    extraction.
    """

    model_id: str

    def generate(self, prompt: str) -> str:
        match = _DOI_IN_PROMPT.search(prompt)
        if not match:
            raise ValueError(
                "EchoModelClient could not find an 'Article DOI:' line in the prompt; "
                "the prompt template may have changed."
            )
        doi = match.group(1)
        row = {
            "doi": doi,
            "sample_id": 1,
            "gender_reported": 0,
            "gender_pct": {},
            "race_reported": 0,
            "race_pct": {},
            "education_reported": 0,
            "education_pct": {},
            "ses_reported": 0,
            "ses_value": None,
        }
        return json.dumps([row])


def build_client(model_id: str, backend: str = "vllm", **kwargs):
    if backend == "vllm":
        return VLLMModelClient(model_id=model_id, **kwargs)
    if backend == "transformers":
        return TransformersModelClient(model_id=model_id, **kwargs)
    if backend == "echo":
        return EchoModelClient(model_id=model_id, **kwargs)
    raise ValueError(f"Unknown backend {backend!r}, expected 'vllm', 'transformers', or 'echo'")
