"""Local GPU inference backends for Rivanna, where compute isn't the
constraint — no hosted API, no rate limits. All implement the ModelClient
interface from src/extract_demographics.py (a `.generate(prompt) -> str`
method), so any drops straight into extract_demographics()/run_pilot.py/run_pipeline.py.

vLLM is the recommended path for anything beyond the 46-article pilot: it
batches requests and uses paged attention, which matters once you're running
the full filtered corpus. The transformers backend is a simpler fallback for
small allocations or debugging on a single GPU. The `echo` backend loads no
model at all — a GPU-free dry-run to validate the pipeline wiring before
requesting a GPU allocation (see EchoModelClient).

Neither backend has been exercised in this sandbox (no GPU, no model
weights, no network to huggingface.co) — see docs/RUNNING_ON_RIVANNA.md for
how to validate this on an actual Rivanna GPU node before trusting output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

DEFAULT_MAX_NEW_TOKENS = 3072
DEFAULT_TEMPERATURE = 0.0  # deterministic extraction, not creative generation
# Greedy decoding on this task can degenerate into repeated filler ("I hope
# this helps...") that runs to the token cap and breaks JSON parsing. A mild
# frequency penalty curbs that loop without meaningfully distorting the short,
# structured JSON we want.
DEFAULT_FREQUENCY_PENALTY = 0.4


@dataclass
class VLLMModelClient:
    """Batch-oriented local inference via vLLM's offline LLM API.

    tensor_parallel_size should match the number of GPUs requested in the
    SLURM job (see slurm/run_pipeline.slurm) — e.g. 4 for a 70B model split
    across 4x A100-80GB.
    """

    model_id: str
    tensor_parallel_size: int = 1
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY
    dtype: str = "bfloat16"
    _llm: object = field(default=None, init=False, repr=False)

    def _load(self):
        if self._llm is None:
            from vllm import LLM  # deferred: heavy import, GPU-only

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
        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            frequency_penalty=self.frequency_penalty,
        )
        outputs = llm.generate(prompts, params)
        # vLLM does not guarantee output order matches input order; sort by
        # the prompt's original request index it attaches internally.
        outputs = sorted(outputs, key=lambda o: o.request_id)
        return [o.outputs[0].text for o in outputs]


@dataclass
class TransformersModelClient:
    """Simpler single-process fallback via Hugging Face `transformers`.

    Loads the model once per process with device_map="auto" (splits across
    all visible GPUs automatically). Fine for the 46-article pilot; for a
    full-corpus run prefer VLLMModelClient's batching.
    """

    model_id: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    _pipeline: object = field(default=None, init=False, repr=False)

    def _load(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline  # deferred: heavy import

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
    """GPU-free dry-run backend: loads no model and needs no network.

    Returns a schema-valid "all not reported" response for whatever DOI the
    prompt carries (extract_demographics.EXTRACTION_PROMPT_TEMPLATE emits an
    `Article DOI: <doi>` line). This lets `run_pilot`/`run_pipeline` exercise the
    whole path — CSV load, XML read, prompt build, parse/validate, JSON write —
    on the login node before requesting GPUs. Output is a wiring check, NOT a
    real extraction.
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
            "gender": {"reported": 0, "pct": {}},
            "race": {"reported": 0, "pct": {}},
            "education": {"reported": 0, "pct": {}},
            "ses": {"reported": 0, "value": None},
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
