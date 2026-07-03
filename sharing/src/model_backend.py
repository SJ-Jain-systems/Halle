# ============================================================================
# PLAIN-ENGLISH NOTES (for colleagues reading this file)
#
# What this file is for: the plumbing that actually runs the AI model on the
# cluster's graphics cards (GPUs). Everything else in the project produces a
# prompt; this is what turns a prompt into the model's answer.
#
# There are two ways to run it:
#   - VLLMModelClient: the fast one. It's built for running lots of prompts and
#     splitting one huge model across several GPUs. This is what we use for the
#     real 58,000-paper run.
#   - TransformersModelClient: the simple one. Fine for the small pilot or for
#     debugging on a single GPU.
#
# You don't need to read the internals unless you're changing how the model
# runs. The key knob is "tensor_parallel_size" = how many GPUs to spread the
# 70-billion-parameter model across (it's too big to fit on one).
#
# "temperature = 0" means the model answers as deterministically as possible.
# We want faithful extraction, not creative writing, so we turn creativity off.
# ============================================================================

"""Local GPU inference backends for Rivanna, where compute isn't the
constraint — no hosted API, no rate limits. Both implement the ModelClient
interface from src/extract_demographics.py (a `.generate(prompt) -> str`
method), so either drops straight into extract_demographics()/run_pilot.py/run_pipeline.py.

vLLM is the recommended path for anything beyond the 10-article pilot: it
batches requests and uses paged attention, which matters once you're running
the full filtered corpus. The transformers backend is a simpler fallback for
small allocations or debugging on a single GPU.

Neither backend has been exercised in this sandbox (no GPU, no model
weights, no network to huggingface.co) — see docs/RUNNING_ON_RIVANNA.md for
how to validate this on an actual Rivanna GPU node before trusting output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0  # deterministic extraction, not creative generation


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
    dtype: str = "bfloat16"
    _llm: object = field(default=None, init=False, repr=False)

    def _load(self):
        # Load the model once, the first time it's needed (it's ~140GB, so we
        # don't want to load it twice). The "import" is inside the function on
        # purpose - it's a heavy, GPU-only library we only touch when actually
        # running on a GPU node.
        if self._llm is None:
            from vllm import LLM  # deferred: heavy import, GPU-only

            self._llm = LLM(
                model=self.model_id,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=self.dtype,
            )
        return self._llm

    def generate(self, prompt: str) -> str:
        # Single prompt in, single answer out.
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        # Many prompts at once - much more efficient on a GPU than one at a time.
        from vllm import SamplingParams

        llm = self._load()
        params = SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens)
        outputs = llm.generate(prompts, params)
        # vLLM does not guarantee output order matches input order; sort by
        # the prompt's original request index it attaches internally.
        outputs = sorted(outputs, key=lambda o: o.request_id)
        return [o.outputs[0].text for o in outputs]


@dataclass
class TransformersModelClient:
    """Simpler single-process fallback via Hugging Face `transformers`.

    Loads the model once per process with device_map="auto" (splits across
    all visible GPUs automatically). Fine for the 10-article pilot; for a
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
    # Pick which runner to use by name. Defaults to the fast one (vllm).
    if backend == "vllm":
        return VLLMModelClient(model_id=model_id, **kwargs)
    if backend == "transformers":
        return TransformersModelClient(model_id=model_id, **kwargs)
    raise ValueError(f"Unknown backend {backend!r}, expected 'vllm' or 'transformers'")
