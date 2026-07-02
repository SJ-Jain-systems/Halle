"""Local-corpus access layer (docs/DECISIONS.md #1, revised): allofplos only,
no Solr calls anywhere in this pipeline.

allofplos maintains a directory of every PLOS article as JATS XML. We treat
that directory as a static local dataset: download/sync it once (see
docs/RUNNING_ON_RIVANNA.md step 1), then every later pipeline stage
(indexing, sampling, extraction) reads XML files off disk. Nothing here
makes a network call.

Uses the `PLOS_CORPUS` env var name — that's allofplos's own convention
(see `allofplos.get_corpus_dir()` in the installed package), not something
we invented, so the same env var works for both the corpus-sync step and
everything downstream here.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CORPUS_DIR = os.environ.get("PLOS_CORPUS", os.path.expanduser("~/allofplos_corpus"))


class CorpusNotFoundError(RuntimeError):
    pass


def ensure_corpus_available(corpus_dir: str = DEFAULT_CORPUS_DIR) -> Path:
    path = Path(corpus_dir)
    if not path.is_dir() or not any(path.glob("*.xml")):
        raise CorpusNotFoundError(
            f"No article XML found under {corpus_dir!r}. Sync the allofplos corpus first "
            "(docs/RUNNING_ON_RIVANNA.md, step 1) or set PLOS_CORPUS to point at it."
        )
    return path


def iter_corpus_xml(corpus_dir: str = DEFAULT_CORPUS_DIR):
    """Yield paths to every article XML file in the local corpus."""
    path = ensure_corpus_available(corpus_dir)
    yield from sorted(str(p) for p in path.glob("*.xml"))


def corpus_size(corpus_dir: str = DEFAULT_CORPUS_DIR) -> int:
    return sum(1 for _ in iter_corpus_xml(corpus_dir))
