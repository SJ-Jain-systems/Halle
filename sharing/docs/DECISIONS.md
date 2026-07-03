# Project decisions

Source brief: `PLOS_ONE_PROJECT.pdf` — study of demographic representativeness
in psychology research articles published in *PLOS ONE*, 2010–2026.

This document records the three open decisions from the brief and the
reasoning behind each. Code implementing these decisions lives in `src/`;
how to actually run it on Rivanna is in `docs/RUNNING_ON_RIVANNA.md`.

## 1. Data access: Solr for discovery, `allofplos` for full text

**Decision: enumerate the psychology article set from PLOS's Solr index
(`src/solr_client.py`, `src/build_index_solr.py`), and use the local
`allofplos` XML corpus only for full-text extraction.**

This landed on the brief's *original* architecture, but only after trying and
abandoning an allofplos-only approach — the reasoning is worth recording
because it's a real data-quality finding:

1. **First tried: allofplos-only, parse taxonomy from local XML.** The idea
   was to sync the whole corpus once and read PLOS's subject taxonomy straight
   from each article's JATS `<subj-group>` blocks (`src/jats_xml.py`,
   `src/build_corpus_index.py`), making the pipeline fully offline after the
   one-time download.
2. **Why it failed:** validated against the real corpus (391k articles), the
   XML is missing the subject taxonomy for a large, *year-dependent* share of
   articles — ≈99% of 2015 and ≈37% of 2013 have only a "Research Article"
   heading and no discipline tags at all (see `scripts/diagnose_2015.py`). An
   XML-only scan therefore silently dropped almost all of 2015, which would
   gut the "middle" (2015–2019) stage of the analysis. This isn't fixable by
   better parsing — the data simply isn't in those files.
3. **Fix: Solr for discovery.** PLOS's Solr index carries the authoritative
   taxonomy for *every* article. `src/build_index_solr.py` queries it for
   every PLOS ONE research article 2010–2026 tagged under Psychology
   (`subject:"Psychology"`, with the mandatory `doc_type:full` and strict `fq`
   filters — see `scripts/test_solr.py`), parses the subfield from the subject
   *paths*, and maps each DOI to its local XML file for full-text. Gap-free
   and uniform across all years.

Subject metadata (brief item 4.c.i: `subject` / `subject_level_1`) comes from
Solr's subject paths; full text (Methods/Participants) still comes from the
local XML via `src/jats_xml.py::get_extraction_text()`, which is present even
for the taxonomy-less articles.

Net effect: one network step (Solr enumeration, on the login node — ~10-20
min, not the ~1.5h XML scan) plus the one-time corpus download; everything
downstream reads local XML. The XML-scan code
(`src/build_corpus_index.py`, `slurm/build_index.slurm`) is kept for
reference but superseded.

## 2. Pilot sample

**Decision: stratify 2 articles per psychology subfield, over *all* subfields
PLOS actually tags (not a hardcoded five), drawn at random with a fixed seed
for reproducibility.**

The brief (item 3.c) named five subfields — social, cognitive, developmental,
clinical, quantitative psychology — but validating against the real corpus
showed PLOS's taxonomy doesn't use "Quantitative psychology" as a term at all,
and *does* use several the brief didn't list (Experimental psychology,
Psychometrics, ...). Rather than force the brief's list onto a taxonomy that
doesn't match it, the pipeline is taxonomy-driven: it captures every article
tagged under the Psychology node and records whatever subfield(s) PLOS
assigned (see decision #1 / `jats_xml.get_psychology_subfields`). The pilot
then stratifies over whatever subfields the index actually contains.

Why stratify instead of pooling and sampling uniformly at random: a uniform
draw risks a high-volume subfield (e.g. social/cognitive) crowding out the
others, and the point of the pilot is to sanity-check the chosen model's
extraction quality *across* subfields before spending GPU time on the full
run. 2-per-subfield guarantees coverage. (This makes the pilot 2×N articles
for N subfields present, rather than a fixed 10 — the brief's "10" assumed
exactly 5 subfields.)

Implementation: `src/sample_articles.py`, function `stratified_sample()`
(auto-derives subfields from the index, skips any with < 2 articles), reading
from `data/corpus_index.csv` (no network). Seed defaults to `42`; override
with `--seed`.

## 3. Model choice

**Decision: `meta-llama/Llama-3.3-70B-Instruct`. Final — this is the only
model the pipeline runs.**

The brief named two candidates and asked for one to be picked:

| | Mistral-7B-Instruct-v0.3 | Llama-3.3-70B-Instruct |
|---|---|---|
| Parameters | 7B | 70B |
| Context window | 32K | 128K |
| Structured extraction / instruction-following accuracy | Noticeably weaker on multi-field JSON extraction and numeric reasoning buried in prose | Materially stronger; better at not hallucinating percentages or missing a demographic subgroup |
| Compute/cost | Runs on a single consumer GPU or free-tier hosted inference | Needs multiple GPUs — a real constraint on hosted/shared infrastructure, not on a dedicated HPC allocation |

Llama-3.3-70B-Instruct wins on the axis that actually matters here: a missed
or hallucinated demographic percentage directly biases the
representativeness analysis that is the whole point of this project, so
extraction accuracy dominates the decision. The only reason to hedge toward
Mistral-7B would be compute cost, and running on Rivanna (dedicated GPU
allocation, no per-token API cost, no rate limits) removes that constraint
entirely — there's no real tradeoff left to weigh, so there's no reason to
keep Mistral in the loop as a second candidate.

`src/run_pilot.py` (`slurm/run_pilot.slurm`) still runs the 10-article pilot
through Llama-3.3-70B-Instruct before the full corpus run — not to compare
it against anything, but as a QA spot-check (`docs/SCORING_RUBRIC.md`) that
extraction quality looks right across the psychology subfields before spending real
GPU time on ~thousands of articles.
