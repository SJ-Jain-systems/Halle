# Project decisions

Source brief: `PLOS_ONE_PROJECT.pdf` — study of demographic representativeness
in psychology research articles published in *PLOS ONE*, 2010–2026.

This document records the three open decisions from the brief and the
reasoning behind each. Code implementing these decisions lives in `src/`;
how to actually run it on Rivanna is in `docs/RUNNING_ON_RIVANNA.md`.

## 1. Data access: `allofplos` only (revised — Solr dropped)

**Decision: `allofplos` (github.com/PLOS/allofplos) is the sole data
source. No Solr calls anywhere in this pipeline.**

The original plan (see git history) split this into "Solr for discovery,
allofplos for full text," using Solr's `subject_level_1` field to filter by
psychology subfield. That's no longer how this pipeline works — the whole
corpus (or the PLOS ONE slice of it) is synced locally once, and every
filtering/discovery step that would have been a Solr query is now a local
scan of the XML instead:

- **Subject taxonomy** (brief item 4.c.i: `subject` and `subject_level_1`):
  PLOS tags every article's JATS XML with `<subj-group
  subj-group-type="Discipline">` blocks — nested `<subject>` elements
  encoding the same taxonomy Solr exposed as `subject`/`subject_level_1`.
  `src/jats_xml.py::get_subjects()` parses this directly: `subject_level_1`
  is the top `<subject>` of each Discipline branch, `subject` is every
  `<subject>` term found at any depth. This is a direct re-derivation of the
  Solr fields, not a proxy for them — same semantics, no network call.
- **Subfield / article-type / journal / date filtering** (brief item 3):
  `src/build_corpus_index.py` scans every XML file in the local corpus once
  and writes the matching population to `data/corpus_index.csv`.
- **Full text for demographic extraction**: already local, since the whole
  point of allofplos is a local mirror — `src/jats_xml.py::get_extraction_text()`
  pulls Methods/Participants section text straight off disk.

Net effect: this pipeline makes zero calls to `api.plos.org` after the
one-time corpus sync. Tradeoff versus the Solr-hybrid approach: the initial
corpus download is a large one-time cost (see `docs/RUNNING_ON_RIVANNA.md`
step 1 for size/storage guidance), and taxonomy parsing now depends on JATS
XML structure being consistent across ~15 years of PLOS ONE articles — spot
check `data/corpus_index.csv` against a handful of known articles after the
first index build to confirm subfield tagging looks right before trusting it
at scale.

## 2. Ten-article pilot sample

**Decision: stratify 2 articles per subfield across the 5 subfields named in
the brief, drawn independently at random with a fixed seed for
reproducibility.**

Subfields (brief item 3.c): social, cognitive, developmental, clinical,
quantitative psychology.

Why stratify instead of pooling and sampling 10 uniformly at random: a
uniform draw over all matching articles risks a subfield with more PLOS ONE
output (e.g. social/cognitive) crowding out the others, and the whole point
of the pilot is to see how each candidate LLM performs *across* subfields
before committing to one model for the full run. 2×5 guarantees coverage.

Implementation: `src/sample_articles.py`, function `stratified_sample()`,
reading from `data/corpus_index.csv` (no network). Seed defaults to `42`;
override with `--seed` for a different draw.

## 3. Model choice

**Decision: `meta-llama/Llama-3.3-70B-Instruct`, pending confirmation from
the 10-article pilot.**

The brief names two candidates and asks for one:

| | Mistral-7B-Instruct-v0.3 | Llama-3.3-70B-Instruct |
|---|---|---|
| Parameters | 7B | 70B |
| Context window | 32K | 128K |
| Structured extraction / instruction-following accuracy | Noticeably weaker on multi-field JSON extraction and numeric reasoning buried in prose | Materially stronger; better at not hallucinating percentages or missing a demographic subgroup |
| Compute/cost | Runs on a single consumer GPU or free-tier hosted inference | Needs multiple GPUs — a real constraint on hosted/shared infrastructure, not on a dedicated HPC allocation |

Reasoning: extraction accuracy is the variable that matters most — a missed
or hallucinated percentage directly biases the representativeness analysis
that is the actual research question. The original writeup flagged compute
cost as the reason this pick was tentative; running on Rivanna removes that
constraint (dedicated GPU allocation, no per-token API cost), so the
accuracy case for Llama-3.3-70B-Instruct now stands on its own without a
cost tradeoff to weigh against it.

**Still a reasoned default, not a validated result** — confirm it against
the pilot before committing to a full-corpus run. `src/compare_llms.py` runs
both models on the 10 sampled articles (`slurm/pilot_comparison_llama.slurm`,
`slurm/pilot_comparison_mistral.slurm`); score the output against
`docs/SCORING_RUBRIC.md` and switch `default_model` in `config.yaml` if
Mistral-7B wins the pilot.
