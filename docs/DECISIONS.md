# Project decisions

Source brief: `PLOS_ONE_PROJECT.pdf` — study of demographic representativeness
in psychology research articles published in *PLOS ONE*, 2010–2026.

This document records the three open decisions from the brief and the
reasoning behind each. Code implementing these decisions lives in `src/`.

## 1. Data access: Solr vs. `allofplos` (GitHub repo)

**Decision: use both, for different jobs — Solr for discovery/filtering, `allofplos` for full-text retrieval.**

The brief lists these as alternatives, but they solve different problems and
the brief's own metadata section (`subject` / `subject_level_1` "via Solr")
already assumes Solr is in the loop:

| | PLOS Search API (Solr) | `allofplos` (github.com/PLOS/allofplos) |
|---|---|---|
| What it's good at | Fielded queries: filter by `subject_level_1` (subfield taxonomy), `article_type`, `publication_date`, journal, and page through results | Reliable, rate-limit-free full-text JATS XML for every PLOS article, kept in sync locally |
| Weak point | Not meant for bulk full-text harvesting — PLOS explicitly discourages hammering Solr for full article bodies, and results are metadata/abstract-oriented | No query/faceting layer; you need a DOI list before it's useful |
| Cost to us | Free, instant, no local storage | One-time local sync (large first download), then free/offline |

Our extraction task needs both capabilities: we must **find** articles by
psychology subfield and time-stage, and then **read** their full
Methods/Participants sections to pull out demographic percentages — Solr's
metadata/abstract fields aren't sufficient for the second part.

**Pipeline:**
1. `src/plos_client.py :: search_articles()` queries Solr per subfield to get
   candidate DOIs + metadata (`subject`, `subject_level_1`, `publication_date`,
   author institution).
2. `src/plos_client.py :: fetch_fulltext()` resolves each selected DOI to full
   JATS XML via `allofplos`, for demographic extraction.

This also directly satisfies brief item 4.c.i: "subject & subject_level_1 (via
Solr)."

**Caveat found while building this:** this sandbox's outbound network policy
blocks both `api.plos.org` and `huggingface.co`. The client code is written
and unit-tested against mocked responses, but nobody has run it against the
live API yet — do that first, from an environment with normal internet
access, before trusting `data/sampled_articles.csv`.

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

Implementation: `src/sample_articles.py`, function `stratified_sample()`.
Filters applied per brief item 3: `article_type:"Research Article"`, journal
= PLOS ONE, no sample-size restriction. Seed defaults to `42`; override with
`--seed` for a different draw.

This step also has not been run against live data for the reason above —
running `python -m src.sample_articles` from a networked machine will
populate `data/sampled_articles.csv`.

## 3. Model choice

**Decision: `meta-llama/Llama-3.3-70B-Instruct`, pending confirmation from the
10-article pilot.**

The brief names two candidates and asks for one:

| | Mistral-7B-Instruct-v0.3 | Llama-3.3-70B-Instruct |
|---|---|---|
| Parameters | 7B | 70B |
| Context window | 32K | 128K |
| Structured extraction / instruction-following accuracy | Noticeably weaker on multi-field JSON extraction and numeric reasoning buried in prose | Materially stronger; better at not hallucinating percentages or missing a demographic subgroup |
| Compute/cost | Runs on a single consumer GPU or free-tier hosted inference — cheap to scale to hundreds of articles | Needs a paid hosted endpoint or multi-GPU box |

Reasoning: extraction accuracy is the variable that matters most here — a
missed or hallucinated percentage directly biases the representativeness
analysis that is the actual research question. At pilot scale (10 articles)
the cost gap between the two models is negligible, so we should optimize for
accuracy now and revisit cost only if/when scaling to the full corpus makes
Llama-3.3-70B's compute cost a real constraint. If that happens, fall back to
Mistral-7B-Instruct-v0.3 and expect a manual-QA/spot-check step to catch the
accuracy gap.

**This is a reasoned default, not a validated result** — nobody has run
either model against this repo's Hugging Face IDs (no credentials/network in
this sandbox). `src/compare_llms.py` is the harness for actually running both
models on the 10 sampled articles and scoring them against
`docs/SCORING_RUBRIC.md`; use its output to confirm or overturn this choice
before scaling up.
