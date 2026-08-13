# Halle

Pipeline for studying the representativeness of demographic reporting
(gender, race, education, socioeconomic status) in *PLOS ONE* psychology
articles, 2010–2026, year by year and by stage (early / middle / COVID /
post-COVID).

Source brief: see the project decisions and their reasoning in
[`docs/DECISIONS.md`](docs/DECISIONS.md):

1. **Data access** — `allofplos` only (github.com/PLOS/allofplos). The whole
   corpus is synced locally once; every filter (subfield, article type,
   date) and every full-text read after that is a local scan, no network.
2. **Pilot sample** — 100 articles for human validation (7/15 meeting): a floor
   of 2 per psychology subfield (all subfields PLOS tags — social, cognitive,
   clinical, developmental, experimental psychology, psychometrics, ...) then
   topped up proportionally, fixed random seed for reproducibility. The model's
   extraction is validated against a multi-coder human gold set on
   recall/precision/accuracy (each ≥ 0.90) before the full run — see
   [`docs/SCORING_RUBRIC.md`](docs/SCORING_RUBRIC.md) and
   [`docs/references.md`](docs/references.md).
3. **Model** — `meta-llama/Llama-3.3-70B-Instruct`, run locally on Rivanna
   GPU nodes. Final choice, on accuracy grounds — see
   [`docs/DECISIONS.md`](docs/DECISIONS.md) #3. The 46-article pilot still
   runs through it as a QA spot-check
   ([`docs/SCORING_RUBRIC.md`](docs/SCORING_RUBRIC.md)) before the full run,
   not as a model comparison.

**To actually run this on Rivanna, start with
[`docs/RUNNING_ON_RIVANNA.md`](docs/RUNNING_ON_RIVANNA.md)** — it walks
through every stage from a bare account to the final trend summaries,
including SLURM scripts, storage/GPU allocation guidance, and
troubleshooting.

## Pipeline

```
allofplos corpus → build_corpus_index → sample_articles (100-article pilot)
                                       → run_pilot → merge_gold → score_pilot
                                         (recall/precision/accuracy ≥ 0.90 gate)
                                       → run_pipeline (full corpus)
                                       → merge_shards → analyze_trends
```

## Layout

```
src/
  jats_xml.py              low-level JATS XML parsing (subjects, dates, institutions, section text)
  allofplos_client.py      local corpus directory access — no network calls
  subfields.py             time-stage buckets (subfields are taxonomy-driven, not hardcoded)
  build_corpus_index.py    scans the corpus, applies inclusion criteria, writes data/corpus_index.csv
  subject_filter.py        drops non-human (animal-model) studies by subject taxonomy
  sample_articles.py       draws the 100-article stratified pilot sample (floor + proportional top-up)
  extract_demographics.py  prompt + schema + validation for demographic rows
  model_backend.py         local GPU inference (vLLM / transformers) for Rivanna
  run_pilot.py             runs the pilot sample through the chosen model
  make_gold_template.py    blank multi-coder hand-coding sheet for the pilot gold set
  merge_gold.py            merges coders -> inter-rater agreement + consensus gold
  score_pilot.py           recall/precision/accuracy gate (≥0.90) vs the gold set
  run_pipeline.py          full-scale extraction over the entire filtered corpus
  merge_shards.py          combines SLURM-array shard outputs into one table
  analyze_trends.py        year-by-year / by-stage aggregation — the actual research answer
docs/
  DECISIONS.md             the project decisions (data access, sampling, model, validation metrics)
  RUNNING_ON_RIVANNA.md    step-by-step guide to running the full pipeline on Rivanna
  SCORING_RUBRIC.md        recall/precision/accuracy validation gate + multi-coder gold workflow
  references.md            papers behind the validation thresholds (7/15 meeting)
slurm/                     SLURM batch scripts for every GPU/CPU stage
tests/                     unit tests (fixture XML + mocked model calls, no GPU/network needed)
```

## Running it

Full instructions (Rivanna-specific: modules, GPU allocation, storage) are
in [`docs/RUNNING_ON_RIVANNA.md`](docs/RUNNING_ON_RIVANNA.md). Short version:

```
pip install -r requirements.txt

# 1. Sync the allofplos corpus locally (one-time, large — see the doc above)
# 2. Build the filtered population index
python -m src.build_corpus_index --out data/corpus_index.csv

# 3. Draw the 46-article pilot sample
python -m src.sample_articles --index data/corpus_index.csv --out data/sampled_articles.csv

# 4. QA the chosen model on the pilot before scaling up
python -m src.run_pilot --sample data/sampled_articles.csv
#    score results/ against docs/SCORING_RUBRIC.md

# 5. Run the full pipeline
python -m src.run_pipeline --index data/corpus_index.csv --out data/demographics_table.csv

# 6. Answer the research question
python -m src.analyze_trends --table data/demographics_table.csv --out-dir data/analysis
```

## Tests

```
pytest
```

All tests offline: fixture JATS XML in `tests/fixtures/` stands in for
the real corpus, and a fake `ModelClient` stands in for real model calls —
nothing here requires a GPU or network access.
