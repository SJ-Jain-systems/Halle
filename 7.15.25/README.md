# Halle

A pipeline for looking at how well *PLOS ONE* psychology articles report who was
actually in their samples (gender, race, education, socioeconomic status), from
2010 to 2026, broken down year by year and by stage (early, middle, COVID,
post-COVID).

The three big decisions and the reasoning behind them live in
[`docs/DECISIONS.md`](docs/DECISIONS.md):

1. **Data access.** Just `allofplos` (github.com/PLOS/allofplos). We sync the
   whole corpus locally once, and after that every filter (subfield, article
   type, date) and every full-text read is a local scan with no network.
2. **Pilot sample.** 46 articles, 2 per psychology subfield (across every
   subfield PLOS tags: social, cognitive, clinical, developmental, experimental
   psychology, psychometrics, and so on), with a fixed random seed so it's
   reproducible.
3. **Model.** `meta-llama/Llama-3.3-70B-Instruct`, run locally on Rivanna GPU
   nodes. We picked it on accuracy grounds (see
   [`docs/DECISIONS.md`](docs/DECISIONS.md) #3). The 46-article pilot still runs
   through it as a QA spot-check ([`docs/SCORING_RUBRIC.md`](docs/SCORING_RUBRIC.md))
   before the full run. It's not a model comparison.

**If you actually want to run this on Rivanna, start with
[`docs/RUNNING_ON_RIVANNA.md`](docs/RUNNING_ON_RIVANNA.md).** It walks through
every stage, from a bare account all the way to the final trend summaries,
including the SLURM scripts, the storage and GPU allocation notes, and
troubleshooting.

## Pipeline

```
allofplos corpus → build_corpus_index → sample_articles (pilot)
                                       → run_pilot (QA spot-check)
                                       → run_pipeline (full corpus)
                                       → merge_shards → analyze_trends
```

## Layout

```
src/
  jats_xml.py              low-level JATS XML parsing (subjects, dates, institutions, section text)
  allofplos_client.py      local corpus directory access, no network calls
  subfields.py             time-stage buckets (subfields come from the taxonomy, not a hardcoded list)
  build_corpus_index.py    scans the corpus, applies the inclusion criteria, writes data/corpus_index.csv
  sample_articles.py       draws the 46-article stratified pilot sample
  extract_demographics.py  prompt + schema + validation for the demographic rows
  model_backend.py         local GPU inference (vLLM / transformers) for Rivanna
  run_pilot.py             runs the pilot sample through the model for a QA spot-check
  run_pipeline.py          full extraction over the whole filtered corpus
  merge_shards.py          combines the SLURM array shard outputs into one table
  analyze_trends.py        year by year and by stage aggregation, the actual research answer
docs/
  DECISIONS.md             the three decisions above, with the reasoning
  RUNNING_ON_RIVANNA.md    step by step guide to running the whole thing on Rivanna
  SCORING_RUBRIC.md        QA rubric for the pilot run
slurm/                     SLURM batch scripts for every GPU/CPU stage
tests/                     unit tests (fixture XML + mocked model calls, no GPU or network needed)
```

## Running it

The full instructions for Rivanna (modules, GPU allocation, storage) are in
[`docs/RUNNING_ON_RIVANNA.md`](docs/RUNNING_ON_RIVANNA.md). The short version:

```
pip install -r requirements.txt

# 1. Sync the allofplos corpus locally (one time, it's big, see the doc above)
# 2. Build the filtered population index
python -m src.build_corpus_index --out data/corpus_index.csv

# 3. Draw the 46-article pilot sample
python -m src.sample_articles --index data/corpus_index.csv --out data/sampled_articles.csv

# 4. QA the model on the pilot before scaling up
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

25 tests, all offline: the fixture JATS XML in `tests/fixtures/` stands in for
the real corpus, and a fake `ModelClient` stands in for real model calls, so
nothing here needs a GPU or network access.
