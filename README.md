# Halle

Pipeline for studying the representativeness of demographic reporting
(gender, race, education, socioeconomic status) in *PLOS ONE* psychology
articles, 2010–2026.

Source brief: see the three open decisions and their reasoning in
[`docs/DECISIONS.md`](docs/DECISIONS.md):

1. **Data access** — PLOS Solr API for discovery/filtering by subfield, date,
   article type; `allofplos` for full-text retrieval of selected articles.
2. **Pilot sample** — 10 articles, stratified 2 per psychology subfield
   (social, cognitive, developmental, clinical, quantitative), fixed random
   seed for reproducibility.
3. **Model** — `meta-llama/Llama-3.3-70B-Instruct`, pending confirmation from
   the pilot comparison against `mistralai/Mistral-7B-Instruct-v0.3`. See
   [`docs/SCORING_RUBRIC.md`](docs/SCORING_RUBRIC.md).

## Layout

```
src/
  subfields.py            psychology subfields + time-stage buckets
  plos_client.py          Solr search + allofplos full-text fetch
  sample_articles.py      draws the 10-article stratified pilot sample
  extract_demographics.py prompt + schema + validation for demographic rows
  compare_llms.py         runs the pilot sample through each candidate model
docs/
  DECISIONS.md            the three decisions above, with reasoning
  SCORING_RUBRIC.md        rubric for picking a winner from the pilot
tests/                    unit tests (mocked network/model calls)
```

## Running it

```
pip install -r requirements.txt

# 1. Draw the pilot sample (needs network access to api.plos.org)
python -m src.sample_articles --out data/sampled_articles.csv

# 2. Run both candidate models over it and score with docs/SCORING_RUBRIC.md
#    (wire src/extract_demographics.py::ModelClient to a real inference
#    backend first — see its docstring)
python -m src.compare_llms --sample data/sampled_articles.csv
```

**Network note:** this repo was scaffolded in a sandbox whose outbound proxy
blocks `api.plos.org` and `huggingface.co`, so the client code is written and
unit-tested against mocked responses but has not been exercised against the
live APIs. Run the two commands above from a machine with normal internet
access (and Hugging Face credentials, for step 2) before trusting their
output.

## Tests

```
pytest
```
