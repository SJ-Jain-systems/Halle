# Pilot QA rubric

This is how we sanity-check `meta-llama/Llama-3.3-70B-Instruct`'s output from
`src/run_pilot.py` against a "gold" answer we code by hand for the same pilot
articles (46 of them, 2 per subfield across the 23 subfields the index has),
before we spend GPU time on the full corpus run (`src/run_pipeline.py`). Score
each article 0 to 4 on each axis, then average.

## Producing the gold set

The gold answer is our ground truth: a human reads each pilot article and codes
the same schema the model spits out (`src/extract_demographics.py`, REQUIRED_KEYS)
by hand.

1. Make a blank template, one row per pilot article:
   ```
   python -m src.make_gold_template \
       --sample data/sampled_articles.csv \
       --out data/pilot_gold_template.csv \
       --text-dir data/pilot_text        # optional: also dumps the text the model reads
   ```
   We code per sample, not per article, so copy a row and bump `sample_id` for
   each extra participant sample an article reports. The `*_pct` columns take a
   JSON object like `{"male": 45, "female": 55}`.
2. Fill in every row by hand, save it as `data/pilot_gold.csv`.
3. Run the model over the pilot (`src/run_pilot.py`), then score it against the
   gold set:
   ```
   python -m src.score_pilot --gold data/pilot_gold.csv --out results/pilot_accuracy.csv
   ```
   That reports per-field agreement on the four axes below, matched up by
   `(doi, sample_id)`, into `results/pilot_accuracy.csv` plus a summary. The 0 to
   4 scores below are still the qualitative record; the script is the number side
   of it.

| Axis | 0 | 2 | 4 |
|---|---|---|---|
| **Coverage** | Missed a reported demographic (say race is in the article but not in the output) | Caught most of them, missed one minor one | Caught every reported demographic |
| **Numeric accuracy** | Percentages wrong or made up | Percentages there but off by rounding or a typo | Percentages match the article exactly |
| **Schema adherence** | Output isn't valid JSON, or it's missing keys | Valid JSON, small field-name drift | Matches the schema in `src/extract_demographics.py` exactly |
| **Multi-sample handling** | Squashed several study samples into one row | Split them but mislabeled `sample_id` | One row per sample, all sharing the article DOI |

If the mean drops below about 3 out of 4 on **numeric accuracy** or **schema
adherence** across the pilot, stop and go fix the prompt
(`src/extract_demographics.py`, EXTRACTION_PROMPT_TEMPLATE) instead of barreling
into the full run. Mistakes here feed straight into the representativeness
numbers.

Record the raw scores in `results/scores.csv` (doi, coverage, numeric, schema,
multi_sample, notes).
