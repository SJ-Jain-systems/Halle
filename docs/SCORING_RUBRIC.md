# Pilot QA rubric

Used to spot-check `meta-llama/Llama-3.3-70B-Instruct`'s output from
`src/run_pilot.py` against a human-coded "gold" answer for the same pilot
articles (46 = 2 per subfield across the 23 subfields the index contains),
before committing GPU time to the full corpus run (`src/run_pipeline.py`).
Score each article 0-4 on each axis, then average.

## Producing the gold set

The gold answer is the ground truth: a human reads each pilot article and hand-
codes the same schema the model emits (`src/extract_demographics.py::REQUIRED_KEYS`).

1. Generate a blank template, one row per pilot article:
   ```
   python -m src.make_gold_template \
       --sample data/sampled_articles.csv \
       --out data/pilot_gold_template.csv \
       --text-dir data/pilot_text        # optional: dumps the same text the model reads
   ```
   The unit of analysis is the *sample*, not the article — duplicate a row and
   bump `sample_id` for each additional participant sample an article reports.
   Each demographic is one combined column in the flat, human-readable form,
   e.g. `gender` = `1, 45% Male, 55% Female`, `race` = `1, 60% White, 40% Black`
   (0 = not reported, 1 = reported); `ses` = `2, 30000` on the 0/1/2 detail
   scale.
2. Hand-code every row, save as `data/pilot_gold.csv`.
3. Run the model over the pilot (`src/run_pilot.py`), then score it against the
   gold set:
   ```
   python -m src.score_pilot --gold data/pilot_gold.csv --out results/pilot_accuracy.csv
   ```
   This reports per-field agreement mapped to the four axes below, aligned by
   `(doi, sample_id)`, into `results/pilot_accuracy.csv` plus an aggregate
   summary. The 0-4 axis scores below remain the qualitative record; the script
   is the quantitative complement.

| Axis | 0 | 2 | 4 |
|---|---|---|---|
| **Coverage** | Missed a reported demographic (e.g. race reported in the article but omitted from output) | Caught most reported demographics, missed one minor one | Every reported demographic captured |
| **Numeric accuracy** | Percentages wrong or fabricated | Percentages present but off by rounding/transcription | Percentages match the source article exactly |
| **Schema adherence** | Output isn't valid JSON / missing required keys | Valid JSON, minor field-naming drift | Matches the schema in `src/extract_demographics.py` exactly |
| **Multi-sample handling** | Collapsed multiple study samples into one row | Split samples but mislabeled `sample_id` | Correctly emitted one row per distinct sample, all sharing the article DOI |

A mean score below ~3/4 on **numeric accuracy** or **schema adherence**
across the pilot articles is the signal to stop and debug the prompt
(`src/extract_demographics.py::EXTRACTION_PROMPT_TEMPLATE`) rather than
proceeding straight to the full-corpus run — errors here propagate directly
into the representativeness analysis.

Record raw scores in `results/scores.csv` (doi, coverage, numeric, schema,
multi_sample, notes).
