# Pilot QA rubric (10-article pilot)

Used to spot-check `meta-llama/Llama-3.3-70B-Instruct`'s output from
`src/run_pilot.py` against a human-coded "gold" answer for the same 10
articles, before committing GPU time to the full corpus run
(`src/run_pipeline.py`). Score each article 0-4 on each axis, then average.

| Axis | 0 | 2 | 4 |
|---|---|---|---|
| **Coverage** | Missed a reported demographic (e.g. race reported in the article but omitted from output) | Caught most reported demographics, missed one minor one | Every reported demographic captured |
| **Numeric accuracy** | Percentages wrong or fabricated | Percentages present but off by rounding/transcription | Percentages match the source article exactly |
| **Schema adherence** | Output isn't valid JSON / missing required keys | Valid JSON, minor field-naming drift | Matches the schema in `src/extract_demographics.py` exactly |
| **Multi-sample handling** | Collapsed multiple study samples into one row | Split samples but mislabeled `sample_id` | Correctly emitted one row per distinct sample, all sharing the article DOI |

A mean score below ~3/4 on **numeric accuracy** or **schema adherence**
across the 10 pilot articles is the signal to stop and debug the prompt
(`src/extract_demographics.py::EXTRACTION_PROMPT_TEMPLATE`) rather than
proceeding straight to the full-corpus run — errors here propagate directly
into the representativeness analysis.

Record raw scores in `results/scores.csv` (doi, coverage, numeric, schema,
multi_sample, notes).
