# LLM comparison rubric (10-article pilot)

Used to score each model's output from `src/compare_llms.py` against a
human-coded "gold" answer for the same 10 articles. Score each article 0-4 on
each axis, then average.

| Axis | 0 | 2 | 4 |
|---|---|---|---|
| **Coverage** | Missed a reported demographic (e.g. race reported in the article but omitted from output) | Caught most reported demographics, missed one minor one | Every reported demographic captured |
| **Numeric accuracy** | Percentages wrong or fabricated | Percentages present but off by rounding/transcription | Percentages match the source article exactly |
| **Schema adherence** | Output isn't valid JSON / missing required keys | Valid JSON, minor field-naming drift | Matches the schema in `src/extract_demographics.py` exactly |
| **Multi-sample handling** | Collapsed multiple study samples into one row | Split samples but mislabeled `sample_id` | Correctly emitted one row per distinct sample, all sharing the article DOI |

A model "wins" the pilot if it has the higher mean total across the 10
articles. Ties go to the cheaper model (Mistral-7B) per `docs/DECISIONS.md`.

Record raw scores in `results/scores.csv` (doi, model, coverage, numeric,
schema, multi_sample, notes).
