# Pilot scoring rubric

How we grade the model on the pilot batch before running it on everything. For
each pilot paper, compare the model's output against a hand-coded correct answer
for the same paper. Score 0 to 4 on each of the four axes below, then average.

| Axis | 0 | 2 | 4 |
|---|---|---|---|
| Coverage | Missed a demographic the paper reported | Caught most, missed one minor one | Caught every demographic reported |
| Numeric accuracy | Percentages wrong or made up | Present but off by rounding or transcription | Match the paper exactly |
| Format | Not valid JSON, or missing required fields | Valid, minor field-name drift | Matches the required fields exactly |
| Multiple samples | Collapsed several samples into one row | Split them but mislabeled the sample id | One row per sample, all sharing the paper's ID |

If numeric accuracy or format averages below about 3 out of 4 across the batch,
stop. Fix the instruction (the prompt in `src/extract_demographics.py`) and
re-test. Do not run the full 58,000 with a model that fabricates numbers or
breaks format, because those errors go straight into the final trends.

Write the raw scores to `results/scores.csv`: paper id, coverage, numeric,
format, multi-sample, notes.
