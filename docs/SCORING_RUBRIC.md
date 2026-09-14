# Pilot QA rubric

Used to validate `meta-llama/Llama-3.3-70B-Instruct`'s output from
`src/run_pilot.py` against a human-coded "gold" answer for the same pilot
articles (100 articles — a floor of 2 per subfield across the ~23 subfields the
index contains, then topped up proportionally; `docs/DECISIONS.md` #2), before
committing GPU time to the full corpus run (`src/run_pipeline.py`).

## Primary gate: recall / precision / accuracy

Following the 7/15 meeting (`docs/DECISIONS.md` #4, `docs/references.md`), the
model's extraction is validated with the standard information-retrieval metrics.
For each demographic's *reported* flag, the human gold is the truth and the
model is the classifier:

- **Recall** = TP / (TP + FN) — of the demographics actually reported in the
  articles, what share did the model catch? Low recall = the model *misses*
  reported demographics.
- **Precision** = TP / (TP + FP) — of the demographics the model reported, what
  share were really in the article? Low precision = the model *hallucinates*.
- **Accuracy** = (TP + TN) / all — overall correctness of the reported/not
  decision.

**Thresholds (must pass, per variable and overall):** recall ≥ **0.90**,
precision ≥ **0.90**, accuracy ≥ **0.90** (`config.yaml`
`validation.thresholds`). These come from the cited literature — see
`docs/references.md`: the benchmark paper's "study information" tier (recall
≥0.80 / precision ≥0.90), GPT as a second reviewer (precision 0.91 / recall
0.89), and Elicit's ~0.87 accuracy gate — rounded to a single 0.90 bar.
`src/score_pilot.py` fails (exit 1) if any variable or the overall micro-average
misses its threshold.

SES is expected to be sparse (~2/3 of articles omit it), so it carries many true
negatives; the scorer prints the SES reported-rate alongside the metrics so a
high SES accuracy isn't mistaken for good coverage.

**SES is scored on presence only** — whether an article reports it or not — and
not on its numeric value or 0/1/2 detail level. It is too sparse (~10% of pilot
articles) to grade the reported value reliably, and presence is what we care
about. This is a standing decision: `src/score_pilot.py` counts SES in the
reported/not gate like the other demographics and does not compare the SES
value.

## Producing the gold set (multiple coders)

The gold answer is the ground truth: humans read each pilot article and hand-code
the same schema the model emits (`src/extract_demographics.py::REQUIRED_KEYS`).
To keep the ground truth from being skewed by one coder, **several blind coders
code the same articles** and their sheets are merged into a consensus.

1. Generate a blank template, one row per pilot article:
   ```
   python -m src.make_gold_template \
       --sample data/sampled_articles.csv \
       --out data/pilot_gold_template.csv \
       --text-dir data/pilot_text        # optional: dumps the same text the model reads
   ```
   The unit of analysis is the *sample*, not the article — duplicate a row and
   bump `sample_id` for each additional participant sample an article reports.
   The `coder` column takes each coder's initials. Each demographic is one
   combined column in the flat, human-readable form, e.g.
   `gender` = `1, 45% Male, 55% Female`, `race` = `1, 60% White, 40% Black`
   (0 = not reported, 1 = reported); `ses` = `2, 30000` on the 0/1/2 detail
   scale.
2. Each coder fills in their copy; save one CSV per coder.
3. Merge the coders into a consensus gold set and report inter-rater agreement:
   ```
   python -m src.merge_gold \
       --inputs data/gold_alice.csv data/gold_bob.csv \
       --out data/pilot_gold.csv \
       --agreement-out results/inter_rater.csv \
       --disagreements-out results/gold_disagreements.csv
   ```
   This writes per-variable **percent agreement** and **Cohen's kappa** (on the
   reported flags) and a majority-vote consensus (`pct` = per-subgroup median
   across the coders who reported it). Samples where coders split are flagged in
   the disagreements file for the team to adjudicate.
4. Run the model over the pilot (`src/run_pilot.py`), then score it against the
   consensus gold:
   ```
   python -m src.score_pilot \
       --gold data/pilot_gold.csv \
       --out results/pilot_accuracy.csv \
       --metrics-out results/pilot_metrics.csv
   ```
   `results/pilot_metrics.csv` is the per-variable + overall recall/precision/
   accuracy gate; `results/pilot_accuracy.csv` is the per-article detail
   (numeric accuracy, sample match, schema).

## Secondary / qualitative axes

Beyond the pass/fail gate, keep these as a qualitative record (score 0–4, then
average):

| Axis | 0 | 2 | 4 |
|---|---|---|---|
| **Numeric accuracy** | Percentages wrong or fabricated | Percentages present but off by rounding/transcription | Percentages match the source article exactly |
| **Schema adherence** | Output isn't valid JSON / missing required keys | Valid JSON, minor field-naming drift | Matches the schema in `src/extract_demographics.py` exactly |
| **Multi-sample handling** | Collapsed multiple study samples into one row | Split samples but mislabeled `sample_id` | Correctly emitted one row per distinct sample, all sharing the article DOI |

A recall/precision/accuracy miss on the gate — or a mean below ~3/4 on numeric
accuracy or schema adherence — is the signal to stop and debug the prompt
(`src/extract_demographics.py::EXTRACTION_PROMPT_TEMPLATE`) rather than
proceeding straight to the full-corpus run — errors here propagate directly into
the representativeness analysis.
