# References — validation thresholds

The recall/precision/accuracy gate (`docs/DECISIONS.md` #4,
`docs/SCORING_RUBRIC.md`) was set in the 7/15 meeting from three papers on LLM
data extraction for systematic reviews. This file records what each contributes
so the thresholds are traceable when we write up.

> Note: the figures below are as discussed in the meeting (the transcript is the
> primary record of the team's decision). Confirm exact numbers and page
> references against the PDFs before citing them in the manuscript.

## Decision (7/15 meeting)

- **Metrics:** recall, precision, accuracy on each demographic's *reported*
  flag (TP/FP/FN/TN), per variable and micro-averaged overall.
- **Thresholds:** recall ≥ 0.90, precision ≥ 0.90, accuracy ≥ 0.90 — a single
  0.90 bar chosen so we don't cite a threshold below what these papers already
  report.
- **Validation set:** 100 pilot articles (papers span 30–900).
- **Ground truth:** multiple blind coders → inter-rater agreement + consensus.

## Papers

1. **"What level of automation is good enough? A benchmark of large language
   models for meta-analysis data extraction."**
   Establishes tiered minimum thresholds by extraction category — *statistical*,
   *quality assessment*, and *study information*. Our demographics fall under
   **study information**, for which the benchmark sets **recall ≥ 0.80 and
   precision ≥ 0.90**. Uses a large validation set (~900 items). This is the main
   source for the tiered-threshold framing.

2. **"Using Elicit AI research assistant for data extraction in systematic
   reviews: a feasibility study across environmental and life sciences."**
   Uses an accuracy gate of **~87%** in a development phase — only proceeding to
   extract the full set once the tool clears it. Validation on the order of ~90
   articles. Source for the accuracy-gate idea (rounded up to 0.90 here).

3. **"Using Artificial Intelligence Tools as a Second Reviewer for Data
   Extraction in Systematic Reviews."**
   Compares Elicit against a GPT model; reports GPT at **precision ≈ 0.91,
   recall ≈ 0.89** (with F1). Validation on ~30 articles. Source for the
   precision/recall bar — 0.90 sits right at these reported values.

4. **Meeting transcript, 7/15/26.** The team's discussion adopting the metrics,
   the 0.90 thresholds, the 100-article validation set, the multi-coder ground
   truth, and the "1/0 reported + percentage breakdown" output format.

## SES sparsity

A practical note from the same meeting: most articles don't report socioeconomic
status (~15 of 46 in a test batch reported it). SES therefore carries many true
negatives; `src/score_pilot.py` prints the SES reported-rate next to its metrics
so a high SES accuracy isn't mistaken for good coverage.
