"""Score the model's pilot output against the hand-coded gold set
(docs/SCORING_RUBRIC.md).

Reads the completed gold CSV (produced from src/make_gold_template.py and coded
by hand) and the model's per-article JSON output written by src/run_pilot.py
(results/<model_id>/<doi>.json), aligns rows by (doi, sample_id), and reports:

  - Recall / precision / accuracy : the primary validation gate (7/15 meeting,
    docs/DECISIONS.md #4). For each demographic's *reported* flag, treat the
    human gold as truth and the model as the classifier, count TP/FP/FN/TN, and
    compute recall, precision, and accuracy per variable and micro-averaged
    overall. Each must clear the config threshold (default 90%). Recall = the
    model doesn't miss demographics that ARE reported; precision = it doesn't
    hallucinate ones that aren't.
  - Numeric accuracy : do the reported `pct` breakdowns agree (within a
    percentage-point tolerance)?
  - Multi-sample handling : one gold row per sample matched by the model?
  - Schema adherence : model rows carry exactly the required keys?

Usage:
    python -m src.score_pilot \\
        --gold data/pilot_gold.csv \\
        --results-dir results \\
        --model meta-llama/Llama-3.3-70B-Instruct \\
        --out results/pilot_accuracy.csv \\
        --metrics-out results/pilot_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict

from src.extract_demographics import (
    DEMOGRAPHIC_FIELDS,
    PCT_FIELDS,
    REQUIRED_KEYS,
    parse_field,
)

FLAG_FIELDS = list(DEMOGRAPHIC_FIELDS)  # gender, race, education, ses
PCT_DEMOGRAPHICS = list(PCT_FIELDS)     # gender, race, education

# Golden-standard QA gate (config.yaml `validation.thresholds`, docs/DECISIONS.md
# #4). Overridable via load_thresholds(); this is the fallback if no config.
DEFAULT_THRESHOLDS = {"recall": 0.90, "precision": 0.90, "accuracy": 0.90}


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _num(x) -> float:
    """Best-effort numeric value. SES thresholds are hand-coded as free text
    (e.g. "< RM 4,359", "$30,000"), so pull the first number out rather than
    assuming a clean float. No number found -> 0.0."""
    if x is None or x == "":
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    m = _NUM_RE.search(str(x).replace(",", ""))
    return float(m.group()) if m else 0.0


def _to_int(x):
    if x is None or x == "":
        return None
    return int(float(x))


def _reported(field) -> int | None:
    return (field or {}).get("reported")


def _reported_binary(field) -> int:
    """Collapse a demographic's reported flag to 0/1 for the confusion matrix.
    SES uses a 0/1/2 detail scale, so anything >= 1 counts as reported."""
    return 1 if ((field or {}).get("reported") or 0) >= 1 else 0


def _safe_div(numer: float, denom: float) -> float | None:
    return round(numer / denom, 3) if denom else None


def load_gold(gold_csv: str) -> list[dict]:
    """Parse the hand-coded gold CSV into typed rows, turning each combined
    demographic column (e.g. gender "1, 45% Male, 55% Female") into its
    {"reported", "pct"} / {"reported", "value"} dict via parse_field."""
    with open(gold_csv, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    rows = []
    for r in raw_rows:
        if not r.get("doi"):
            continue
        rows.append(
            {
                "doi": r["doi"],
                "sample_id": _to_int(r.get("sample_id")) or 1,
                "gender": parse_field("gender", r.get("gender")),
                "race": parse_field("race", r.get("race")),
                "education": parse_field("education", r.get("education")),
                "ses": parse_field("ses", r.get("ses")),
            }
        )
    return rows


def load_model_outputs(results_dir: str, model_id: str) -> dict[str, dict]:
    """Read run_pilot.py's per-article JSON into {doi: {"status", "rows"}}."""
    model_dir = os.path.join(results_dir, model_id.replace("/", "__"))
    outputs: dict[str, dict] = {}
    if not os.path.isdir(model_dir):
        return outputs
    for name in os.listdir(model_dir):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(model_dir, name), encoding="utf-8") as f:
            payload = json.load(f)
        outputs[payload["doi"]] = payload
    return outputs


def _pct_equal(a: dict, b: dict, tol: float) -> bool:
    keys = set(a) | set(b)
    return all(abs(_num(a.get(k)) - _num(b.get(k))) <= tol for k in keys)


def _by_sample(rows: list[dict]) -> dict[int, dict]:
    return {int(r["sample_id"]): r for r in rows}


def confusion_counts(gold_rows: list[dict], model_outputs: dict[str, dict]) -> dict[str, dict]:
    """Per-variable (and micro-averaged `overall`) TP/FP/FN/TN on the *reported*
    flag, treating the human gold as truth.

    The gold set is the universe of (doi, sample_id) pairs scored. A gold
    article with no/failed model output, or a gold sample the model didn't
    produce, counts as the model saying "not reported" (0) for every variable —
    so a genuinely-reported demographic there becomes a false negative rather
    than being silently dropped. Model samples with no gold counterpart are not
    counted here (over-/under-production is tracked separately by
    sample_count_match_rate in score()).
    """
    gold_by_doi: dict[str, list[dict]] = defaultdict(list)
    for r in gold_rows:
        gold_by_doi[r["doi"]].append(r)

    counts = {v: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for v in FLAG_FIELDS}
    for doi, rows in gold_by_doi.items():
        gold = _by_sample(rows)
        payload = model_outputs.get(doi)
        ok = payload is not None and payload.get("status") == "ok"
        model = _by_sample(payload["rows"]) if ok else {}
        for sid, g in gold.items():
            m = model.get(sid, {})
            for v in FLAG_FIELDS:
                gb = _reported_binary(g.get(v))
                mb = _reported_binary(m.get(v))
                cell = counts[v]
                if gb == 1 and mb == 1:
                    cell["tp"] += 1
                elif gb == 0 and mb == 1:
                    cell["fp"] += 1
                elif gb == 1 and mb == 0:
                    cell["fn"] += 1
                else:
                    cell["tn"] += 1

    overall = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for v in FLAG_FIELDS:
        for k in overall:
            overall[k] += counts[v][k]
    counts["overall"] = overall
    return counts


def _passes(value: float | None, threshold: float) -> bool | None:
    """None (metric undefined, e.g. recall with no gold positives) is treated as
    not-applicable rather than a failure."""
    return None if value is None else value >= threshold


def metrics_from_counts(counts: dict[str, dict], thresholds: dict | None = None) -> dict[str, dict]:
    """recall / precision / accuracy / f1 (+ per-metric and per-variable
    pass/fail) from confusion_counts()."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    metrics: dict[str, dict] = {}
    for v, c in counts.items():
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        recall = _safe_div(tp, tp + fn)
        precision = _safe_div(tp, tp + fp)
        accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
        f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
        passes = {
            "recall_pass": _passes(recall, thresholds["recall"]),
            "precision_pass": _passes(precision, thresholds["precision"]),
            "accuracy_pass": _passes(accuracy, thresholds["accuracy"]),
        }
        applicable = [p for p in passes.values() if p is not None]
        metrics[v] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "precision": precision, "accuracy": accuracy, "f1": f1,
            **passes,
            "passed": all(applicable),
        }
    return metrics


def score(
    gold_rows: list[dict],
    model_outputs: dict[str, dict],
    tol: float = 1.0,
    thresholds: dict | None = None,
):
    """Score model output against gold, aligned by (doi, sample_id).

    Returns (per_doi_records, summary). The gold set defines the universe of
    articles scored; a gold article with no model output is scored as a total
    miss on coverage/schema. `summary["metrics"]` holds the per-variable and
    overall recall/precision/accuracy gate, and `summary["passed"]` is the
    overall pass/fail.
    """
    gold_by_doi: dict[str, list[dict]] = defaultdict(list)
    for r in gold_rows:
        gold_by_doi[r["doi"]].append(r)

    per_doi = []
    tot_flag_correct = tot_flag = 0
    tot_num_correct = tot_num = 0
    n_sample_match = n_schema_ok = 0

    for doi in sorted(gold_by_doi):
        gold = _by_sample(gold_by_doi[doi])
        payload = model_outputs.get(doi)
        notes: list[str] = []

        if payload is None or payload.get("status") != "ok":
            reason = "no model output" if payload is None else f"model status={payload.get('status')}"
            per_doi.append(
                {
                    "doi": doi,
                    "n_gold_samples": len(gold),
                    "n_model_samples": 0,
                    "reporting_flag_accuracy": 0.0,
                    "numeric_accuracy": "",
                    "sample_count_match": False,
                    "schema_ok": False,
                    "notes": reason,
                }
            )
            continue

        model = _by_sample(payload["rows"])

        schema_ok = all(REQUIRED_KEYS <= set(row.keys()) for row in payload["rows"])
        if not schema_ok:
            notes.append("model row missing required keys")
        n_schema_ok += schema_ok

        sample_match = set(gold) == set(model)
        if not sample_match:
            notes.append(f"sample ids gold={sorted(gold)} model={sorted(model)}")
        n_sample_match += sample_match

        common = sorted(set(gold) & set(model))
        flag_correct = flag_total = 0
        num_correct = num_total = 0
        for sid in common:
            g, m = gold[sid], model[sid]
            for name in FLAG_FIELDS:
                flag_total += 1
                gf, mf = _reported(g.get(name)), _reported(m.get(name))
                flag_correct += int(gf == mf)
                if gf != mf:
                    notes.append(f"s{sid} {name}.reported: gold={gf} model={mf}")
            for name in PCT_DEMOGRAPHICS:
                gd, md = g.get(name) or {}, m.get(name) or {}
                if gd.get("reported") == 1 and md.get("reported") == 1:
                    num_total += 1
                    ok = _pct_equal(gd.get("pct") or {}, md.get("pct") or {}, tol)
                    num_correct += int(ok)
                    if not ok:
                        notes.append(f"s{sid} {name}.pct: gold={gd.get('pct')} model={md.get('pct')}")
            gs, ms = g.get("ses") or {}, m.get("ses") or {}
            if gs.get("reported") == 2 and ms.get("reported") == 2:
                num_total += 1
                ok = abs(_num(gs.get("value")) - _num(ms.get("value"))) <= tol
                num_correct += int(ok)

        per_doi.append(
            {
                "doi": doi,
                "n_gold_samples": len(gold),
                "n_model_samples": len(model),
                "reporting_flag_accuracy": round(flag_correct / flag_total, 3) if flag_total else "",
                "numeric_accuracy": round(num_correct / num_total, 3) if num_total else "",
                "sample_count_match": sample_match,
                "schema_ok": schema_ok,
                "notes": "; ".join(notes),
            }
        )
        tot_flag_correct += flag_correct
        tot_flag += flag_total
        tot_num_correct += num_correct
        tot_num += num_total

    counts = confusion_counts(gold_rows, model_outputs)
    metrics = metrics_from_counts(counts, thresholds)

    n_docs = len(gold_by_doi)
    summary = {
        "n_articles": n_docs,
        "reporting_flag_accuracy": round(tot_flag_correct / tot_flag, 3) if tot_flag else None,
        "numeric_accuracy": round(tot_num_correct / tot_num, 3) if tot_num else None,
        "sample_count_match_rate": round(n_sample_match / n_docs, 3) if n_docs else None,
        "schema_ok_rate": round(n_schema_ok / n_docs, 3) if n_docs else None,
        "metrics": metrics,
        "passed": metrics["overall"]["passed"],
    }
    return per_doi, summary


def write_per_doi(per_doi: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = [
        "doi",
        "n_gold_samples",
        "n_model_samples",
        "reporting_flag_accuracy",
        "numeric_accuracy",
        "sample_count_match",
        "schema_ok",
        "notes",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_doi)


METRICS_FIELDS = [
    "variable", "tp", "fp", "fn", "tn",
    "recall", "precision", "accuracy", "f1",
    "recall_pass", "precision_pass", "accuracy_pass", "passed",
]
# Order rows so the four demographics come first and `overall` last.
_METRICS_ROW_ORDER = FLAG_FIELDS + ["overall"]


def write_metrics(metrics: dict[str, dict], out_path: str) -> None:
    """Write the per-variable + overall recall/precision/accuracy gate to CSV."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for variable in _METRICS_ROW_ORDER:
            if variable in metrics:
                writer.writerow({"variable": variable, **metrics[variable]})


def load_thresholds(config_path: str) -> dict:
    """Read validation.thresholds from config.yaml, falling back to defaults."""
    import yaml

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except OSError:
        return dict(DEFAULT_THRESHOLDS)
    thresholds = (config.get("validation") or {}).get("thresholds") or {}
    return {**DEFAULT_THRESHOLDS, **thresholds}


def _fmt_gate(value, passed) -> str:
    if value is None:
        return "  n/a"
    tag = "PASS" if passed else "FAIL" if passed is not None else "n/a"
    return f"{value:.3f} {tag}"


def _print_summary(summary: dict) -> None:
    print(f"Scored {summary['n_articles']} pilot articles against the gold set:")
    metrics = summary.get("metrics", {})
    if metrics:
        gate = "PASS" if summary.get("passed") else "FAIL"
        print(f"  Recall / precision / accuracy gate (overall: {gate}):")
        header = f"    {'variable':<12} {'recall':>12} {'precision':>12} {'accuracy':>12}"
        print(header)
        for variable in _METRICS_ROW_ORDER:
            m = metrics.get(variable)
            if not m:
                continue
            print(
                f"    {variable:<12} "
                f"{_fmt_gate(m['recall'], m['recall_pass']):>12} "
                f"{_fmt_gate(m['precision'], m['precision_pass']):>12} "
                f"{_fmt_gate(m['accuracy'], m['accuracy_pass']):>12}"
            )
        ses = metrics.get("ses", {})
        ses_total = ses.get("tp", 0) + ses.get("fp", 0) + ses.get("fn", 0) + ses.get("tn", 0)
        ses_reported = ses.get("tp", 0) + ses.get("fn", 0)
        if ses_total:
            print(
                f"    (SES reported in {ses_reported}/{ses_total} gold samples "
                f"= {ses_reported / ses_total:.0%} — often sparse)"
            )
    print("  Secondary axes:")
    print(f"    Numeric accuracy (% breakdowns)  : {summary['numeric_accuracy']}")
    print(f"    Multi-sample handling (id match) : {summary['sample_count_match_rate']}")
    print(f"    Schema adherence (valid rows)    : {summary['schema_ok_rate']}")


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", default="data/pilot_gold.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None, help="Override config.yaml's default_model")
    parser.add_argument("--out", default="results/pilot_accuracy.csv")
    parser.add_argument("--metrics-out", default="results/pilot_metrics.csv",
                        help="Per-variable recall/precision/accuracy gate CSV")
    parser.add_argument("--tol", type=float, default=1.0, help="Tolerance for percentage-point agreement")
    args = parser.parse_args()

    if args.model:
        model_id = args.model
    else:
        with open(args.config, encoding="utf-8") as f:
            model_id = yaml.safe_load(f)["default_model"]

    thresholds = load_thresholds(args.config)
    gold_rows = load_gold(args.gold)
    model_outputs = load_model_outputs(args.results_dir, model_id)
    per_doi, summary = score(gold_rows, model_outputs, tol=args.tol, thresholds=thresholds)
    write_per_doi(per_doi, args.out)
    write_metrics(summary["metrics"], args.metrics_out)
    _print_summary(summary)
    print(f"Wrote per-article scores to {args.out}")
    print(f"Wrote recall/precision/accuracy gate to {args.metrics_out}")
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
