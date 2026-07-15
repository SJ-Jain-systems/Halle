"""Score the model's pilot output against the hand-coded gold set
(docs/SCORING_RUBRIC.md).

Reads the completed gold CSV (produced from src/make_gold_template.py and coded
by hand) and the model's per-article JSON output written by src/run_pilot.py
(results/<model_id>/<doi>.json), aligns rows by (doi, sample_id), and reports
per-field agreement mapped to the rubric's four axes:

  - Coverage / reporting flags : does gold `*_reported` match the model's?
  - Numeric accuracy           : do the reported `*_pct` breakdowns agree?
  - Multi-sample handling       : one gold row per sample matched by the model?
  - Schema adherence           : model rows carry exactly the required keys?

Usage:
    python -m src.score_pilot \\
        --gold data/pilot_gold.csv \\
        --results-dir results \\
        --model meta-llama/Llama-3.3-70B-Instruct \\
        --out results/pilot_accuracy.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

from src.extract_demographics import (
    DEMOGRAPHIC_FIELDS,
    PCT_FIELDS,
    REQUIRED_KEYS,
    parse_field,
)

FLAG_FIELDS = list(DEMOGRAPHIC_FIELDS)  # gender, race, education, ses
PCT_DEMOGRAPHICS = list(PCT_FIELDS)     # gender, race, education


def _num(x) -> float:
    if x is None or x == "":
        return 0.0
    return float(x)


def _to_int(x):
    if x is None or x == "":
        return None
    return int(float(x))


def _reported(field) -> int | None:
    return (field or {}).get("reported")


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


def score(gold_rows: list[dict], model_outputs: dict[str, dict], tol: float = 1.0):
    """Score model output against gold, aligned by (doi, sample_id).

    Returns (per_doi_records, summary). The gold set defines the universe of
    articles scored; a gold article with no model output is scored as a total
    miss on coverage/schema.
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

    n_docs = len(gold_by_doi)
    summary = {
        "n_articles": n_docs,
        "reporting_flag_accuracy": round(tot_flag_correct / tot_flag, 3) if tot_flag else None,
        "numeric_accuracy": round(tot_num_correct / tot_num, 3) if tot_num else None,
        "sample_count_match_rate": round(n_sample_match / n_docs, 3) if n_docs else None,
        "schema_ok_rate": round(n_schema_ok / n_docs, 3) if n_docs else None,
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


def _print_summary(summary: dict) -> None:
    print(f"Scored {summary['n_articles']} pilot articles against the gold set:")
    print(f"  Coverage (reporting-flag accuracy) : {summary['reporting_flag_accuracy']}")
    print(f"  Numeric accuracy (% breakdowns)    : {summary['numeric_accuracy']}")
    print(f"  Multi-sample handling (id match)   : {summary['sample_count_match_rate']}")
    print(f"  Schema adherence (valid rows)      : {summary['schema_ok_rate']}")


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", default="data/pilot_gold.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None, help="Override config.yaml's default_model")
    parser.add_argument("--out", default="results/pilot_accuracy.csv")
    parser.add_argument("--tol", type=float, default=1.0, help="Tolerance for percentage-point agreement")
    args = parser.parse_args()

    if args.model:
        model_id = args.model
    else:
        with open(args.config, encoding="utf-8") as f:
            model_id = yaml.safe_load(f)["default_model"]

    gold_rows = load_gold(args.gold)
    model_outputs = load_model_outputs(args.results_dir, model_id)
    per_doi, summary = score(gold_rows, model_outputs, tol=args.tol)
    write_per_doi(per_doi, args.out)
    _print_summary(summary)
    print(f"Wrote per-article scores to {args.out}")


if __name__ == "__main__":
    main()
