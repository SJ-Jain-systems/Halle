"""Check the model's pilot output against the gold answers we coded by hand
(see docs/SCORING_RUBRIC.md).

It reads the filled-in gold CSV (from src/make_gold_template.py) and the model's
per-article JSON from src/run_pilot.py (results/<model_id>/<doi>.json), lines the
rows up by (doi, sample_id), and reports how well they agree on the rubric's four
axes:

  Coverage / reporting flags : do the gold *_reported flags match the model's?
  Numeric accuracy           : do the reported *_pct breakdowns line up?
  Multi-sample handling       : did the model produce one row per sample like the gold?
  Schema adherence           : do the model rows have all the required keys?

Run it like:
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

from src.extract_demographics import REQUIRED_KEYS

FLAG_COLS = ["gender_reported", "race_reported", "education_reported", "ses_reported"]
PCT_DEMOGRAPHICS = [
    ("gender_reported", "gender_pct"),
    ("race_reported", "race_pct"),
    ("education_reported", "education_pct"),
]


def _num(x) -> float:
    if x is None or x == "":
        return 0.0
    return float(x)


def _to_int(x):
    if x is None or x == "":
        return None
    return int(float(x))


def _to_pct(x) -> dict:
    if isinstance(x, dict):
        return x
    if x is None or x == "":
        return {}
    return json.loads(x)


def load_gold(gold_csv: str) -> list[dict]:
    """Read the hand-coded gold CSV into typed rows (flags become ints, *_pct
    becomes a dict), keeping just the schema columns."""
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
                "gender_reported": _to_int(r.get("gender_reported")),
                "gender_pct": _to_pct(r.get("gender_pct")),
                "race_reported": _to_int(r.get("race_reported")),
                "race_pct": _to_pct(r.get("race_pct")),
                "education_reported": _to_int(r.get("education_reported")),
                "education_pct": _to_pct(r.get("education_pct")),
                "ses_reported": _to_int(r.get("ses_reported")),
                "ses_value": r.get("ses_value") if r.get("ses_value") else None,
            }
        )
    return rows


def load_model_outputs(results_dir: str, model_id: str) -> dict[str, dict]:
    """Load run_pilot.py's per-article JSON into {doi: {"status", "rows"}}."""
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
    """Score the model against the gold rows, matched up by (doi, sample_id).

    Returns (per_doi_records, summary). The gold set is what we're scoring
    against, so an article that's in the gold but has no model output counts as a
    total miss on coverage and schema.
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
            for col in FLAG_COLS:
                flag_total += 1
                flag_correct += int(g.get(col) == m.get(col))
                if g.get(col) != m.get(col):
                    notes.append(f"s{sid} {col}: gold={g.get(col)} model={m.get(col)}")
            for flag_col, pct_col in PCT_DEMOGRAPHICS:
                if g.get(flag_col) == 1 and m.get(flag_col) == 1:
                    num_total += 1
                    ok = _pct_equal(g.get(pct_col) or {}, m.get(pct_col) or {}, tol)
                    num_correct += int(ok)
                    if not ok:
                        notes.append(f"s{sid} {pct_col}: gold={g.get(pct_col)} model={m.get(pct_col)}")
            if g.get("ses_reported") == 2 and m.get("ses_reported") == 2:
                num_total += 1
                ok = abs(_num(g.get("ses_value")) - _num(m.get("ses_value"))) <= tol
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
    parser.add_argument("--tol", type=float, default=1.0, help="How many percentage points of slack to allow")
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
