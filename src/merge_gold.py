"""Combine several blind coders' gold sheets into one consensus gold set and
report inter-rater agreement (docs/SCORING_RUBRIC.md; 7/15 meeting).

Multiple coders code the same pilot articles to keep the ground truth from being
skewed by any one coder. This tool aligns their sheets by (doi, sample_id) and:

  - inter_rater_agreement(): per-variable percent agreement and Cohen's kappa on
    the *reported* flags (pairwise, averaged over coder pairs when >2 coders);
  - consensus(): majority vote on each demographic's reported flag, with the
    percentage breakdown taken as the per-subgroup median across the coders who
    marked it reported. Rows where coders disagree are recorded in a
    disagreements report for the team to adjudicate.

The consensus CSV is written in exactly the schema src/score_pilot.py::load_gold
consumes, so scoring the model is unchanged downstream.

Inputs may be one CSV per coder, or a single CSV carrying a `coder` column
(src/make_gold_template.py emits that column). When a sheet has no `coder`
value, the coder id falls back to the file's basename.

Usage:
    python -m src.merge_gold \\
        --inputs data/gold_alice.csv data/gold_bob.csv \\
        --out data/pilot_gold.csv \\
        --agreement-out results/inter_rater.csv \\
        --disagreements-out results/gold_disagreements.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
from collections import defaultdict

from src.extract_demographics import (
    DEMOGRAPHIC_FIELDS,
    format_field,
    parse_field,
)

CONSENSUS_FIELDS = ["doi", "sample_id", "coder", "gender", "race", "education", "ses"]


def _reported_binary(field) -> int:
    """0/1 reported flag; SES's 0/1/2 detail scale collapses to reported = >=1
    (same convention as src/score_pilot.py)."""
    return 1 if ((field or {}).get("reported") or 0) >= 1 else 0


def _coder_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_coded(paths: list[str]) -> tuple[dict, list[str]]:
    """Read coder sheets into {(doi, sample_id): {coder: {var: parsed field}}}
    plus the list of coder ids in first-seen order."""
    units: dict = defaultdict(dict)
    coders: list[str] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("doi"):
                    continue
                coder = (r.get("coder") or "").strip() or _coder_from_path(path)
                try:
                    sid = int(float(r.get("sample_id") or 1))
                except (TypeError, ValueError):
                    sid = 1
                units[(r["doi"], sid)][coder] = {
                    v: parse_field(v, r.get(v)) for v in DEMOGRAPHIC_FIELDS
                }
                if coder not in coders:
                    coders.append(coder)
    return units, coders


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float | None:
    """Cohen's kappa for two aligned label sequences. Returns None if there's
    nothing to compare; 1.0 when raters fully agree with no label variance
    (the degenerate p_e == 1 case)."""
    n = len(labels_a)
    if n == 0:
        return None
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    categories = set(labels_a) | set(labels_b)
    pe = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories)
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return round((po - pe) / (1 - pe), 3)


def inter_rater_agreement(units: dict, coders: list[str]) -> dict[str, dict]:
    """Per-variable (and `overall`) percent agreement + mean pairwise Cohen's
    kappa on the reported flags."""
    pairs = list(itertools.combinations(coders, 2))
    result: dict[str, dict] = {}
    tot_agree = tot_pairs_units = 0
    all_kappas: list[float] = []
    for v in DEMOGRAPHIC_FIELDS:
        agree = total = 0
        kappas: list[float] = []
        for a, b in pairs:
            la, lb = [], []
            for byc in units.values():
                if a in byc and b in byc:
                    la.append(_reported_binary(byc[a][v]))
                    lb.append(_reported_binary(byc[b][v]))
            agree += sum(int(x == y) for x, y in zip(la, lb))
            total += len(la)
            k = cohen_kappa(la, lb)
            if k is not None:
                kappas.append(k)
        result[v] = {
            "percent_agreement": round(agree / total, 3) if total else None,
            "cohen_kappa": round(sum(kappas) / len(kappas), 3) if kappas else None,
            "n_pairs": len(pairs),
            "n_compared": total,
        }
        tot_agree += agree
        tot_pairs_units += total
        all_kappas.extend(kappas)
    result["overall"] = {
        "percent_agreement": round(tot_agree / tot_pairs_units, 3) if tot_pairs_units else None,
        "cohen_kappa": round(sum(all_kappas) / len(all_kappas), 3) if all_kappas else None,
        "n_pairs": len(pairs),
        "n_compared": tot_pairs_units,
    }
    return result


def _median(nums: list[float]):
    nums = sorted(nums)
    n = len(nums)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2


def _median_pct(pcts: list[dict]) -> dict:
    """Per-subgroup median across coders who reported the demographic."""
    keys = []
    for d in pcts:
        for k in d:
            if k not in keys:
                keys.append(k)
    out = {}
    for k in keys:
        vals = [d[k] for d in pcts if k in d]
        m = _median(vals)
        if m is None:
            continue
        out[k] = int(m) if float(m).is_integer() else m
    return out


def consensus(units: dict, coders: list[str]) -> tuple[list[dict], list[dict]]:
    """Majority-vote consensus per (doi, sample_id), plus a disagreements report.

    A demographic is consensus-reported when a strict majority of coders marked
    it reported; its percentage breakdown is the per-subgroup median across those
    coders. Even splits (no majority) default to not-reported and are flagged for
    adjudication, as is any variable the coders split on.
    """
    rows: list[dict] = []
    disagreements: list[dict] = []
    for key in sorted(units):
        doi, sid = key
        byc = units[key]
        n = len(byc)
        row = {"doi": doi, "sample_id": sid, "coder": "consensus"}
        notes: list[str] = []
        for v in DEMOGRAPHIC_FIELDS:
            flags = [_reported_binary(byc[c][v]) for c in byc]
            ones = sum(flags)
            reported = 1 if ones * 2 > n else 0
            if len(set(flags)) > 1:
                notes.append(f"{v}:split{flags}")
            reporting = [byc[c][v] for c in byc if _reported_binary(byc[c][v])]
            if v == "ses":
                if reported:
                    levels = [(f or {}).get("reported") or 0 for f in reporting]
                    level = int(_median(levels)) if levels else 1
                    numeric = [
                        float((f or {}).get("value"))
                        for f in reporting
                        if ((f or {}).get("reported") or 0) >= 2 and (f or {}).get("value") not in (None, "")
                    ]
                    value = _median(numeric) if numeric else None
                    if value is not None and float(value).is_integer():
                        value = int(value)
                    row[v] = format_field("ses", {"reported": level, "value": value})
                else:
                    row[v] = format_field("ses", {"reported": 0, "value": None})
            else:
                if reported:
                    pct = _median_pct([(f or {}).get("pct") or {} for f in reporting])
                    row[v] = format_field(v, {"reported": 1, "pct": pct})
                else:
                    row[v] = format_field(v, {"reported": 0, "pct": {}})
        rows.append(row)
        if notes:
            disagreements.append({"doi": doi, "sample_id": sid, "notes": "; ".join(notes)})
    return rows, disagreements


def write_consensus(rows: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONSENSUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_agreement(agreement: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    order = list(DEMOGRAPHIC_FIELDS) + ["overall"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variable", "percent_agreement", "cohen_kappa", "n_pairs", "n_compared"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for v in order:
            if v in agreement:
                writer.writerow({"variable": v, **agreement[v]})


def write_disagreements(disagreements: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doi", "sample_id", "notes"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(disagreements)


def _print_agreement(agreement: dict, coders: list[str]) -> None:
    print(f"Inter-rater agreement over {len(coders)} coder(s): {', '.join(coders)}")
    print(f"    {'variable':<12} {'% agree':>9} {'kappa':>8}")
    for v in list(DEMOGRAPHIC_FIELDS) + ["overall"]:
        m = agreement.get(v)
        if not m:
            continue
        pa = "n/a" if m["percent_agreement"] is None else f"{m['percent_agreement']:.3f}"
        kp = "n/a" if m["cohen_kappa"] is None else f"{m['cohen_kappa']:.3f}"
        print(f"    {v:<12} {pa:>9} {kp:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="Coder gold CSVs")
    parser.add_argument("--out", default="data/pilot_gold.csv", help="Consensus gold CSV")
    parser.add_argument("--agreement-out", default="results/inter_rater.csv")
    parser.add_argument("--disagreements-out", default="results/gold_disagreements.csv")
    args = parser.parse_args()

    units, coders = load_coded(args.inputs)
    agreement = inter_rater_agreement(units, coders)
    rows, disagreements = consensus(units, coders)

    write_consensus(rows, args.out)
    write_agreement(agreement, args.agreement_out)
    write_disagreements(disagreements, args.disagreements_out)

    _print_agreement(agreement, coders)
    print(f"Wrote consensus gold ({len(rows)} samples) to {args.out}")
    print(f"Flagged {len(disagreements)} sample(s) with coder disagreement -> {args.disagreements_out}")


if __name__ == "__main__":
    main()
