# NOTES
# The payoff. Takes the finished demographics spreadsheet and turns it into the
# actual answers. For each demographic (gender, race, education, socioeconomic
# status) it computes two things:
#   Reporting rate: what fraction of samples reported it, by year and by the four
#   time stages. This is the headline number. "X% of psychology papers in 2020
#   reported participant race."
#   Average composition: among papers that did report it, the average breakdown,
#   like mean percent female. This describes who the samples were.
#
# It writes one small summary spreadsheet per demographic per grouping, plus
# optional trend charts. Socioeconomic status is handled on its own because it
# uses a 0/1/2 detail scale instead of a plain yes/no.
#
# Standard data summarizing with pandas. No AI here.
"""Turn the demographics table into reporting-rate trends by year and stage.

Usage:
    python -m src.analyze_trends --table data/demographics_table.csv --out-dir data/analysis
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

# The three demographics that use a plain yes/no plus a percentage breakdown.
# Socioeconomic status is handled separately further down.
CATEGORIES = {
    "gender": ("gender_reported", "gender_pct"),
    "race": ("race_reported", "race_pct"),
    "education": ("education_reported", "education_pct"),
}
STAGE_ORDER = ["early", "middle", "covid", "post_covid"]


def load_table(table_csv: str) -> pd.DataFrame:
    # Load the table. The percentage columns are stored as JSON text, like
    # '{"male": 45, "female": 55}', so turn those back into real objects.
    df = pd.read_csv(table_csv)
    for _, pct_col in CATEGORIES.values():
        df[pct_col] = df[pct_col].apply(_safe_json_loads)
    return df


def _safe_json_loads(value):
    # Turn the stored text back into a dict. Return {} if it won't parse.
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


def reporting_rate(df: pd.DataFrame, reported_col: str, group_col: str) -> pd.DataFrame:
    # For each year (or stage), what percent of samples had reported == 1. This
    # is the core "how often is it reported" number.
    return (
        df.groupby(group_col)[reported_col]
        .apply(lambda s: 100 * (s == 1).sum() / len(s))
        .rename("reporting_rate_pct")
        .reset_index()
    )


def mean_subgroup_pct(df: pd.DataFrame, reported_col: str, pct_col: str, group_col: str) -> pd.DataFrame:
    # Among samples that did report this demographic, average each subgroup's
    # percentage (mean percent female, mean percent white) by year or stage.
    reported = df[df[reported_col] == 1]
    records = []
    for _, row in reported.iterrows():
        pct_dict = row[pct_col] or {}
        for subgroup, pct_value in pct_dict.items():
            records.append({group_col: row[group_col], "subgroup": subgroup, "pct": pct_value})
    if not records:
        return pd.DataFrame(columns=[group_col, "subgroup", "mean_pct", "n"])
    long_df = pd.DataFrame(records)
    long_df["pct"] = pd.to_numeric(long_df["pct"], errors="coerce")
    return (
        long_df.groupby([group_col, "subgroup"])["pct"]
        .agg(mean_pct="mean", n="count")
        .reset_index()
    )


def order_stage_column(df: pd.DataFrame) -> pd.DataFrame:
    # Make the four stages sort in time order, not alphabetically.
    if "stage" in df.columns:
        df["stage"] = pd.Categorical(df["stage"], categories=STAGE_ORDER, ordered=True)
        df = df.sort_values("stage")
    return df


def run(table_csv: str, out_dir: str, make_plots: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = load_table(table_csv)

    # For each demographic, and both groupings (by year and by stage), write the
    # reporting-rate table, the average-composition table, and a chart.
    for category, (reported_col, pct_col) in CATEGORIES.items():
        for group_col in ("year", "stage"):
            rate_df = reporting_rate(df, reported_col, group_col)
            if group_col == "stage":
                rate_df = order_stage_column(rate_df)
            rate_path = os.path.join(out_dir, f"{category}_reporting_rate_by_{group_col}.csv")
            rate_df.to_csv(rate_path, index=False)

            mean_df = mean_subgroup_pct(df, reported_col, pct_col, group_col)
            if group_col == "stage":
                mean_df = order_stage_column(mean_df)
            mean_path = os.path.join(out_dir, f"{category}_mean_pct_by_{group_col}.csv")
            mean_df.to_csv(mean_path, index=False)

            if make_plots:
                _plot(rate_df, group_col, "reporting_rate_pct", category,
                      os.path.join(out_dir, f"{category}_reporting_rate_by_{group_col}.png"))

    # SES uses a 0/1/2 detail scale, not reported/not, so it gets its own
    # summary. Here I report the share of samples at each level (0/1/2) per period.
    for group_col in ("year", "stage"):
        ses_df = (
            df.groupby(group_col)["ses_reported"]
            .value_counts(normalize=True)
            .rename("share")
            .mul(100)
            .reset_index()
        )
        if group_col == "stage":
            ses_df = order_stage_column(ses_df)
        ses_df.to_csv(os.path.join(out_dir, f"ses_reporting_detail_by_{group_col}.csv"), index=False)

    print(f"Wrote trend summaries to {out_dir}")


def _plot(df: pd.DataFrame, x_col: str, y_col: str, category: str, out_path: str) -> None:
    # Draw a simple trend line of reporting rate over time and save it as a PNG.
    # Skips quietly if matplotlib isn't installed or there's no data.
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display needed, just save image files
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df[x_col].astype(str), df[y_col], marker="o")
    ax.set_title(f"{category.title()} reporting rate by {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel("% of samples reporting")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data/demographics_table.csv")
    parser.add_argument("--out-dir", default="data/analysis")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run(args.table, args.out_dir, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
