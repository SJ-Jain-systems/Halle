"""Answer the brief's main goal: how does demographic reporting in PLOS ONE
psychology articles change over time, year by year and by stage.

Reads data/demographics_table.csv (src/run_pipeline.py output) and writes,
for each of gender/race/education/ses:
  - reporting rate (% of samples that report it) by year and by stage
  - mean reported percentage per subgroup (e.g. mean %female) by year and
    by stage, among samples that do report it
plus optional trend-line plots.

Usage:
    python -m src.analyze_trends --table data/demographics_table.csv --out-dir data/analysis
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

CATEGORIES = {
    "gender": ("gender_reported", "gender_pct"),
    "race": ("race_reported", "race_pct"),
    "education": ("education_reported", "education_pct"),
}
STAGE_ORDER = ["early", "middle", "covid", "post_covid"]


def load_table(table_csv: str) -> pd.DataFrame:
    df = pd.read_csv(table_csv)
    for _, pct_col in CATEGORIES.values():
        df[pct_col] = df[pct_col].apply(_safe_json_loads)
    return df


def _safe_json_loads(value):
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


def reporting_rate(df: pd.DataFrame, reported_col: str, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col)[reported_col]
        .apply(lambda s: 100 * (s == 1).sum() / len(s))
        .rename("reporting_rate_pct")
        .reset_index()
    )


def mean_subgroup_pct(df: pd.DataFrame, reported_col: str, pct_col: str, group_col: str) -> pd.DataFrame:
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
    if "stage" in df.columns:
        df["stage"] = pd.Categorical(df["stage"], categories=STAGE_ORDER, ordered=True)
        df = df.sort_values("stage")
    return df


def run(table_csv: str, out_dir: str, make_plots: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = load_table(table_csv)

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

    # SES uses a 0/1/2 reporting-detail scale rather than reported/not, so it
    # gets its own summary instead of reusing reporting_rate()/mean_subgroup_pct().
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
    try:
        import matplotlib

        matplotlib.use("Agg")
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
