from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "second_revision"
OUT = RESULTS / "summary"


def mean_std(frame: pd.DataFrame, value: str) -> str:
    return f"{frame[value].mean():.4f} +/- {frame[value].std():.4f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(RESULTS / "full_critical_metrics.csv")
    expected = 3 * 3 * 5
    if len(full) < expected:
        raise RuntimeError(
            f"Full critical comparison incomplete: {len(full)} of {expected} rows."
        )

    full_summary = (
        full.groupby(["split", "model"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            conditional_pcs_mean=("conditional_pcs", "mean"),
            conditional_pcs_std=("conditional_pcs", "std"),
            pred_r2_from_label_mean=("pred_r2_from_label", "mean"),
            pred_r2_from_label_and_priors_mean=(
                "pred_r2_from_label_and_priors",
                "mean",
            ),
            incremental_prior_r2_mean=("incremental_prior_r2_given_label", "mean"),
        )
    )
    full_summary.to_csv(OUT / "full_critical_summary.csv", index=False)

    audit = pd.read_csv(RESULTS / "full_data_split_audit.csv")
    audit.to_csv(OUT / "full_split_audit.csv", index=False)

    pareto = pd.read_csv(RESULTS / "cpb_pareto_metrics.csv")
    pareto_summary = (
        pareto.groupby(
            ["split", "control", "k_pair", "k_cell", "strength"],
            as_index=False,
        )
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            conditional_pcs_mean=("conditional_pcs", "mean"),
            conditional_pcs_std=("conditional_pcs", "std"),
        )
    )
    pareto_summary.to_csv(OUT / "cpb_pareto_summary.csv", index=False)

    synthetic = pd.read_csv(RESULTS / "synthetic_sanity_summary.csv")
    synthetic.to_csv(OUT / "synthetic_sanity_summary.csv", index=False)

    random_identity = full[
        (full["split"] == "random") & (full["model"] == "identity_ridge")
    ]
    random_feature = full[
        (full["split"] == "random") & (full["model"] == "feature_ridge")
    ]
    cold_both = audit[audit["split"] == "cold_both_drugs"]

    lines = [
        "# Second Revision Evidence Summary",
        "",
        "## Full-table critical comparison",
        "",
        full_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Split audit",
        "",
        audit.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Synthetic sanity check",
        "",
        synthetic[
            [
                "case",
                "rmse_mean",
                "pearson_mean",
                "pcs_mean",
                "conditional_pcs_mean",
                "incremental_prior_r2_given_label_mean",
                "pcs_excess_vs_moment_matched_null_mean",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Evidence-bound conclusions",
        "",
        f"- Full-table random identity RMSE: {mean_std(random_identity, 'rmse')}.",
        f"- Full-table random feature-ridge RMSE: {mean_std(random_feature, 'rmse')}.",
        (
            "- Cold-both test rows range from "
            f"{cold_both['n_test'].min():,} to {cold_both['n_test'].max():,}; "
            f"{cold_both['n_discarded_cross_group'].min():,} to "
            f"{cold_both['n_discarded_cross_group'].max():,} cross-group rows are discarded."
        ),
        "- Learned CPB is not Pareto-superior to random or shuffled controls at matched RMSE.",
        "- The synthetic check separates label-only, direct-prior, and nonlinear label-prior cases using cPCS, incremental prior R2, and the moment-matched null.",
    ]
    (OUT / "second_revision_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"Wrote second-revision summaries to {OUT}")


if __name__ == "__main__":
    main()
