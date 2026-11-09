from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import run_major_revision_experiments as revision
import run_synergy_prior_experiments as base


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "major_revision"
OUT = RESULTS / "summary"


def summarize_metrics(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    metrics = [
        "rmse",
        "pearson",
        "pcs",
        "conditional_pcs",
        "pcs_fixed_denominator",
        "pcs_components_used",
        "incremental_prior_r2_given_label",
        "pred_r2_from_label",
        "pred_r2_from_label_and_priors",
        "residual_target_rmse",
        "residual_target_pearson",
        "pcs_excess_vs_moment_matched_null",
        "top5pct_enrichment",
        "elapsed_sec",
        "working_set_peak_mb",
        "working_set_delta_mb",
    ]
    available = [x for x in metrics if x in df.columns]
    grouped = df.groupby(["split", "model"])[available].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join([str(x) for x in col if x]) for col in grouped.columns]
    grouped.insert(0, "phase", phase)
    return grouped


def pair_blocks(group: pd.DataFrame) -> list[pd.DataFrame]:
    return [block.reset_index(drop=True) for _, block in group.groupby("pair_id", sort=False)]


def sampled_blocks(blocks: list[pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    chosen = rng.integers(0, len(blocks), size=len(blocks))
    return pd.concat([blocks[i] for i in chosen], ignore_index=True)


def metrics_from_predictions(group: pd.DataFrame) -> dict[str, float]:
    y = group["y_true"].to_numpy(dtype=float)
    pred = group["y_pred"].to_numpy(dtype=float)
    priors = {
        "drug": group["drug_prior"].to_numpy(dtype=float),
        "pair": group["pair_prior"].to_numpy(dtype=float),
        "cell": group["cell_prior"].to_numpy(dtype=float),
    }
    raw_corrs = []
    partial_corrs = []
    r_pred_y = base.finite_corr(pred, y)
    for values in priors.values():
        r_pred_prior = base.finite_corr(pred, values)
        r_y_prior = base.finite_corr(y, values)
        if np.isfinite(r_pred_prior):
            raw_corrs.append(abs(r_pred_prior))
        denominator = np.sqrt(max(0.0, (1.0 - r_pred_y**2) * (1.0 - r_y_prior**2)))
        if np.isfinite(r_pred_prior) and np.isfinite(r_y_prior) and denominator > 1e-12:
            partial_corrs.append(abs((r_pred_prior - r_pred_y * r_y_prior) / denominator))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "pearson": r_pred_y,
        "pcs": float(np.mean(raw_corrs)) if raw_corrs else float("nan"),
        "conditional_pcs": float(np.mean(partial_corrs)) if partial_corrs else float("nan"),
    }


def hierarchical_pair_bootstrap(pred: pd.DataFrame, phase: str, n_boot: int = 100, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (split, model), model_df in pred.groupby(["split", "model"], sort=False):
        by_seed = {int(k): v.reset_index(drop=True) for k, v in model_df.groupby("seed")}
        blocks_by_seed = {key: pair_blocks(group) for key, group in by_seed.items()}
        point_values = [metrics_from_predictions(group) for group in by_seed.values()]
        point = {key: float(np.nanmean([x[key] for x in point_values])) for key in point_values[0]}
        draws = {key: [] for key in point}
        for _ in range(n_boot):
            per_seed = []
            for blocks in blocks_by_seed.values():
                per_seed.append(metrics_from_predictions(sampled_blocks(blocks, rng)))
            for key in draws:
                draws[key].append(float(np.nanmean([x[key] for x in per_seed])))
        row: dict[str, object] = {"phase": phase, "split": split, "model": model, "n_seeds": len(by_seed)}
        for key, value in point.items():
            row[key] = value
            row[f"{key}_ci_low"] = float(np.nanpercentile(draws[key], 2.5))
            row[f"{key}_ci_high"] = float(np.nanpercentile(draws[key], 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_rmse_bootstrap(
    pred: pd.DataFrame,
    phase: str,
    baseline: str = "feature_ridge",
    n_boot: int = 100,
    seed: int = 712,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    keys = ["seed", "split", "drug1", "drug2", "pair_id", "cell", "y_true"]
    base_df = pred[pred["model"] == baseline][keys + ["y_pred"]].rename(columns={"y_pred": "baseline_pred"})
    for (split, model), model_df in pred[pred["model"] != baseline].groupby(["split", "model"], sort=False):
        merged = model_df[keys + ["y_pred"]].merge(base_df, on=keys, how="inner", validate="one_to_one")
        by_seed = {int(k): v.reset_index(drop=True) for k, v in merged.groupby("seed")}
        blocks_by_seed = {key: pair_blocks(group) for key, group in by_seed.items()}

        def delta(group: pd.DataFrame) -> float:
            y = group["y_true"].to_numpy(dtype=float)
            model_rmse = float(np.sqrt(mean_squared_error(y, group["y_pred"])))
            base_rmse = float(np.sqrt(mean_squared_error(y, group["baseline_pred"])))
            return model_rmse - base_rmse

        point = float(np.mean([delta(group) for group in by_seed.values()]))
        draws = []
        for _ in range(n_boot):
            draws.append(float(np.mean([delta(sampled_blocks(blocks, rng)) for blocks in blocks_by_seed.values()])))
        rows.append(
            {
                "phase": phase,
                "split": split,
                "model": model,
                "baseline": baseline,
                "n_seeds": len(by_seed),
                "delta_rmse": point,
                "delta_rmse_ci_low": float(np.percentile(draws, 2.5)),
                "delta_rmse_ci_high": float(np.percentile(draws, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    core_summary: pd.DataFrame,
    cpb_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    deltas: pd.DataFrame,
    scale: pd.DataFrame,
) -> None:
    core_view = core_summary[[
        "split",
        "model",
        "rmse_mean",
        "rmse_std",
        "pcs_mean",
        "conditional_pcs_mean",
        "pcs_excess_vs_moment_matched_null_mean",
    ]]
    cpb_view = cpb_summary[[
        "split",
        "model",
        "rmse_mean",
        "rmse_std",
        "pcs_mean",
        "pcs_std",
        "conditional_pcs_mean",
        "conditional_pcs_std",
    ]]
    lines = [
        "# Major Revision Experiment Summary",
        "",
        "## Core multi-seed results",
        "",
        core_view.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## CPB matched controls",
        "",
        cpb_view.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Hierarchical unordered-pair bootstrap",
        "",
        bootstrap.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired RMSE deltas versus feature ridge",
        "",
        deltas.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Scale and resource sensitivity",
        "",
        scale.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Evidence-bound interpretation",
        "",
        "- PCS is reported as prior coupling, not proof of shortcut use.",
        "- Conditional PCS controls linearly for the observed label and remains a diagnostic association rather than a causal quantity.",
        "- Cold-both-drugs removes both drug and exact-pair reuse; cell priors remain available and their component count is reported.",
        "- The CPB sweep does not establish learned-prototype Pareto superiority over matched controls at comparable RMSE cost.",
        "- The distributed TDC task table contains no study/source/assay field, so a source split cannot be estimated from this artifact.",
    ]
    (OUT / "major_revision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    core_metrics = pd.read_csv(RESULTS / "core_metrics.csv")
    cpb_metrics = pd.read_csv(RESULTS / "cpb_metrics.csv")
    scale = pd.read_csv(RESULTS / "scale_metrics.csv")
    core_pred = pd.read_csv(RESULTS / "core_predictions.csv")
    cpb_pred = pd.read_csv(RESULTS / "cpb_predictions.csv")

    core_summary = summarize_metrics(core_metrics, "core")
    cpb_summary = summarize_metrics(cpb_metrics, "cpb")
    core_ci = core_pred[
        core_pred["split"].isin(["random", "cold_pair", "cold_both_drugs", "molecular_cluster"])
        & core_pred["model"].isin(["identity_ridge", "additive_prior", "feature_ridge", "dual_view_mlp"])
    ]
    bootstrap = pd.concat(
        [
            hierarchical_pair_bootstrap(core_ci, "core"),
            hierarchical_pair_bootstrap(cpb_pred, "cpb"),
        ],
        ignore_index=True,
    )
    deltas = pd.concat(
        [
            paired_rmse_bootstrap(core_ci, "core"),
            paired_rmse_bootstrap(cpb_pred, "cpb"),
        ],
        ignore_index=True,
    )
    core_summary.to_csv(OUT / "core_multiseed_summary.csv", index=False)
    cpb_summary.to_csv(OUT / "cpb_multiseed_summary.csv", index=False)
    bootstrap.to_csv(OUT / "pair_cluster_bootstrap_ci.csv", index=False)
    deltas.to_csv(OUT / "paired_delta_vs_feature_cluster_bootstrap.csv", index=False)
    scale.to_csv(OUT / "scale_resource_summary.csv", index=False)
    metadata = {
        "bootstrap_unit": "unordered drug pair",
        "bootstrap_hierarchy": "resample unordered pairs within each split seed, then average seed-level metrics",
        "bootstrap_replicates": 100,
        "split_seeds": sorted(core_metrics["seed"].unique().astype(int).tolist()),
    }
    (OUT / "statistical_protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(core_summary, cpb_summary, bootstrap, deltas, scale)
    print(f"Wrote summaries to {OUT}")


if __name__ == "__main__":
    main()
