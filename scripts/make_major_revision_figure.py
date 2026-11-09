from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "major_revision"
SUMMARY = RESULTS / "summary"
FIGURES = ROOT / "figures" / "synergy_manuscript"


MODEL_LABELS = {
    "identity_ridge": "Identity ridge",
    "additive_prior": "Additive prior",
    "feature_ridge": "Feature ridge",
    "dual_view_mlp": "Dual-view MLP",
    "cpb_learned_anchor": "Learned + anchors",
    "cpb_learned_no_anchor": "Learned residual",
    "cpb_shuffled_assignment": "Shuffled assignment",
    "cpb_random_clusters": "Random clusters",
}

COLORS = {
    "identity_ridge": "#2F4B7C",
    "additive_prior": "#008B8B",
    "feature_ridge": "#5B5B5B",
    "dual_view_mlp": "#E07A5F",
    "cpb_learned_anchor": "#007C91",
    "cpb_learned_no_anchor": "#2A9D8F",
    "cpb_shuffled_assignment": "#D97706",
    "cpb_random_clusters": "#9C6644",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.15, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    source_dir = FIGURES / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    style()

    boot = pd.read_csv(SUMMARY / "pair_cluster_bootstrap_ci.csv")
    cpb = pd.read_csv(RESULTS / "cpb_metrics.csv")
    scale = pd.read_csv(RESULTS / "scale_metrics.csv")

    splits = ["random", "cold_pair", "cold_both_drugs", "molecular_cluster"]
    split_labels = ["Random", "Cold pair", "Cold both*", "Mol. cluster"]
    models = ["identity_ridge", "additive_prior", "feature_ridge", "dual_view_mlp"]
    core = boot[(boot["phase"] == "core") & boot["split"].isin(splits) & boot["model"].isin(models)].copy()
    core.to_csv(source_dir / "fig_major_revision_core_bootstrap.csv", index=False)

    cpb_feature = cpb[cpb["model"] == "feature_ridge"][["seed", "split", "rmse", "conditional_pcs"]].rename(
        columns={"rmse": "feature_rmse", "conditional_pcs": "feature_conditional_pcs"}
    )
    cpb_delta = cpb[cpb["model"] != "feature_ridge"].merge(cpb_feature, on=["seed", "split"], validate="many_to_one")
    cpb_delta["delta_rmse"] = cpb_delta["rmse"] - cpb_delta["feature_rmse"]
    cpb_delta["delta_conditional_pcs"] = cpb_delta["conditional_pcs"] - cpb_delta["feature_conditional_pcs"]
    cpb_stats = (
        cpb_delta.groupby(["split", "model"])[["delta_rmse", "delta_conditional_pcs"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    cpb_stats.columns = ["_".join(str(x) for x in col if x) for col in cpb_stats.columns]
    cpb_stats.to_csv(source_dir / "fig_major_revision_cpb_tradeoff.csv", index=False)
    scale.to_csv(source_dir / "fig_major_revision_scale_resources.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.25), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    offsets = np.linspace(-0.24, 0.24, len(models))
    for offset, model in zip(offsets, models):
        values = core[core["model"] == model].set_index("split").reindex(splits)
        ax_a.errorbar(
            np.arange(len(splits)) + offset,
            values["rmse"],
            yerr=[values["rmse"] - values["rmse_ci_low"], values["rmse_ci_high"] - values["rmse"]],
            fmt="o",
            color=COLORS[model],
            ms=3.4,
            capsize=1.8,
            lw=0.9,
            label=MODEL_LABELS[model],
        )
    ax_a.set_xticks(range(len(splits)), split_labels)
    ax_a.set_ylabel("RMSE (lower is better)")
    ax_a.set_title("Predictive error across split stress tests", pad=5)
    ax_a.grid(axis="y", color="#D9D9D9", lw=0.5)
    ax_a.legend(frameon=False, ncol=2, loc="upper left", handletextpad=0.4, columnspacing=0.8)
    panel_label(ax_a, "a")

    for offset, model in zip(offsets, models):
        values = core[core["model"] == model].set_index("split").reindex(splits)
        ax_b.errorbar(
            np.arange(len(splits)) + offset,
            values["conditional_pcs"],
            yerr=[
                values["conditional_pcs"] - values["conditional_pcs_ci_low"],
                values["conditional_pcs_ci_high"] - values["conditional_pcs"],
            ],
            fmt="o",
            color=COLORS[model],
            ms=3.4,
            capsize=1.8,
            lw=0.9,
        )
    ax_b.set_xticks(range(len(splits)), split_labels)
    ax_b.set_ylabel("Conditional PCS (lower is less coupled)")
    ax_b.set_ylim(-0.02, 1.04)
    ax_b.set_title("Prior coupling after controlling for the label", pad=5)
    ax_b.grid(axis="y", color="#D9D9D9", lw=0.5)
    ax_b.text(0.99, 0.02, "* only the cell-prior component is available", transform=ax_b.transAxes, ha="right", va="bottom", fontsize=5.8, color="#555555")
    panel_label(ax_b, "b")

    markers = {"random": "o", "cold_pair": "s"}
    for _, row in cpb_stats.iterrows():
        model = row["model"]
        split = row["split"]
        ax_c.errorbar(
            row["delta_rmse_mean"],
            row["delta_conditional_pcs_mean"],
            xerr=row["delta_rmse_std"],
            yerr=row["delta_conditional_pcs_std"],
            fmt=markers[split],
            color=COLORS[model],
            mec="white",
            mew=0.5,
            ms=5.2,
            capsize=1.7,
            lw=0.8,
            alpha=0.95,
        )
    ax_c.axhline(0, color="#777777", lw=0.6, ls="--")
    ax_c.axvline(0, color="#777777", lw=0.6, ls="--")
    ax_c.set_xlabel(r"$\Delta$RMSE versus feature ridge")
    ax_c.set_ylabel(r"$\Delta$conditional PCS")
    ax_c.set_title("CPB controls reveal a non-specific random-split tradeoff", pad=5)
    ax_c.grid(color="#E3E3E3", lw=0.45)
    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=COLORS[m], markeredgecolor="white", label=MODEL_LABELS[m])
        for m in ["cpb_learned_anchor", "cpb_learned_no_anchor", "cpb_shuffled_assignment", "cpb_random_clusters"]
    ]
    handles += [
        plt.Line2D([], [], marker="o", color="#555555", linestyle="none", label="Row-random split"),
        plt.Line2D([], [], marker="s", color="#555555", linestyle="none", label="Cold pair"),
    ]
    ax_c.legend(handles=handles, frameon=False, ncol=2, loc="lower left", handletextpad=0.3, columnspacing=0.7)
    panel_label(ax_c, "c")

    scale_models = ["identity_ridge", "additive_prior", "feature_ridge", "cpb_learned_no_anchor"]
    for model in scale_models:
        values = scale[scale["model"] == model].sort_values("sample_size")
        label = MODEL_LABELS.get(model, model)
        ax_d.plot(values["sample_size"] / 1000.0, values["elapsed_sec"], marker="o", ms=3.4, lw=1.1, color=COLORS[model], label=label)
    ax_d.set_yscale("log")
    ax_d.set_xlabel("Sample size (thousands)")
    ax_d.set_ylabel("Fit + inference time (s, log scale)")
    ax_d.set_title("Full-table audit finishes in under 100 seconds", pad=5)
    ax_d.grid(axis="both", color="#D9D9D9", lw=0.5)
    ax_d.legend(frameon=False, ncol=2, loc="upper left", handletextpad=0.4, columnspacing=0.8)
    full_scale = scale[scale["sample_size"] == scale["sample_size"].max()].set_index("model")
    if "working_set_peak_mb" in full_scale.columns:
        ax_d.text(
            0.98,
            0.07,
            f"Peak working set: feature {full_scale.loc['feature_ridge', 'working_set_peak_mb'] / 1024:.2f} GB; "
            f"CPB {full_scale.loc['cpb_learned_no_anchor', 'working_set_peak_mb'] / 1024:.2f} GB",
            transform=ax_d.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            color="#555555",
        )
    panel_label(ax_d, "d")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    stem = FIGURES / "fig_major_revision_stress_tests"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"Wrote {stem}.pdf/.svg/.png/.tiff and source data")


if __name__ == "__main__":
    main()
