from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "second_revision"
FIGURES = ROOT / "figures" / "synergy_manuscript"

MODEL_LABELS = {
    "identity_ridge": "Identity ridge",
    "additive_prior": "Additive prior",
    "feature_ridge": "Feature ridge",
    "cpb_learned_no_anchor": "Learned CPB",
    "cpb_random_clusters": "Random-cluster CPB",
}

COLORS = {
    "identity_ridge": "#2F4B7C",
    "additive_prior": "#008B8B",
    "feature_ridge": "#666666",
    "cpb_learned_no_anchor": "#2A9D8F",
    "cpb_random_clusters": "#D97706",
    "learned_no_anchor": "#2A9D8F",
    "shuffled_assignment": "#B56576",
    "random_clusters": "#D97706",
}

CONTROL_LABELS = {
    "learned_no_anchor": "Learned prototypes",
    "shuffled_assignment": "Shuffled assignment",
    "random_clusters": "Random clusters",
}

MODEL_MARKERS = {
    "identity_ridge": "o",
    "additive_prior": "s",
    "feature_ridge": "D",
    "cpb_learned_no_anchor": "^",
    "cpb_random_clusters": "X",
}

K_MARKERS = {
    (8, 4): "o",
    (16, 8): "s",
    (32, 16): "^",
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
            "legend.fontsize": 6.0,
            "legend.title_fontsize": 5.8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def pareto_front(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["rmse", "conditional_pcs"])
    keep: list[int] = []
    best_cpcs = np.inf
    for index, row in ordered.iterrows():
        if row["conditional_pcs"] < best_cpcs:
            keep.append(index)
            best_cpcs = row["conditional_pcs"]
    return ordered.loc[keep].sort_values("rmse")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    source_dir = FIGURES / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    style()

    full = pd.read_csv(RESULTS / "full_critical_metrics.csv")
    expected = 3 * 3 * 5
    if len(full) < expected:
        raise RuntimeError(
            f"Full critical run is incomplete: found {len(full)} rows, expected {expected}."
        )

    models = list(MODEL_LABELS)
    splits = ["random", "cold_pair", "cold_both_drugs"]
    split_labels = ["Random", "Cold pair", "Cold both"]
    full_summary = (
        full[full["model"].isin(models)]
        .assign(
            conditional_pcs_fixed=lambda frame: (
                frame["conditional_pcs"] * frame["pcs_components_used"] / 3.0
            )
        )
        .groupby(["split", "model"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            conditional_pcs_mean=("conditional_pcs", "mean"),
            conditional_pcs_std=("conditional_pcs", "std"),
            conditional_pcs_fixed_mean=("conditional_pcs_fixed", "mean"),
            conditional_pcs_fixed_std=("conditional_pcs_fixed", "std"),
        )
    )
    full_summary.to_csv(source_dir / "fig_second_revision_full_summary.csv", index=False)

    pareto = pd.read_csv(RESULTS / "cpb_pareto_metrics.csv")
    pareto = pareto[pareto["split"] == "cold_pair"].copy()
    pareto_summary = (
        pareto.groupby(
            ["control", "k_pair", "k_cell", "strength"],
            as_index=False,
        )
        .agg(
            rmse=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            conditional_pcs=("conditional_pcs", "mean"),
            conditional_pcs_std=("conditional_pcs", "std"),
        )
    )
    pareto_summary.to_csv(source_dir / "fig_second_revision_cpb_pareto.csv", index=False)

    synthetic = pd.read_csv(RESULTS / "synthetic_sanity_summary.csv")
    synthetic.to_csv(source_dir / "fig_second_revision_synthetic.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.2), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    offsets = np.linspace(-0.27, 0.27, len(models))
    x = np.arange(len(splits))
    for offset, model in zip(offsets, models):
        values = (
            full_summary[full_summary["model"] == model]
            .set_index("split")
            .reindex(splits)
        )
        ax_a.errorbar(
            x + offset,
            values["rmse_mean"],
            yerr=values["rmse_std"],
            fmt=MODEL_MARKERS[model],
            ms=3.6,
            capsize=1.8,
            lw=0.9,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax_a.axvspan(1.62, 2.38, color="#ECECEC", zorder=-3)
    ax_a.set_xticks(x, split_labels)
    ax_a.set_ylabel("RMSE (lower is better)")
    ax_a.set_title("Full-table critical comparison (three seeds)", pad=5)
    ax_a.grid(axis="y", color="#D9D9D9", lw=0.5)
    ax_a.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        handletextpad=0.35,
        columnspacing=0.7,
    )
    ax_a.text(
        2,
        0.97,
        "unstable retained graph\n111 / 154 / 138 test pairs",
        transform=ax_a.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=5.7,
        color="#555555",
    )
    panel_label(ax_a, "a")

    for offset, model in zip(offsets, models):
        values = (
            full_summary[full_summary["model"] == model]
            .set_index("split")
            .reindex(splits)
        )
        ax_b.errorbar(
            x + offset,
            values["conditional_pcs_fixed_mean"],
            yerr=values["conditional_pcs_fixed_std"],
            fmt=MODEL_MARKERS[model],
            ms=3.6,
            capsize=1.8,
            lw=0.9,
            color=COLORS[model],
        )
    ax_b.set_xticks(x, split_labels)
    ax_b.set_ylim(-0.03, 0.72)
    ax_b.set_ylabel(r"Fixed-denominator cPCS(3)")
    ax_b.set_title("Conditional coupling on a common three-component scale", pad=5)
    ax_b.grid(axis="y", color="#D9D9D9", lw=0.5)
    panel_label(ax_b, "b")

    shared_rmse_min = max(
        group["rmse"].min() for _, group in pareto_summary.groupby("control")
    )
    shared_rmse_max = min(
        group["rmse"].max() for _, group in pareto_summary.groupby("control")
    )
    ax_c.axvspan(
        shared_rmse_min,
        shared_rmse_max,
        color="#E6E6E6",
        alpha=0.8,
        zorder=-4,
    )
    for control, group in pareto_summary.groupby("control"):
        for (k_pair, k_cell), k_group in group.groupby(["k_pair", "k_cell"]):
            in_shared = k_group["rmse"].between(shared_rmse_min, shared_rmse_max)
            sizes = 14 + 34 * k_group["strength"]
            marker = K_MARKERS[(int(k_pair), int(k_cell))]
            ax_c.scatter(
                k_group.loc[~in_shared, "rmse"],
                k_group.loc[~in_shared, "conditional_pcs"],
                s=sizes.loc[~in_shared],
                marker=marker,
                alpha=0.16,
                color=COLORS[control],
                edgecolor="none",
            )
            ax_c.scatter(
                k_group.loc[in_shared, "rmse"],
                k_group.loc[in_shared, "conditional_pcs"],
                s=sizes.loc[in_shared],
                marker=marker,
                alpha=0.72,
                color=COLORS[control],
                edgecolor="white",
                linewidth=0.35,
            )
        shared_group = group[group["rmse"].between(shared_rmse_min, shared_rmse_max)]
        front = pareto_front(shared_group)
        ax_c.plot(
            front["rmse"],
            front["conditional_pcs"],
            color=COLORS[control],
            lw=1.0,
            alpha=0.95,
        )
    ax_c.set_xlabel("RMSE")
    ax_c.set_ylabel("Conditional PCS")
    ax_c.set_title("Matched controls cover the learned frontier", pad=5)
    ax_c.grid(color="#E1E1E1", lw=0.45)

    control_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.5,
            markerfacecolor=COLORS[key],
            markeredgecolor="none",
            label=CONTROL_LABELS[key],
        )
        for key in CONTROL_LABELS
    ]
    control_legend = ax_c.legend(
        handles=control_handles,
        frameon=False,
        loc="upper right",
        handletextpad=0.3,
    )
    ax_c.add_artist(control_legend)

    k_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markersize=4.2,
            markerfacecolor="#666666",
            markeredgecolor="none",
            label=f"{k_pair}/{k_cell}",
        )
        for (k_pair, k_cell), marker in K_MARKERS.items()
    ]
    k_legend = ax_c.legend(
        handles=k_handles,
        title=r"$K_P/K_C$",
        frameon=False,
        ncol=3,
        loc="lower left",
        handletextpad=0.2,
        columnspacing=0.55,
        borderaxespad=0.25,
    )
    ax_c.add_artist(k_legend)

    strength_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=np.sqrt(14 + 34 * strength) / 1.4,
            markerfacecolor="#777777",
            markeredgecolor="white",
            label=f"{strength:.2f}",
        )
        for strength in (0.25, 0.50, 1.00)
    ]
    ax_c.legend(
        handles=strength_handles,
        title="Removal",
        frameon=False,
        ncol=3,
        loc="lower right",
        handletextpad=0.1,
        columnspacing=0.35,
        borderaxespad=0.25,
    )
    ax_c.text(
        (shared_rmse_min + shared_rmse_max) / 2,
        0.995,
        "shared RMSE range",
        transform=ax_c.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=5.7,
        color="#555555",
    )
    panel_label(ax_c, "c")

    case_order = [
        "A_label_only",
        "B_direct_prior_same_rmse",
        "C_nonlinear_label_prior",
    ]
    case_labels = [
        "A: label-only",
        "B: direct prior",
        "C: nonlinear label",
    ]
    metric_columns = [
        "conditional_pcs_mean",
        "incremental_prior_r2_given_label_mean",
        "pcs_excess_vs_moment_matched_null_mean",
    ]
    error_columns = [
        "conditional_pcs_std",
        "incremental_prior_r2_given_label_std",
        "pcs_excess_vs_moment_matched_null_std",
    ]
    metric_labels = ["cPCS", r"$\Delta R^2_P$", "PCS excess"]
    metric_colors = ["#2F4B7C", "#2A9D8F", "#D97706"]
    metric_markers = ["o", "s", "D"]
    synthetic = synthetic.set_index("case").reindex(case_order)
    point_x = np.arange(len(case_order))
    width = 0.22
    for index, (column, error_column, label, color, marker) in enumerate(
        zip(
            metric_columns,
            error_columns,
            metric_labels,
            metric_colors,
            metric_markers,
        )
    ):
        positions = point_x + (index - 1) * width
        values = synthetic[column].to_numpy()
        errors = synthetic[error_column].to_numpy()
        ax_d.errorbar(
            positions,
            values,
            yerr=errors,
            fmt=marker,
            ms=4.2,
            capsize=2.0,
            lw=0.85,
            color=color,
            label=label,
        )
        for xpos, value, error in zip(positions, values, errors):
            if abs(value) < 0.02:
                y_text = 0.025
                va = "bottom"
            elif value > 0.55:
                y_text = value - max(error, 0.008) - 0.018
                va = "top"
            else:
                y_text = value + max(error, 0.008) + 0.012
                va = "bottom"
            ax_d.text(
                xpos,
                y_text,
                f"{value:.3f}",
                ha="center",
                va=va,
                fontsize=5.2,
                color=color,
                rotation=90 if abs(value) < 0.02 else 0,
            )
    ax_d.axhline(0, color="#888888", lw=0.6)
    ax_d.set_xticks(point_x, case_labels)
    ax_d.set_ylim(-0.04, 0.72)
    ax_d.set_ylabel("Diagnostic value")
    ax_d.set_title("Synthetic sanity check (mean and SD, 20 seeds)", pad=5)
    ax_d.grid(axis="y", color="#D9D9D9", lw=0.5)
    ax_d.legend(
        frameon=False,
        ncol=3,
        loc="upper left",
        handletextpad=0.35,
        columnspacing=0.7,
    )
    panel_label(ax_d, "d")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    stem = FIGURES / "fig_second_revision_validation"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"Wrote {stem}.pdf/.svg/.png/.tiff and source data")


if __name__ == "__main__":
    main()
