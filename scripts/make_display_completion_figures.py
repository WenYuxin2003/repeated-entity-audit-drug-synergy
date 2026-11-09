from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "synergy_manuscript"
SOURCE = FIGURES / "source_data"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_synergy_prior_experiments as base  # noqa: E402


SPLITS = [
    "random",
    "cold_pair",
    "cold_one_drug",
    "cold_both_drugs",
    "cold_cell",
    "molecular_cluster",
]

SPLIT_LABELS = {
    "random": "Random",
    "cold_pair": "Cold pair",
    "cold_one_drug": r"Cold $\geq$1 drug",
    "cold_both_drugs": "Cold both",
    "cold_cell": "Cold cell",
    "molecular_cluster": "SMILES hash",
}

MODELS = [
    "identity_ridge",
    "additive_prior",
    "ecfp_ridge",
    "feature_ridge",
    "descriptor_deep_mlp",
    "dual_view_mlp",
]

MODEL_LABELS = {
    "identity_ridge": "Identity ridge",
    "additive_prior": "Additive prior",
    "ecfp_ridge": "ECFP ridge",
    "feature_ridge": "Hash ridge",
    "descriptor_deep_mlp": "Descriptor MLP",
    "dual_view_mlp": "Dual-view MLP",
}

MODEL_COLORS = {
    "identity_ridge": "#2F4B7C",
    "additive_prior": "#008B8B",
    "ecfp_ridge": "#7A5195",
    "feature_ridge": "#666666",
    "descriptor_deep_mlp": "#B56576",
    "dual_view_mlp": "#D97706",
    "cpb_learned_no_anchor": "#2A9D8F",
}

MODEL_MARKERS = {
    "identity_ridge": "o",
    "additive_prior": "s",
    "ecfp_ridge": "v",
    "feature_ridge": "D",
    "descriptor_deep_mlp": "^",
    "dual_view_mlp": "X",
    "cpb_learned_no_anchor": "P",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(
    ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.07
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def annotate_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    fmt: str,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    suffix: str = "",
) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            red, green, blue, _ = cmap(norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "white" if luminance < 0.48 else "#222222"
            ax.text(
                col,
                row,
                f"{format(value, fmt)}{suffix}",
                ha="center",
                va="center",
                fontsize=5.8,
                color=color,
            )


def save_figure(fig: plt.Figure, stem: Path) -> None:
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


def make_six_split_figure() -> None:
    audit = pd.read_csv(RESULTS / "second_revision" / "full_data_split_audit.csv")
    metrics = pd.read_csv(RESULTS / "major_revision" / "core_metrics.csv")
    ecfp = pd.read_csv(RESULTS / "third_revision" / "ecfp_metrics.csv")
    ecfp = ecfp[ecfp["suite"] == "common30k"].copy()
    metrics = pd.concat([metrics, ecfp], ignore_index=True, sort=False)
    metrics = metrics[metrics["model"].isin(MODELS)].copy()
    metrics["conditional_pcs_fixed"] = (
        metrics["conditional_pcs"] * metrics["pcs_components_used"] / 3.0
    )

    overlap = (
        audit.groupby("split", as_index=False)[
            ["seen_drug_ratio", "seen_pair_ratio", "seen_cell_ratio"]
        ]
        .mean()
        .set_index("split")
        .reindex(SPLITS)
    )
    overlap.to_csv(SOURCE / "fig_six_split_overlap.csv")

    summary = (
        metrics.groupby(["model", "split"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            cpcs3_mean=("conditional_pcs_fixed", "mean"),
            cpcs3_std=("conditional_pcs_fixed", "std"),
        )
    )
    summary.to_csv(SOURCE / "fig_six_split_metrics.csv", index=False)

    size_summary = (
        audit.groupby("split", as_index=False)
        .agg(
            n_test_mean=("n_test", "mean"),
            n_test_std=("n_test", "std"),
            discarded_mean=("n_discarded_cross_group", "mean"),
            discarded_std=("n_discarded_cross_group", "std"),
            test_pairs_mean=("test_pairs", "mean"),
            test_pairs_std=("test_pairs", "std"),
        )
        .set_index("split")
        .reindex(SPLITS)
        .reset_index()
    )
    size_summary.to_csv(SOURCE / "fig_six_split_sizes.csv", index=False)

    rmse = (
        summary.pivot(index="model", columns="split", values="rmse_mean")
        .reindex(index=MODELS, columns=SPLITS)
    )
    cpcs = (
        summary.pivot(index="model", columns="split", values="cpcs3_mean")
        .reindex(index=MODELS, columns=SPLITS)
    )

    fig = plt.figure(figsize=(7.16, 4.65), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[0.92, 1.08])
    top = outer[0].subgridspec(1, 2, width_ratios=[0.82, 1.45])
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])

    overlap_values = overlap.to_numpy()
    cmap_a = mpl.colormaps["YlGnBu"]
    norm_a = Normalize(0, 1)
    im_a = ax_a.imshow(
        overlap_values,
        cmap=cmap_a,
        norm=norm_a,
        aspect="auto",
    )
    ax_a.set_xticks(range(3), ["Drug", "Pair", "Cell"])
    ax_a.set_yticks(range(len(SPLITS)), [SPLIT_LABELS[key] for key in SPLITS])
    ax_a.set_title("Train-test entity overlap")
    annotate_heatmap(ax_a, overlap_values, ".2f", cmap_a, norm_a)
    cbar_a = fig.colorbar(im_a, ax=ax_a, fraction=0.048, pad=0.03)
    cbar_a.set_label("Seen ratio")
    panel_label(ax_a, "a", x=-0.25)

    rmse_values = rmse.to_numpy()
    cmap_b = mpl.colormaps["Blues_r"]
    norm_b = Normalize(rmse_values.min() - 0.05, rmse_values.max() + 0.05)
    im_b = ax_b.imshow(
        rmse_values,
        cmap=cmap_b,
        norm=norm_b,
        aspect="auto",
    )
    ax_b.set_xticks(range(len(SPLITS)), [SPLIT_LABELS[key] for key in SPLITS], rotation=25, ha="right")
    ax_b.set_yticks(range(len(MODELS)), [MODEL_LABELS[key] for key in MODELS])
    ax_b.set_title("RMSE across six evaluation protocols")
    annotate_heatmap(ax_b, rmse_values, ".2f", cmap_b, norm_b)
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.035, pad=0.02)
    cbar_b.set_label("RMSE")
    panel_label(ax_b, "b", x=-0.14)

    cpcs_values = cpcs.to_numpy()
    cmap_c = mpl.colormaps["PuBuGn"]
    norm_c = Normalize(0, max(0.65, cpcs_values.max()))
    im_c = ax_c.imshow(
        cpcs_values,
        cmap=cmap_c,
        norm=norm_c,
        aspect="auto",
    )
    ax_c.set_xticks(range(len(SPLITS)), [SPLIT_LABELS[key] for key in SPLITS], rotation=25, ha="right")
    ax_c.set_yticks(range(len(MODELS)), [MODEL_LABELS[key] for key in MODELS])
    ax_c.set_title(r"Fixed-denominator conditional coupling, cPCS$^{(3)}$")
    annotate_heatmap(ax_c, cpcs_values, ".2f", cmap_c, norm_c)
    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.035, pad=0.02)
    cbar_c.set_label("cPCS(3)")
    panel_label(ax_c, "c", x=-0.14)

    for ax in [ax_a, ax_b, ax_c]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    stem = FIGURES / "fig_six_split_audit"
    save_figure(fig, stem)
    plt.close(fig)


def make_support_figure() -> None:
    raw = pd.read_pickle(ROOT / "data" / "raw" / "tdc" / "drugcomb.pkl")
    data = base.normalize_synergy_table(raw, label_column="Synergy_ZIP")

    drug_counts = pd.concat([data[base.D1], data[base.D2]], ignore_index=True).value_counts()
    pair_counts = data[base.PAIR].value_counts()
    cell_counts = data[base.CELL].value_counts()
    frequency = pd.concat(
        [
            pd.DataFrame({"entity": "Drug", "count": drug_counts.to_numpy()}),
            pd.DataFrame({"entity": "Pair", "count": pair_counts.to_numpy()}),
            pd.DataFrame({"entity": "Cell", "count": cell_counts.to_numpy()}),
        ],
        ignore_index=True,
    )
    frequency.to_csv(SOURCE / "fig_support_entity_frequency.csv", index=False)

    runtime = pd.read_csv(RESULTS / "major_revision" / "scale_metrics.csv")
    runtime.to_csv(SOURCE / "fig_support_runtime.csv", index=False)

    label_gap = pd.read_csv(
        RESULTS / "second_revision" / "symmetric_label_robustness.csv"
    )
    label_gap.to_csv(SOURCE / "fig_support_label_robustness.csv", index=False)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.58),
        gridspec_kw={"width_ratios": [1.08, 1.18, 0.98]},
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(
        w_pad=2.5 / 72,
        h_pad=2.5 / 72,
        wspace=0.10,
        hspace=0.02,
    )
    ax_a, ax_b, ax_c = axes

    frequency_colors = {"Drug": "#008B8B", "Pair": "#2F4B7C", "Cell": "#D97706"}
    frequency_markers = {"Drug": "o", "Pair": "s", "Cell": "D"}
    for entity in ["Drug", "Pair", "Cell"]:
        counts = np.sort(frequency.loc[frequency["entity"] == entity, "count"].to_numpy())
        ecdf = np.arange(1, len(counts) + 1) / len(counts)
        ax_a.plot(
            counts,
            ecdf,
            color=frequency_colors[entity],
            marker=frequency_markers[entity],
            markevery=max(1, len(counts) // 8),
            ms=2.8,
            lw=1.1,
            label=entity,
        )
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Occurrences per entity (log scale)")
    ax_a.set_ylabel("Empirical cumulative fraction")
    ax_a.set_title("Repeated-entity frequencies", pad=8)
    ax_a.grid(color="#E1E1E1", lw=0.45)
    ax_a.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0,
    )
    panel_label(ax_a, "a", x=-0.20, y=1.17)

    runtime_models = [
        "identity_ridge",
        "additive_prior",
        "feature_ridge",
        "cpb_learned_no_anchor",
    ]
    size_order = [25000, 50000, 100000, 287268]
    x = np.arange(len(size_order))
    for model in runtime_models:
        frame = runtime[runtime["model"] == model].set_index("sample_size").reindex(size_order)
        ax_b.plot(
            x,
            frame["elapsed_sec"],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            ms=4,
            lw=1.1,
            label={
                "identity_ridge": "Identity",
                "additive_prior": "Additive",
                "feature_ridge": "Hash",
                "cpb_learned_no_anchor": "CPB",
            }[model],
        )
    ax_b.set_yscale("log")
    ax_b.set_xticks(x, ["25k", "50k", "100k", "287k"])
    ax_b.set_xlabel("DrugComb rows")
    ax_b.set_ylabel("Fit + inference time (s, log scale)")
    ax_b.set_title("Single-threaded scaling", pad=8)
    ax_b.grid(color="#E1E1E1", lw=0.45)
    ax_b.legend(frameon=False, ncol=2, loc="upper left", handletextpad=0.3, columnspacing=0.6)
    panel_label(ax_b, "b", x=-0.20, y=1.17)

    label_order = ["ZIP", "Bliss", "Loewe", "HSA"]
    gap_split_order = ["cold_pair", "cold_cell", "cold_one_drug"]
    gap_labels = ["Cold pair", "Cold cell", r"Cold $\geq$1 drug"]
    gap = (
        label_gap.pivot(index="label", columns="split", values="rmse_increase_pct")
        .reindex(index=label_order, columns=gap_split_order)
    )
    values = gap.to_numpy()
    cmap_gap = mpl.colormaps["YlOrRd"]
    norm_gap = Normalize(0, max(25, values.max()))
    im = ax_c.imshow(
        values,
        cmap=cmap_gap,
        norm=norm_gap,
        aspect="auto",
    )
    ax_c.set_xticks(range(len(gap_split_order)), gap_labels, rotation=25, ha="right")
    ax_c.set_yticks(range(len(label_order)), label_order)
    ax_c.set_title("Label robustness", pad=8)
    annotate_heatmap(ax_c, values, ".1f", cmap_gap, norm_gap, suffix="%")
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.055, pad=0.03)
    cbar.set_label("Increase from random split (%)")
    panel_label(ax_c, "c", x=-0.24, y=1.17)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    stem = FIGURES / "fig_scaling_frequency_label_support"
    save_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    style()
    make_six_split_figure()
    make_support_figure()
    print("Wrote six-split and supporting audit figures with source data.")


if __name__ == "__main__":
    main()
