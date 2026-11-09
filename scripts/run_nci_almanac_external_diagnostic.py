"""Lightweight external prior-shortcut diagnostic on NCI-ALMANAC.

This script deliberately avoids chemistry and expression dependencies. It reads
the CellMiner DTP ALMANAC combo-score workbook with Python stdlib XML parsing,
converts the wide NCI-60 matrix to a long table, and evaluates prior-only plus
identity baselines under random, cold-pair, and cold-cell splits.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "nci_almanac"
ZIP_PATH = DATA_DIR / "DTP_NCI60_ALMANAC_COMBO_SCORE.zip"
XLSX_PATH = DATA_DIR / "unzipped" / "output" / "DTP_NCI60_ALMANAC_COMBO_SCORE.xlsx"
PROCESSED_PATH = ROOT / "data" / "processed" / "nci_almanac_combo_score_long.csv"
DEFAULT_OUT = ROOT / "results" / "nci_almanac_external_diagnostic"

NCI_ALMANAC_URL = "https://discover.nci.nih.gov/cellminer/download/processeddataset/DTP_NCI60_ALMANAC_COMBO_SCORE.zip"

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def finite_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(x, y).correlation)
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": finite_corr(y_true, y_pred),
        "spearman": finite_corr(y_true, y_pred, method="spearman"),
    }


def topk_metrics(y_true: np.ndarray, y_pred: np.ndarray, high_threshold: float) -> dict[str, float]:
    high = y_true >= high_threshold
    prevalence = float(np.mean(high)) if len(high) else float("nan")
    out = {"high_threshold": float(high_threshold), "high_prevalence": prevalence}
    order = np.argsort(-y_pred)
    total_high = int(high.sum())
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(math.ceil(len(y_true) * frac)))
        selected = order[:k]
        hits = int(high[selected].sum())
        precision = hits / k
        suffix = f"top{int(frac * 100)}pct"
        out[f"{suffix}_precision"] = float(precision)
        out[f"{suffix}_recall"] = float(hits / total_high) if total_high else float("nan")
        out[f"{suffix}_enrichment"] = float(precision / prevalence) if prevalence > 0 else float("nan")
        out[f"{suffix}_mean_true"] = float(np.mean(y_true[selected]))
    return out


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


def ensure_nci_workbook() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        log(f"Downloading NCI-ALMANAC combo score workbook -> {ZIP_PATH}")
        urllib.request.urlretrieve(NCI_ALMANAC_URL, ZIP_PATH)
    if not XLSX_PATH.exists():
        log(f"Extracting {ZIP_PATH}")
        with ZipFile(ZIP_PATH) as zf:
            zf.extractall(DATA_DIR / "unzipped")


def col_index(cell_ref: str) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref).group(1)
    value = 0
    for char in letters:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def load_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values = []
    for si in root.findall("a:si", NS):
        parts = []
        for node in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"):
            parts.append(node.text or "")
        values.append("".join(parts))
    return values


def load_first_sheet_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as zf:
        shared = load_shared_strings(zf)
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.find("a:sheetData", NS).findall("a:row", NS):
            cells: dict[int, str] = {}
            max_idx = -1
            for cell in row.findall("a:c", NS):
                idx = col_index(cell.attrib["r"])
                max_idx = max(max_idx, idx)
                value_node = cell.find("a:v", NS)
                if value_node is None:
                    cells[idx] = ""
                    continue
                raw = value_node.text or ""
                if cell.attrib.get("t") == "s":
                    cells[idx] = shared[int(raw)]
                else:
                    cells[idx] = raw
            rows.append([cells.get(i, "") for i in range(max_idx + 1)])
    return rows


def workbook_metadata(rows: list[list[str]]) -> dict[str, str]:
    meta = {}
    for row in rows[:8]:
        if len(row) >= 2 and str(row[0]).strip():
            key = str(row[0]).strip().rstrip(":")
            if key.startswith("NSC #"):
                continue
            meta[key] = str(row[1]).strip()
    return meta


def build_long_table(force: bool = False) -> pd.DataFrame:
    if PROCESSED_PATH.exists() and not force:
        log(f"Loading cached long table: {PROCESSED_PATH}")
        return pd.read_csv(PROCESSED_PATH)

    ensure_nci_workbook()
    rows = load_first_sheet_rows(XLSX_PATH)
    meta = workbook_metadata(rows)
    header_idx = None
    for i, row in enumerate(rows):
        if len(row) > 9 and str(row[0]).startswith("NSC") and str(row[1]).strip() == "Drug name":
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not locate NCI-ALMANAC header row.")

    header = rows[header_idx]
    cell_columns = [(j, str(name).strip()) for j, name in enumerate(header[8:], start=8) if str(name).strip()]

    records = []
    for raw in rows[header_idx + 1 :]:
        if len(raw) < 8:
            continue
        drug1 = str(raw[1]).strip()
        drug2 = str(raw[5]).strip()
        if not drug1 or not drug2:
            continue
        nsc1 = str(raw[0]).strip()
        nsc2 = str(raw[4]).strip()
        d_low, d_high = sorted([drug1, drug2])
        n_low, n_high = sorted([nsc1, nsc2])
        pair_id = f"{d_low}||{d_high}"
        nsc_pair_id = f"{n_low}||{n_high}"
        for col_idx, cell in cell_columns:
            value = raw[col_idx] if col_idx < len(raw) else ""
            score = pd.to_numeric(value, errors="coerce")
            if pd.isna(score):
                continue
            records.append(
                {
                    "drug1": d_low,
                    "drug2": d_high,
                    "nsc1": n_low,
                    "nsc2": n_high,
                    "pair_id": pair_id,
                    "nsc_pair_id": nsc_pair_id,
                    "cell": cell,
                    "combo_score": float(score),
                }
            )
    df = pd.DataFrame.from_records(records)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    (PROCESSED_PATH.with_suffix(".manifest.json")).write_text(
        json.dumps(
            {
                "source_url": NCI_ALMANAC_URL,
                "zip_path": str(ZIP_PATH),
                "xlsx_path": str(XLSX_PATH),
                "metadata": meta,
                "n_rows": int(len(df)),
                "n_drugs": int(pd.unique(pd.concat([df["drug1"], df["drug2"]], ignore_index=True)).size),
                "n_pairs": int(df["pair_id"].nunique()),
                "n_cells": int(df["cell"].nunique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"Wrote long table: {PROCESSED_PATH} ({len(df):,} rows)")
    return df


def split_indices(df: pd.DataFrame, split: str, seed: int, test_frac: float = 0.10, val_frac: float = 0.10) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if split == "random":
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        n_val = int(round(len(idx) * val_frac))
        return {"test": idx[:n_test], "val": idx[n_test : n_test + n_val], "train": idx[n_test + n_val :]}

    key_col = {"cold_pair": "pair_id", "cold_cell": "cell"}[split]
    keys = np.array(sorted(df[key_col].unique()))
    rng.shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_frac)))
    n_val = max(1, int(round(len(keys) * val_frac)))
    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test : n_test + n_val])
    test = df.index[df[key_col].isin(test_keys)].to_numpy()
    val = df.index[df[key_col].isin(val_keys)].to_numpy()
    train = df.index[~df[key_col].isin(test_keys | val_keys)].to_numpy()
    return {"train": train, "val": val, "test": test}


def shrinkage_map(frame: pd.DataFrame, key_col: str, label_col: str, global_mean: float, lam: float) -> dict[str, float]:
    stats = frame.groupby(key_col)[label_col].agg(["mean", "count"])
    values = global_mean + stats["count"] / (stats["count"] + lam) * (stats["mean"] - global_mean)
    return values.to_dict()


def fit_priors(train: pd.DataFrame, lam: float) -> dict[str, object]:
    global_mean = float(train["combo_score"].mean())
    drug_rows = pd.concat(
        [
            train[["drug1", "combo_score"]].rename(columns={"drug1": "drug"}),
            train[["drug2", "combo_score"]].rename(columns={"drug2": "drug"}),
        ],
        ignore_index=True,
    )
    return {
        "global_mean": global_mean,
        "drug": shrinkage_map(drug_rows, "drug", "combo_score", global_mean, lam),
        "pair": shrinkage_map(train, "pair_id", "combo_score", global_mean, lam),
        "cell": shrinkage_map(train, "cell", "combo_score", global_mean, lam),
    }


def apply_priors(frame: pd.DataFrame, priors: dict[str, object]) -> pd.DataFrame:
    out = frame.copy()
    mu = float(priors["global_mean"])
    drug_map = priors["drug"]
    pair_map = priors["pair"]
    cell_map = priors["cell"]
    out["global_mean"] = mu
    out["drug_prior"] = [
        (float(drug_map.get(d1, mu)) + float(drug_map.get(d2, mu))) / 2.0
        for d1, d2 in zip(out["drug1"], out["drug2"])
    ]
    out["pair_prior"] = [float(pair_map.get(pair, mu)) for pair in out["pair_id"]]
    out["cell_prior"] = [float(cell_map.get(cell, mu)) for cell in out["cell"]]
    out["additive_prior"] = out["global_mean"] + (out["drug_prior"] - mu) + (out["pair_prior"] - mu) + (out["cell_prior"] - mu)
    return out


def identity_matrix(
    frame: pd.DataFrame,
    drug_encoder: OneHotEncoder,
    context_encoder: OneHotEncoder,
) -> sparse.csr_matrix:
    drug1 = drug_encoder.transform(frame[["drug1"]].rename(columns={"drug1": "drug"}))
    drug2 = drug_encoder.transform(frame[["drug2"]].rename(columns={"drug2": "drug"}))
    context = context_encoder.transform(frame[["pair_id", "cell"]])
    return sparse.hstack([drug1 + drug2, context], format="csr")


def fit_identity_ridge(
    train: pd.DataFrame,
    alpha: float,
) -> tuple[OneHotEncoder, OneHotEncoder, Ridge]:
    drug_encoder = one_hot_encoder()
    drug_values = pd.concat(
        [
            train[["drug1"]].rename(columns={"drug1": "drug"}),
            train[["drug2"]].rename(columns={"drug2": "drug"}),
        ],
        ignore_index=True,
    )
    drug_encoder.fit(drug_values)
    context_encoder = one_hot_encoder()
    context_encoder.fit(train[["pair_id", "cell"]])
    x_train = identity_matrix(train, drug_encoder, context_encoder)
    model = Ridge(alpha=alpha, random_state=0)
    model.fit(x_train, train["combo_score"].to_numpy(dtype=float))
    return drug_encoder, context_encoder, model


def prior_coupling(part: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    r_drug = finite_corr(pred, part["drug_prior"].to_numpy(dtype=float))
    r_pair = finite_corr(pred, part["pair_prior"].to_numpy(dtype=float))
    r_cell = finite_corr(pred, part["cell_prior"].to_numpy(dtype=float))
    vals = [abs(v) for v in (r_drug, r_pair, r_cell) if np.isfinite(v)]
    return {
        "corr_pred_drug_prior": r_drug,
        "corr_pred_pair_prior": r_pair,
        "corr_pred_cell_prior": r_cell,
        "pcs": float(np.mean(vals)) if vals else float("nan"),
    }


def evaluate_split(df: pd.DataFrame, split: str, seed: int, alpha: float, lam: float, save_predictions: bool) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    idx = split_indices(df, split=split, seed=seed)
    train = df.iloc[idx["train"]].reset_index(drop=True)
    val = df.iloc[idx["val"]].reset_index(drop=True)
    test = df.iloc[idx["test"]].reset_index(drop=True)
    priors = fit_priors(train, lam=lam)
    train_p = apply_priors(train, priors)
    val_p = apply_priors(val, priors)
    test_p = apply_priors(test, priors)
    drug_encoder, context_encoder, identity_model = fit_identity_ridge(train_p, alpha=alpha)
    high_threshold = float(np.quantile(train_p["combo_score"], 0.90))

    seen_drugs = set(train_p["drug1"]) | set(train_p["drug2"])
    seen_pairs = set(train_p["pair_id"])
    seen_cells = set(train_p["cell"])

    rows: list[dict[str, object]] = []
    pred_frames: list[pd.DataFrame] = []
    models = ["global_mean", "drug_prior", "pair_prior", "cell_prior", "additive_prior", "identity_ridge"]

    for partition, part in (("val", val_p), ("test", test_p)):
        y = part["combo_score"].to_numpy(dtype=float)
        for model_name in models:
            if model_name == "identity_ridge":
                x_part = identity_matrix(part, drug_encoder, context_encoder)
                pred = identity_model.predict(x_part)
            else:
                pred = part[model_name].to_numpy(dtype=float)
            metric = {
                "dataset": "nci_almanac",
                "split": split,
                "model": model_name,
                "partition": partition,
                "n_train": len(train_p),
                "n_val": len(val_p),
                "n_test": len(test_p),
                "n_train_drugs": len(seen_drugs),
                "n_test_drugs": len(set(test_p["drug1"]) | set(test_p["drug2"])),
                "n_train_pairs": len(seen_pairs),
                "n_test_pairs": test_p["pair_id"].nunique(),
                "n_train_cells": len(seen_cells),
                "n_test_cells": test_p["cell"].nunique(),
                "seen_drug_ratio": float(np.mean([(d1 in seen_drugs) and (d2 in seen_drugs) for d1, d2 in zip(test_p["drug1"], test_p["drug2"])])),
                "seen_pair_ratio": float(np.mean(test_p["pair_id"].isin(seen_pairs))),
                "seen_cell_ratio": float(np.mean(test_p["cell"].isin(seen_cells))),
            }
            metric.update(regression_metrics(y, pred))
            metric.update(topk_metrics(y, pred, high_threshold))
            metric.update(prior_coupling(part, pred))
            rows.append(metric)
            if save_predictions and partition == "test":
                pred_frames.append(
                    pd.DataFrame(
                        {
                            "dataset": "nci_almanac",
                            "split": split,
                            "model": model_name,
                            "row_id": np.arange(len(part)),
                            "drug1": part["drug1"].to_numpy(),
                            "drug2": part["drug2"].to_numpy(),
                            "cell": part["cell"].to_numpy(),
                            "pair_id": part["pair_id"].to_numpy(),
                            "y_true": y,
                            "y_pred": pred,
                            "drug_prior": part["drug_prior"].to_numpy(dtype=float),
                            "pair_prior": part["pair_prior"].to_numpy(dtype=float),
                            "cell_prior": part["cell_prior"].to_numpy(dtype=float),
                            "additive_prior": part["additive_prior"].to_numpy(dtype=float),
                            "high_threshold": high_threshold,
                        }
                    )
                )
    return rows, pred_frames


def write_report(metrics: pd.DataFrame, out_dir: Path, manifest: dict[str, object]) -> None:
    test = metrics[metrics["partition"] == "test"].copy()
    cols = [
        "split",
        "model",
        "n_test",
        "seen_drug_ratio",
        "seen_pair_ratio",
        "seen_cell_ratio",
        "rmse",
        "pearson",
        "pcs",
        "top5pct_precision",
        "top5pct_enrichment",
    ]
    lines = [
        "# NCI-ALMANAC External Prior Diagnostic",
        "",
        "## Source",
        "",
        f"- URL: {NCI_ALMANAC_URL}",
        f"- CellMiner metadata: {manifest.get('metadata', {})}",
        f"- Long table rows: {manifest['n_rows']:,}",
        f"- Drugs / pairs / cells: {manifest['n_drugs']} / {manifest['n_pairs']} / {manifest['n_cells']}",
        "",
        "## Test Metrics",
        "",
        test[cols].sort_values(["split", "rmse"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- This is an external dataset diagnostic, not a direct DrugComb-to-NCI transfer experiment.",
        "- It tests whether the same repeated-entity prior shortcut exists in NCI-ALMANAC combo scores.",
        "- Only prior and identity baselines are evaluated because the CellMiner workbook does not provide the same SMILES/expression feature table used by the DrugComb experiments.",
    ]
    (out_dir / "nci_almanac_external_diagnostic.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional resource cap after long-table conversion.")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--shrinkage", type=float, default=10.0)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_long_table(force=args.force_prepare)
    manifest_path = PROCESSED_PATH.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        log(f"Using resource-capped sample: {len(df):,} rows")
    else:
        df = df.reset_index(drop=True)

    all_rows = []
    all_pred = []
    for split in ("random", "cold_pair", "cold_cell"):
        log(f"Evaluating split={split}")
        rows, pred_frames = evaluate_split(
            df,
            split=split,
            seed=args.seed,
            alpha=args.alpha,
            lam=args.shrinkage,
            save_predictions=args.save_predictions,
        )
        all_rows.extend(rows)
        all_pred.extend(pred_frames)

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    if all_pred:
        pd.concat(all_pred, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    run_manifest = {
        **manifest,
        "script": str(Path(__file__).relative_to(ROOT)),
        "seed": args.seed,
        "max_samples": args.max_samples,
        "alpha": args.alpha,
        "shrinkage": args.shrinkage,
    }
    (out_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    write_report(metrics, out_dir, run_manifest)
    log(f"Wrote NCI-ALMANAC external diagnostic to {out_dir}")


if __name__ == "__main__":
    main()
