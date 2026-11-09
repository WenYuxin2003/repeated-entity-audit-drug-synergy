from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR


D1 = "drug1"
D2 = "drug2"
S1 = "smiles1"
S2 = "smiles2"
CELL = "cell"
EXPR = "cell_expression"
LABEL = "synergy"
PAIR = "pair_id"

MODEL_PRESETS = {
    "diagnostic": [
        "global_mean",
        "drug_prior",
        "pair_prior",
        "cell_prior",
        "additive_prior",
        "identity_ridge",
    ],
    "feature": [
        "global_mean",
        "drug_prior",
        "pair_prior",
        "cell_prior",
        "additive_prior",
        "identity_ridge",
        "feature_ridge",
    ],
    "cpb_light": [
        "feature_ridge",
        "cpb_cell_ridge",
        "cpb_pair_ridge",
        "cpb_synergy_hybrid",
        "cpb_synergy_pure",
    ],
    "baseline_light": [
        "global_mean",
        "additive_prior",
        "identity_ridge",
        "feature_ridge",
        "compact_ridge",
        "prior_feature_ridge",
        "mlp_regressor",
        "hist_gbdt",
        "extra_trees",
    ],
    "classic_ml": [
        "global_mean",
        "additive_prior",
        "identity_ridge",
        "feature_ridge",
        "elastic_net",
        "linear_svr",
        "knn_regressor",
        "random_forest",
        "gradient_boosting",
        "hist_gbdt",
        "extra_trees",
        "mlp_regressor",
    ],
    "full": [
        "global_mean",
        "drug_prior",
        "pair_prior",
        "cell_prior",
        "additive_prior",
        "identity_ridge",
        "feature_ridge",
        "compact_ridge",
        "prior_feature_ridge",
        "elastic_net",
        "linear_svr",
        "knn_regressor",
        "random_forest",
        "gradient_boosting",
        "mlp_regressor",
        "hist_gbdt",
        "extra_trees",
        "cpb_cell_ridge",
        "cpb_pair_ridge",
        "cpb_synergy_hybrid",
        "cpb_synergy_pure",
    ],
}

DEFAULT_BIAS_SCENARIOS = [
    "original",
    "random_drug_a",
    "random_drug_b",
    "random_both_drugs",
    "random_cell",
    "random_all",
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def finite_corr(x: Iterable[float], y: Iterable[float], method: str = "pearson") -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return float("nan")
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(x_arr, y_arr).correlation)
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": finite_corr(y_true, y_pred),
        "spearman": finite_corr(y_true, y_pred, method="spearman"),
    }


def topk_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    high_threshold: float,
    fracs: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    high = y_true >= high_threshold
    prevalence = float(np.mean(high)) if len(high) else float("nan")
    out: dict[str, float] = {
        "high_threshold": float(high_threshold),
        "high_prevalence": prevalence,
    }
    if len(y_true) == 0:
        return out
    order = np.argsort(-y_pred)
    total_high = int(np.sum(high))
    for frac in fracs:
        k = max(1, int(math.ceil(len(y_true) * frac)))
        selected = order[:k]
        hits = int(np.sum(high[selected]))
        precision = hits / k
        recall = hits / total_high if total_high else float("nan")
        enrichment = precision / prevalence if prevalence > 0 else float("nan")
        suffix = f"top{int(frac * 100)}pct"
        out[f"{suffix}_precision"] = float(precision)
        out[f"{suffix}_recall"] = float(recall)
        out[f"{suffix}_enrichment"] = float(enrichment)
        out[f"{suffix}_mean_true"] = float(np.mean(y_true[selected]))
    return out


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


def pick_column(df: pd.DataFrame, candidates: list[str], semantic: str, required: bool = True) -> str | None:
    lower_to_col = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_col:
            return lower_to_col[candidate.lower()]
    if required:
        raise ValueError(f"Could not find {semantic} column. Available columns: {list(df.columns)}")
    return None


def vector_like(value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, (list, tuple, np.ndarray, pd.Series)) and pd.isna(value):
        return False
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value, dtype=object)
        return arr.size > 1 or (arr.size == 1 and isinstance(arr.ravel()[0], (list, tuple, np.ndarray)))
    return False


def to_vector(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        if value.dtype == np.float32 and value.ndim == 1:
            return value
        if value.dtype != object:
            return value.astype(np.float32, copy=False).ravel()
        if value.dtype == object and value.size > 0 and isinstance(value.ravel()[0], (list, tuple, np.ndarray)):
            parts = [np.asarray(part, dtype=float).ravel() for part in value.ravel()]
            return np.concatenate(parts).astype(np.float32)
        return np.asarray(value, dtype=np.float32).ravel()
    if isinstance(value, (list, tuple, pd.Series)):
        if len(value) == 0:
            return None
        if isinstance(value[0], (list, tuple, np.ndarray, pd.Series)):
            parts = [np.asarray(part, dtype=float).ravel() for part in value]
            return np.concatenate(parts).astype(np.float32)
        return np.asarray(value, dtype=float).ravel().astype(np.float32)
    return None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def download_dataverse_file(file_id: int, out_path: Path) -> None:
    import requests

    url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading Dataverse file {file_id} -> {out_path}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with out_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def normalize_synergy_table(raw: pd.DataFrame, label_column: str | None = None) -> pd.DataFrame:
    d1_id_col = pick_column(raw, ["Drug1_ID", "drug1_id", "drug_a_id", "ID1"], "drug 1 id", required=False)
    d2_id_col = pick_column(raw, ["Drug2_ID", "drug2_id", "drug_b_id", "ID2"], "drug 2 id", required=False)
    s1_col = pick_column(raw, ["Drug1", "drug1", "SMILES1", "smiles1", "Drug_A", "X1"], "drug 1 SMILES")
    s2_col = pick_column(raw, ["Drug2", "drug2", "SMILES2", "smiles2", "Drug_B", "X2"], "drug 2 SMILES")
    cell_col = pick_column(
        raw,
        ["Cell_Line_ID", "cell_line_id", "CellLine_ID", "cell_id", "Cell_ID", "cell", "Cell"],
        "cell line id",
    )
    if label_column:
        label_col = pick_column(raw, [label_column], "requested label")
    else:
        label_col = pick_column(
            raw,
            [
                "Y",
                "y",
                "Synergy_ZIP",
                "ZIP",
                "zip",
                "Synergy_Bliss",
                "Synergy_Loewe",
                "Synergy_HSA",
                "CSS",
                "synergy",
                "Score",
                "score",
                "Label",
                "label",
            ],
            "label",
        )

    expr_col = None
    for candidate in ["CellLine", "cell_expression", "CellLineExpression", "Expression", "expression", "X3"]:
        col = pick_column(raw, [candidate], "cell expression", required=False)
        if col is None:
            continue
        sample = raw[col].dropna().head(20).to_list()
        if any(vector_like(value) for value in sample):
            expr_col = col
            break

    expr_map = None
    if expr_col:
        unique_expr = raw[[cell_col, expr_col]].drop_duplicates(subset=[cell_col], keep="first")
        expr_map = {
            str(cell): to_vector(value)
            for cell, value in zip(unique_expr[cell_col].to_numpy(), unique_expr[expr_col].to_numpy())
        }

    out = pd.DataFrame(
        {
            D1: raw[d1_id_col].astype(str) if d1_id_col else raw[s1_col].astype(str),
            D2: raw[d2_id_col].astype(str) if d2_id_col else raw[s2_col].astype(str),
            S1: raw[s1_col].astype(str),
            S2: raw[s2_col].astype(str),
            CELL: raw[cell_col].astype(str),
            LABEL: pd.to_numeric(raw[label_col], errors="coerce"),
        }
    )
    out[EXPR] = None
    out = out.dropna(subset=[D1, D2, S1, S2, CELL, LABEL]).reset_index(drop=True)

    swap = out[D1].astype(str) > out[D2].astype(str)
    for left, right in [(D1, D2), (S1, S2)]:
        tmp = out.loc[swap, left].copy()
        out.loc[swap, left] = out.loc[swap, right].to_numpy()
        out.loc[swap, right] = tmp.to_numpy()
    out[PAIR] = out[D1].astype(str) + "||" + out[D2].astype(str)

    grouped = (
        out.groupby([D1, D2, PAIR, CELL], as_index=False)
        .agg({S1: "first", S2: "first", LABEL: "median", EXPR: "first"})
        .reset_index(drop=True)
    )
    if expr_map:
        grouped[EXPR] = grouped[CELL].map(expr_map)
    return grouped


def make_synthetic_synergy(seed: int, n_samples: int = 2400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_drugs = 90
    n_cells = 32
    latent_dim = 24
    expr_dim = 96
    drugs = np.array([f"D{i:03d}" for i in range(n_drugs)])
    cells = np.array([f"CELL{i:03d}" for i in range(n_cells)])
    smiles = {drug: "C" * (3 + i % 8) + "N" * (1 + i % 5) + f"O{i % 7}" for i, drug in enumerate(drugs)}
    drug_latent = {drug: rng.normal(size=latent_dim) for drug in drugs}
    cell_latent = {cell: rng.normal(size=latent_dim) for cell in cells}
    cell_expr = {cell: rng.normal(size=expr_dim).astype(np.float32) for cell in cells}
    drug_prior = {drug: rng.normal(scale=0.8) for drug in drugs}
    cell_prior = {cell: rng.normal(scale=0.9) for cell in cells}

    rows = []
    for _ in range(n_samples):
        d1, d2 = rng.choice(drugs, size=2, replace=False)
        if d1 > d2:
            d1, d2 = d2, d1
        cell = str(rng.choice(cells))
        pair_vec = drug_latent[d1] + drug_latent[d2]
        interaction = 0.35 * float(np.dot(pair_vec, cell_latent[cell]) / latent_dim)
        pair_prior = 0.45 * float(np.dot(drug_latent[d1], drug_latent[d2]) / latent_dim)
        y = 5.0 + drug_prior[d1] + drug_prior[d2] + pair_prior + cell_prior[cell] + interaction + rng.normal(scale=0.55)
        rows.append(
            {
                D1: d1,
                D2: d2,
                S1: smiles[d1],
                S2: smiles[d2],
                CELL: cell,
                EXPR: cell_expr[cell],
                LABEL: y,
                PAIR: f"{d1}||{d2}",
            }
        )
    return pd.DataFrame(rows)


def load_synergy_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.synthetic_smoke:
        return make_synthetic_synergy(args.seed, n_samples=args.synthetic_samples)

    if args.data_path.exists():
        raw = read_table(args.data_path)
    elif args.download_tdc:
        download_dataverse_file(args.tdc_dataverse_id, args.data_path)
        raw = read_table(args.data_path)
    else:
        raise FileNotFoundError(
            f"{args.data_path} not found. Use --download-tdc or --synthetic-smoke for a local smoke test."
        )
    return normalize_synergy_table(raw, label_column=args.label_column)


def partition_values(values: np.ndarray, seed: int, frac: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> tuple[set, set, set]:
    rng = np.random.default_rng(seed)
    values = np.array(sorted(pd.unique(values)))
    rng.shuffle(values)
    n_train = int(len(values) * frac[0])
    n_val = int(len(values) * frac[1])
    return set(values[:n_train]), set(values[n_train : n_train + n_val]), set(values[n_train + n_val :])


def split_random(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(len(idx) * 0.8)
    n_val = int(len(idx) * 0.1)
    return {"train": idx[:n_train], "val": idx[n_train : n_train + n_val], "test": idx[n_train + n_val :]}


def split_by_column(df: pd.DataFrame, col: str, seed: int) -> dict[str, np.ndarray]:
    train_values, val_values, test_values = partition_values(df[col].to_numpy(), seed)
    values = df[col].to_numpy()
    return {
        "train": np.flatnonzero(np.isin(values, list(train_values))),
        "val": np.flatnonzero(np.isin(values, list(val_values))),
        "test": np.flatnonzero(np.isin(values, list(test_values))),
    }


def split_tdc_combination(df: pd.DataFrame, seed: int, frac: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    cell_pairs = [set(group[PAIR].to_numpy()) for _, group in df.groupby(CELL)]
    common_pairs = set.intersection(*cell_pairs) if cell_pairs else set()
    if len(common_pairs) < 10:
        return split_by_column(df, PAIR, seed)
    common_pairs_arr = np.array(sorted(common_pairs))
    rng.shuffle(common_pairs_arr)
    n_test_pairs = max(1, int(len(common_pairs_arr) * frac[2]))
    n_val_pairs = max(1, int(len(common_pairs_arr) * frac[1]))
    test_pairs = set(common_pairs_arr[:n_test_pairs])
    val_pairs = set(common_pairs_arr[n_test_pairs : n_test_pairs + n_val_pairs])
    pair_values = df[PAIR].to_numpy()
    test_mask = np.isin(pair_values, list(test_pairs))
    val_mask = (~test_mask) & np.isin(pair_values, list(val_pairs))
    train_mask = (~test_mask) & (~val_mask)
    return {
        "train": np.flatnonzero(train_mask),
        "val": np.flatnonzero(val_mask),
        "test": np.flatnonzero(test_mask),
    }


def split_cell_cluster(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    unique = df[[CELL, EXPR]].drop_duplicates(subset=[CELL], keep="first").reset_index(drop=True)
    expr_values = [to_vector(value) for value in unique[EXPR].to_numpy()]
    valid = [(cell, value) for cell, value in zip(unique[CELL].to_numpy(), expr_values) if value is not None]
    if len(valid) < 12:
        return split_by_column(df, CELL, seed)
    cells, vectors = zip(*valid)
    expr = np.vstack(vectors).astype(np.float32)
    expr = StandardScaler().fit_transform(expr)
    n_pca = int(min(8, expr.shape[0], expr.shape[1]))
    reduced = PCA(n_components=max(1, n_pca), svd_solver="randomized", random_state=seed).fit_transform(expr)
    n_clusters = int(min(6, max(3, len(cells) // 8)))
    labels = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=min(64, len(cells)), n_init=10).fit_predict(reduced)
    cluster_sizes = pd.Series(labels).value_counts().sort_values()
    test_cluster = int(cluster_sizes.index[-1])
    val_cluster = int(cluster_sizes.index[-2]) if len(cluster_sizes) > 1 else int(cluster_sizes.index[-1])
    cluster_map = {cell: int(label) for cell, label in zip(cells, labels)}
    clusters = df[CELL].map(cluster_map).fillna(-1).to_numpy(dtype=int)
    test_mask = clusters == test_cluster
    val_mask = clusters == val_cluster
    train_mask = (~test_mask) & (~val_mask)
    return {
        "train": np.flatnonzero(train_mask),
        "val": np.flatnonzero(val_mask),
        "test": np.flatnonzero(test_mask),
    }


def split_cold_drug(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    drugs = np.unique(np.concatenate([df[D1].to_numpy(), df[D2].to_numpy()]))
    train_drugs, val_drugs, test_drugs = partition_values(drugs, seed)
    d1 = df[D1].to_numpy()
    d2 = df[D2].to_numpy()
    test_mask = np.isin(d1, list(test_drugs)) | np.isin(d2, list(test_drugs))
    val_mask = (~test_mask) & (np.isin(d1, list(val_drugs)) | np.isin(d2, list(val_drugs)))
    train_mask = np.isin(d1, list(train_drugs)) & np.isin(d2, list(train_drugs))
    return {
        "train": np.flatnonzero(train_mask),
        "val": np.flatnonzero(val_mask),
        "test": np.flatnonzero(test_mask),
    }


def build_splits(df: pd.DataFrame, split_names: list[str], seed: int) -> list["Experiment"]:
    experiments = []
    for split in split_names:
        if split == "random":
            split_idx = split_random(df, seed)
        elif split == "cold_pair":
            split_idx = split_by_column(df, PAIR, seed)
        elif split == "tdc_combination":
            split_idx = split_tdc_combination(df, seed)
        elif split == "cold_cell":
            split_idx = split_by_column(df, CELL, seed)
        elif split == "cell_cluster":
            split_idx = split_cell_cluster(df, seed)
        elif split == "cold_drug":
            split_idx = split_cold_drug(df, seed)
        else:
            raise ValueError(split)
        if min(len(split_idx["train"]), len(split_idx["val"]), len(split_idx["test"])) == 0:
            log(f"Skipping split={split}; one partition is empty")
            continue
        experiments.append(Experiment(dataset="drugcomb" if not df.empty else "unknown", split=split, df=df, splits=split_idx))
    return experiments


def overlap_audit(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    train_drugs = set(train_df[D1]).union(set(train_df[D2]))
    test_drug_occ = np.concatenate([test_df[D1].to_numpy(), test_df[D2].to_numpy()])
    train_pairs = set(train_df[PAIR])
    train_cells = set(train_df[CELL])
    return {
        "seen_drug_ratio": float(np.mean(np.isin(test_drug_occ, list(train_drugs)))) if len(test_drug_occ) else float("nan"),
        "seen_pair_ratio": float(np.mean(test_df[PAIR].isin(train_pairs))) if len(test_df) else float("nan"),
        "seen_cell_ratio": float(np.mean(test_df[CELL].isin(train_cells))) if len(test_df) else float("nan"),
    }


class SynergyPrior:
    def __init__(self, smoothing: float = 10.0) -> None:
        self.smoothing = smoothing

    def fit(self, train_df: pd.DataFrame) -> "SynergyPrior":
        self.global_mean_ = float(train_df[LABEL].mean())
        self.drug_prior_ = self._fit_drug_tendency(train_df)
        self.pair_prior_ = self._shrunk_series(train_df, PAIR, LABEL, self.global_mean_)
        self.cell_prior_ = self._shrunk_series(train_df, CELL, LABEL, self.global_mean_)

        residual = train_df[LABEL].to_numpy(dtype=float) - self.global_mean_
        drug_rows = pd.DataFrame(
            {
                "drug": np.concatenate([train_df[D1].to_numpy(), train_df[D2].to_numpy()]),
                "_res": np.concatenate([residual / 2.0, residual / 2.0]),
            }
        )
        self.drug_effect_ = self._shrunk_series(drug_rows, "drug", "_res", 0.0)
        after_drug = residual - self._drug_effect_sum(train_df)
        pair_df = pd.DataFrame({PAIR: train_df[PAIR].to_numpy(), "_res": after_drug})
        self.pair_effect_ = self._shrunk_series(pair_df, PAIR, "_res", 0.0)
        after_pair = after_drug - train_df[PAIR].map(self.pair_effect_).fillna(0.0).to_numpy(dtype=float)
        cell_df = pd.DataFrame({CELL: train_df[CELL].to_numpy(), "_res": after_pair})
        self.cell_effect_ = self._shrunk_series(cell_df, CELL, "_res", 0.0)
        return self

    def _shrunk_series(self, df: pd.DataFrame, key: str, value: str, center: float) -> pd.Series:
        stats = df.groupby(key)[value].agg(["mean", "count"])
        weight = stats["count"] / (stats["count"] + self.smoothing)
        return center + weight * (stats["mean"] - center)

    def _fit_drug_tendency(self, train_df: pd.DataFrame) -> pd.Series:
        stacked = pd.DataFrame(
            {
                "drug": np.concatenate([train_df[D1].to_numpy(), train_df[D2].to_numpy()]),
                LABEL: np.concatenate([train_df[LABEL].to_numpy(dtype=float), train_df[LABEL].to_numpy(dtype=float)]),
            }
        )
        return self._shrunk_series(stacked, "drug", LABEL, self.global_mean_)

    def _drug_effect_sum(self, df: pd.DataFrame) -> np.ndarray:
        d1 = df[D1].map(self.drug_effect_).fillna(0.0).to_numpy(dtype=float)
        d2 = df[D2].map(self.drug_effect_).fillna(0.0).to_numpy(dtype=float)
        return d1 + d2

    def drug_tendency(self, df: pd.DataFrame) -> np.ndarray:
        d1 = df[D1].map(self.drug_prior_).fillna(self.global_mean_).to_numpy(dtype=float)
        d2 = df[D2].map(self.drug_prior_).fillna(self.global_mean_).to_numpy(dtype=float)
        return (d1 + d2) / 2.0

    def pair_tendency(self, df: pd.DataFrame) -> np.ndarray:
        return df[PAIR].map(self.pair_prior_).fillna(self.global_mean_).to_numpy(dtype=float)

    def cell_tendency(self, df: pd.DataFrame) -> np.ndarray:
        return df[CELL].map(self.cell_prior_).fillna(self.global_mean_).to_numpy(dtype=float)

    def predict(self, df: pd.DataFrame, mode: str) -> np.ndarray:
        if mode == "global_mean":
            return np.full(len(df), self.global_mean_, dtype=float)
        if mode == "drug_prior":
            return self.global_mean_ + self._drug_effect_sum(df)
        if mode == "pair_prior":
            return self.pair_tendency(df)
        if mode == "cell_prior":
            return self.cell_tendency(df)
        if mode == "additive_prior":
            pair = df[PAIR].map(self.pair_effect_).fillna(0.0).to_numpy(dtype=float)
            cell = df[CELL].map(self.cell_effect_).fillna(0.0).to_numpy(dtype=float)
            return self.global_mean_ + self._drug_effect_sum(df) + pair + cell
        raise ValueError(mode)


class PriorRegressor:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "PriorRegressor":
        self.prior_ = SynergyPrior().fit(train_df)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.prior_.predict(df, self.mode)


class IdentityRidgeRegressor:
    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "IdentityRidgeRegressor":
        self.encoder_ = one_hot_encoder()
        x_train = self.encoder_.fit_transform(train_df[[D1, D2, PAIR, CELL]])
        self.model_ = Ridge(alpha=self.alpha, solver="lsqr")
        self.model_.fit(x_train, train_df[LABEL].to_numpy(dtype=float))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.encoder_.transform(df[[D1, D2, PAIR, CELL]]))


class SynergyFeatureBuilder:
    def __init__(self, n_smiles_features: int = 2048, n_latent: int = 64, seed: int = 42) -> None:
        self.n_smiles_features = n_smiles_features
        self.n_latent = n_latent
        self.seed = seed

    def fit(self, train_df: pd.DataFrame) -> "SynergyFeatureBuilder":
        self.vectorizer_ = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            n_features=self.n_smiles_features,
            alternate_sign=False,
            norm="l2",
            lowercase=False,
        )
        rng = np.random.default_rng(self.seed)
        self.pair_projection_ = rng.normal(
            loc=0.0,
            scale=1.0 / math.sqrt(self.n_smiles_features),
            size=(self.n_smiles_features, self.n_latent),
        ).astype(np.float32)

        expression_sample = train_df[EXPR].head(200).to_numpy()
        self.uses_expression_ = any(vector_like(value) for value in expression_sample)
        if self.uses_expression_:
            unique = train_df[[CELL, EXPR]].drop_duplicates(subset=[CELL], keep="first")
            expr_values = [to_vector(value) for value in unique[EXPR].to_numpy()]
            valid = [(cell, value) for cell, value in zip(unique[CELL].to_numpy(), expr_values) if value is not None]
            if not valid:
                self.uses_expression_ = False
            else:
                cells, vectors = zip(*valid)
                expr = np.vstack(vectors).astype(np.float32)
                self.expr_width_ = expr.shape[1]
                self.expr_scaler_ = StandardScaler()
                scaled = self.expr_scaler_.fit_transform(expr)
                n_pca = int(min(self.n_latent, scaled.shape[0], scaled.shape[1]))
                self.expr_pca_n_ = max(1, n_pca)
                self.expr_pca_ = PCA(n_components=self.expr_pca_n_, svd_solver="randomized", random_state=self.seed)
                emb = self._pad(self.expr_pca_.fit_transform(scaled).astype(np.float32))
                self.cell_embedding_ = {cell: emb[i] for i, cell in enumerate(cells)}
                self.fallback_cell_embedding_ = np.mean(emb, axis=0)
        if not getattr(self, "uses_expression_", False):
            self.cell_encoder_ = one_hot_encoder()
            one_hot = self.cell_encoder_.fit_transform(train_df[[CELL]])
            self.cell_projection_ = rng.normal(
                loc=0.0,
                scale=1.0 / math.sqrt(max(1, one_hot.shape[1])),
                size=(one_hot.shape[1], self.n_latent),
            ).astype(np.float32)
            self.fallback_cell_embedding_ = np.zeros(self.n_latent, dtype=np.float32)
        return self

    def _pad(self, values: np.ndarray) -> np.ndarray:
        if values.shape[1] == self.n_latent:
            return values.astype(np.float32)
        out = np.zeros((values.shape[0], self.n_latent), dtype=np.float32)
        out[:, : values.shape[1]] = values
        return out

    def _drug_hash(self, smiles: pd.Series) -> sparse.csr_matrix:
        return self.vectorizer_.transform(smiles.fillna("").astype(str).to_numpy())

    def pair_hash_blocks(self, df: pd.DataFrame) -> sparse.csr_matrix:
        x1 = self._drug_hash(df[S1])
        x2 = self._drug_hash(df[S2])
        x_sum = x1 + x2
        x_diff = x1 - x2
        x_diff.data = np.abs(x_diff.data)
        x_prod = x1.multiply(x2)
        return sparse.hstack([x_sum, x_diff, x_prod], format="csr")

    def pair_latent(self, df: pd.DataFrame) -> np.ndarray:
        x1 = self._drug_hash(df[S1])
        x2 = self._drug_hash(df[S2])
        return np.asarray((x1 + x2) @ self.pair_projection_, dtype=np.float32)

    def _transform_expr_values(self, values: list[np.ndarray | None], cells: Iterable[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        transform_values: list[np.ndarray] = []
        transform_pos: list[int] = []
        for i, (cell, value) in enumerate(zip(cells, values)):
            if value is not None and len(value) == self.expr_width_:
                rows.append(np.zeros(self.n_latent, dtype=np.float32))
                transform_values.append(value)
                transform_pos.append(i)
            else:
                rows.append(self.cell_embedding_.get(cell, self.fallback_cell_embedding_))
        if transform_values:
            raw = np.vstack(transform_values).astype(np.float32)
            emb = self._pad(self.expr_pca_.transform(self.expr_scaler_.transform(raw)).astype(np.float32))
            for pos, value in zip(transform_pos, emb):
                rows[pos] = value
        return np.vstack(rows).astype(np.float32)

    def cell_dense(self, df: pd.DataFrame) -> np.ndarray:
        if self.uses_expression_:
            rows = [self.cell_embedding_.get(cell, self.fallback_cell_embedding_) for cell in df[CELL].to_numpy()]
            return np.vstack(rows).astype(np.float32)
        one_hot = self.cell_encoder_.transform(df[[CELL]])
        return np.asarray(one_hot @ self.cell_projection_, dtype=np.float32)

    def feature_matrix(self, df: pd.DataFrame) -> sparse.csr_matrix:
        pair_hash = self.pair_hash_blocks(df)
        pair_latent = self.pair_latent(df)
        cell = self.cell_dense(df)
        interaction = pair_latent * cell
        return sparse.hstack(
            [pair_hash, sparse.csr_matrix(cell), sparse.csr_matrix(interaction)],
            format="csr",
        )


class FeatureRidgeRegressor:
    def __init__(self, alpha: float = 10.0, n_smiles_features: int = 2048, n_latent: int = 64, seed: int = 42) -> None:
        self.alpha = alpha
        self.builder = SynergyFeatureBuilder(n_smiles_features=n_smiles_features, n_latent=n_latent, seed=seed)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "FeatureRidgeRegressor":
        self.builder.fit(train_df)
        self.model_ = Ridge(alpha=self.alpha, solver="lsqr")
        self.model_.fit(self.builder.feature_matrix(train_df), train_df[LABEL].to_numpy(dtype=float))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.builder.feature_matrix(df))


class CompactFeatureRegressor:
    def __init__(
        self,
        model_type: str,
        alpha: float = 10.0,
        n_smiles_features: int = 512,
        n_latent: int = 16,
        n_estimators: int = 80,
        max_tree_depth: int = 12,
        seed: int = 42,
    ) -> None:
        self.model_type = model_type
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.max_tree_depth = max_tree_depth
        self.seed = seed
        self.builder = SynergyFeatureBuilder(n_smiles_features=n_smiles_features, n_latent=n_latent, seed=seed)

    def _dense_features(self, df: pd.DataFrame) -> np.ndarray:
        pair = self.builder.pair_latent(df)
        cell = self.builder.cell_dense(df)
        interaction = pair * cell
        return np.hstack([pair, cell, interaction]).astype(np.float32)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "CompactFeatureRegressor":
        self.builder.fit(train_df)
        x_train = self._dense_features(train_df)
        y_train = train_df[LABEL].to_numpy(dtype=float)
        if self.model_type == "compact_ridge":
            self.scaler_ = StandardScaler()
            self.model_ = Ridge(alpha=self.alpha, solver="lsqr")
            self.model_.fit(self.scaler_.fit_transform(x_train), y_train)
        elif self.model_type == "elastic_net":
            self.scaler_ = StandardScaler()
            self.model_ = ElasticNet(alpha=0.002, l1_ratio=0.15, max_iter=4000, random_state=self.seed)
            self.model_.fit(self.scaler_.fit_transform(x_train), y_train)
        elif self.model_type == "linear_svr":
            self.scaler_ = StandardScaler()
            self.model_ = LinearSVR(C=1.0, epsilon=0.1, max_iter=4000, random_state=self.seed)
            self.model_.fit(self.scaler_.fit_transform(x_train), y_train)
        elif self.model_type == "knn_regressor":
            self.scaler_ = StandardScaler()
            self.model_ = KNeighborsRegressor(n_neighbors=25, weights="distance", metric="minkowski", p=2, n_jobs=1)
            self.model_.fit(self.scaler_.fit_transform(x_train), y_train)
        elif self.model_type == "random_forest":
            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_tree_depth,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=self.seed,
                n_jobs=1,
            )
            self.model_.fit(x_train, y_train)
        elif self.model_type == "gradient_boosting":
            self.model_ = GradientBoostingRegressor(
                n_estimators=min(self.n_estimators, 80),
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=10,
                subsample=0.85,
                random_state=self.seed,
            )
            self.model_.fit(x_train, y_train)
        elif self.model_type == "hist_gbdt":
            self.model_ = HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=self.seed,
            )
            self.model_.fit(x_train, y_train)
        elif self.model_type == "extra_trees":
            self.model_ = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_tree_depth,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=self.seed,
                n_jobs=1,
            )
            self.model_.fit(x_train, y_train)
        elif self.model_type == "mlp_regressor":
            self.scaler_ = StandardScaler()
            self.model_ = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size=256,
                learning_rate_init=0.001,
                max_iter=160,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=12,
                random_state=self.seed,
            )
            self.model_.fit(self.scaler_.fit_transform(x_train), y_train)
        else:
            raise ValueError(self.model_type)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = self._dense_features(df)
        if self.model_type in {"compact_ridge", "elastic_net", "linear_svr", "knn_regressor", "mlp_regressor"}:
            x = self.scaler_.transform(x)
        return self.model_.predict(x)


class PriorFeatureRidgeRegressor(FeatureRidgeRegressor):
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "PriorFeatureRidgeRegressor":
        self.builder.fit(train_df)
        self.prior_ = SynergyPrior().fit(train_df)
        self.model_ = Ridge(alpha=self.alpha, solver="lsqr")
        self.model_.fit(self._feature_matrix(train_df), train_df[LABEL].to_numpy(dtype=float))
        return self

    def _prior_block(self, df: pd.DataFrame) -> sparse.csr_matrix:
        values = np.vstack(
            [
                self.prior_.drug_tendency(df),
                self.prior_.pair_tendency(df),
                self.prior_.cell_tendency(df),
                self.prior_.predict(df, "additive_prior"),
            ]
        ).T.astype(np.float32)
        return sparse.csr_matrix(values)

    def _feature_matrix(self, df: pd.DataFrame) -> sparse.csr_matrix:
        return sparse.hstack([self.builder.feature_matrix(df), self._prior_block(df)], format="csr")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._feature_matrix(df))


class CPBSynergyRidgeRegressor(FeatureRidgeRegressor):
    def __init__(
        self,
        variant: str,
        alpha: float = 10.0,
        n_smiles_features: int = 2048,
        n_latent: int = 64,
        n_pair_confounders: int = 32,
        n_cell_confounders: int = 16,
        seed: int = 42,
    ) -> None:
        super().__init__(alpha=alpha, n_smiles_features=n_smiles_features, n_latent=n_latent, seed=seed)
        self.variant = variant
        self.n_pair_confounders = n_pair_confounders
        self.n_cell_confounders = n_cell_confounders
        self.seed = seed

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "CPBSynergyRidgeRegressor":
        self.builder.fit(train_df)
        unique_pairs = train_df.drop_duplicates(PAIR, keep="first").reset_index(drop=True)
        unique_cells = train_df.drop_duplicates(CELL, keep="first").reset_index(drop=True)
        pair_latent = self.builder.pair_latent(unique_pairs)
        cell_latent = self.builder.cell_dense(unique_cells)
        self._fit_pair_prototypes(pair_latent)
        self._fit_cell_prototypes(cell_latent)
        self.model_ = Ridge(alpha=self.alpha, solver="lsqr")
        self.model_.fit(self._cpb_features(train_df), train_df[LABEL].to_numpy(dtype=float))
        return self

    def _fit_pair_prototypes(self, pair_latent: np.ndarray) -> None:
        n_clusters = int(min(self.n_pair_confounders, max(1, len(pair_latent))))
        self.pair_kmeans_ = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.seed, batch_size=min(256, len(pair_latent)), n_init=10)
        labels = self.pair_kmeans_.fit_predict(pair_latent)
        self.pair_prototypes_ = self.pair_kmeans_.cluster_centers_.astype(np.float32)
        weights = np.bincount(labels, minlength=n_clusters).astype(np.float32)
        weights = weights / weights.sum()
        self.pair_prototype_mean_ = np.average(self.pair_prototypes_, axis=0, weights=weights).astype(np.float32)

    def _fit_cell_prototypes(self, cell_latent: np.ndarray) -> None:
        n_clusters = int(min(self.n_cell_confounders, max(1, len(cell_latent))))
        self.cell_kmeans_ = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.seed, batch_size=min(256, len(cell_latent)), n_init=10)
        labels = self.cell_kmeans_.fit_predict(cell_latent)
        self.cell_prototypes_ = self.cell_kmeans_.cluster_centers_.astype(np.float32)
        weights = np.bincount(labels, minlength=n_clusters).astype(np.float32)
        weights = weights / weights.sum()
        self.cell_prototype_mean_ = np.average(self.cell_prototypes_, axis=0, weights=weights).astype(np.float32)

    @staticmethod
    def _nearest(values: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        diff = values[:, None, :] - prototypes[None, :, :]
        return np.argmin(np.sum(diff * diff, axis=2), axis=1)

    def _cpb_features(self, df: pd.DataFrame) -> sparse.csr_matrix:
        pair_latent = self.builder.pair_latent(df)
        cell_latent = self.builder.cell_dense(df)
        pair_assigned = self.pair_prototypes_[self._nearest(pair_latent, self.pair_prototypes_)]
        cell_assigned = self.cell_prototypes_[self._nearest(cell_latent, self.cell_prototypes_)]
        pair_residual = pair_latent - pair_assigned
        cell_residual = cell_latent - cell_assigned
        residual_interaction = pair_residual * cell_residual
        pair_backdoor = np.broadcast_to(self.pair_prototype_mean_, pair_residual.shape) * cell_residual
        cell_backdoor = pair_residual * np.broadcast_to(self.cell_prototype_mean_, cell_residual.shape)

        blocks: list[sparse.spmatrix] = []
        if self.variant == "cpb_cell":
            blocks.append(self.builder.pair_hash_blocks(df))
            blocks.extend([sparse.csr_matrix(cell_residual), sparse.csr_matrix(pair_latent * cell_residual)])
        elif self.variant == "cpb_pair":
            blocks.extend([sparse.csr_matrix(pair_residual), sparse.csr_matrix(cell_latent), sparse.csr_matrix(pair_residual * cell_latent)])
        elif self.variant == "cpb_synergy_hybrid":
            blocks.append(self.builder.pair_hash_blocks(df))
            blocks.append(sparse.csr_matrix(cell_latent))
            blocks.extend(
                [
                    sparse.csr_matrix(pair_residual),
                    sparse.csr_matrix(cell_residual),
                    sparse.csr_matrix(residual_interaction),
                    sparse.csr_matrix(pair_backdoor),
                    sparse.csr_matrix(cell_backdoor),
                ]
            )
        elif self.variant == "cpb_synergy_pure":
            blocks.extend(
                [
                    sparse.csr_matrix(pair_residual),
                    sparse.csr_matrix(cell_residual),
                    sparse.csr_matrix(residual_interaction),
                    sparse.csr_matrix(pair_backdoor),
                    sparse.csr_matrix(cell_backdoor),
                ]
            )
        else:
            raise ValueError(self.variant)
        return sparse.hstack(blocks, format="csr")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._cpb_features(df))


def make_model(name: str, args: argparse.Namespace):
    if name in {"global_mean", "drug_prior", "pair_prior", "cell_prior", "additive_prior"}:
        return PriorRegressor(name)
    if name == "identity_ridge":
        return IdentityRidgeRegressor(alpha=args.ridge_alpha)
    if name == "feature_ridge":
        return FeatureRidgeRegressor(
            alpha=args.ridge_alpha,
            n_smiles_features=args.n_smiles_features,
            n_latent=args.n_latent,
            seed=args.seed,
        )
    if name in {
        "compact_ridge",
        "elastic_net",
        "linear_svr",
        "knn_regressor",
        "random_forest",
        "gradient_boosting",
        "hist_gbdt",
        "extra_trees",
        "mlp_regressor",
    }:
        return CompactFeatureRegressor(
            name,
            alpha=args.ridge_alpha,
            n_smiles_features=args.n_smiles_features,
            n_latent=args.n_latent,
            n_estimators=args.n_estimators,
            max_tree_depth=args.max_tree_depth,
            seed=args.seed,
        )
    if name == "prior_feature_ridge":
        return PriorFeatureRidgeRegressor(
            alpha=args.ridge_alpha,
            n_smiles_features=args.n_smiles_features,
            n_latent=args.n_latent,
            seed=args.seed,
        )
    if name == "cpb_cell_ridge":
        return CPBSynergyRidgeRegressor(
            "cpb_cell",
            alpha=args.ridge_alpha,
            n_smiles_features=args.n_smiles_features,
            n_latent=args.n_latent,
            n_pair_confounders=args.n_pair_confounders,
            n_cell_confounders=args.n_cell_confounders,
            seed=args.seed,
        )
    if name == "cpb_pair_ridge":
        return CPBSynergyRidgeRegressor(
            "cpb_pair",
            alpha=args.ridge_alpha,
            n_smiles_features=args.n_smiles_features,
            n_latent=args.n_latent,
            n_pair_confounders=args.n_pair_confounders,
            n_cell_confounders=args.n_cell_confounders,
            seed=args.seed,
        )
    if name == "cpb_synergy_hybrid":
        return CPBSynergyRidgeRegressor(
            "cpb_synergy_hybrid",
            alpha=args.ridge_alpha,
            n_smiles_features=args.n_smiles_features,
            n_latent=args.n_latent,
            n_pair_confounders=args.n_pair_confounders,
            n_cell_confounders=args.n_cell_confounders,
            seed=args.seed,
        )
    if name == "cpb_synergy_pure":
        return CPBSynergyRidgeRegressor(
            "cpb_synergy_pure",
            alpha=args.ridge_alpha,
            n_smiles_features=args.n_smiles_features,
            n_latent=args.n_latent,
            n_pair_confounders=args.n_pair_confounders,
            n_cell_confounders=args.n_cell_confounders,
            seed=args.seed,
        )
    raise ValueError(name)


def add_prior_coupling(metrics: dict[str, float], y_pred: np.ndarray, df: pd.DataFrame, prior: SynergyPrior) -> None:
    metrics["corr_pred_drug_prior"] = finite_corr(y_pred, prior.drug_tendency(df))
    metrics["corr_pred_pair_prior"] = finite_corr(y_pred, prior.pair_tendency(df))
    metrics["corr_pred_cell_prior"] = finite_corr(y_pred, prior.cell_tendency(df))
    values = [
        abs(metrics[key])
        for key in ["corr_pred_drug_prior", "corr_pred_pair_prior", "corr_pred_cell_prior"]
        if np.isfinite(metrics[key])
    ]
    metrics["pcs"] = float(np.mean(values)) if values else float("nan")


def perturb_df(df: pd.DataFrame, scenario: str, seed: int, expression_available: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy(deep=True)

    def permute(cols: list[str]) -> None:
        perm = rng.permutation(len(out))
        for col in cols:
            out[col] = out[col].to_numpy()[perm]

    if scenario == "original":
        return out
    if scenario == "random_drug_a":
        permute([S1])
    elif scenario == "random_drug_b":
        permute([S2])
    elif scenario == "random_both_drugs":
        permute([S1])
        permute([S2])
    elif scenario == "random_cell":
        if expression_available:
            permute([CELL, EXPR])
        else:
            permute([CELL])
    elif scenario == "random_all":
        permute([S1])
        permute([S2])
        if expression_available:
            permute([CELL, EXPR])
        else:
            permute([CELL])
    else:
        raise ValueError(scenario)
    return out


@dataclass
class Experiment:
    dataset: str
    split: str
    df: pd.DataFrame
    splits: dict[str, np.ndarray]


def evaluate_model(
    model,
    model_name: str,
    experiment: Experiment,
    seed: int,
    bias_scenarios: list[str],
    topk_quantile: float,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    df = experiment.df
    train_df = df.iloc[experiment.splits["train"]].reset_index(drop=True)
    val_df = df.iloc[experiment.splits["val"]].reset_index(drop=True)
    test_df = df.iloc[experiment.splits["test"]].reset_index(drop=True)
    prior = SynergyPrior().fit(train_df)
    audit = overlap_audit(train_df, test_df)
    expression_available = any(to_vector(value) is not None for value in train_df[EXPR].head(50).to_numpy())
    high_threshold = float(np.quantile(train_df[LABEL].to_numpy(dtype=float), topk_quantile))

    model.fit(train_df, val_df)
    rows: list[dict] = []
    bias_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    for partition, part_df in [("val", val_df), ("test", test_df)]:
        y_true = part_df[LABEL].to_numpy(dtype=float)
        y_pred = model.predict(part_df)
        metrics = regression_metrics(y_true, y_pred)
        metrics.update(topk_metrics(y_true, y_pred, high_threshold=high_threshold))
        add_prior_coupling(metrics, y_pred, part_df, prior)
        rows.append(
            {
                "dataset": experiment.dataset,
                "split": experiment.split,
                "model": model_name,
                "partition": partition,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "n_train_drugs": int(len(set(train_df[D1]).union(set(train_df[D2])))),
                "n_test_drugs": int(len(set(test_df[D1]).union(set(test_df[D2])))),
                "n_train_pairs": int(train_df[PAIR].nunique()),
                "n_test_pairs": int(test_df[PAIR].nunique()),
                "n_train_cells": int(train_df[CELL].nunique()),
                "n_test_cells": int(test_df[CELL].nunique()),
                **audit,
                **metrics,
            }
        )
        if partition == "test":
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "dataset": experiment.dataset,
                        "split": experiment.split,
                        "model": model_name,
                        "row_id": part_df.index.to_numpy(),
                        D1: part_df[D1].to_numpy(),
                        D2: part_df[D2].to_numpy(),
                        CELL: part_df[CELL].to_numpy(),
                        PAIR: part_df[PAIR].to_numpy(),
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "drug_prior": prior.drug_tendency(part_df),
                        "pair_prior": prior.pair_tendency(part_df),
                        "cell_prior": prior.cell_tendency(part_df),
                        "additive_prior": prior.predict(part_df, "additive_prior"),
                        "high_threshold": high_threshold,
                    }
                )
            )

    original_pred = model.predict(test_df)
    for scenario in bias_scenarios:
        perturbed = perturb_df(test_df, scenario, seed + 991, expression_available=expression_available)
        pred = model.predict(perturbed)
        metrics = regression_metrics(test_df[LABEL].to_numpy(dtype=float), pred)
        metrics.update(topk_metrics(test_df[LABEL].to_numpy(dtype=float), pred, high_threshold=high_threshold))
        add_prior_coupling(metrics, pred, test_df, prior)
        bias_rows.append(
            {
                "dataset": experiment.dataset,
                "split": experiment.split,
                "model": model_name,
                "scenario": scenario,
                "n": len(test_df),
                "mean_pred": float(np.mean(pred)),
                "std_pred": float(np.std(pred)),
                "mean_abs_delta_from_original": float(np.mean(np.abs(pred - original_pred))),
                "corr_with_original_pred": finite_corr(pred, original_pred),
                **metrics,
            }
        )
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return rows, bias_rows, predictions


def dataset_stats(df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "samples": int(len(df)),
        "drugs": int(len(set(df[D1]).union(set(df[D2])))),
        "pairs": int(df[PAIR].nunique()),
        "cells": int(df[CELL].nunique()),
        "mean_label": float(df[LABEL].mean()),
        "std_label": float(df[LABEL].std()),
        "has_expression": bool(any(to_vector(value) is not None for value in df[EXPR].head(100).to_numpy())),
    }


def downsample_df(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def write_markdown_report(out_dir: Path, metrics_df: pd.DataFrame, bias_df: pd.DataFrame, stats: dict, config: dict) -> None:
    test_df = metrics_df[metrics_df["partition"] == "test"].copy()
    lines = [
        "# CPB-Synergy Prior Shortcut Experiment Report",
        "",
        f"Generated: {now()}",
        "",
        "## Dataset",
        "",
        "```json",
        json.dumps(stats, indent=2),
        "```",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
        "## Entity Overlap Audit",
        "",
    ]
    audit_cols = ["split", "n_train", "n_test", "seen_drug_ratio", "seen_pair_ratio", "seen_cell_ratio"]
    lines.append(test_df[audit_cols].drop_duplicates().sort_values("split").to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Test Performance And Prior Coupling", ""])
    perf_cols = [
        "split",
        "model",
        "rmse",
        "mae",
        "pearson",
        "spearman",
        "corr_pred_drug_prior",
        "corr_pred_pair_prior",
        "corr_pred_cell_prior",
        "pcs",
        "top5pct_precision",
        "top5pct_enrichment",
        "top5pct_mean_true",
    ]
    lines.append(test_df[perf_cols].sort_values(["split", "rmse"]).to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Random-Input Shortcut Tests", ""])
    bias_cols = [
        "split",
        "model",
        "scenario",
        "rmse",
        "mean_abs_delta_from_original",
        "corr_with_original_pred",
        "corr_pred_drug_prior",
        "corr_pred_pair_prior",
        "corr_pred_cell_prior",
        "pcs",
    ]
    lines.append(bias_df[bias_cols].sort_values(["split", "model", "scenario"]).to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- High prior-only performance on `random` indicates repeated-entity shortcut risk.",
            "- `seen_pair_ratio` should be 0 for `cold_pair`; `seen_cell_ratio` should be 0 for `cold_cell`.",
            "- `pcs` is the mean absolute prediction-prior coupling over drug, pair, and cell priors.",
            "- In stress tests, high coupling after randomizing one input suggests the retained side still carries shortcut information.",
        ]
    )
    (out_dir / "synergy_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/tdc/drugcomb.pkl"))
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional raw label column to use, e.g. Synergy_ZIP, Synergy_Bliss, Synergy_Loewe, Synergy_HSA, or CSS.",
    )
    parser.add_argument("--tdc-root", type=Path, default=Path("external/TDC-main"))
    parser.add_argument("--tdc-dataverse-id", type=int, default=4215720)
    parser.add_argument("--download-tdc", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=2400)
    parser.add_argument("--out-dir", type=Path, default=Path("results/synergy_prior_bias"))
    parser.add_argument("--splits", nargs="+", default=["random", "cold_pair", "cold_cell", "cold_drug"])
    parser.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS),
        default="diagnostic",
        help="Resource-aware model set. Default 'diagnostic' avoids heavy feature/CPB models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Override --preset with an explicit model list.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Randomly cap samples after normalization. 0 means no cap.")
    parser.add_argument(
        "--bias-scenarios",
        nargs="+",
        choices=DEFAULT_BIAS_SCENARIOS,
        default=DEFAULT_BIAS_SCENARIOS,
        help="Random-input stress tests to run. Use fewer scenarios for quick CPB iteration.",
    )
    parser.add_argument(
        "--allow-full-heavy",
        action="store_true",
        help="Allow full-data CPB runs. Without this, CPB models are automatically capped on large datasets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--n-smiles-features", type=int, default=512)
    parser.add_argument("--n-latent", type=int, default=16)
    parser.add_argument("--n-pair-confounders", type=int, default=16)
    parser.add_argument("--n-cell-confounders", type=int, default=8)
    parser.add_argument("--topk-quantile", type=float, default=0.90, help="Training-label quantile used to define high-synergy positives.")
    parser.add_argument("--n-estimators", type=int, default=80, help="Number of trees for lightweight tree baselines.")
    parser.add_argument("--max-tree-depth", type=int, default=12, help="Maximum tree depth for lightweight tree baselines.")
    parser.add_argument("--save-predictions", action="store_true", help="Write per-sample test predictions for case studies and bootstrap CIs.")
    args = parser.parse_args()
    if args.models is None:
        args.models = MODEL_PRESETS[args.preset]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading synergy data")
    df = load_synergy_data(args)
    has_cpb_model = any(name.startswith("cpb_") for name in args.models)
    if has_cpb_model and len(df) > 60000 and args.max_samples <= 0 and not args.allow_full_heavy:
        args.max_samples = 60000
        log("CPB models requested on a large dataset; applying --max-samples 60000. Use --allow-full-heavy to override.")
    if args.max_samples > 0:
        before = len(df)
        df = downsample_df(df, args.max_samples, args.seed)
        log(f"Downsampled data from {before} to {len(df)} samples")
    stats = dataset_stats(df)
    config = vars(args).copy()
    config["data_path"] = str(args.data_path)
    config["tdc_root"] = str(args.tdc_root)
    config["out_dir"] = str(args.out_dir)
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"Loaded {stats['samples']} samples, {stats['pairs']} pairs, {stats['cells']} cells")

    experiments = build_splits(df, args.splits, args.seed)
    metric_rows: list[dict] = []
    bias_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    for experiment in experiments:
        log(f"Experiment split={experiment.split}")
        for model_name in args.models:
            log(f"Training/evaluating model={model_name}")
            start = time.time()
            model = make_model(model_name, args)
            rows, bias, predictions = evaluate_model(
                model, model_name, experiment, args.seed, args.bias_scenarios, args.topk_quantile
            )
            elapsed = time.time() - start
            for row in rows:
                row["elapsed_sec"] = elapsed
            for row in bias:
                row["elapsed_sec"] = elapsed
            metric_rows.extend(rows)
            bias_rows.extend(bias)
            if args.save_predictions and not predictions.empty:
                prediction_rows.append(predictions)
            pd.DataFrame(metric_rows).to_csv(args.out_dir / "metrics.csv", index=False)
            pd.DataFrame(bias_rows).to_csv(args.out_dir / "bias_tests.csv", index=False)
            if args.save_predictions and prediction_rows:
                pd.concat(prediction_rows, ignore_index=True).to_csv(args.out_dir / "predictions.csv", index=False)
            log(f"Finished {model_name} in {elapsed:.1f}s")

    metrics_df = pd.DataFrame(metric_rows)
    bias_df = pd.DataFrame(bias_rows)
    metrics_df.to_csv(args.out_dir / "metrics.csv", index=False)
    bias_df.to_csv(args.out_dir / "bias_tests.csv", index=False)
    if args.save_predictions and prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(args.out_dir / "predictions.csv", index=False)
    write_markdown_report(args.out_dir, metrics_df, bias_df, stats, config)
    log(f"Done. Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
