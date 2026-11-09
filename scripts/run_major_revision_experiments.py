from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import run_synergy_prior_experiments as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "major_revision"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def working_set_mb() -> float:
    if not hasattr(ctypes, "windll"):
        return float("nan")
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    get_info = ctypes.windll.kernel32.K32GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    ok = get_info(handle, ctypes.byref(counters), counters.cb)
    return float(counters.WorkingSetSize / 2**20) if ok else float("nan")


class MemorySampler:
    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.peak_mb = working_set_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            value = working_set_mb()
            if np.isfinite(value):
                self.peak_mb = max(self.peak_mb, value)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        value = working_set_mb()
        if np.isfinite(value):
            self.peak_mb = max(self.peak_mb, value)


def nested_stratified_sample(df: pd.DataFrame, n: int, seed: int = 2026) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.copy().reset_index(drop=True)
    work = df.copy()
    bins = pd.qcut(work[base.LABEL], q=10, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    work["_sample_key"] = rng.random(len(work))
    work["_sample_bin"] = bins.to_numpy()
    selected: list[pd.DataFrame] = []
    remaining = n
    groups = list(work.groupby("_sample_bin", sort=True))
    for i, (_, group) in enumerate(groups):
        if i == len(groups) - 1:
            take = remaining
        else:
            take = int(round(n * len(group) / len(work)))
            take = min(take, remaining)
        selected.append(group.nsmallest(take, "_sample_key"))
        remaining -= take
    out = pd.concat(selected, ignore_index=False).sort_index()
    return out.drop(columns=["_sample_key", "_sample_bin"]).reset_index(drop=True)


def split_cold_one_drug(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    return base.split_cold_drug(df, seed)


def split_cold_both_drugs(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    drugs = np.unique(np.concatenate([df[base.D1].to_numpy(), df[base.D2].to_numpy()]))
    train_drugs, val_drugs, test_drugs = base.partition_values(drugs, seed, frac=(0.70, 0.15, 0.15))
    d1 = df[base.D1].to_numpy()
    d2 = df[base.D2].to_numpy()
    return {
        "train": np.flatnonzero(np.isin(d1, list(train_drugs)) & np.isin(d2, list(train_drugs))),
        "val": np.flatnonzero(np.isin(d1, list(val_drugs)) & np.isin(d2, list(val_drugs))),
        "test": np.flatnonzero(np.isin(d1, list(test_drugs)) & np.isin(d2, list(test_drugs))),
    }


def _drug_cluster_map(df: pd.DataFrame, seed: int, n_clusters: int = 10) -> dict[str, int]:
    left = df[[base.D1, base.S1]].rename(columns={base.D1: "drug", base.S1: "smiles"})
    right = df[[base.D2, base.S2]].rename(columns={base.D2: "drug", base.S2: "smiles"})
    unique = pd.concat([left, right], ignore_index=True).drop_duplicates("drug").sort_values("drug")
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        n_features=1024,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
    )
    x = vectorizer.transform(unique["smiles"].fillna("").astype(str))
    n_components = max(2, min(24, x.shape[0] - 1, x.shape[1] - 1))
    z = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(x)
    labels = MiniBatchKMeans(
        n_clusters=min(n_clusters, len(unique)),
        random_state=seed,
        batch_size=min(64, len(unique)),
        n_init=20,
    ).fit_predict(z)
    return dict(zip(unique["drug"].astype(str), labels.astype(int)))


def split_molecular_cluster(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    mapping = _drug_cluster_map(df, seed)
    labels = np.array(sorted(set(mapping.values())))
    train_clusters, val_clusters, test_clusters = base.partition_values(labels, seed, frac=(0.70, 0.15, 0.15))
    c1 = df[base.D1].map(mapping).to_numpy()
    c2 = df[base.D2].map(mapping).to_numpy()
    test_mask = np.isin(c1, list(test_clusters)) | np.isin(c2, list(test_clusters))
    val_mask = (~test_mask) & (np.isin(c1, list(val_clusters)) | np.isin(c2, list(val_clusters)))
    train_mask = np.isin(c1, list(train_clusters)) & np.isin(c2, list(train_clusters))
    return {
        "train": np.flatnonzero(train_mask),
        "val": np.flatnonzero(val_mask),
        "test": np.flatnonzero(test_mask),
    }


def build_split(df: pd.DataFrame, name: str, seed: int) -> dict[str, np.ndarray]:
    if name == "random":
        return base.split_random(df, seed)
    if name == "cold_pair":
        return base.split_by_column(df, base.PAIR, seed)
    if name == "cold_cell":
        return base.split_by_column(df, base.CELL, seed)
    if name == "cold_one_drug":
        return split_cold_one_drug(df, seed)
    if name == "cold_both_drugs":
        return split_cold_both_drugs(df, seed)
    if name == "molecular_cluster":
        return split_molecular_cluster(df, seed)
    raise ValueError(name)


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    x = np.asarray(covariates, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    out = np.full(len(y), np.nan, dtype=float)
    if mask.sum() < x.shape[1] + 3:
        return out
    design = np.column_stack([np.ones(mask.sum()), x[mask]])
    coef, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
    out[mask] = y[mask] - design @ coef
    return out


def linear_r2(outcome: np.ndarray, predictors: np.ndarray) -> float:
    y = np.asarray(outcome, dtype=float)
    x = np.asarray(predictors, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if mask.sum() < x.shape[1] + 3 or np.std(y[mask]) < 1e-12:
        return float("nan")
    design = np.column_stack([np.ones(mask.sum()), x[mask]])
    coef, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
    return float(r2_score(y[mask], design @ coef))


def coupling_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    priors: dict[str, np.ndarray],
    additive_prior: np.ndarray,
    seed: int,
    n_null: int = 64,
) -> dict[str, float]:
    out: dict[str, float] = {}
    available: list[np.ndarray] = []
    raw_corrs: list[float] = []
    partial_corrs: list[float] = []
    label_corrs: list[float] = []
    pred_resid_y = residualize(y_pred, y_true)
    for name, values in priors.items():
        raw = base.finite_corr(y_pred, values)
        label = base.finite_corr(y_true, values)
        prior_resid_y = residualize(values, y_true)
        partial = base.finite_corr(pred_resid_y, prior_resid_y)
        out[f"corr_pred_{name}_prior"] = raw
        out[f"corr_label_{name}_prior"] = label
        out[f"partial_corr_pred_{name}_prior_given_y"] = partial
        if np.isfinite(raw):
            available.append(values)
            raw_corrs.append(abs(raw))
        if np.isfinite(partial):
            partial_corrs.append(abs(partial))
        if np.isfinite(label):
            label_corrs.append(abs(label))
    out["pcs"] = float(np.mean(raw_corrs)) if raw_corrs else float("nan")
    out["pcs_fixed_denominator"] = float(np.sum(raw_corrs) / 3.0)
    out["pcs_components_used"] = int(len(raw_corrs))
    out["conditional_pcs"] = float(np.mean(partial_corrs)) if partial_corrs else float("nan")
    out["label_prior_alignment"] = float(np.mean(label_corrs)) if label_corrs else float("nan")

    prior_matrix = np.column_stack(available) if available else np.empty((len(y_true), 0))
    base_r2 = linear_r2(y_pred, y_true)
    full_r2 = linear_r2(y_pred, np.column_stack([y_true, prior_matrix])) if available else base_r2
    out["pred_r2_from_label"] = base_r2
    out["pred_r2_from_label_and_priors"] = full_r2
    out["incremental_prior_r2_given_label"] = full_r2 - base_r2 if np.isfinite(base_r2) and np.isfinite(full_r2) else float("nan")

    target_residual = y_true - additive_prior
    pred_residual = y_pred - additive_prior
    out["residual_target_rmse"] = float(np.sqrt(mean_squared_error(target_residual, pred_residual)))
    out["residual_target_pearson"] = base.finite_corr(target_residual, pred_residual)

    if available and np.isfinite(out["pcs"]) and n_null > 0:
        rng = np.random.default_rng(seed)
        null_pcs = []
        y_centered = y_true - np.mean(y_true)
        y_scale = float(np.std(y_centered))
        pred_mean = float(np.mean(y_pred))
        pred_scale = float(np.std(y_pred))
        pred_label_corr = base.finite_corr(y_true, y_pred)
        y_unit = y_centered / max(y_scale, 1e-12)
        for _ in range(n_null):
            noise = rng.normal(size=len(y_true))
            noise -= np.mean(noise)
            noise -= np.dot(noise, y_unit) / max(np.dot(y_unit, y_unit), 1e-12) * y_unit
            noise /= max(float(np.std(noise)), 1e-12)
            rho = float(np.clip(pred_label_corr, -0.999999, 0.999999))
            null_pred = pred_mean + pred_scale * (rho * y_unit + math.sqrt(1.0 - rho**2) * noise)
            corr = [abs(base.finite_corr(null_pred, values)) for values in available]
            null_pcs.append(float(np.nanmean(corr)))
        out["moment_matched_null_pcs_mean"] = float(np.nanmean(null_pcs))
        out["moment_matched_null_pcs_low"] = float(np.nanpercentile(null_pcs, 2.5))
        out["moment_matched_null_pcs_high"] = float(np.nanpercentile(null_pcs, 97.5))
        out["pcs_excess_vs_moment_matched_null"] = out["pcs"] - out["moment_matched_null_pcs_mean"]
        # Backward-compatible aliases for earlier result readers.
        out["accuracy_matched_null_pcs_mean"] = out["moment_matched_null_pcs_mean"]
        out["accuracy_matched_null_pcs_low"] = out["moment_matched_null_pcs_low"]
        out["accuracy_matched_null_pcs_high"] = out["moment_matched_null_pcs_high"]
        out["pcs_excess_vs_accuracy_matched_null"] = out["pcs_excess_vs_moment_matched_null"]
    else:
        for key in [
            "moment_matched_null_pcs_mean",
            "moment_matched_null_pcs_low",
            "moment_matched_null_pcs_high",
            "pcs_excess_vs_moment_matched_null",
            "accuracy_matched_null_pcs_mean",
            "accuracy_matched_null_pcs_low",
            "accuracy_matched_null_pcs_high",
            "pcs_excess_vs_accuracy_matched_null",
        ]:
            out[key] = float("nan")
    return out


class SymmetricMLPRegressor:
    def __init__(self, mode: str, seed: int, n_smiles_features: int = 512, n_latent: int = 24) -> None:
        self.mode = mode
        self.seed = seed
        self.builder = base.SynergyFeatureBuilder(n_smiles_features=n_smiles_features, n_latent=n_latent, seed=seed)

    def _drug_latent(self, smiles: pd.Series) -> np.ndarray:
        return np.asarray(self.builder._drug_hash(smiles) @ self.builder.pair_projection_, dtype=np.float32)

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        d1 = self._drug_latent(df[base.S1])
        d2 = self._drug_latent(df[base.S2])
        cell = self.builder.cell_dense(df)
        if self.mode == "descriptor_deep_mlp":
            pair_sum = d1 + d2
            pair_diff = np.abs(d1 - d2)
            blocks = [pair_sum, pair_diff, d1 * d2, cell, pair_sum * cell, pair_diff * cell]
        elif self.mode == "dual_view_mlp":
            view1 = np.hstack([d1, cell, d1 * cell])
            view2 = np.hstack([d2, cell, d2 * cell])
            blocks = [view1 + view2, np.abs(view1 - view2), view1 * view2]
        else:
            raise ValueError(self.mode)
        return np.hstack(blocks).astype(np.float32)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "SymmetricMLPRegressor":
        self.builder.fit(train_df)
        x_train = self._features(train_df)
        x_val = self._features(val_df)
        y_train = train_df[base.LABEL].to_numpy(dtype=float)
        y_val = val_df[base.LABEL].to_numpy(dtype=float)
        self.scaler_ = StandardScaler()
        x_train = self.scaler_.fit_transform(x_train)
        x_val = self.scaler_.transform(x_val)
        candidates = [
            ((128, 64), 1e-3),
            ((128, 64, 32), 1e-4),
        ]
        best_rmse = float("inf")
        best_model = None
        best_config = None
        for i, (hidden, alpha) in enumerate(candidates):
            model = MLPRegressor(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                alpha=alpha,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=120,
                early_stopping=True,
                validation_fraction=0.10,
                n_iter_no_change=10,
                random_state=self.seed + i,
            )
            model.fit(x_train, y_train)
            rmse = float(np.sqrt(mean_squared_error(y_val, model.predict(x_val))))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_config = {"hidden": list(hidden), "alpha": alpha, "validation_rmse": rmse}
        self.model_ = best_model
        self.selected_config_ = best_config
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.scaler_.transform(self._features(df)))


class SymmetricIdentityRidgeRegressor:
    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "SymmetricIdentityRidgeRegressor":
        self.drug_encoder_ = base.one_hot_encoder()
        drug_values = pd.concat([train_df[base.D1], train_df[base.D2]], ignore_index=True).to_frame("drug")
        self.drug_encoder_.fit(drug_values)
        self.pair_encoder_ = base.one_hot_encoder()
        self.cell_encoder_ = base.one_hot_encoder()
        self.pair_encoder_.fit(train_df[[base.PAIR]])
        self.cell_encoder_.fit(train_df[[base.CELL]])
        self.model_ = Ridge(alpha=self.alpha, solver="lsqr")
        self.model_.fit(self._features(train_df), train_df[base.LABEL].to_numpy(dtype=float))
        return self

    def _features(self, df: pd.DataFrame) -> sparse.csr_matrix:
        d1 = self.drug_encoder_.transform(df[[base.D1]].rename(columns={base.D1: "drug"}))
        d2 = self.drug_encoder_.transform(df[[base.D2]].rename(columns={base.D2: "drug"}))
        pair = self.pair_encoder_.transform(df[[base.PAIR]])
        cell = self.cell_encoder_.transform(df[[base.CELL]])
        return sparse.hstack([d1 + d2, pair, cell], format="csr")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._features(df))


def _nearest(values: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    distances = np.sum(values * values, axis=1, keepdims=True) - 2.0 * values @ prototypes.T
    distances += np.sum(prototypes * prototypes, axis=1)[None, :]
    return np.argmin(distances, axis=1)


def _stable_random_indices(keys: Iterable[str], n: int, seed: int) -> np.ndarray:
    series = pd.Series([f"{seed}|{key}" for key in keys], dtype="string")
    hashes = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype=np.uint64)
    return (hashes % np.uint64(max(1, n))).astype(int)


def _random_partition_centroids(values: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.arange(len(values)) % n_clusters
    rng.shuffle(labels)
    return np.vstack([values[labels == k].mean(axis=0) for k in range(n_clusters)]).astype(np.float32)


class ControlledResidualRidge:
    def __init__(self, control: str, seed: int, n_smiles_features: int = 512, n_latent: int = 16, k_pair: int = 16, k_cell: int = 8) -> None:
        self.control = control
        self.seed = seed
        self.k_pair = k_pair
        self.k_cell = k_cell
        self.builder = base.SynergyFeatureBuilder(n_smiles_features=n_smiles_features, n_latent=n_latent, seed=seed)

    def _fit_channel(self, values: np.ndarray, k: int, seed: int) -> dict[str, object]:
        n_clusters = min(k, len(values))
        learned = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=min(256, len(values)),
            n_init=10,
        ).fit(values)
        learned_centers = learned.cluster_centers_.astype(np.float32)
        learned_assigned = learned_centers[learned.predict(values)]
        target_energy = float(np.mean(np.sum(learned_assigned**2, axis=1)))
        random_centers = _random_partition_centroids(values, n_clusters, seed + 103)
        random_assigned = random_centers[_nearest(values, random_centers)]
        random_energy = float(np.mean(np.sum(random_assigned**2, axis=1)))
        random_scale = math.sqrt(target_energy / max(random_energy, 1e-12))
        weights = np.bincount(learned.labels_, minlength=n_clusters).astype(float)
        anchor = np.average(learned_centers, axis=0, weights=weights).astype(np.float32)
        return {
            "learned": learned_centers,
            "random": random_centers,
            "random_scale": random_scale,
            "target_energy": target_energy,
            "anchor": anchor,
        }

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "ControlledResidualRidge":
        self.builder.fit(train_df)
        pairs = train_df.drop_duplicates(base.PAIR, keep="first")
        cells = train_df.drop_duplicates(base.CELL, keep="first")
        self.pair_state_ = self._fit_channel(self.builder.pair_latent(pairs), self.k_pair, self.seed)
        self.cell_state_ = self._fit_channel(self.builder.cell_dense(cells), self.k_cell, self.seed + 1)
        self.model_ = Ridge(alpha=10.0, solver="lsqr")
        self.model_.fit(self._features(train_df), train_df[base.LABEL].to_numpy(dtype=float))
        return self

    def _assigned(self, values: np.ndarray, keys: Iterable[str], state: dict[str, object], seed: int) -> np.ndarray:
        learned = np.asarray(state["learned"])
        if self.control in {"learned_anchor", "learned_no_anchor"}:
            return learned[_nearest(values, learned)]
        if self.control == "shuffled_assignment":
            idx = _stable_random_indices(keys, len(learned), seed)
            assigned = learned[idx]
            target = float(state["target_energy"])
            actual = float(np.mean(np.sum(assigned**2, axis=1)))
            return assigned * math.sqrt(target / max(actual, 1e-12))
        if self.control == "random_clusters":
            centers = np.asarray(state["random"])
            return centers[_nearest(values, centers)] * float(state["random_scale"])
        raise ValueError(self.control)

    def _features(self, df: pd.DataFrame) -> sparse.csr_matrix:
        pair = self.builder.pair_latent(df)
        cell = self.builder.cell_dense(df)
        pair_center = self._assigned(pair, df[base.PAIR].astype(str), self.pair_state_, self.seed + 201)
        cell_center = self._assigned(cell, df[base.CELL].astype(str), self.cell_state_, self.seed + 301)
        rp = pair - pair_center
        rc = cell - cell_center
        blocks = [rp, rc, rp * rc]
        if self.control == "learned_anchor":
            blocks.extend([
                np.broadcast_to(np.asarray(self.pair_state_["anchor"]), rc.shape) * rc,
                rp * np.broadcast_to(np.asarray(self.cell_state_["anchor"]), rp.shape),
            ])
        return sparse.csr_matrix(np.hstack(blocks).astype(np.float32))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._features(df))

    def diagnostics(self) -> dict[str, float]:
        return {
            "pair_removed_energy_target": float(self.pair_state_["target_energy"]),
            "cell_removed_energy_target": float(self.cell_state_["target_energy"]),
        }


def make_model(name: str, seed: int, n_smiles_features: int, n_latent: int):
    if name == "identity_ridge":
        return SymmetricIdentityRidgeRegressor(alpha=10.0)
    if name == "additive_prior":
        return base.PriorRegressor("additive_prior")
    if name == "feature_ridge":
        return base.FeatureRidgeRegressor(alpha=10.0, n_smiles_features=n_smiles_features, n_latent=n_latent, seed=seed)
    if name in {"descriptor_deep_mlp", "dual_view_mlp"}:
        return SymmetricMLPRegressor(name, seed=seed, n_smiles_features=n_smiles_features, n_latent=max(24, n_latent))
    control_map = {
        "cpb_learned_anchor": "learned_anchor",
        "cpb_learned_no_anchor": "learned_no_anchor",
        "cpb_shuffled_assignment": "shuffled_assignment",
        "cpb_random_clusters": "random_clusters",
    }
    if name in control_map:
        return ControlledResidualRidge(
            control_map[name],
            seed=seed,
            n_smiles_features=n_smiles_features,
            n_latent=n_latent,
            k_pair=16,
            k_cell=8,
        )
    raise ValueError(name)


def evaluate_model(
    df: pd.DataFrame,
    split_name: str,
    split: dict[str, np.ndarray],
    model_name: str,
    seed: int,
    sample_size: int,
    n_smiles_features: int,
    n_latent: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    train = df.iloc[split["train"]].reset_index(drop=True)
    val = df.iloc[split["val"]].reset_index(drop=True)
    test = df.iloc[split["test"]].reset_index(drop=True)
    if min(len(train), len(val), len(test)) < 30:
        raise ValueError(f"Split {split_name} is too small: {len(train)}/{len(val)}/{len(test)}")
    prior = base.SynergyPrior().fit(train)
    model = make_model(model_name, seed, n_smiles_features, n_latent)
    start = time.perf_counter()
    baseline_mb = working_set_mb()
    with MemorySampler() as memory:
        model.fit(train, val)
        pred = model.predict(test)
    elapsed = time.perf_counter() - start
    y = test[base.LABEL].to_numpy(dtype=float)
    threshold = float(np.quantile(train[base.LABEL].to_numpy(dtype=float), 0.90))
    priors = {
        "drug": prior.drug_tendency(test),
        "pair": prior.pair_tendency(test),
        "cell": prior.cell_tendency(test),
    }
    metrics: dict[str, object] = {
        "seed": seed,
        "sample_size": sample_size,
        "split": split_name,
        "model": model_name,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        **base.overlap_audit(train, test),
        **base.regression_metrics(y, pred),
        **base.topk_metrics(y, pred, threshold),
        **coupling_metrics(y, pred, priors, prior.predict(test, "additive_prior"), seed + 404),
        "elapsed_sec": elapsed,
        "working_set_start_mb": baseline_mb,
        "working_set_peak_mb": memory.peak_mb,
        "working_set_delta_mb": memory.peak_mb - baseline_mb if np.isfinite(baseline_mb) else float("nan"),
    }
    if hasattr(model, "selected_config_"):
        metrics["selected_config"] = json.dumps(model.selected_config_, sort_keys=True)
    if hasattr(model, "diagnostics"):
        metrics.update(model.diagnostics())
    predictions = pd.DataFrame(
        {
            "seed": seed,
            "sample_size": sample_size,
            "split": split_name,
            "model": model_name,
            "drug1": test[base.D1].to_numpy(),
            "drug2": test[base.D2].to_numpy(),
            "pair_id": test[base.PAIR].to_numpy(),
            "cell": test[base.CELL].to_numpy(),
            "y_true": y,
            "y_pred": pred,
            "drug_prior": priors["drug"],
            "pair_prior": priors["pair"],
            "cell_prior": priors["cell"],
            "additive_prior": prior.predict(test, "additive_prior"),
        }
    )
    return metrics, predictions


def frequency_summary(df: pd.DataFrame, entity: str, values: pd.Series) -> dict[str, object]:
    counts = values.value_counts().to_numpy(dtype=float)
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    gini = float((2 * np.sum((np.arange(1, n + 1)) * sorted_counts) / (n * sorted_counts.sum())) - (n + 1) / n)
    return {
        "entity": entity,
        "n_entities": n,
        "count_min": float(np.min(counts)),
        "count_q25": float(np.quantile(counts, 0.25)),
        "count_median": float(np.median(counts)),
        "count_q75": float(np.quantile(counts, 0.75)),
        "count_q90": float(np.quantile(counts, 0.90)),
        "count_q99": float(np.quantile(counts, 0.99)),
        "count_max": float(np.max(counts)),
        "gini": gini,
    }


def split_detail_row(df: pd.DataFrame, split_name: str, seed: int, split: dict[str, np.ndarray]) -> dict[str, object]:
    train = df.iloc[split["train"]]
    val = df.iloc[split["val"]]
    test = df.iloc[split["test"]]
    train_drugs = set(train[base.D1]).union(train[base.D2])
    test_drugs = set(test[base.D1]).union(test[base.D2])
    d1_seen = test[base.D1].isin(train_drugs).to_numpy()
    d2_seen = test[base.D2].isin(train_drugs).to_numpy()
    one_new = np.logical_xor(~d1_seen, ~d2_seen)
    both_new = (~d1_seen) & (~d2_seen)
    row = {
        "seed": seed,
        "split": split_name,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_discarded_cross_group": len(df) - len(train) - len(val) - len(test),
        "train_drugs": len(train_drugs),
        "test_drugs": len(test_drugs),
        "train_pairs": train[base.PAIR].nunique(),
        "test_pairs": test[base.PAIR].nunique(),
        "train_cells": train[base.CELL].nunique(),
        "test_cells": test[base.CELL].nunique(),
        "test_exactly_one_new_drug_ratio": float(np.mean(one_new)) if len(test) else float("nan"),
        "test_both_drugs_new_ratio": float(np.mean(both_new)) if len(test) else float("nan"),
        **base.overlap_audit(train, test),
    }
    if split_name == "molecular_cluster":
        held_out = sorted(test_drugs - train_drugs)
        train_unique = pd.concat(
            [
                train[[base.D1, base.S1]].rename(columns={base.D1: "drug", base.S1: "smiles"}),
                train[[base.D2, base.S2]].rename(columns={base.D2: "drug", base.S2: "smiles"}),
            ],
            ignore_index=True,
        ).drop_duplicates("drug")
        test_unique = pd.concat(
            [
                test[[base.D1, base.S1]].rename(columns={base.D1: "drug", base.S1: "smiles"}),
                test[[base.D2, base.S2]].rename(columns={base.D2: "drug", base.S2: "smiles"}),
            ],
            ignore_index=True,
        ).drop_duplicates("drug")
        test_unique = test_unique[test_unique["drug"].isin(held_out)]
        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            n_features=1024,
            alternate_sign=False,
            norm="l2",
            lowercase=False,
        )
        train_x = vectorizer.transform(train_unique["smiles"].astype(str))
        test_x = vectorizer.transform(test_unique["smiles"].astype(str))
        nearest = np.asarray((test_x @ train_x.T).max(axis=1).toarray()).ravel() if len(test_unique) else np.array([])
        row["heldout_drug_nearest_train_hash_cosine_mean"] = float(np.mean(nearest)) if len(nearest) else float("nan")
        row["heldout_drug_nearest_train_hash_cosine_max"] = float(np.max(nearest)) if len(nearest) else float("nan")
    return row


def run_audit(df: pd.DataFrame, out: Path, seeds: list[int]) -> None:
    rows = [
        frequency_summary(df, "drug", pd.concat([df[base.D1], df[base.D2]], ignore_index=True)),
        frequency_summary(df, "pair", df[base.PAIR]),
        frequency_summary(df, "cell", df[base.CELL]),
        frequency_summary(df, "pair_cell", df[base.PAIR].astype(str) + "||" + df[base.CELL].astype(str)),
    ]
    pd.DataFrame(rows).to_csv(out / "entity_frequency_summary.csv", index=False)
    overlap_rows = []
    for seed in seeds:
        for name in ["random", "cold_pair", "cold_one_drug", "cold_both_drugs", "cold_cell", "molecular_cluster"]:
            split = build_split(df, name, seed)
            overlap_rows.append(split_detail_row(df, name, seed, split))
    pd.DataFrame(overlap_rows).to_csv(out / "full_data_split_audit.csv", index=False)
    raw = pd.read_pickle(ROOT / "data" / "raw" / "tdc" / "drugcomb.pkl")
    candidate_fields = ["study", "source", "assay", "dose", "dose_grid", "batch", "publication"]
    metadata = {
        "raw_rows": int(len(raw)),
        "raw_columns": list(map(str, raw.columns)),
        "candidate_batch_fields_available": [c for c in candidate_fields if c in {str(x).lower() for x in raw.columns}],
        "source_split_status": "not_estimable_from_the_distributed_TDC_task_table",
    }
    (out / "metadata_audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def run_grid(
    df: pd.DataFrame,
    out: Path,
    phase: str,
    seeds: list[int],
    sample_size: int,
    split_names: list[str],
    models: list[str],
    n_smiles_features: int,
    n_latent: int,
) -> None:
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    sample = nested_stratified_sample(df, sample_size, seed=2026)
    for seed in seeds:
        for split_name in split_names:
            split = build_split(sample, split_name, seed)
            log(f"phase={phase} seed={seed} split={split_name} sizes={len(split['train'])}/{len(split['val'])}/{len(split['test'])}")
            for model_name in models:
                log(f"training {model_name}")
                try:
                    metrics, predictions = evaluate_model(
                        sample,
                        split_name,
                        split,
                        model_name,
                        seed,
                        len(sample),
                        n_smiles_features,
                        n_latent,
                    )
                except ValueError as exc:
                    log(f"skipped {model_name}: {exc}")
                    continue
                metrics_rows.append(metrics)
                prediction_frames.append(predictions)
                append_csv(out / f"{phase}_metrics.csv", metrics_rows)
                pd.concat(prediction_frames, ignore_index=True).to_csv(out / f"{phase}_predictions.csv", index=False)
                log(f"finished {model_name}: RMSE={metrics['rmse']:.4f}, PCS={metrics['pcs']:.4f}, conditional={metrics['conditional_pcs']:.4f}")


def run_scale(df: pd.DataFrame, out: Path, sizes: list[int], n_smiles_features: int, n_latent: int) -> None:
    rows: list[dict[str, object]] = []
    for requested in sizes:
        size = len(df) if requested <= 0 else min(requested, len(df))
        sample = nested_stratified_sample(df, size, seed=2026)
        split = build_split(sample, "random", 42)
        for model_name in ["identity_ridge", "additive_prior", "feature_ridge", "cpb_learned_no_anchor"]:
            log(f"scale size={size} model={model_name}")
            metrics, _ = evaluate_model(
                sample,
                "random",
                split,
                model_name,
                42,
                size,
                n_smiles_features,
                n_latent,
            )
            rows.append(metrics)
            append_csv(out / "scale_metrics.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Major-revision experiments for the prior-coupling audit paper.")
    parser.add_argument("--phase", choices=["audit", "core", "cpb", "scale", "all"], default="all")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "raw" / "tdc" / "drugcomb.pkl")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 123])
    parser.add_argument("--core-samples", type=int, default=30000)
    parser.add_argument("--cpb-samples", type=int, default=40000)
    parser.add_argument("--scale-sizes", nargs="+", type=int, default=[25000, 50000, 100000, 0], help="0 denotes full data.")
    parser.add_argument("--n-smiles-features", type=int, default=512)
    parser.add_argument("--n-latent", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_pickle(args.data_path)
    df = base.normalize_synergy_table(raw, label_column="Synergy_ZIP")
    config = vars(args).copy()
    config.update({"data_path": str(args.data_path), "out_dir": str(args.out_dir), "normalized_rows": len(df)})
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log(f"loaded {len(df)} normalized rows")

    if args.phase in {"audit", "all"}:
        run_audit(df, args.out_dir, args.seeds)
    if args.phase in {"core", "all"}:
        run_grid(
            df,
            args.out_dir,
            "core",
            args.seeds,
            args.core_samples,
            ["random", "cold_pair", "cold_one_drug", "cold_both_drugs", "cold_cell", "molecular_cluster"],
            ["identity_ridge", "additive_prior", "feature_ridge", "descriptor_deep_mlp", "dual_view_mlp"],
            args.n_smiles_features,
            args.n_latent,
        )
    if args.phase in {"cpb", "all"}:
        run_grid(
            df,
            args.out_dir,
            "cpb",
            args.seeds,
            args.cpb_samples,
            ["random", "cold_pair"],
            ["feature_ridge", "cpb_learned_anchor", "cpb_learned_no_anchor", "cpb_shuffled_assignment", "cpb_random_clusters"],
            args.n_smiles_features,
            args.n_latent,
        )
    if args.phase in {"scale", "all"}:
        run_scale(df, args.out_dir, args.scale_sizes, args.n_smiles_features, args.n_latent)
    log(f"completed phase={args.phase}; outputs={args.out_dir}")


if __name__ == "__main__":
    main()
