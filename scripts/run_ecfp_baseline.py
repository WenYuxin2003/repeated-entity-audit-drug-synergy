from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.linear_model import Ridge

import run_major_revision_experiments as revision
import run_second_revision_experiments as second
import run_synergy_prior_experiments as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "third_revision"
SEEDS = [0, 42, 123]
COMMON_SPLITS = [
    "random",
    "cold_pair",
    "cold_one_drug",
    "cold_both_drugs",
    "cold_cell",
    "molecular_cluster",
]
FULL_SPLITS = ["random", "cold_pair", "cold_both_drugs"]


class ECFPRidgeRegressor:
    def __init__(self, radius: int = 2, n_bits: int = 2048, alpha: float = 10.0) -> None:
        self.radius = radius
        self.n_bits = n_bits
        self.alpha = alpha
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
        )
        self.cache: dict[str, sparse.csr_matrix] = {}

    def _fingerprint(self, smiles: str) -> sparse.csr_matrix:
        key = str(smiles or "")
        if key in self.cache:
            return self.cache[key]
        mol = Chem.MolFromSmiles(key)
        values = np.zeros(self.n_bits, dtype=np.float32)
        if mol is not None:
            fp = self.generator.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, values)
        row = sparse.csr_matrix(values.reshape(1, -1))
        self.cache[key] = row
        return row

    def _drug_matrix(self, smiles: pd.Series) -> sparse.csr_matrix:
        return sparse.vstack(
            [self._fingerprint(value) for value in smiles.fillna("").astype(str)],
            format="csr",
        )

    def _features(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        left = self._drug_matrix(frame[base.S1])
        right = self._drug_matrix(frame[base.S2])
        pair_sum = left + right
        pair_diff = left - right
        pair_diff.data = np.abs(pair_diff.data)
        pair_product = left.multiply(right)
        cell = self.cell_encoder_.transform(frame[[base.CELL]])
        return sparse.hstack(
            [pair_sum, pair_diff, pair_product, cell],
            format="csr",
            dtype=np.float32,
        )

    def fit(self, train: pd.DataFrame) -> "ECFPRidgeRegressor":
        self.cell_encoder_ = base.one_hot_encoder()
        self.cell_encoder_.fit(train[[base.CELL]])
        self.model_ = Ridge(alpha=self.alpha, solver="lsqr")
        self.model_.fit(
            self._features(train),
            train[base.LABEL].to_numpy(dtype=float),
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._features(frame))


def append_result(path: Path, row: dict[str, object]) -> None:
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    key = (row["suite"], row["seed"], row["split"], row["model"])
    if not current.empty:
        keys = list(
            zip(
                current["suite"],
                current["seed"],
                current["split"],
                current["model"],
            )
        )
        if key in keys:
            return
    pd.concat([current, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def run_suite(
    frame: pd.DataFrame,
    suite: str,
    splits: list[str],
    seeds: list[int],
) -> None:
    path = OUT / "ecfp_metrics.csv"
    for seed in seeds:
        for split_name in splits:
            split = revision.build_split(frame, split_name, seed)
            train = frame.iloc[split["train"]].reset_index(drop=True)
            test = frame.iloc[split["test"]].reset_index(drop=True)
            started = time.perf_counter()
            model = ECFPRidgeRegressor().fit(train)
            prediction = model.predict(test)
            elapsed = time.perf_counter() - started
            row = second.evaluate_prediction(
                train,
                test,
                prediction,
                split_name,
                "ecfp_ridge",
                seed,
                elapsed,
                float("nan"),
                {
                    "suite": suite,
                    "ecfp_radius": 2,
                    "ecfp_bits": 2048,
                    "pair_blocks": "sum|absolute_difference|product",
                    "cell_features": "training-fitted one-hot",
                    "ridge_alpha": 10.0,
                    "ridge_solver": "lsqr",
                },
            )
            append_result(path, row)
            print(
                f"{suite} seed={seed} split={split_name} "
                f"RMSE={row['rmse']:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["smoke", "common", "full", "all"],
        default="smoke",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_pickle(ROOT / "data" / "raw" / "tdc" / "drugcomb.pkl")
    full = base.normalize_synergy_table(raw, label_column="Synergy_ZIP")
    common = revision.nested_stratified_sample(full, 30000, seed=2026)
    if args.suite == "smoke":
        run_suite(common, "common30k", ["random"], [0])
    if args.suite in {"common", "all"}:
        run_suite(common, "common30k", COMMON_SPLITS, SEEDS)
    if args.suite in {"full", "all"}:
        run_suite(full, "full", FULL_SPLITS, SEEDS)


if __name__ == "__main__":
    main()
