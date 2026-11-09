from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import run_major_revision_experiments as revision
import run_synergy_prior_experiments as base


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "second_revision"
MAJOR = ROOT / "results" / "major_revision"
SEEDS = [0, 42, 123]
FULL_SPLITS = ["random", "cold_pair", "cold_both_drugs"]
FULL_MODELS = [
    "identity_ridge",
    "additive_prior",
    "feature_ridge",
    "cpb_learned_no_anchor",
    "cpb_random_clusters",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def append_row(path: Path, row: dict[str, object]) -> None:
    frame = pd.DataFrame([row])
    if not path.exists():
        frame.to_csv(path, index=False)
        return
    existing = pd.read_csv(path)
    columns = list(existing.columns)
    columns.extend(column for column in frame.columns if column not in existing.columns)
    existing = existing.reindex(columns=columns)
    frame = frame.reindex(columns=columns)
    pd.concat([existing, frame], ignore_index=True).to_csv(path, index=False)


def completed_keys(path: Path, columns: list[str]) -> set[tuple[object, ...]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    return {tuple(row[col] for col in columns) for _, row in frame.iterrows()}


def evaluate_prediction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prediction: np.ndarray,
    split_name: str,
    model_name: str,
    seed: int,
    elapsed: float,
    peak_mb: float,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    prior = base.SynergyPrior().fit(train)
    y = test[base.LABEL].to_numpy(dtype=float)
    priors = {
        "drug": prior.drug_tendency(test),
        "pair": prior.pair_tendency(test),
        "cell": prior.cell_tendency(test),
    }
    threshold = float(np.quantile(train[base.LABEL].to_numpy(dtype=float), 0.90))
    row: dict[str, object] = {
        "seed": seed,
        "sample_size": len(train) + len(test),
        "split": split_name,
        "model": model_name,
        "n_train": len(train),
        "n_test": len(test),
        **base.overlap_audit(train, test),
        **base.regression_metrics(y, prediction),
        **base.topk_metrics(y, prediction, threshold),
        **revision.coupling_metrics(y, prediction, priors, prior.predict(test, "additive_prior"), seed + 900),
        "elapsed_sec": elapsed,
        "working_set_peak_mb": peak_mb,
    }
    if extra:
        row.update(extra)
    return row


def prepared_cpb(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    n_smiles_features: int = 512,
    n_latent: int = 16,
    k_pair: int = 16,
    k_cell: int = 8,
) -> dict[str, object]:
    helper = revision.ControlledResidualRidge(
        "learned_no_anchor",
        seed=seed,
        n_smiles_features=n_smiles_features,
        n_latent=n_latent,
        k_pair=k_pair,
        k_cell=k_cell,
    )
    builder = helper.builder
    builder.fit(train)
    pair_entities = train.drop_duplicates(base.PAIR, keep="first")
    cell_entities = train.drop_duplicates(base.CELL, keep="first")
    pair_state = helper._fit_channel(builder.pair_latent(pair_entities), k_pair, seed)
    cell_state = helper._fit_channel(builder.cell_dense(cell_entities), k_cell, seed + 1)
    return {
        "helper": helper,
        "builder": builder,
        "pair_state": pair_state,
        "cell_state": cell_state,
        "train_pair": builder.pair_latent(train),
        "test_pair": builder.pair_latent(test),
        "train_cell": builder.cell_dense(train),
        "test_cell": builder.cell_dense(test),
    }


def assigned_centers(
    prepared: dict[str, object],
    frame: pd.DataFrame,
    channel: str,
    control: str,
    seed: int,
) -> np.ndarray:
    helper = prepared["helper"]
    state = prepared[f"{channel}_state"]
    values = prepared[f"{'train' if frame is prepared.get('train_frame') else 'test'}_{channel}"]
    keys = frame[base.PAIR if channel == "pair" else base.CELL].astype(str)
    helper.control = control
    return helper._assigned(values, keys, state, seed)


def residual_features(
    pair: np.ndarray,
    cell: np.ndarray,
    pair_centers: np.ndarray,
    cell_centers: np.ndarray,
    strength: float,
) -> sparse.csr_matrix:
    rp = pair - strength * pair_centers
    rc = cell - strength * cell_centers
    return sparse.csr_matrix(np.hstack([rp, rc, rp * rc]).astype(np.float32))


def run_full(df: pd.DataFrame, out: Path) -> None:
    path = out / "full_critical_metrics.csv"
    done = completed_keys(path, ["seed", "split", "model"])
    for seed in SEEDS:
        for split_name in FULL_SPLITS:
            split = revision.build_split(df, split_name, seed)
            train = df.iloc[split["train"]].reset_index(drop=True)
            val = df.iloc[split["val"]].reset_index(drop=True)
            test = df.iloc[split["test"]].reset_index(drop=True)
            log(f"full seed={seed} split={split_name} sizes={len(train)}/{len(val)}/{len(test)}")

            for model_name in ["identity_ridge", "additive_prior", "feature_ridge"]:
                if (seed, split_name, model_name) in done:
                    continue
                model = revision.make_model(model_name, seed, 512, 16)
                start = time.perf_counter()
                with revision.MemorySampler() as memory:
                    model.fit(train, val)
                    pred = model.predict(test)
                row = evaluate_prediction(
                    train,
                    test,
                    pred,
                    split_name,
                    model_name,
                    seed,
                    time.perf_counter() - start,
                    memory.peak_mb,
                    {"n_val": len(val)},
                )
                append_row(path, row)
                done.add((seed, split_name, model_name))
                log(f"finished {model_name}: RMSE={row['rmse']:.4f}, cPCS={row['conditional_pcs']:.4f}")

            cpb_needed = [
                model
                for model in ["cpb_learned_no_anchor", "cpb_random_clusters"]
                if (seed, split_name, model) not in done
            ]
            if cpb_needed:
                start_prepare = time.perf_counter()
                with revision.MemorySampler() as memory:
                    prepared = prepared_cpb(train, test, seed)
                    prepared["train_frame"] = train
                    train_pair = prepared["train_pair"]
                    test_pair = prepared["test_pair"]
                    train_cell = prepared["train_cell"]
                    test_cell = prepared["test_cell"]
                    for model_name in cpb_needed:
                        control = "learned_no_anchor" if model_name == "cpb_learned_no_anchor" else "random_clusters"
                        helper = prepared["helper"]
                        helper.control = control
                        train_pair_centers = helper._assigned(
                            train_pair, train[base.PAIR].astype(str), prepared["pair_state"], seed + 201
                        )
                        test_pair_centers = helper._assigned(
                            test_pair, test[base.PAIR].astype(str), prepared["pair_state"], seed + 201
                        )
                        train_cell_centers = helper._assigned(
                            train_cell, train[base.CELL].astype(str), prepared["cell_state"], seed + 301
                        )
                        test_cell_centers = helper._assigned(
                            test_cell, test[base.CELL].astype(str), prepared["cell_state"], seed + 301
                        )
                        x_train = residual_features(
                            train_pair, train_cell, train_pair_centers, train_cell_centers, 1.0
                        )
                        x_test = residual_features(test_pair, test_cell, test_pair_centers, test_cell_centers, 1.0)
                        model = Ridge(alpha=10.0, solver="lsqr")
                        model.fit(x_train, train[base.LABEL].to_numpy(dtype=float))
                        pred = model.predict(x_test)
                        elapsed = time.perf_counter() - start_prepare
                        row = evaluate_prediction(
                            train,
                            test,
                            pred,
                            split_name,
                            model_name,
                            seed,
                            elapsed,
                            memory.peak_mb,
                            {
                                "n_val": len(val),
                                "pair_removed_energy_target": prepared["pair_state"]["target_energy"],
                                "cell_removed_energy_target": prepared["cell_state"]["target_energy"],
                            },
                        )
                        append_row(path, row)
                        done.add((seed, split_name, model_name))
                        log(f"finished {model_name}: RMSE={row['rmse']:.4f}, cPCS={row['conditional_pcs']:.4f}")


def run_pareto(df: pd.DataFrame, out: Path) -> None:
    path = out / "cpb_pareto_metrics.csv"
    key_columns = ["seed", "split", "control", "k_pair", "k_cell", "strength"]
    done = completed_keys(path, key_columns)
    sample = revision.nested_stratified_sample(df, 40000, seed=2026)
    controls = ["learned_no_anchor", "shuffled_assignment", "random_clusters"]
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    cluster_settings = [(8, 4), (16, 8), (32, 16)]
    for seed in SEEDS:
        for split_name in ["random", "cold_pair"]:
            split = revision.build_split(sample, split_name, seed)
            train = sample.iloc[split["train"]].reset_index(drop=True)
            test = sample.iloc[split["test"]].reset_index(drop=True)
            prior = base.SynergyPrior().fit(train)
            y = test[base.LABEL].to_numpy(dtype=float)
            priors = {
                "drug": prior.drug_tendency(test),
                "pair": prior.pair_tendency(test),
                "cell": prior.cell_tendency(test),
            }
            additive = prior.predict(test, "additive_prior")
            threshold = float(np.quantile(train[base.LABEL].to_numpy(dtype=float), 0.90))
            for k_pair, k_cell in cluster_settings:
                prepared = prepared_cpb(train, test, seed, k_pair=k_pair, k_cell=k_cell)
                train_pair = prepared["train_pair"]
                test_pair = prepared["test_pair"]
                train_cell = prepared["train_cell"]
                test_cell = prepared["test_cell"]
                helper = prepared["helper"]
                for control in controls:
                    helper.control = control
                    pair_train_centers = helper._assigned(
                        train_pair, train[base.PAIR].astype(str), prepared["pair_state"], seed + 201
                    )
                    pair_test_centers = helper._assigned(
                        test_pair, test[base.PAIR].astype(str), prepared["pair_state"], seed + 201
                    )
                    cell_train_centers = helper._assigned(
                        train_cell, train[base.CELL].astype(str), prepared["cell_state"], seed + 301
                    )
                    cell_test_centers = helper._assigned(
                        test_cell, test[base.CELL].astype(str), prepared["cell_state"], seed + 301
                    )
                    for strength in strengths:
                        key = (seed, split_name, control, k_pair, k_cell, strength)
                        if key in done:
                            continue
                        start = time.perf_counter()
                        x_train = residual_features(
                            train_pair, train_cell, pair_train_centers, cell_train_centers, strength
                        )
                        x_test = residual_features(
                            test_pair, test_cell, pair_test_centers, cell_test_centers, strength
                        )
                        model = Ridge(alpha=10.0, solver="lsqr")
                        model.fit(x_train, train[base.LABEL].to_numpy(dtype=float))
                        pred = model.predict(x_test)
                        metrics = {
                            "seed": seed,
                            "split": split_name,
                            "control": control,
                            "k_pair": k_pair,
                            "k_cell": k_cell,
                            "strength": strength,
                            "n_train": len(train),
                            "n_test": len(test),
                            **base.regression_metrics(y, pred),
                            **base.topk_metrics(y, pred, threshold),
                            **revision.coupling_metrics(y, pred, priors, additive, seed + 1700, n_null=32),
                            "elapsed_sec": time.perf_counter() - start,
                        }
                        append_row(path, metrics)
                        done.add(key)
                log(f"pareto seed={seed} split={split_name} K={k_pair}/{k_cell} complete")


def standardized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.mean(values)) / max(float(np.std(values)), 1e-12)


def fixed_rmse_prediction(y: np.ndarray, residual: np.ndarray, target_rmse: float = 1.0) -> np.ndarray:
    residual = residual - np.mean(residual)
    residual *= target_rmse / max(float(np.sqrt(np.mean(residual**2))), 1e-12)
    return y + residual


def run_synthetic(out: Path) -> None:
    rows = []
    n = 20000
    for seed in range(20):
        rng = np.random.default_rng(seed)
        z = rng.normal(size=n)
        prior_linear = standardized(z)
        y_linear = 1.4 * prior_linear + rng.normal(scale=1.0, size=n)
        independent_error = rng.normal(size=n)
        pred_a = fixed_rmse_prediction(y_linear, independent_error)

        prior_residual = revision.residualize(prior_linear, y_linear)
        direct_error = 0.65 * standardized(prior_residual) + math.sqrt(1.0 - 0.65**2) * standardized(
            rng.normal(size=n)
        )
        pred_b = fixed_rmse_prediction(y_linear, direct_error)

        y_nonlinear = 0.9 * z + 1.1 * (z**2 - 1.0) + rng.normal(scale=0.8, size=n)
        pred_c = fixed_rmse_prediction(y_nonlinear, rng.normal(size=n))

        cases = [
            ("A_label_only", y_linear, pred_a, prior_linear),
            ("B_direct_prior_same_rmse", y_linear, pred_b, prior_linear),
            ("C_nonlinear_label_prior", y_nonlinear, pred_c, prior_linear),
        ]
        for case, y, pred, prior in cases:
            constants = np.full(n, np.mean(prior))
            metrics = revision.coupling_metrics(
                y,
                pred,
                {"drug": prior, "pair": constants, "cell": constants},
                np.full(n, np.mean(y)),
                seed + 2500,
                n_null=64,
            )
            rows.append(
                {
                    "seed": seed,
                    "case": case,
                    "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                    "pearson": base.finite_corr(y, pred),
                    **metrics,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "synthetic_sanity_metrics.csv", index=False)
    summary = frame.groupby("case").agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x) for col in summary.columns]
    summary.to_csv(out / "synthetic_sanity_summary.csv", index=False)


def refresh_core_identity(df: pd.DataFrame) -> None:
    metrics_path = MAJOR / "core_metrics.csv"
    pred_path = MAJOR / "core_predictions.csv"
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(pred_path)
    metrics = metrics[metrics["model"] != "identity_ridge"].copy()
    predictions = predictions[predictions["model"] != "identity_ridge"].copy()
    sample = revision.nested_stratified_sample(df, 30000, seed=2026)
    new_metrics = []
    new_predictions = []
    for seed in SEEDS:
        for split_name in [
            "random",
            "cold_pair",
            "cold_one_drug",
            "cold_both_drugs",
            "cold_cell",
            "molecular_cluster",
        ]:
            split = revision.build_split(sample, split_name, seed)
            row, pred = revision.evaluate_model(
                sample, split_name, split, "identity_ridge", seed, len(sample), 512, 16
            )
            new_metrics.append(row)
            new_predictions.append(pred)
    pd.concat([metrics, pd.DataFrame(new_metrics)], ignore_index=True).to_csv(metrics_path, index=False)
    pd.concat([predictions, *new_predictions], ignore_index=True).to_csv(pred_path, index=False)
    log("replaced core identity results with symmetric shared-drug encoding")


def refresh_moment_null(phase: str) -> None:
    metrics_path = MAJOR / f"{phase}_metrics.csv"
    pred_path = MAJOR / f"{phase}_predictions.csv"
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(pred_path)
    updates = []
    for (seed, split_name, model_name), group in predictions.groupby(["seed", "split", "model"], sort=False):
        y = group["y_true"].to_numpy(dtype=float)
        pred = group["y_pred"].to_numpy(dtype=float)
        priors = {
            "drug": group["drug_prior"].to_numpy(dtype=float),
            "pair": group["pair_prior"].to_numpy(dtype=float),
            "cell": group["cell_prior"].to_numpy(dtype=float),
        }
        coupling = revision.coupling_metrics(
            y,
            pred,
            priors,
            group["additive_prior"].to_numpy(dtype=float),
            int(seed) + 3100,
            n_null=64,
        )
        updates.append({"seed": seed, "split": split_name, "model": model_name, **coupling})
    update_frame = pd.DataFrame(updates)
    coupling_cols = [col for col in update_frame.columns if col not in {"seed", "split", "model"}]
    metrics = metrics.drop(columns=[col for col in coupling_cols if col in metrics.columns])
    metrics = metrics.merge(update_frame, on=["seed", "split", "model"], how="left", validate="one_to_one")
    metrics.to_csv(metrics_path, index=False)
    log(f"refreshed {phase} coupling metrics with the moment-matched null")


def write_environment(out: Path) -> None:
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "unknown"
    memory_bytes = None
    if hasattr(revision.ctypes, "windll"):
        class MemoryStatus(revision.ctypes.Structure):
            _fields_ = [
                ("dwLength", revision.ctypes.c_ulong),
                ("dwMemoryLoad", revision.ctypes.c_ulong),
                ("ullTotalPhys", revision.ctypes.c_ulonglong),
                ("ullAvailPhys", revision.ctypes.c_ulonglong),
                ("ullTotalPageFile", revision.ctypes.c_ulonglong),
                ("ullAvailPageFile", revision.ctypes.c_ulonglong),
                ("ullTotalVirtual", revision.ctypes.c_ulonglong),
                ("ullAvailVirtual", revision.ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", revision.ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = revision.ctypes.sizeof(status)
        if revision.ctypes.windll.kernel32.GlobalMemoryStatusEx(revision.ctypes.byref(status)):
            memory_bytes = int(status.ullTotalPhys)
    cpu_name = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                cpu_name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    environment = {
        "cpu": cpu_name,
        "python": sys.version.split()[0],
        "scikit_learn": sklearn_version,
        "total_physical_memory_gb": memory_bytes / 2**30 if memory_bytes else None,
        "thread_policy": "OMP/MKL/OpenBLAS/NumExpr restricted to one thread",
        "runtime_scope": "feature construction, fitting, and test inference",
    }
    (out / "compute_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["audit", "synthetic", "refresh", "pareto", "full", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_pickle(ROOT / "data" / "raw" / "tdc" / "drugcomb.pkl")
    df = base.normalize_synergy_table(raw, label_column="Synergy_ZIP")
    write_environment(RESULTS)
    if args.phase in {"audit", "all"}:
        revision.run_audit(df, RESULTS, SEEDS)
    if args.phase in {"synthetic", "all"}:
        run_synthetic(RESULTS)
    if args.phase in {"refresh", "all"}:
        refresh_core_identity(df)
        refresh_moment_null("core")
        refresh_moment_null("cpb")
    if args.phase in {"pareto", "all"}:
        run_pareto(df, RESULTS)
    if args.phase in {"full", "all"}:
        run_full(df, RESULTS)
    log(f"completed phase={args.phase}")


if __name__ == "__main__":
    main()
