from __future__ import annotations

import os
from pathlib import Path

for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import run_major_revision_experiments as revision
import run_synergy_prior_experiments as base


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "tdc" / "drugcomb.pkl"
OUTPUT = ROOT / "results" / "second_revision" / "symmetric_label_robustness.csv"

LABELS = {
    "ZIP": "Synergy_ZIP",
    "Bliss": "Synergy_Bliss",
    "Loewe": "Synergy_Loewe",
    "HSA": "Synergy_HSA",
}

SPLITS = ["random", "cold_pair", "cold_one_drug", "cold_cell"]
SEED = 42


def main() -> None:
    raw = pd.read_pickle(DATA)
    rows: list[dict[str, object]] = []
    for label, column in LABELS.items():
        data = base.normalize_synergy_table(raw, label_column=column)
        for split_name in SPLITS:
            split = revision.build_split(data, split_name, SEED)
            train = data.iloc[split["train"]].reset_index(drop=True)
            test = data.iloc[split["test"]].reset_index(drop=True)
            model = revision.SymmetricIdentityRidgeRegressor(alpha=10.0)
            model.fit(train)
            prediction = model.predict(test)
            rows.append(
                {
                    "label": label,
                    "split": split_name,
                    "seed": SEED,
                    "n_train": len(train),
                    "n_test": len(test),
                    "rmse": float(np.sqrt(mean_squared_error(test[base.LABEL], prediction))),
                    **base.overlap_audit(train, test),
                }
            )
            print(f"{label:5s} {split_name:15s} complete")

    frame = pd.DataFrame(rows)
    random_rmse = frame[frame["split"] == "random"].set_index("label")["rmse"]
    frame["random_rmse"] = frame["label"].map(random_rmse)
    frame["rmse_increase_pct"] = 100.0 * (frame["rmse"] - frame["random_rmse"]) / frame["random_rmse"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
