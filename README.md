# Repeated-Entity Audit for Drug Synergy Benchmarks

Official code and lightweight result artifacts for:

> **What Does Random Splitting Evaluate in Drug Synergy Benchmarks? A Repeated-Entity Audit**

The repository audits how repeated drugs, unordered drug pairs, and cell lines shape evaluation on large drug-synergy tables. It compares train-only prior and identity controls with molecular feature and compact neural baselines, then measures how their relative performance changes across random, cold-pair, cold-drug, cold-cell, and molecular-cluster splits.

## Main components

- Full-table train--test entity-overlap and frequency audits.
- Global, drug, pair, cell, additive-prior, and symmetric identity-ridge controls.
- Hash-feature ridge, Morgan-fingerprint ridge, and compact neural baselines.
- Six split protocols: random, cold pair, at least one unseen drug, both drugs unseen, cold cell, and SMILES-hash cluster.
- Conditional prediction--prior coupling diagnostics and prototype-residual stress tests.
- Label robustness across ZIP, Bliss, Loewe, and HSA scores.
- An external repeated-entity check on NCI-ALMANAC.

## Repository layout

```text
data/                 Public-data download instructions; raw data are excluded
scripts/              Audit, model, robustness, summary, and figure scripts
results/              Aggregate metrics and source tables reported in the paper
figures/              Final manuscript figures and their source-data CSV files
paper/                Accepted manuscript source and PDF
```

Large per-example prediction files, local environments, caches, and raw public datasets are excluded. The included aggregate tables are sufficient to inspect the reported values and regenerate the final plots.

## Environment

Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux and macOS users can replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Obtain DrugComb

The primary table is the DrugComb task table distributed through Therapeutics Data Commons. The experiment entry point can download the public Harvard Dataverse file (file ID `4215720`) directly:

```powershell
.\.venv\Scripts\python.exe scripts\run_synergy_prior_experiments.py `
  --download-tdc `
  --data-path data\raw\tdc\drugcomb.pkl `
  --models global_mean `
  --splits random `
  --max-samples 2000 `
  --out-dir results\download_check
```

The normalized primary analysis uses `Synergy_ZIP`. Label-robustness experiments additionally use `Synergy_Bliss`, `Synergy_Loewe`, and `Synergy_HSA` from the same table.

## Quick smoke test

The synthetic smoke test checks preprocessing, splitting, prior controls, identity ridge, and metric export without downloading data:

```powershell
.\.venv\Scripts\python.exe scripts\run_synergy_prior_experiments.py `
  --synthetic-smoke `
  --synthetic-samples 2400 `
  --models global_mean drug_prior pair_prior cell_prior additive_prior identity_ridge `
  --splits random cold_pair cold_cell `
  --out-dir results\smoke
```

## Reproduce the DrugComb audit

Run the six-split audit, compact model comparison, CPB controls, and scaling experiment:

```powershell
.\.venv\Scripts\python.exe scripts\run_major_revision_experiments.py --phase all
.\.venv\Scripts\python.exe scripts\summarize_major_revision.py
```

Run the full-table critical controls, conditional-coupling sanity check, and matched prototype-removal sweeps:

```powershell
.\.venv\Scripts\python.exe scripts\run_second_revision_experiments.py --phase all
.\.venv\Scripts\python.exe scripts\summarize_second_revision.py
```

Run the Morgan-fingerprint ridge baseline:

```powershell
.\.venv\Scripts\python.exe scripts\run_ecfp_baseline.py --suite all
```

Run the four-label robustness check:

```powershell
.\.venv\Scripts\python.exe scripts\run_symmetric_label_robustness.py
```

The experiment scripts restrict dense ablations to deterministic 30k or 40k subsets and reserve full-table runs for lightweight controls. Seeds are `0`, `42`, and `123` unless a script states otherwise.

## Reproduce the NCI-ALMANAC check

The external script downloads and parses the public CellMiner combination-score workbook, constructs the long table, and evaluates the train-only controls:

```powershell
foreach ($seed in 0, 42, 123) {
  .\.venv\Scripts\python.exe scripts\run_nci_almanac_external_diagnostic.py `
    --seed $seed `
    --out-dir "results\nci_seed$seed"
}
```

## Regenerate figures

```powershell
.\.venv\Scripts\python.exe scripts\make_major_revision_figure.py
.\.venv\Scripts\python.exe scripts\make_second_revision_figure.py
.\.venv\Scripts\python.exe scripts\make_display_completion_figures.py
```

The final plot inputs are also available under `figures/synergy_manuscript/source_data/`.

## Key result files

- `results/major_revision/summary/core_multiseed_summary.csv`
- `results/major_revision/summary/pair_cluster_bootstrap_ci.csv`
- `results/second_revision/summary/full_critical_summary.csv`
- `results/second_revision/summary/cpb_pareto_summary.csv`
- `results/second_revision/symmetric_label_robustness.csv`
- `results/second_revision/nci_three_seed_metrics.csv`
- `results/third_revision/ecfp_metrics.csv`

## Funding

This research was funded by the National Science and Technology Council, Taiwan (grant number NSTC115-2221-E-038-012-MY3).

## Citation

Please cite the accepted paper when using this audit or its result artifacts. Citation metadata are provided in `CITATION.cff`.

