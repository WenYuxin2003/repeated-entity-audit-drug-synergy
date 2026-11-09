# Public data

Raw datasets are intentionally excluded from the repository.

## DrugComb through Therapeutics Data Commons

The primary experiments use the DrugComb task table distributed through Therapeutics Data Commons and hosted on Harvard Dataverse as file ID `4215720`.

The expected local path is:

```text
data/raw/tdc/drugcomb.pkl
```

Download and validate it through the primary experiment script as described in the repository README.

## NCI-ALMANAC through CellMiner

`scripts/run_nci_almanac_external_diagnostic.py` downloads the public CellMiner NCI-ALMANAC combination-score archive from:

```text
https://discover.nci.nih.gov/cellminer/download/processeddataset/DTP_NCI60_ALMANAC_COMBO_SCORE.zip
```

The script median-aggregates duplicate drug-pair--cell records and stores its derived long table under `data/processed/`.

