# NCI-ALMANAC External Prior Diagnostic

## Source

- URL: https://discover.nci.nih.gov/cellminer/download/processeddataset/DTP_NCI60_ALMANAC_COMBO_SCORE.zip
- CellMiner metadata: {'CellMiner Address': 'https://discover.nci.nih.gov/cellminer/', 'Download normalized data set': 'Compound activity almanac combo scores a', 'CellMiner Database Version': '2.15', 'Human Genome Version': 'HG-19', 'Date': '09-17-2025'}
- Long table rows: 306,365
- Drugs / pairs / cells: 104 / 5233 / 59

## Test Metrics

| split     | model          |   n_test |   seen_drug_ratio |   seen_pair_ratio |   seen_cell_ratio |    rmse |   pearson |      pcs |   top5pct_precision |   top5pct_enrichment |
|:----------|:---------------|---------:|------------------:|------------------:|------------------:|--------:|----------:|---------:|--------------------:|---------------------:|
| cold_cell | identity_ridge |    31369 |            1.0000 |            1.0000 |            0.0000 | 51.5187 |    0.5863 |   0.7603 |              0.6546 |               6.1958 |
| cold_cell | pair_prior     |    31369 |            1.0000 |            1.0000 |            0.0000 | 51.5912 |    0.5875 |   0.7271 |              0.6501 |               6.1535 |
| cold_cell | additive_prior |    31369 |            1.0000 |            1.0000 |            0.0000 | 51.8760 |    0.5756 |   0.8051 |              0.6488 |               6.1415 |
| cold_cell | drug_prior     |    31369 |            1.0000 |            1.0000 |            0.0000 | 61.6165 |    0.2731 |   0.7271 |              0.3907 |               3.6982 |
| cold_cell | global_mean    |    31369 |            1.0000 |            1.0000 |            0.0000 | 63.4353 |  nan      | nan      |              0.0382 |               0.3620 |
| cold_cell | cell_prior     |    31369 |            1.0000 |            1.0000 |            0.0000 | 63.4353 |  nan      | nan      |              0.0382 |               0.3620 |
| cold_pair | identity_ridge |    30720 |            0.9982 |            0.0000 |            1.0000 | 62.3977 |    0.2616 |   0.6547 |              0.2702 |               2.8750 |
| cold_pair | additive_prior |    30720 |            0.9982 |            0.0000 |            1.0000 | 62.5949 |    0.2572 |   0.7042 |              0.2637 |               2.8057 |
| cold_pair | drug_prior     |    30720 |            0.9982 |            0.0000 |            1.0000 | 63.0454 |    0.2375 |   0.5004 |              0.2546 |               2.7087 |
| cold_pair | cell_prior     |    30720 |            0.9982 |            0.0000 |            1.0000 | 64.1585 |    0.1166 |   0.5004 |              0.1803 |               1.9189 |
| cold_pair | global_mean    |    30720 |            0.9982 |            0.0000 |            1.0000 | 64.5988 |  nan      | nan      |              0.0358 |               0.3810 |
| cold_pair | pair_prior     |    30720 |            0.9982 |            0.0000 |            1.0000 | 64.5988 |  nan      | nan      |              0.0358 |               0.3810 |
| random    | identity_ridge |    30636 |            1.0000 |            1.0000 |            1.0000 | 51.8108 |    0.5938 |   0.5662 |              0.6240 |               6.2090 |
| random    | additive_prior |    30636 |            1.0000 |            1.0000 |            1.0000 | 52.1664 |    0.5845 |   0.5902 |              0.6195 |               6.1635 |
| random    | pair_prior     |    30636 |            1.0000 |            1.0000 |            1.0000 | 52.5114 |    0.5811 |   0.4894 |              0.6149 |               6.1181 |
| random    | drug_prior     |    30636 |            1.0000 |            1.0000 |            1.0000 | 62.3428 |    0.2807 |   0.4875 |              0.3479 |               3.4617 |
| random    | cell_prior     |    30636 |            1.0000 |            1.0000 |            1.0000 | 63.8285 |    0.1184 |   0.3383 |              0.1906 |               1.8965 |
| random    | global_mean    |    30636 |            1.0000 |            1.0000 |            1.0000 | 64.2811 |  nan      | nan      |              0.0960 |               0.9547 |

## Interpretation

- This is an external dataset diagnostic, not a direct DrugComb-to-NCI transfer experiment.
- It tests whether the same repeated-entity prior shortcut exists in NCI-ALMANAC combo scores.
- Only prior and identity baselines are evaluated because the CellMiner workbook does not provide the same SMILES/expression feature table used by the DrugComb experiments.