# NCI-ALMANAC External Prior Diagnostic

## Source

- URL: https://discover.nci.nih.gov/cellminer/download/processeddataset/DTP_NCI60_ALMANAC_COMBO_SCORE.zip
- CellMiner metadata: {'CellMiner Address': 'https://discover.nci.nih.gov/cellminer/', 'Download normalized data set': 'Compound activity almanac combo scores a', 'CellMiner Database Version': '2.15', 'Human Genome Version': 'HG-19', 'Date': '09-17-2025'}
- Long table rows: 306,365
- Drugs / pairs / cells: 104 / 5233 / 59

## Test Metrics

| split     | model          |   n_test |   seen_drug_ratio |   seen_pair_ratio |   seen_cell_ratio |    rmse |   pearson |      pcs |   top5pct_precision |   top5pct_enrichment |
|:----------|:---------------|---------:|------------------:|------------------:|------------------:|--------:|----------:|---------:|--------------------:|---------------------:|
| cold_cell | pair_prior     |    31342 |            1.0000 |            1.0000 |            0.0000 | 56.2442 |    0.5858 |   0.7281 |              0.6020 |               5.9731 |
| cold_cell | identity_ridge |    31342 |            1.0000 |            1.0000 |            0.0000 | 56.2451 |    0.5817 |   0.7617 |              0.6033 |               5.9858 |
| cold_cell | additive_prior |    31342 |            1.0000 |            1.0000 |            0.0000 | 56.7310 |    0.5668 |   0.8059 |              0.5855 |               5.8086 |
| cold_cell | drug_prior     |    31342 |            1.0000 |            1.0000 |            0.0000 | 67.1925 |    0.2436 |   0.7281 |              0.3253 |               3.2270 |
| cold_cell | global_mean    |    31342 |            1.0000 |            1.0000 |            0.0000 | 68.8338 |  nan      | nan      |              0.0491 |               0.4872 |
| cold_cell | cell_prior     |    31342 |            1.0000 |            1.0000 |            0.0000 | 68.8338 |  nan      | nan      |              0.0491 |               0.4872 |
| cold_pair | identity_ridge |    30481 |            1.0000 |            0.0000 |            1.0000 | 67.5131 |    0.3016 |   0.6569 |              0.3180 |               3.2367 |
| cold_pair | additive_prior |    30481 |            1.0000 |            0.0000 |            1.0000 | 68.3719 |    0.2858 |   0.7051 |              0.3161 |               3.2167 |
| cold_pair | drug_prior     |    30481 |            1.0000 |            0.0000 |            1.0000 | 68.7886 |    0.2844 |   0.5004 |              0.3167 |               3.2234 |
| cold_pair | cell_prior     |    30481 |            1.0000 |            0.0000 |            1.0000 | 70.3961 |    0.1070 |   0.5004 |              0.1948 |               1.9821 |
| cold_pair | global_mean    |    30481 |            1.0000 |            0.0000 |            1.0000 | 70.8015 |  nan      | nan      |              0.0374 |               0.3804 |
| cold_pair | pair_prior     |    30481 |            1.0000 |            0.0000 |            1.0000 | 70.8015 |  nan      | nan      |              0.0374 |               0.3804 |
| random    | identity_ridge |    30636 |            1.0000 |            1.0000 |            1.0000 | 52.8333 |    0.6223 |   0.5653 |              0.6410 |               6.2719 |
| random    | additive_prior |    30636 |            1.0000 |            1.0000 |            1.0000 | 53.0719 |    0.6126 |   0.5891 |              0.6279 |               6.1442 |
| random    | pair_prior     |    30636 |            1.0000 |            1.0000 |            1.0000 | 53.5701 |    0.6120 |   0.4831 |              0.6292 |               6.1570 |
| random    | drug_prior     |    30636 |            1.0000 |            1.0000 |            1.0000 | 65.1186 |    0.2862 |   0.4839 |              0.3401 |               3.3276 |
| random    | cell_prior     |    30636 |            1.0000 |            1.0000 |            1.0000 | 66.6539 |    0.1172 |   0.3343 |              0.2030 |               1.9863 |
| random    | global_mean    |    30636 |            1.0000 |            1.0000 |            1.0000 | 67.1164 |  nan      | nan      |              0.1018 |               0.9964 |

## Interpretation

- This is an external dataset diagnostic, not a direct DrugComb-to-NCI transfer experiment.
- It tests whether the same repeated-entity prior shortcut exists in NCI-ALMANAC combo scores.
- Only prior and identity baselines are evaluated because the CellMiner workbook does not provide the same SMILES/expression feature table used by the DrugComb experiments.