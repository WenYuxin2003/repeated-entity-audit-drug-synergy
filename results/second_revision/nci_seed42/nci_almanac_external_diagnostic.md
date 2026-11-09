# NCI-ALMANAC External Prior Diagnostic

## Source

- URL: https://discover.nci.nih.gov/cellminer/download/processeddataset/DTP_NCI60_ALMANAC_COMBO_SCORE.zip
- CellMiner metadata: {'CellMiner Address': 'https://discover.nci.nih.gov/cellminer/', 'Download normalized data set': 'Compound activity almanac combo scores a', 'CellMiner Database Version': '2.15', 'Human Genome Version': 'HG-19', 'Date': '09-17-2025'}
- Long table rows: 306,365
- Drugs / pairs / cells: 104 / 5233 / 59

## Test Metrics

| split     | model          |   n_test |   seen_drug_ratio |   seen_pair_ratio |   seen_cell_ratio |    rmse |   pearson |      pcs |   top5pct_precision |   top5pct_enrichment |
|:----------|:---------------|---------:|------------------:|------------------:|------------------:|--------:|----------:|---------:|--------------------:|---------------------:|
| cold_cell | identity_ridge |    31159 |            1.0000 |            1.0000 |            0.0000 | 61.0142 |    0.5672 |   0.7587 |              0.6258 |               5.8103 |
| cold_cell | pair_prior     |    31159 |            1.0000 |            1.0000 |            0.0000 | 61.1613 |    0.5685 |   0.7255 |              0.6322 |               5.8699 |
| cold_cell | additive_prior |    31159 |            1.0000 |            1.0000 |            0.0000 | 61.2244 |    0.5561 |   0.8032 |              0.6142 |               5.7030 |
| cold_cell | drug_prior     |    31159 |            1.0000 |            1.0000 |            0.0000 | 71.7439 |    0.2580 |   0.7255 |              0.3517 |               3.2657 |
| cold_cell | global_mean    |    31159 |            1.0000 |            1.0000 |            0.0000 | 73.5272 |  nan      | nan      |              0.0693 |               0.6436 |
| cold_cell | cell_prior     |    31159 |            1.0000 |            1.0000 |            0.0000 | 73.5272 |  nan      | nan      |              0.0693 |               0.6436 |
| cold_pair | identity_ridge |    30993 |            1.0000 |            0.0000 |            1.0000 | 64.5359 |    0.2492 |   0.6543 |              0.2748 |               2.8983 |
| cold_pair | additive_prior |    30993 |            1.0000 |            0.0000 |            1.0000 | 64.7119 |    0.2421 |   0.7035 |              0.2781 |               2.9323 |
| cold_pair | drug_prior     |    30993 |            1.0000 |            0.0000 |            1.0000 | 65.0939 |    0.2266 |   0.5011 |              0.2839 |               2.9935 |
| cold_pair | cell_prior     |    30993 |            1.0000 |            0.0000 |            1.0000 | 66.1608 |    0.1059 |   0.5011 |              0.1858 |               1.9594 |
| cold_pair | global_mean    |    30993 |            1.0000 |            0.0000 |            1.0000 | 66.5320 |  nan      | nan      |              0.0477 |               0.5035 |
| cold_pair | pair_prior     |    30993 |            1.0000 |            0.0000 |            1.0000 | 66.5320 |  nan      | nan      |              0.0477 |               0.5035 |
| random    | identity_ridge |    30636 |            1.0000 |            1.0000 |            1.0000 | 52.5630 |    0.6028 |   0.5680 |              0.6142 |               6.1839 |
| random    | additive_prior |    30636 |            1.0000 |            1.0000 |            1.0000 | 52.8701 |    0.5927 |   0.5920 |              0.6031 |               6.0722 |
| random    | pair_prior     |    30636 |            1.0000 |            1.0000 |            1.0000 | 53.2280 |    0.5927 |   0.4869 |              0.6136 |               6.1773 |
| random    | drug_prior     |    30636 |            1.0000 |            1.0000 |            1.0000 | 63.6915 |    0.2802 |   0.4862 |              0.3290 |               3.3121 |
| random    | cell_prior     |    30636 |            1.0000 |            1.0000 |            1.0000 | 65.2114 |    0.1138 |   0.3356 |              0.1971 |               1.9846 |
| random    | global_mean    |    30636 |            1.0000 |            1.0000 |            1.0000 | 65.6377 |  nan      | nan      |              0.1044 |               1.0515 |

## Interpretation

- This is an external dataset diagnostic, not a direct DrugComb-to-NCI transfer experiment.
- It tests whether the same repeated-entity prior shortcut exists in NCI-ALMANAC combo scores.
- Only prior and identity baselines are evaluated because the CellMiner workbook does not provide the same SMILES/expression feature table used by the DrugComb experiments.