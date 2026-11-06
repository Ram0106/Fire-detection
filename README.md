# PYRAVISION Research Paper Dataset Release: TabResNet & GraphSAGE

This repository provides the complete, publication-ready dataset package and evaluation outputs supporting the PYRAVISION research paper on satellite-based thermal-event classification.

## Paper Focus & Architecture Scope

This release contains the verified datasets for the two primary architectures evaluated in the paper:
1. **TabResNet** (Deep Tabular Residual Network with 24 engineered features)
2. **GraphSAGE** (Spatial Graph Neural Network with inductive $K=5$ nearest-neighbor edges)

Both models were evaluated on the identical **195-event canonical development cohort** under identical **frozen 5-fold spatial-grouped cross-validation**.

## Directory Layout

```text
github_dataset_release/
│
├── README.md                          # Release overview & quick-start guide
├── DATASET_MANIFEST.csv               # Complete inventory with SHA-256 hashes, sizes & descriptions
├── DATASET_LINEAGE.md                 # Full data engineering pipeline & experiment lineages
├── DATASET_LICENSES.md                # Licensing, NASA attribution & compliance documentation
├── checksums.sha256                   # Cryptographic SHA-256 signatures for every file
│
├── tabresnet/                         # TabResNet Experiment Data & Evaluation Outputs
│   ├── dataset/                       # 195-event canonical cohort (CSV and Parquet format)
│   ├── splits/                        # Frozen 5-fold spatial-grouped cross-validation splits
│   └── evaluation/                    # Predictions, fold metrics, training history & confusion matrix
│
├── graphsage/                         # GraphSAGE Experiment Data & Evaluation Outputs
│   ├── dataset/                       # 195-event node features & coordinates for KNN graph construction
│   ├── splits/                        # Same frozen 5-fold spatial splits for inductive evaluation
│   └── evaluation/                    # Graph predictions, fold metrics, training history & confusion matrix
│
└── supporting_data/                   # Shared Schemas & Manifests
    ├── canonical_feature_schema.json  # 24 F3 feature definitions and physical branches
    ├── canonical_event_manifest.csv   # Event IDs, class labels, and spatial cluster assignments
    ├── canonical_label_mapping.json   # Four-class string-to-integer mapping
    ├── canonical_split_manifest.json  # Detailed 5-fold cluster partition metadata
    └── dataset_report.md              # Dataset status, baseline constraints & documentation
```

## Dataset Summary & Performance

| Metric | TabResNet | GraphSAGE |
| :--- | :---: | :---: |
| **Cohort Size** | 195 events | 195 events |
| **Input Features** | 24 tabular features (F3) | 24 tabular features + coordinates |
| **Graph Construction** | None | Inductive Spatial KNN ($K=5$) |
| **Validation Scheme** | Frozen 5-Fold Spatial Group CV | Frozen 5-Fold Spatial Group CV |
| **Accuracy** | **80.51%** (157 / 195) | **75.90%** (148 / 195) |
| **Balanced Accuracy** | **81.70%** | **76.78%** |
| **Macro-F1 Score** | **81.21%** | **76.43%** |
| **Weighted-F1 Score** | **80.43%** | **75.86%** |

## Verification

To verify the integrity of all staged files:

```bash
sha256sum -c checksums.sha256
```
