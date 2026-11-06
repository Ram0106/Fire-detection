# PYRAVISION Dataset Lineage & Methodology (Paper Cohort)

This document details the exact experimental lineage, data pipelines, and validation protocols for the two evaluated architectures in the research paper:
1. **TabResNet** (Deep Tabular Residual Network)
2. **GraphSAGE** (Spatial Inductive Graph Neural Network)

Both models were evaluated on the **exact same 195-event canonical development cohort** under identical **frozen 5-fold spatial-grouped cross-validation**.

```mermaid
flowchart TD
    FIRMS[NASA FIRMS Multi-Satellite Archive<br/>33.8M thermal soundings] --> ENGINE[Spatial & Temporal Aggregation Engine]
    ENGINE --> UNIFIED[tabresnet/dataset/unified_event_registry.csv<br/>195 Canonical Events | 24 F3 Features]
    
    UNIFIED --> SPLITS[frozen_5fold_splits.json<br/>5-Fold Stratified Group K-Fold<br/>143 Spatial Groups | Zero Facility Overlap]
    
    SPLITS --> TABRES[TabResNet Architecture<br/>4 Configurations T1-T4 | 2 Seeds 42, 123<br/>Selected: T4 (128 hidden, 2 blocks, dropout 0.3)]
    TABRES --> TAB_EVAL[tabresnet/evaluation/predictions.csv<br/>Accuracy: 80.51% | Balanced Acc: 81.70% | Macro-F1: 81.21%]
    
    SPLITS --> GSAGE[GraphSAGE Spatial KNN<br/>Dynamic Geodesic KNN K=5 on Coordinates<br/>Split-Local Inductive Graphs | Zero Leakage]
    GSAGE --> GSAGE_EVAL[graphsage/evaluation/predictions.csv<br/>Accuracy: 75.90% | Balanced Acc: 76.78% | Macro-F1: 76.43%]
```

## 1. Canonical 195-Event Benchmark Cohort
- **File**: `tabresnet/dataset/unified_event_registry.csv` and `tabresnet/dataset/canonical_training_dataset.parquet`.
- **Cohort Size**: Exactly 195 unique physical events.
- **Four-Class Composition**:
  - **Industrial Fire**: 60 events
  - **Forest Fire**: 53 events
  - **Agricultural Fire**: 47 events
  - **Persistent / Other Thermal Source**: 35 events
  - **Total**: **195 events**
- **Feature Schema (F3, 24 Features)**:
  1. *Historical Causal Recurrence (5 cont + 3 masks)*: `log_hist_count_365d`, `log_hist_count_90d`, `hist_persistence_ratio`, `log_days_since_obs`, `gini_seasonal_concentration`, `mask_m_hist`, `mask_m_gini`, `mask_m_days`.
  2. *Contemporaneous Acute Radiometry (6 cont + 1 mask)*: `bright_ti4`, `bright_ti5`, `delta_ti4_ti5`, `frp`, `scan`, `track`, `mask_m_acute`.
  3. *Diurnal & Long-Term Stability (6 cont + 3 masks)*: `acute_lst_hours`, `acute_is_solar_night`, `night_recurrence_ratio`, `day_recurrence_ratio`, `day_night_stability`, `active_months_count_365d`, `inter_observation_regularity`, `mask_m_daynight`, `mask_m_regularity`.

## 2. Frozen Spatial-Grouped Cross-Validation Protocol
- **File**: `tabresnet/splits/frozen_5fold_splits.json` and `supporting_data/canonical_split_manifest.json`.
- **Strategy**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
- **Grouping Unit**: `spatial_group_id` (143 unique spatial clusters across India).
- **Leakage Audit**: All 195 events are assigned to validation exactly once across the 5 folds. Pairwise validation intersection between all fold pairs is strictly empty ($\emptyset$). Preprocessing standard scalers were fitted strictly in-fold on training splits; zero validation statistics leaked into training.

## 3. TabResNet Evaluation Lineage
- **Configuration**: Evaluated across 4 trials ($T1$ to $T4$).
- **Selected Configuration ($T4$)**:
  - Hidden dimension: 128
  - Residual blocks: 2
  - Dropout: 0.30
  - Learning rate: 0.0005 (AdamW, weight decay 0.0003)
  - Label smoothing: 0.05, weighted cross-entropy
  - Replicates: 5 folds $	imes$ 2 seeds (`42` and `123`) = 10 runs
- **Verified Metrics**:
  - Accuracy: **80.51%** (157 / 195)
  - Balanced Accuracy: **81.70%**
  - Macro-Precision: **80.84%**
  - Macro-Recall: **81.70%**
  - Macro-F1: **81.21%**
  - Weighted-F1: **80.43%**
- **Confusion Matrix**:
  ```
                 Predicted Class
                 Ind   For   Agr   Per  | Total
  Industrial      46     5     4     5  |   60
  Forest           5    42     6     0  |   53
  Agricultural     5     6    36     0  |   47
  Persistent       0     1     1    33  |   35
  --------------------------------------+------
  Total           56    54    47    38  |  195
  ```

## 4. GraphSAGE Spatial KNN Evaluation Lineage
- **Input Data**: The exact same 195 events and 24 F3 features.
- **Graph Topology**: Constructed using split-local spatial K-Nearest Neighbours ($K=5$) based on event latitude and longitude.
- **Inductive Graph Separation**: For each fold, edges are formed strictly within the training nodes ($tr$) and strictly within the validation nodes ($va$). Cross-partition edges are strictly zero (`cross_partition_edges = 0`), guaranteeing valid inductive evaluation.
- **Selected Configuration ($G\_K5$)**:
  - Mode: Spatial KNN ($K=5$)
  - Hidden dimension: 64
  - Dropout: 0.20
  - Learning rate: 0.001 (AdamW, weight decay 0.0001)
  - Replicates: 5 folds $	imes$ 2 seeds (`42` and `123`) = 10 runs
- **Verified Metrics**:
  - Accuracy: **75.90%** (148 / 195)
  - Balanced Accuracy: **76.78%**
  - Macro-Precision: **76.16%**
  - Macro-Recall: **76.78%**
  - Macro-F1: **76.43%**
  - Weighted-F1: **75.86%**
- **Confusion Matrix**:
  ```
                 Predicted Class
                 Ind   For   Agr   Per  | Total
  Industrial      45     8     3     4  |   60
  Forest           4    37     9     3  |   53
  Agricultural     4     7    36     0  |   47
  Persistent       4     0     1    30  |   35
  --------------------------------------+------
  Total           57    52    49    37  |  195
  ```

## 5. Comparative Evaluation Summary
Both models were evaluated on the identical 195 development events using the identical 5 spatial folds:

| Architecture | Input Representation | Neighborhood / Gating | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **TabResNet (T4)** | 24 F3 Features | None (Pure Tabular Residual) | **80.51%** | **81.70%** | **81.21%** | **80.43%** |
| **GraphSAGE (G_K5)** | 24 F3 Features + Coords | Inductive Spatial KNN ($K=5$) | **75.90%** | **76.78%** | **76.43%** | **75.86%** |

*Note: Models outside the scope of this research paper (Stage 8.52 Compact MLP and Stage 8.73 DualHead Classifier) and un-evaluated temporal exploratory models have been excluded from this release package.*
