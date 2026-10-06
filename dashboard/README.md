# Power BI Dashboard Data

These CSVs provide Power BI with the original CM1 software metrics, calculated dataset summaries, stored model evaluation results, and feature importances already recorded in the executed evaluation notebook. The generator does not train models or alter the source dataset.

## Files

- `data/cm1_dashboard_data.csv`: all source CM1 rows and metrics, plus `Defect_Status` (`Defective` or `Non-Defective`). Original target values remain in `defects`.
- `data/dataset_summary.csv`: module counts, calculated defect rate, exact duplicate count, and feature count.
- `data/model_comparison.csv`: the four models' stored holdout metrics; `Selected` identifies the model chosen by training-only cross-validation.
- `data/feature_importance.csv`: stored top feature importances for Random Forest and XGBoost, with rank calculated from their descending stored values.

All files are UTF-8 CSVs without an index column. The summary's duplicate count reflects preprocessing, while the CM1 dashboard data retains the original rows for faithful source-data exploration.

## Regenerate

Run from the project root:

```powershell
python dashboard/generate_dashboard_data.py
```

The script reads `data/cm1.csv` and the executed outputs in `notebooks/05_model_evaluation.ipynb`. It raises an error rather than filling in missing evaluation or feature-importance values if those saved notebook outputs are unavailable.

## Recommended Pages

### Page 1: QA Defect Overview

- Total Modules, Defective Modules, Non-Defective Modules, and Defect Rate cards
- Defective vs Non-Defective chart
- Software metric distributions

### Page 2: ML Model Performance

- Accuracy, Precision, Recall, F1, and ROC-AUC comparisons
- Selected model indicator

### Page 3: Defect Risk & Feature Analysis

- Feature importance by model and rank
- Important software metrics
- Defect probability or risk analysis only when backed by a separate set of actual model predictions; no per-module probability output is included in these CSVs.