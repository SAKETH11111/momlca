# Baseline Models

The descriptor baselines live in `src/gnn/baselines/` and provide the traditional-ML comparison point for PFASBench experiments.

`DescriptorExtractor` is the shared feature pipeline. It uses RDKit's public `Descriptors.descList`, supports curated descriptor subsets (`all`, `2d`, `physicochemical`, `topological`), and can append Morgan, RDKit, or MACCS fingerprints. The preferred API is molecule-first through `extract()` and `extract_batch()`, with `extract_from_smiles()` kept for dataset code that still starts from strings.

`RandomForestBaseline` and `XGBoostBaseline` both train one regressor per property so they can handle partially missing labels and expose property-wise feature importances. The function APIs (`train_rf_baseline`, `train_xgboost_baseline`, plus save/load helpers) are the simplest integration surface for scripts and notebooks.

`ModelComparison` registers models once, evaluates them on one or more named splits, and exports CSV, Markdown, LaTeX, and W&B-friendly summaries. The default regression metrics are MAE, RMSE, R², Pearson, and Spearman, computed per property and averaged across properties.

For an end-to-end baseline run, use:

```bash
make compare-baselines
```

That command runs `scripts/compare_baselines.py`, trains Random Forest and XGBoost on PFASBench descriptor features, and writes artifacts into `outputs/baselines/`.
