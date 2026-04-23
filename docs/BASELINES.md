# Baseline Models

The descriptor baselines live in `src/gnn/baselines/` and provide the traditional-ML comparison point for PFASBench experiments.

`DescriptorExtractor` is the shared feature pipeline. It uses RDKit's public `Descriptors.descList`, supports curated descriptor subsets (`all`, `2d`, `physicochemical`, `topological`), and can append Morgan, RDKit, or MACCS fingerprints. The preferred API is molecule-first through `extract()` and `extract_batch()`, where a single molecule returns a `dict[str, float]` and batches return a labeled `pandas.DataFrame`.

NaN handling is controlled with the `handle_nan` modes:

- `drop`: remove any feature columns that contain missing values in the fitted batch
- `impute_mean`: fill missing values with the column mean
- `impute_median`: fill missing values with the column median
- `impute_zero`: fill missing values with zeros

`extract_from_smiles()` accepts either a single SMILES string or a list. If you need fingerprints directly, use `compute_fingerprints(mol, fp_type="morgan" | "rdkit" | "maccs")`.

For PFASBench-wide exports, use `export_pfasbench_descriptors(...)` to create `data/pfasbench/processed/descriptors.parquet`. Writing parquet files requires an installed parquet engine such as `pyarrow` or `fastparquet`.

`train_rf_baseline`, `predict_rf`, `save_rf_model`, and `load_rf_model` expose the sklearn `RandomForestRegressor` workflow directly while preserving feature-name metadata for interpretability. When multi-target labels are sparse, the helper intentionally falls back to the per-property wrapper so partially labeled PFASBench rows are kept instead of discarded. `train_xgb_baseline`, `predict_xgb`, `save_xgb_model`, and `load_xgb_model` provide a compact XGBoost interface, with native `XGBRegressor` behavior for single-target use and the project wrapper retained for sparse multi-target comparison runs.

`ModelComparison` collects results across one or more named splits and exports CSV, Markdown, LaTeX, and W&B-friendly summaries. The report includes an overall summary table, per-split MAE rankings, and best-model-per-target sections. The default regression metrics are MAE, RMSE, R², Pearson, and Spearman, computed per property and averaged across properties.

For an end-to-end baseline run, use:

```bash
make compare-baselines
```

The CLI supports multiple split presets in one run and accepts either train-in-place aliases or saved model artifacts:

```bash
poetry run python scripts/compare_baselines.py \
  --models rf xgb \
  --splits scaffold random pfas_ood_chain pfas_ood_headgroup \
  --output reports/baseline_comparison.md \
  --wandb-mode disabled
```

Artifact-backed comparisons use `name=<kind>:<path>` specs:

```bash
poetry run python scripts/compare_baselines.py \
  --models SavedRF=rf:artifacts/rf.joblib SavedXGB=xgb:artifacts/xgb_model \
  --splits scaffold random \
  --output reports/baseline_comparison.md
```

Checkpoint-backed GNN adapters can be supplied the same way with `name=gnn:/path/to/model.ckpt` plus `--gnn-loader package.module:load_predictor`. The loader should return an object with either `predict(X)` or `predict_dataset(dataset, split_name=...)`.

Use `--wandb-mode offline` to log locally to Weights & Biases without requiring an online run.

## Checkpoint Ablation Comparison

Use `scripts/compare_ablations.py` to compare multiple GNN checkpoints on the same held-out test split using the canonical evaluation/export pipeline from `src/eval.py`.

```bash
poetry run python scripts/compare_ablations.py \
  --models \
    GIN2D=gnn:/path/to/gin2d.ckpt \
    PaiNN3D=gnn:/path/to/painn3d.ckpt \
  --eval-overrides data=pfasbench_scaffold seed=42 \
  --output-dir reports/ablation_comparison
```

The workflow writes deterministic local artifacts under `--output-dir`:

- `ablation-<split>-<run_id>-comparison.csv` (aggregate regression metrics from `ModelComparison`)
- `ablation-<split>-<run_id>-significance.csv` (paired significance on aligned per-example absolute errors)
- `ablation-<split>-<run_id>-report.md` (summary report with significance table)

For pre-exported prediction payloads from prior eval runs, provide `--prediction-exports name=/path/to/export.json` entries to skip checkpoint re-evaluation.
