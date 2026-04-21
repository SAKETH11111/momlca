# momlca

[![CI](https://github.com/SAKETH11111/momlca/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SAKETH11111/momlca/actions/workflows/ci.yml)

MoML-CA research project scaffold based on lightning-hydra-template.

## Requirements
- Python 3.10+
- PyTorch 2.0+
- Lightning 2.1+
- Hydra 1.3+

## Setup

### pip

```bash
pip install -r requirements.txt
```

### conda

```bash
conda env create -f environment.yaml -n momlca
conda activate momlca
```

## Run

```bash
python src/train.py
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```

## Data Versioning with DVC

This project uses [DVC (Data Version Control)](https://dvc.org) to track datasets. Data files are stored separately from git and pulled on-demand.

### Setup (First Time)

After cloning the repository, pull the data:

```bash
# Install dependencies first
poetry install

# Pull all data files
poetry run dvc pull
```

### Common DVC Commands

```bash
# Pull latest data (after git pull)
poetry run dvc pull

# Check data status
poetry run dvc status

# Push new/updated data (after adding or modifying data files)
# Requires an explicitly configured writable remote.
poetry run dvc push -r <your-writable-remote>

# Add new data file to DVC tracking
poetry run dvc add data/path/to/file.csv
git add data/path/to/file.csv.dvc data/path/to/.gitignore
```

### How It Works

- Raw data: `data/pfasbench/raw/pfasbench.csv` (tracked by DVC)
- Processed data: `data/pfasbench/processed/` (tracked by DVC)
- DVC files (`.dvc`) are committed to git, actual data is stored in remote storage
- Use `dvc pull` after `git pull` to sync data with the current branch

### Remote Storage

The checked-in default remote is a public read-only GitHub mirror so that fresh clones can run `poetry run dvc pull` without machine-local path surgery.

If you need to publish new DVC objects, configure a writable remote locally in `.dvc/config.local` (or with `dvc remote add --local`) and push to that remote explicitly. The checked-in `githubraw` mirror is pull-only and will reject plain `dvc push` writes.

```bash
# Example: local writable override
poetry run dvc remote add --local localremote /path/to/local/storage
poetry run dvc push -r localremote
```

For production/team use, you can also configure a cloud remote locally:

```bash
# Example: Add S3 remote
pip install dvc-s3
poetry run dvc remote add --local s3remote s3://your-bucket/dvc-storage

# Example: Add GCS remote
pip install dvc-gs
poetry run dvc remote add --local gcsremote gs://your-bucket/dvc-storage
```

See `.dvc/config.example` for more remote configuration options.

## Project Structure

- configs/: Hydra configs
- src/: application code
- tests/: pytest suite
- scripts/: utilities
- notebooks/: exploratory notebooks
- data/: local datasets (gitignored)
- logs/: run artifacts (gitignored)

## Notes

- The `.project-root` file is used to detect the repository root.
- Add private environment variables to `.env` (see `.env.example`).
