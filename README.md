# momlca

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
