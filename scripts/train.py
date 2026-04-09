"""User-facing Hydra training launcher.

Examples:
    python scripts/train.py model=painn data=pfasbench
    python scripts/train.py --multirun seed=42,43,44 model=painn data=pfasbench
    python scripts/train.py model=painn data=pfasbench trainer.max_epochs=5
"""

from __future__ import annotations

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.train import train
from src.utils import extras, get_metric_value


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> float | None:
    """Run model training through the canonical config entrypoint."""
    extras(cfg)
    metric_dict, _ = train(cfg)
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
