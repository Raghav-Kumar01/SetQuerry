from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    """
    Common training settings for SetQuerry ML models.
    """

    epochs: int = 10

    batch_size: int = 8

    # Authoritative Model 1 baseline, matching ml/configs/base_config.yaml.
    learning_rate: float = 0.0001

    weight_decay: float = 0.0001

    num_workers: int = 0

    checkpoint_dir: str = "ml/checkpoints"

    validation_frequency: int = 1

    best_checkpoint_metric: str = "validation_accuracy"

    best_checkpoint_mode: str = "max"

    @classmethod
    def from_yaml(cls, config_path):
        """Load the common training settings from the project YAML."""

        with Path(config_path).open("r", encoding="utf-8") as file:
            values = yaml.safe_load(file) or {}

        training = values.get("training", {})
        paths = values.get("paths", {})

        return cls(
            epochs=training.get("epochs", cls.epochs),
            batch_size=training.get("batch_size", cls.batch_size),
            learning_rate=training.get("learning_rate", cls.learning_rate),
            weight_decay=training.get("weight_decay", cls.weight_decay),
            num_workers=training.get("num_workers", cls.num_workers),
            checkpoint_dir=paths.get("checkpoints", cls.checkpoint_dir),
            validation_frequency=training.get(
                "validation_frequency", cls.validation_frequency
            ),
            best_checkpoint_metric=training.get(
                "best_checkpoint_metric", cls.best_checkpoint_metric
            ),
            best_checkpoint_mode=training.get(
                "best_checkpoint_mode", cls.best_checkpoint_mode
            ),
        )
