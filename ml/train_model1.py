"""Controlled Model 1 training entry point."""

import argparse
import time
from pathlib import Path

import torch

from ml.common.device import get_device
from ml.common.seed import set_seed
from ml.data.dataloader import create_dataloader
from ml.data.model1_dataset import Model1Dataset, Model1Vocabulary
from ml.models.single_image_model import SingleImageVQAModel
from ml.training.checkpoint import save_checkpoint
from ml.training.config import TrainingConfig
from ml.training.losses import get_classification_loss
from ml.training.model1_checkpoint import build_model1_checkpoint_metadata
from ml.training.optimizers import get_optimizer
from ml.training.trainer import (
    train_classification_epoch,
    validate_classification_epoch,
)


DEFAULT_DATASET_ROOT = str(Path(__file__).resolve().parents[1] / "data" / "raw" / "model1")
DEFAULT_CONFIG_PATH = "ml/configs/base_config.yaml"


def run_training(dataset_root, config_path=DEFAULT_CONFIG_PATH):
    config = TrainingConfig.from_yaml(config_path)
    set_seed(42)
    device = get_device()

    dataset_root = Path(dataset_root)
    train_manifest = dataset_root / "single" / "train" / "manifest.csv"
    val_manifest = dataset_root / "single" / "val" / "manifest.csv"

    vocabulary = Model1Vocabulary.from_manifest(train_manifest)
    train_dataset = Model1Dataset(
        manifest_path=train_manifest,
        dataset_root=dataset_root,
        split="train",
        vocabulary=vocabulary,
    )
    val_dataset = Model1Dataset(
        manifest_path=val_manifest,
        dataset_root=dataset_root,
        split="val",
        vocabulary=vocabulary,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = SingleImageVQAModel(
        vocab_size=vocabulary.question_size,
        num_answers=vocabulary.answer_size,
    ).to(device)
    criterion = get_classification_loss()
    optimizer = get_optimizer(model, config=config)

    best_accuracy = None
    best_epoch = None
    best_path = Path(config.checkpoint_dir) / "model1_best.pt"
    results = []

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        train_result = train_classification_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_result = validate_classification_epoch(
            model,
            val_loader,
            criterion,
            device,
        )
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["accuracy"],
            "evaluated_samples": val_result["evaluated_samples"],
            "excluded_unseen_answers": val_result[
                "excluded_unseen_answers"
            ],
            "elapsed_seconds": elapsed,
        }
        results.append(row)
        print(row, flush=True)

        accuracy = val_result["accuracy"]
        if accuracy is not None and (
            best_accuracy is None or accuracy > best_accuracy
        ):
            best_accuracy = accuracy
            best_epoch = epoch
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric=accuracy,
                checkpoint_path=best_path,
                metadata=build_model1_checkpoint_metadata(
                    vocabulary,
                    model,
                ),
            )

    return {
        "results": results,
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "checkpoint_path": str(best_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    print("MODEL 1 CONTROLLED TRAINING", flush=True)
    print("dataset_root=", args.dataset_root, flush=True)
    print("config=", args.config, flush=True)
    run_training(args.dataset_root, args.config)


if __name__ == "__main__":
    main()
