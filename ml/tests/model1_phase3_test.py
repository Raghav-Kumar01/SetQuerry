import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.data.model1_dataset import Model1Vocabulary
from ml.models.single_image_model import SingleImageVQAModel
from ml.training.checkpoint import load_checkpoint, save_checkpoint
from ml.training.config import TrainingConfig
from ml.training.losses import get_classification_loss
from ml.training.model1_checkpoint import build_model1_checkpoint_metadata
from ml.training.optimizers import get_optimizer
from ml.training.trainer import (
    train_classification_epoch,
    validate_classification_epoch,
)


def test_model1_synthetic_training_contract():
    config = TrainingConfig.from_yaml("ml/configs/base_config.yaml")
    assert config.learning_rate == 0.0001
    assert config.validation_frequency == 1
    assert config.best_checkpoint_metric == "validation_accuracy"
    assert config.best_checkpoint_mode == "max"

    vocabulary = Model1Vocabulary(
        question_to_id={"<pad>": 0, "<unk>": 1, "what": 2},
        answer_to_id={f"answer_{index}": index for index in range(5)},
        max_question_length=8,
    )
    model = SingleImageVQAModel(
        vocab_size=vocabulary.question_size,
        num_answers=vocabulary.answer_size,
    )
    optimizer = get_optimizer(model, config=config)
    assert optimizer.param_groups[0]["lr"] == 0.0001
    criterion = get_classification_loss()

    batch = {
        "image": torch.randn(2, 3, 64, 64),
        "question_ids": torch.ones(2, 8, dtype=torch.long),
        "target": torch.tensor([0, 1], dtype=torch.long),
    }
    loader = DataLoader([batch], batch_size=None)

    train_result = train_classification_epoch(
        model, loader, optimizer, criterion, torch.device("cpu")
    )
    validation_result = validate_classification_epoch(
        model, loader, criterion, torch.device("cpu")
    )
    assert "loss" in train_result
    assert "accuracy" in train_result
    assert "loss" in validation_result
    assert "accuracy" in validation_result

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "best_model1.pt"
        metadata = build_model1_checkpoint_metadata(vocabulary, model)
        save_checkpoint(
            model,
            optimizer,
            epoch=1,
            metric=validation_result["accuracy"],
            checkpoint_path=checkpoint_path,
            metadata=metadata,
        )
        restored_epoch, restored_metric, restored_metadata = load_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            torch.device("cpu"),
            return_metadata=True,
        )
        assert restored_epoch == 1
        assert restored_metric == validation_result["accuracy"]
        assert restored_metadata["answer_to_id"] == metadata["answer_to_id"]


def test_unseen_answers_are_excluded_and_counted():
    vocabulary = Model1Vocabulary(
        question_to_id={"<pad>": 0, "<unk>": 1, "what": 2},
        answer_to_id={"known": 0},
        max_question_length=8,
    )
    model = SingleImageVQAModel(
        vocab_size=vocabulary.question_size,
        num_answers=vocabulary.answer_size,
    )
    criterion = get_classification_loss()

    batch = {
        "image": torch.randn(2, 3, 64, 64),
        "question_ids": torch.ones(2, 8, dtype=torch.long),
        "target": torch.tensor([0, -1], dtype=torch.long),
    }
    result = validate_classification_epoch(
        model,
        DataLoader([batch], batch_size=None),
        criterion,
        torch.device("cpu"),
    )

    assert result["evaluated_samples"] == 1
    assert result["excluded_unseen_answers"] == 1
    assert result["accuracy"] is not None


if __name__ == "__main__":
    test_model1_synthetic_training_contract()
    test_unseen_answers_are_excluded_and_counted()
    print("MODEL 1 PHASE 3 TEST PASSED")
