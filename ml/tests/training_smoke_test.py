from pathlib import Path

import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from ml.common.device import get_device
from ml.common.seed import set_seed

from ml.models.single_image_model import (
    SingleImageVQAModel
)

from ml.models.change_detection_model import (
    ChangeDetectionModel
)

from ml.models.optical_sar_model import (
    OpticalSARFusionModel
)

from ml.training.losses import (
    get_classification_loss,
    get_segmentation_loss
)

from ml.training.optimizers import (
    get_optimizer
)

from ml.training.trainer import (
    train_classification_epoch,
    validate_classification_epoch,
    train_segmentation_epoch,
    validate_segmentation_epoch
)

from ml.training.checkpoint import (
    save_checkpoint,
    load_checkpoint
)


def test_model_1(device):

    print("\n" + "-" * 60)

    print(
        "MODEL 1 TRAINING TEST"
    )

    model = SingleImageVQAModel(
        vocab_size=1000,
        num_answers=20
    ).to(device)

    images = torch.randn(
        4,
        3,
        64,
        64
    )

    questions = torch.randint(
        low=1,
        high=1000,
        size=(4, 12)
    )

    answers = torch.randint(
        low=0,
        high=20,
        size=(4,)
    )

    dataset = TensorDataset(
        images,
        questions,
        answers
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    criterion = (
        get_classification_loss()
    )

    optimizer = get_optimizer(
        model
    )

    train_results = (
        train_classification_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device
        )
    )

    validation_results = (
        validate_classification_epoch(
            model,
            loader,
            criterion,
            device
        )
    )

    print(
        "Train:",
        train_results
    )

    print(
        "Validation:",
        validation_results
    )

    print(
        "MODEL 1 TRAINING TEST PASSED"
    )


def test_model_2(device):

    print("\n" + "-" * 60)

    print(
        "MODEL 2 TRAINING TEST"
    )

    model = (
        ChangeDetectionModel()
        .to(device)
    )

    before = torch.randn(
        4,
        3,
        64,
        64
    )

    after = torch.randn(
        4,
        3,
        64,
        64
    )

    masks = torch.randint(
        low=0,
        high=2,
        size=(
            4,
            1,
            64,
            64
        )
    ).float()

    dataset = TensorDataset(
        before,
        after,
        masks
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    criterion = (
        get_segmentation_loss()
    )

    optimizer = get_optimizer(
        model
    )

    train_results = (
        train_segmentation_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device
        )
    )

    validation_results = (
        validate_segmentation_epoch(
            model,
            loader,
            criterion,
            device
        )
    )

    print(
        "Train:",
        train_results
    )

    print(
        "Validation:",
        validation_results
    )

    print(
        "MODEL 2 TRAINING TEST PASSED"
    )


def test_model_3(device):

    print("\n" + "-" * 60)

    print(
        "MODEL 3 TRAINING TEST"
    )

    model = OpticalSARFusionModel(
        num_classes=10
    ).to(device)

    optical = torch.randn(
        4,
        3,
        64,
        64
    )

    sar = torch.randn(
        4,
        2,
        64,
        64
    )

    labels = torch.randint(
        low=0,
        high=10,
        size=(4,)
    )

    dataset = TensorDataset(
        optical,
        sar,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    criterion = (
        get_classification_loss()
    )

    optimizer = get_optimizer(
        model
    )

    train_results = (
        train_classification_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device
        )
    )

    validation_results = (
        validate_classification_epoch(
            model,
            loader,
            criterion,
            device
        )
    )

    print(
        "Train:",
        train_results
    )

    print(
        "Validation:",
        validation_results
    )

    test_checkpoint(
        model,
        optimizer,
        device
    )

    print(
        "MODEL 3 TRAINING TEST PASSED"
    )


def test_checkpoint(
    model,
    optimizer,
    device
):

    checkpoint_path = Path(
        "ml/checkpoints/"
        "phase3_smoke_test.pt"
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        metric=0.50,
        checkpoint_path=checkpoint_path
    )

    epoch, metric = load_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        device=device
    )

    assert epoch == 1

    assert metric == 0.50

    print(
        "Checkpoint save/load PASSED"
    )

    if checkpoint_path.exists():

        checkpoint_path.unlink()


def main():

    print("=" * 60)

    print(
        "SETQUERY PHASE 3"
    )

    print(
        "TRAINING PIPELINE SMOKE TEST"
    )

    print("=" * 60)

    set_seed(42)

    device = get_device()

    print(
        f"Device: {device}"
    )

    test_model_1(device)

    test_model_2(device)

    test_model_3(device)

    print("\n" + "=" * 60)

    print(
        "PHASE 3 SUCCESSFUL"
    )

    print(
        "ALL TRAINING COMPONENTS PASSED"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()