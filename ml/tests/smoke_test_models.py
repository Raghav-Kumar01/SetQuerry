import torch

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


def test_single_image_model(device):

    print("\nTesting Model 1...")

    model = SingleImageVQAModel(
        vocab_size=1000,
        num_answers=20
    ).to(device)

    fake_images = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    fake_questions = torch.randint(
        low=1,
        high=1000,
        size=(2, 12)
    ).to(device)

    output = model(
        fake_images,
        fake_questions
    )

    print(
        "Model 1 output shape:",
        output.shape
    )

    assert output.shape == (
        2,
        20
    )

    print(
        "Model 1 PASSED"
    )


def test_change_detection_model(device):

    print("\nTesting Model 2...")

    model = ChangeDetectionModel().to(
        device
    )

    fake_before = torch.randn(
        2,
        3,
        256,
        256
    ).to(device)

    fake_after = torch.randn(
        2,
        3,
        256,
        256
    ).to(device)

    output = model(
        fake_before,
        fake_after
    )

    print(
        "Model 2 output shape:",
        output.shape
    )

    assert output.shape == (
        2,
        1,
        256,
        256
    )

    print(
        "Model 2 PASSED"
    )


def test_optical_sar_model(device):

    print("\nTesting Model 3...")

    model = OpticalSARFusionModel(
        num_classes=10
    ).to(device)

    fake_optical = torch.randn(
        2,
        3,
        224,
        224
    ).to(device)

    fake_sar = torch.randn(
        2,
        2,
        224,
        224
    ).to(device)

    output = model(
        fake_optical,
        fake_sar
    )

    print(
        "Model 3 output shape:",
        output.shape
    )

    assert output.shape == (
        2,
        10
    )

    print(
        "Model 3 PASSED"
    )


def main():

    print("=" * 60)

    print(
        "SETQUERY ML MODEL SMOKE TEST"
    )

    print("=" * 60)

    set_seed(42)

    device = get_device()

    print(
        f"Device: {device}"
    )

    test_single_image_model(
        device
    )

    test_change_detection_model(
        device
    )

    test_optical_sar_model(
        device
    )

    print("\n" + "=" * 60)

    print(
        "ALL THREE ML MODELS PASSED"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()