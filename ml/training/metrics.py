import torch


def classification_accuracy(
    logits,
    targets
):
    """
    Calculate classification accuracy.
    """

    predictions = torch.argmax(
        logits,
        dim=1
    )

    correct = (
        predictions == targets
    ).sum().item()

    total = targets.size(0)

    accuracy = correct / total

    return accuracy


def binary_iou(
    logits,
    targets,
    threshold=0.5
):
    """
    Intersection over Union for
    binary segmentation.
    """

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    targets = targets.float()

    intersection = (
        predictions * targets
    ).sum()

    union = (
        predictions
        +
        targets
        -
        predictions * targets
    ).sum()

    if union.item() == 0:
        return 1.0

    iou = (
        intersection / union
    ).item()

    return iou


def binary_dice(
    logits,
    targets,
    threshold=0.5
):
    """
    Dice score for binary segmentation.
    """

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    targets = targets.float()

    intersection = (
        predictions * targets
    ).sum()

    total = (
        predictions.sum()
        +
        targets.sum()
    )

    if total.item() == 0:
        return 1.0

    dice = (
        2.0 * intersection / total
    ).item()

    return dice