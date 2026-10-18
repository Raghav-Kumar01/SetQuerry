import torch

from ml.training.metrics import (
    classification_accuracy,
    binary_iou,
    binary_dice
)


def unpack_segmentation_batch(batch):
    """Normalize tuple and Model 2 dictionary batches explicitly."""

    if isinstance(batch, dict):
        required_keys = {
            "before",
            "after",
            "mask",
        }

        if not required_keys.issubset(batch):
            raise KeyError(
                "Segmentation batch must contain before, after, and mask"
            )

        return (
            batch["before"],
            batch["after"],
            batch["mask"],
        )

    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        return batch

    raise TypeError(
        "Segmentation batch must be a dictionary or a 3-item tuple/list"
    )


def unpack_classification_batch(batch):
    """Normalize Model 1 dictionaries and legacy tuple batches."""

    if isinstance(batch, dict):
        required_keys = {
            "image",
            "question_ids",
            "target",
        }

        if not required_keys.issubset(batch):
            raise KeyError(
                "Classification batch must contain image, question_ids, "
                "and target"
            )

        return (
            batch["image"],
            batch["question_ids"],
            batch["target"],
        )

    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        return batch

    raise TypeError(
        "Classification batch must be a dictionary or a 3-item tuple/list"
    )


def move_inputs_to_device(
    inputs,
    device
):
    """
    Move all model inputs to CPU/GPU.
    """

    return [
        item.to(device)
        for item in inputs
    ]


def train_classification_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device
):
    """
    Train one epoch for a
    classification model.

    Works for:
    Model 1
    Model 3
    """

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    for batch in dataloader:

        inputs = unpack_classification_batch(batch)
        *inputs, targets = inputs

        inputs = move_inputs_to_device(
            inputs,
            device
        )

        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(*inputs)

        loss = criterion(
            logits,
            targets
        )

        loss.backward()

        optimizer.step()

        batch_size = targets.size(0)

        total_loss += (
            loss.item()
            *
            batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        total_correct += (
            predictions == targets
        ).sum().item()

        total_samples += batch_size

    average_loss = (
        total_loss
        /
        total_samples
    )

    accuracy = (
        total_correct
        /
        total_samples
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy
    }


def validate_classification_epoch(
    model,
    dataloader,
    criterion,
    device
):
    """
    Validate one epoch for
    classification.
    """

    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    excluded_unseen_answers = 0

    with torch.no_grad():

        for batch in dataloader:

            inputs = unpack_classification_batch(batch)
            *inputs, targets = inputs

            inputs = move_inputs_to_device(
                inputs,
                device
            )

            targets = targets.to(
                device
            )

            known_answers = targets >= 0
            excluded_unseen_answers += (
                (~known_answers).sum().item()
            )

            if not known_answers.any():
                continue

            inputs = [
                item[known_answers]
                for item in inputs
            ]
            targets = targets[known_answers]

            logits = model(*inputs)

            loss = criterion(
                logits,
                targets
            )

            batch_size = (
                targets.size(0)
            )

            total_loss += (
                loss.item()
                *
                batch_size
            )

            batch_accuracy = (
                classification_accuracy(
                    logits,
                    targets
                )
            )

            total_correct += (
                batch_accuracy
                *
                batch_size
            )

            total_samples += (
                batch_size
            )

    average_loss = (
        total_loss / total_samples
        if total_samples > 0
        else None
    )

    accuracy = (
        total_correct / total_samples
        if total_samples > 0
        else None
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "evaluated_samples": total_samples,
        "excluded_unseen_answers": excluded_unseen_answers,
    }


def train_segmentation_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device
):
    """
    Train one epoch for
    Model 2 change detection.
    """

    model.train()

    total_loss = 0.0

    total_samples = 0

    total_iou = 0.0

    total_dice = 0.0

    for batch in dataloader:

        before_image, after_image, targets = (
            unpack_segmentation_batch(batch)
        )

        before_image = before_image.to(
            device
        )

        after_image = after_image.to(
            device
        )

        targets = targets.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            before_image,
            after_image
        )

        loss = criterion(
            logits,
            targets
        )

        loss.backward()

        optimizer.step()

        batch_size = (
            targets.size(0)
        )

        total_loss += (
            loss.item()
            *
            batch_size
        )

        total_iou += (
            binary_iou(
                logits.detach(),
                targets
            )
            *
            batch_size
        )

        total_dice += (
            binary_dice(
                logits.detach(),
                targets
            )
            *
            batch_size
        )

        total_samples += (
            batch_size
        )

    return {
        "loss":
            total_loss / total_samples,

        "iou":
            total_iou / total_samples,

        "dice":
            total_dice / total_samples
    }


def validate_segmentation_epoch(
    model,
    dataloader,
    criterion,
    device
):
    """
    Validate Model 2.
    """

    model.eval()

    total_loss = 0.0

    total_samples = 0

    total_iou = 0.0

    total_dice = 0.0

    with torch.no_grad():

        for batch in dataloader:

            before_image, after_image, targets = (
                unpack_segmentation_batch(batch)
            )

            before_image = (
                before_image.to(device)
            )

            after_image = (
                after_image.to(device)
            )

            targets = targets.to(
                device
            )

            logits = model(
                before_image,
                after_image
            )

            loss = criterion(
                logits,
                targets
            )

            batch_size = (
                targets.size(0)
            )

            total_loss += (
                loss.item()
                *
                batch_size
            )

            total_iou += (
                binary_iou(
                    logits,
                    targets
                )
                *
                batch_size
            )

            total_dice += (
                binary_dice(
                    logits,
                    targets
                )
                *
                batch_size
            )

            total_samples += (
                batch_size
            )

    return {
        "loss":
            total_loss / total_samples,

        "iou":
            total_iou / total_samples,

        "dice":
            total_dice / total_samples
    }
