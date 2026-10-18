from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    metric,
    checkpoint_path,
    metadata=None
):
    """
    Save model training state.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "metric": metric
    }

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(
        checkpoint,
        checkpoint_path
    )


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    device,
    return_metadata=False
):
    """
    Load a saved model training state.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    epoch = checkpoint["epoch"]

    metric = checkpoint["metric"]

    if return_metadata:
        return epoch, metric, checkpoint.get("metadata")

    return epoch, metric
