import torch

from ml.training.config import TrainingConfig


def get_optimizer(
    model,
    learning_rate=None,
    weight_decay=None,
    config=None
):
    """
    Create the optimizer used to update model weights.
    """

    if config is not None:
        if learning_rate is None:
            learning_rate = config.learning_rate
        if weight_decay is None:
            weight_decay = config.weight_decay
    else:
        defaults = TrainingConfig()
        if learning_rate is None:
            learning_rate = defaults.learning_rate
        if weight_decay is None:
            weight_decay = defaults.weight_decay

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    return optimizer
