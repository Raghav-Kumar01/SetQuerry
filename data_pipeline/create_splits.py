"""Deterministic dataset split helper."""

from collections.abc import Sequence


def create_splits(items: Sequence[str], train_fraction: float = 0.8) -> dict[str, list[str]]:
    """Split an ordered collection without shuffling it."""
    boundary = round(len(items) * train_fraction)
    return {"train": list(items[:boundary]), "validation": list(items[boundary:])}
