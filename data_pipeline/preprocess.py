"""Preprocessing placeholder with an explicit no-op baseline."""

from pathlib import Path


def prepare_image_reference(image_path: str | Path) -> dict[str, str]:
    """Return metadata for a future raster preprocessing pipeline."""
    path = Path(image_path)
    return {"path": str(path), "status": "not_preprocessed"}
