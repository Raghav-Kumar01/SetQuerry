"""Portable inference asset paths and explicit availability errors."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def model_mode() -> str:
    mode = os.getenv("MODEL_MODE", "production").strip().lower()
    if mode not in {"demo", "architecture", "production"}:
        raise ValueError("MODEL_MODE must be demo, architecture or production")
    return mode


def use_demo(number: int) -> bool:
    return model_mode() == "demo" and not checkpoint_path(number).is_file()


def trained_metadata() -> dict:
    return {"model_mode": model_mode(), "inference_kind": "checkpoint", "label": "CHECKPOINT",
            "is_trained": True, "notice": "Checkpoint inference; predictive accuracy and calibration are not verified."}


class ModelUnavailableError(RuntimeError):
    """A required trained checkpoint or vocabulary is unavailable."""


def checkpoint_path(number: int) -> Path:
    configured = os.getenv(f"SETQUERY_MODEL{number}_CHECKPOINT")
    return Path(configured).expanduser() if configured else ROOT / "ml" / "checkpoints" / f"model{number}_baseline.pt"


def require_checkpoint(number: int) -> Path:
    path = checkpoint_path(number)
    if not path.is_file():
        raise ModelUnavailableError(f"Model {number} trained checkpoint is missing: {path}")
    return path


def dataset_root(number: int) -> Path:
    configured = os.getenv(f"SETQUERY_MODEL{number}_DATA_ROOT")
    return Path(configured).expanduser() if configured else ROOT / "data" / "raw" / f"model{number}"


def require_image(path: str) -> Path:
    image = Path(path) if path else Path("")
    if not path or not image.is_file():
        raise FileNotFoundError(f"Image file is missing: {path!r}")
    return image
