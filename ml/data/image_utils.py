"""Image utility functions for loading and tensor conversion."""
import io
from pathlib import Path
from typing import Union
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def load_image_as_tensor(image_path: Union[str, Path], target_size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    """Loads an image from disk, converts to RGB (stripping any 4th alpha/transparency channels),
    resizes, normalizes, and returns a PyTorch tensor [C, H, W].
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file is missing: {path}")

    try:
        # .convert("RGB") ensures 4-channel RGBA images are safely converted to 3-channel RGB
        with Image.open(path) as source:
            image = source.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        tensor = transform(image)
        return tensor
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid image file: {path}") from error


def load_image_bytes_as_tensor(image_bytes: bytes, target_size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    """Loads image tensor directly from raw byte content."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)
    except (OSError, ValueError) as error:
        raise ValueError("Invalid image bytes") from error


def load_binary_mask(source: Union[str, Path, bytes], target_size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    """Read an LEVIR binary target as one unnormalized channel with nearest resize."""
    try:
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as image:
            mask = image.convert("L").resize(target_size, Image.Resampling.NEAREST)
            return torch.from_numpy((np.asarray(mask) > 0).astype(np.float32)).unsqueeze(0)
    except (OSError, ValueError) as error:
        raise ValueError("Invalid binary change mask") from error
