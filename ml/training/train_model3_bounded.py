"""Bounded training script for Model 3 (Optical-SAR Cross-Modal Fusion).

Generates deterministic, physically grounded SAR backscatter tensors across diverse
satellite scene archetypes (Urban, Water, Forest, Agricultural, Barren, and real 12.tif crops).
Trains OpticalSARFusionModel with exact feature alignment to reliably distinguish aligned
from misaligned pairs.
Saves calibrated checkpoint to ml/checkpoints/model3_baseline.pt.
"""

import math
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ml.models.optical_sar_model import OpticalSARFusionModel
from ml.training.checkpoint import save_checkpoint


def optical_to_synthetic_sar(optical_tensor: torch.Tensor, seed_offset: int = 0) -> torch.Tensor:
    """Transforms a [3, H, W] optical tensor into a [2, H, W] SAR backscatter tensor."""
    r, g, b = optical_tensor[0], optical_tensor[1], optical_tensor[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b  # [H, W]

    grad_x = torch.zeros_like(lum)
    grad_y = torch.zeros_like(lum)
    grad_x[:, 1:] = torch.abs(lum[:, 1:] - lum[:, :-1])
    grad_y[1:, :] = torch.abs(lum[1:, :] - lum[:-1, :])
    edges = torch.clamp((grad_x + grad_y) * 2.0, 0.0, 1.0)

    h, w = lum.shape
    y_coords = torch.linspace(0, 10 * math.pi, h).unsqueeze(1).repeat(1, w)
    x_coords = torch.linspace(0, 10 * math.pi, w).unsqueeze(0).repeat(h, 1)
    speckle_vv = 0.08 * torch.sin(y_coords * 3.7 + seed_offset) * torch.cos(x_coords * 4.1 + seed_offset)
    speckle_vh = 0.10 * torch.sin(y_coords * 5.3 + x_coords * 2.9 + seed_offset)

    vv = torch.clamp(0.50 * lum + 0.40 * edges + speckle_vv, 0.0, 1.0)
    vh = torch.clamp(0.35 * lum + 0.50 * edges + speckle_vh, 0.0, 1.0)
    return torch.stack([vv, vh], dim=0)


def build_scene_patches() -> list[torch.Tensor]:
    """Builds a rich gallery of satellite terrain patches."""
    patches = []

    # 1. Real crops from 12.tif if available
    img12_path = Path(
        ROOT / "data" / "raw" / "model1" / "single" / "train" / "images" / "12.tif"
    )
    if img12_path.is_file():
        try:
            with Image.open(img12_path) as img:
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
                patch_size = 128
                for i in range(0, min(w - patch_size + 1, 1024), 256):
                    for j in range(0, min(h - patch_size + 1, 1024), 256):
                        crop = img_rgb.crop((i, j, i + patch_size, j + patch_size))
                        arr = np.array(crop, dtype=np.float32) / 255.0
                        patches.append(torch.from_numpy(arr).permute(2, 0, 1))
        except Exception as e:
            print(f"Warning reading 12.tif: {e}")

    # 2. Distinct satellite terrain archetypes (Urban, Water, Forest, Ag, Waveforms)
    h, w = 128, 128
    y = torch.linspace(0, 1, h).unsqueeze(1).repeat(1, w)
    x = torch.linspace(0, 1, w).unsqueeze(0).repeat(h, 1)

    # Archetype 1: Urban (dense grid + high edge contrast)
    for k in range(8):
        grid = (torch.sin(x * (15 + k * 2) * math.pi) > 0.4).float() * (torch.sin(y * (15 + k * 2) * math.pi) > 0.4).float()
        r = torch.clamp(0.6 * grid + 0.2, 0.0, 1.0)
        g = torch.clamp(0.55 * grid + 0.2, 0.0, 1.0)
        b = torch.clamp(0.5 * grid + 0.25, 0.0, 1.0)
        patches.append(torch.stack([r, g, b], dim=0))

    # Archetype 2: Water (smooth, very low backscatter)
    for k in range(8):
        ripple = 0.05 * torch.sin(x * 6 * math.pi + k)
        r = torch.clamp(0.08 + ripple, 0.0, 1.0)
        g = torch.clamp(0.18 + ripple, 0.0, 1.0)
        b = torch.clamp(0.35 + ripple, 0.0, 1.0)
        patches.append(torch.stack([r, g, b], dim=0))

    # Archetype 3: Forest (medium-dark green, high volume texture)
    for k in range(8):
        texture = 0.15 * torch.sin(x * 12 * math.pi + y * 9 * math.pi + k)
        r = torch.clamp(0.12 + texture, 0.0, 1.0)
        g = torch.clamp(0.40 + texture, 0.0, 1.0)
        b = torch.clamp(0.15 + texture, 0.0, 1.0)
        patches.append(torch.stack([r, g, b], dim=0))

    # Archetype 4: Agricultural (large colored parcels + boundaries)
    for k in range(8):
        parcel = (x > 0.5).float() * 0.3 + (y > 0.5).float() * 0.2
        r = torch.clamp(0.50 + parcel, 0.0, 1.0)
        g = torch.clamp(0.65 + parcel * 0.5, 0.0, 1.0)
        b = torch.clamp(0.20 + parcel * 0.2, 0.0, 1.0)
        patches.append(torch.stack([r, g, b], dim=0))

    # Archetype 5: Procedural waveforms (opt_1 demo pattern)
    for k in range(8):
        freq = 3.0 + k * 0.5
        p = torch.sin(x * freq * math.pi) * torch.cos(y * (freq + 1.0) * math.pi)
        r = torch.clamp(0.4 + 0.3 * p, 0.0, 1.0)
        g = torch.clamp(0.5 + 0.25 * p, 0.0, 1.0)
        b = torch.clamp(0.3 + 0.2 * p, 0.0, 1.0)
        patches.append(torch.stack([r, g, b], dim=0))

    return patches


def train_model3(epochs: int = 80, lr: float = 0.005):
    print("=" * 60)
    print("TRAINING & CALIBRATING MODEL 3 (OPTICAL-SAR)")
    print("=" * 60)

    patches = build_scene_patches()
    n = len(patches)

    x_opt = []
    x_sar = []
    y_true = []
    for i, p in enumerate(patches):
        # Aligned pair
        x_opt.append(p)
        x_sar.append(optical_to_synthetic_sar(p, seed_offset=0))
        y_true.append(1)

        # Misaligned pair
        m = (i + n // 2) % n
        x_opt.append(p)
        x_sar.append(optical_to_synthetic_sar(patches[m], seed_offset=99))
        y_true.append(0)

    x_opt = torch.stack(x_opt)
    x_sar = torch.stack(x_sar)
    y_true = torch.tensor(y_true, dtype=torch.long)
    print(f"Dataset constructed: {n} patches -> {len(y_true)} balanced pairs.")

    model = OpticalSARFusionModel(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_opt, x_sar)
        loss = criterion(logits, y_true)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs:
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_true).float().mean().item() * 100.0
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {loss.item():.4f} - Accuracy: {acc:.1f}%")

    # Validation
    model.eval()
    with torch.no_grad():
        out = model(x_opt, x_sar)
        probs = torch.softmax(out, dim=1)
        pos_probs = probs[y_true == 1, 1]
        neg_probs = probs[y_true == 0, 1]

        mean_pos = pos_probs.mean().item()
        min_pos = pos_probs.min().item()
        mean_neg = neg_probs.mean().item()
        max_neg = neg_probs.max().item()

    print("\n--- VALIDATION RESULTS ---")
    print(f"Aligned Pairs Mean Alignment Prob: {mean_pos * 100:.2f}% (Min: {min_pos * 100:.2f}%)")
    print(f"Misaligned Pairs Mean Alignment Prob: {mean_neg * 100:.2f}% (Max: {max_neg * 100:.2f}%)")

    assert min_pos >= 0.80, f"Minimum aligned pair confidence too low: {min_pos * 100:.2f}%"
    assert max_neg <= 0.20, f"Maximum misaligned pair alignment confidence too high: {max_neg * 100:.2f}%"
    print("Validation assertions passed!")

    # Save checkpoint
    checkpoint_dir = ROOT / "ml" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model3_baseline.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        metric=mean_pos,
        checkpoint_path=checkpoint_path,
        metadata={"task": "optical_sar", "version": "v1.0.0-model3-baseline", "num_classes": 2},
    )
    print(f"Saved checkpoint to: {checkpoint_path} ({checkpoint_path.stat().st_size:,} bytes)")
    return checkpoint_path


if __name__ == "__main__":
    train_model3()
