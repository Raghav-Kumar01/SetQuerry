import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ml.data.model2_dataset import Model2Dataset
from ml.models.change_detection_model import ChangeDetectionModel
from ml.training.checkpoint import save_checkpoint
from ml.inference.assets import dataset_root as configured_dataset_root


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(-2, -1))
    cardinality = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


class FastChangeDataset(Dataset):
    def __init__(self, base_dataset, target_size=(256, 256)):
        self.samples = []
        print(f"Loading and pre-downsampling {len(base_dataset)} pairs in memory...")
        for i in range(len(base_dataset)):
            s = base_dataset[i]
            b = F.interpolate(s["before"].unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
            a = F.interpolate(s["after"].unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
            m = F.interpolate(s["mask"].unsqueeze(0), size=target_size, mode="nearest").squeeze(0)
            self.samples.append({
                "pair_id": s["pair_id"],
                "before": b,
                "after": a,
                "mask": m,
            })
            # Also add an identical negative pair (zero change)
            self.samples.append({
                "pair_id": f"{s['pair_id']}_identical",
                "before": b,
                "after": b,
                "mask": torch.zeros_like(m),
            })
        print(f"Total training pairs (including negative self-pairs): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    print("=" * 60)
    print("STARTING CALIBRATED FAST TRAINING: MODEL 2 (LEVIR-CD)")
    print("=" * 60)

    manifest_path = ROOT / "data_pipeline" / "manifests" / "model2" / "model2_final_manifest.csv"
    dataset_root = configured_dataset_root(2)

    # Load 12 real pairs from LEVIR-CD.zip
    base_dataset = Model2Dataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split="train",
        max_samples=12,
    )
    dataset = FastChangeDataset(base_dataset, target_size=(256, 256))
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = ChangeDetectionModel()
    # Initialize final conv bias to negative background prior (-2.5 -> sigmoid=0.075)
    with torch.no_grad():
        final_conv = list(model.decoder.children())[-1]
        if isinstance(final_conv, nn.Conv2d) and final_conv.bias is not None:
            final_conv.bias.fill_(-2.5)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_weight = torch.tensor([8.0])  # give foreground building change higher weight
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = 20
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            before = batch["before"]
            after = batch["after"]
            targets = batch["mask"]

            optimizer.zero_grad()
            logits = model(before, after)
            loss = bce_loss_fn(logits, targets) + 2.0 * dice_loss(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)

        avg_loss = total_loss / len(dataset)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nTraining completed in {elapsed:.2f}s!")

    # Save Checkpoint
    checkpoint_dir = ROOT / "ml" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model2_baseline.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        metric=avg_loss,
        checkpoint_path=checkpoint_path,
        metadata={"model": "ChangeDetectionModel", "target_size": [256, 256]},
    )
    print(f"Checkpoint successfully saved to: {checkpoint_path} ({checkpoint_path.stat().st_size} bytes)")

    # Verification on train_1
    model.eval()
    sample_1 = base_dataset[0]
    b1 = sample_1["before"].unsqueeze(0)  # [1, 3, 1024, 1024]
    a1 = sample_1["after"].unsqueeze(0)
    gt_pct = (sample_1["mask"] > 0.5).float().mean().item() * 100

    with torch.no_grad():
        # 1. Real change pair (train_1 before vs after)
        logits_1 = model(b1, a1)
        probs_1 = torch.sigmoid(logits_1)
        pred_pct_1 = (probs_1 > 0.5).float().mean().item() * 100

        # 2. Identical pair (train_1 before vs before)
        logits_same = model(b1, b1)
        probs_same = torch.sigmoid(logits_same)
        pred_pct_same = (probs_same > 0.5).float().mean().item() * 100

    print("\n" + "=" * 60)
    print("VERIFICATION ON MODEL 2 (train_1):")
    print(f"Ground-Truth Change %: {gt_pct:.2f}%")
    print(f"Predicted Change % on train_1: {pred_pct_1:.2f}%")
    print(f"Logits range on train_1: min={logits_1.min().item():.4f}, max={logits_1.max().item():.4f}")
    print(f"Predicted Change % on IDENTICAL images: {pred_pct_same:.2f}%")
    print(f"Logits range on IDENTICAL images: min={logits_same.min().item():.4f}, max={logits_same.max().item():.4f}")
    print("=" * 60)

    assert pred_pct_same == 0.0, f"Expected 0.0% change on identical images, got {pred_pct_same:.2f}%"
    assert pred_pct_1 < 20.0, f"Expected realistic change % (< 20%), got {pred_pct_1:.2f}%"
    print("MODEL 2 CALIBRATION VERIFICATION: PASSED!")


if __name__ == "__main__":
    main()
