import sys
from pathlib import Path
import torch

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.model2_dataset import Model2Dataset
from ml.models.change_detection_model import ChangeDetectionModel
from ml.training.checkpoint import load_checkpoint
from ml.inference.assets import dataset_root as configured_dataset_root, require_checkpoint


def main():
    print("=" * 60)
    print("VERIFYING MODEL 2 FORWARD PASS (CHANGE DETECTION)")
    print("=" * 60)

    # 1. Load sample train_1 using Model2Dataset (zero-extraction ZIP loader)
    manifest_path = ROOT / "data_pipeline" / "manifests" / "model2" / "model2_final_manifest.csv"
    dataset_root = configured_dataset_root(2)

    print("Loading pair train_1 from LEVIR-CD.zip...", flush=True)
    dataset = Model2Dataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split="train",
        max_samples=1,
    )
    sample = dataset[0]
    pair_id = sample["pair_id"]
    print(f"Loaded Sample Pair ID: {pair_id}")

    before_tensor = sample["before"].unsqueeze(0)  # [1, 3, H, W]
    after_tensor = sample["after"].unsqueeze(0)    # [1, 3, H, W]
    ground_truth_mask = sample["mask"].unsqueeze(0) # [1, 1, H, W]

    print(f"Before Image shape: {before_tensor.shape}, dtype: {before_tensor.dtype}")
    print(f"After Image shape: {after_tensor.shape}, dtype: {after_tensor.dtype}")
    print(f"Ground Truth Mask shape: {ground_truth_mask.shape}")

    # 2. Instantiate Model 2 and Load Checkpoint
    model = ChangeDetectionModel()
    checkpoint_path = require_checkpoint(2)
    print(f"Loading checkpoint weights from {checkpoint_path}...")
    load_checkpoint(model, optimizer=None, checkpoint_path=checkpoint_path, device=torch.device("cpu"))
    model.eval()

    # 3. Forward Pass
    with torch.no_grad():
        change_logits = model(before_tensor, after_tensor)
        probs = torch.sigmoid(change_logits)
        pred_mask = (probs > 0.5).float()
        change_ratio = pred_mask.mean().item()

    print(f"Change Logits shape: {change_logits.shape}")
    print(f"Predicted Mask shape: {pred_mask.shape}")
    print(f"Predicted Change Percentage: {change_ratio * 100:.2f}%")
    print(f"Logits min: {change_logits.min().item():.4f}, max: {change_logits.max().item():.4f}")
    print(f"Probabilities min: {probs.min().item():.4f}, max: {probs.max().item():.4f}")

    # 4. Assertions
    assert change_logits is not None and change_logits.numel() > 0, "Change logits is empty!"
    expected_shape = (1, 1, *before_tensor.shape[-2:])
    assert change_logits.shape == expected_shape, f"Expected {expected_shape}, got {change_logits.shape}"
    assert pred_mask.shape == expected_shape, f"Expected mask {expected_shape}, got {pred_mask.shape}"

    print("\nMODEL 2 FORWARD PASS: PASS")
    print("=" * 60)
    return change_logits, pred_mask


if __name__ == "__main__":
    main()
