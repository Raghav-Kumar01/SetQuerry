import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from ml.data.image_utils import load_image_as_tensor, load_image_bytes_as_tensor
from ml.models.change_detection_model import ChangeDetectionModel
from ml.training.checkpoint import load_checkpoint
from ml.inference.assets import dataset_root, require_checkpoint

_MODEL2_SINGLETON = None


def get_cached_change_model():
    global _MODEL2_SINGLETON
    if _MODEL2_SINGLETON is not None:
        return _MODEL2_SINGLETON

    checkpoint_path = require_checkpoint(2)
    model = ChangeDetectionModel()
    load_checkpoint(model, optimizer=None, checkpoint_path=checkpoint_path, device=torch.device("cpu"))
    model.eval()

    _MODEL2_SINGLETON = model
    return _MODEL2_SINGLETON


def resolve_change_tensor(img_input: str, default_modality: str = "A") -> tuple[torch.Tensor, str]:
    """Dynamically resolves any image path, pair ID, or zipped pair from LEVIR-CD.zip."""
    if not img_input:
        raise FileNotFoundError("Model 2 image path is empty")

    img_p = Path(img_input)
    # 1. Direct file on local filesystem
    if img_p.is_file():
        t = load_image_as_tensor(img_p, target_size=(256, 256))
        if t.ndim == 3:
            t = t.unsqueeze(0)
        return t, str(img_p)

    # 2. Check inside LEVIR-CD.zip directly without extraction
    zip_path = dataset_root(2) / "LEVIR-CD.zip"
    if zip_path.is_file():
        raw_name = img_p.stem  # e.g., "train_1_A", "train_1_B", "train_1"
        if raw_name.endswith("_A") or raw_name.endswith("_B"):
            modality = raw_name[-1]
            pair_id = raw_name[:-2]
        else:
            modality = default_modality
            pair_id = raw_name

        candidates = []
        for split in ("train", "val", "test"):
            candidates.append(f"LEVIR-CD/{split}/{modality}/{pair_id}.png")
            candidates.append(f"LEVIR-CD/{split}/{modality}/{img_p.name}")

        try:
            with ZipFile(zip_path, "r") as archive:
                namelist = set(archive.namelist())
                for c in candidates:
                    if c in namelist:
                        t = load_image_bytes_as_tensor(archive.read(c), target_size=(256, 256))
                        if t.ndim == 3:
                            t = t.unsqueeze(0)
                        return t, f"LEVIR-CD.zip:{c}"
        except (OSError, KeyError) as error:
            raise ValueError(f"Cannot read LEVIR-CD archive: {zip_path}") from error

    raise FileNotFoundError(f"Model 2 image is missing: {img_input!r}")


def run_change(image_before: str, image_after: str, query: str) -> dict:
    # Check for identical inputs
    is_identical_input = (image_before and image_after and image_before == image_after)

    tensor_before, src_before = resolve_change_tensor(image_before, default_modality="A")
    if is_identical_input:
        tensor_after = tensor_before.clone()
        src_after = src_before
    else:
        tensor_after, src_after = resolve_change_tensor(image_after, default_modality="B")

    model = get_cached_change_model()
    with torch.no_grad():
        change_logits = model(tensor_before, tensor_after)
        probs = torch.sigmoid(change_logits)
        pred_mask = (probs > 0.5).float()
        change_ratio = pred_mask.mean().item()

    change_percentage = round(change_ratio * 100, 2)
    has_change = change_percentage > 0.5
    decision = "Change detected" if has_change else "No significant change detected"
    answer = f"{decision} ({change_percentage}% area change observed)."
    confidence = round(float(torch.where(pred_mask.bool(), probs, 1.0 - probs).mean().item()), 4)
    mask_image = Image.fromarray((pred_mask[0, 0].byte().cpu().numpy() * 255), mode="L")
    mask_buffer = BytesIO()
    mask_image.save(mask_buffer, format="PNG")

    return {
        "answer": answer,
        "confidence": confidence,
        "confidence_semantics": "mean uncalibrated predicted-pixel probability",
        "change_percentage": change_percentage,
        "decision": decision,
        "mask_shape": list(pred_mask.shape),
        "mask_base64": base64.b64encode(mask_buffer.getvalue()).decode("ascii"),
        "logits_min": round(float(change_logits.min().item()), 4),
        "logits_max": round(float(change_logits.max().item()), 4),
        "src_before": src_before,
        "src_after": src_after,
        "model_version": "v1.0.0-model2-baseline",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-before", default="")
    parser.add_argument("--image-after", default="")
    parser.add_argument("--query", default="What changed?")
    args = parser.parse_args()

    result = run_change(args.image_before, args.image_after, args.query)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
