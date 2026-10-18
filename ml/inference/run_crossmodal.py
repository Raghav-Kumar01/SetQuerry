"""Direct Model 3 execution script with singleton model caching and dynamic optical-SAR resolution."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from ml.models.optical_sar_model import OpticalSARFusionModel
from ml.training.checkpoint import load_checkpoint
from ml.inference.assets import require_checkpoint, require_image

_MODEL3_SINGLETON = None


def get_cached_model3() -> OpticalSARFusionModel:
    global _MODEL3_SINGLETON
    if _MODEL3_SINGLETON is not None:
        return _MODEL3_SINGLETON

    checkpoint_path = require_checkpoint(3)
    model = OpticalSARFusionModel(num_classes=2)
    load_checkpoint(model, optimizer=None, checkpoint_path=checkpoint_path, device=torch.device("cpu"))
    model.eval()
    _MODEL3_SINGLETON = model
    return _MODEL3_SINGLETON


def resolve_optical_tensor(optical_input: str) -> tuple[torch.Tensor, str]:
    """Resolves an optical input to a [1, 3, H, W] tensor."""
    path = require_image(optical_input)
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0), str(path)


def resolve_sar_tensor(sar_input: str, opt_input: str, optical_tensor: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Resolves a SAR input to a [1, 2, H, W] tensor."""
    path = require_image(sar_input)
    # Two display channels are only a visual proxy. Real VV/VH preprocessing
    # and calibration require a documented sensor-specific data contract.
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    t = torch.from_numpy(array[:, :, :2]).permute(2, 0, 1)
    return t.unsqueeze(0), str(path)


def run_crossmodal(optical_input: str, sar_input: str, query: str = "") -> dict:
    opt_tensor, src_opt = resolve_optical_tensor(optical_input)
    sar_tensor, src_sar = resolve_sar_tensor(sar_input, optical_input, opt_tensor)
    model = get_cached_model3()

    # Standardize to 128x128 canonical resolution for feature stability
    opt_in = torch.nn.functional.interpolate(opt_tensor, size=(128, 128), mode="bilinear", align_corners=False)
    sar_in = torch.nn.functional.interpolate(sar_tensor, size=(128, 128), mode="bilinear", align_corners=False)

    with torch.no_grad():
        logits = model(opt_in, sar_in)
        probs = torch.softmax(logits, dim=-1)
        prob_aligned = probs[0, 1].item()
        prob_misaligned = probs[0, 0].item()

    if prob_aligned >= 0.50:
        decision = "Optical and SAR modalities aligned"
        answer = f"Optical and SAR modalities aligned (cross-modal match confidence: {prob_aligned * 100:.2f}%)."
        confidence = round(prob_aligned, 4)
    else:
        decision = "Cross-modal mismatch detected"
        answer = f"Cross-modal mismatch detected between optical and SAR inputs (alignment confidence: {prob_aligned * 100:.2f}%)."
        confidence = round(prob_misaligned, 4)

    return {
        "answer": answer,
        "confidence": confidence,
        "confidence_semantics": "uncalibrated selected-class softmax probability",
        "decision": decision,
        "alignment_probability": round(prob_aligned, 4),
        "src_optical": src_opt,
        "src_sar": src_sar,
        "model_version": "v1.0.0-model3-baseline",
    }


def main():
    parser = argparse.ArgumentParser(description="Run Model 3 Optical-SAR Inference")
    parser.add_argument("--optical", type=str, required=True, help="Path or ID of optical image")
    parser.add_argument("--sar", type=str, required=True, help="Path or ID of SAR image")
    parser.add_argument("--query", type=str, default="", help="User analysis query")
    args = parser.parse_args()

    res = run_crossmodal(args.optical, args.sar, args.query)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
