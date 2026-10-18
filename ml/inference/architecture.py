"""Forward-only execution of the repository architectures; no baseline substitution."""

import base64
import hashlib
import io
import json
from functools import lru_cache
from threading import Lock
from time import perf_counter

import torch
from PIL import Image

from .assets import ROOT, ModelUnavailableError, checkpoint_path
from .run_single import resolve_single_image
from .run_change import resolve_change_tensor
from .run_crossmodal import resolve_optical_tensor, resolve_sar_tensor
from ml.data.model1_dataset import Model1Vocabulary
from ml.models.single_image_model import SingleImageVQAModel
from ml.models.change_detection_model import ChangeDetectionModel
from ml.models.optical_sar_model import OpticalSARFusionModel

_LOCK = Lock()


@lru_cache(maxsize=6)
def _load(number, checkpoint, stamp, device):
    """Cache by asset identity, independently of demo and production caches."""
    vocabulary = None
    state = None
    if checkpoint:
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = payload.get("model_state_dict", payload)
        except Exception as error:
            raise ModelUnavailableError(f"Cannot load Model {number} checkpoint: {error}") from error
    if number == 1:
        with (ROOT / "ml/data/model1_vocab_cache.json").open(encoding="utf-8") as stream:
            vocabulary = Model1Vocabulary(**json.load(stream))
    with _LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        if number == 1:
            model = SingleImageVQAModel(vocabulary.question_size, vocabulary.answer_size)
        elif number == 2:
            model = ChangeDetectionModel()
        else:
            if state is not None and "fusion_classifier.3.weight" not in state:
                raise ModelUnavailableError("Incompatible Model 3 checkpoint: missing fusion classifier weights")
            classes = state["fusion_classifier.3.weight"].shape[0] if state is not None else 2
            model = OpticalSARFusionModel(num_classes=classes)
        if state is not None:
            try:
                model.load_state_dict(state, strict=True)
            except Exception as error:
                raise ModelUnavailableError(f"Incompatible Model {number} checkpoint: {error}") from error
        model.to(device).eval()
    return model, vocabulary


def predict(number: int, paths: list[str], query: str) -> dict:
    started = perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asset = checkpoint_path(number)
    present = asset.is_file()
    stamp = (asset.stat().st_mtime_ns, asset.stat().st_size) if present else None
    model, vocabulary = _load(number, str(asset.resolve()) if present else None, stamp, device)
    if number == 1:
        inputs = [resolve_single_image(paths[0])[0], torch.tensor([vocabulary.encode_question(query)], dtype=torch.long)]
    elif number == 2:
        inputs = [resolve_change_tensor(path)[0] for path in paths]
    else:
        optical = resolve_optical_tensor(paths[0])[0]
        inputs = [optical, resolve_sar_tensor(paths[1], paths[0], optical)[0]]
    inputs = [value.to(device) for value in inputs]
    if not all(torch.isfinite(value).all() for value in inputs):
        raise ValueError("Image tensors contain non-finite values")
    with torch.inference_mode():
        output = model(*inputs)
        logits = output.float().cpu()
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite logits")
        scores = torch.sigmoid(logits) if number == 2 else torch.softmax(logits, dim=-1)
    label = "CHECKPOINT / TRAINING UNVERIFIED" if present else "UNTRAINED MODEL"
    result = {
        "task": {1: "single_image", 2: "change_analysis", 3: "optical_sar"}[number],
        "query": query, "model_mode": "architecture", "label": label,
        "inference_kind": "neural_network", "is_trained": None if present else False,
        "weights_source": "checkpoint" if present else "random", "is_mock": False,
        "model": type(model).__name__, "model_version": "architecture-v1", "device": device,
        "notice": f"ARCHITECTURE / {label}. Neural-network output; scores are uncalibrated and accuracy is unverified.",
        "confidence_semantics": "uncalibrated neural-network score; not correctness probability",
        "execution_trace": ["validated_inputs", "loaded_existing_architecture", "neural_forward_inference", "decoded_output"],
        "evidence": ["Existing architecture, eval mode, torch.inference_mode; no training performed."],
        "diagnostics": {"initialization": "success", "input_shapes": [list(value.shape) for value in inputs],
                        "output_shape": list(logits.shape), "eval_mode": not model.training,
                        "requires_grad": output.requires_grad, "seed": None if present else 2026,
                        "logits_min": logits.min().item(), "logits_max": logits.max().item(),
                        "logits_mean": logits.mean().item(), "logits_l2": torch.linalg.vector_norm(logits).item(),
                        "logits_sha256": hashlib.sha256(logits.numpy().tobytes()).hexdigest()},
    }
    if number == 2:
        mask = (scores[0, 0] > 0.5).to(torch.uint8)
        buffer = io.BytesIO()
        Image.fromarray(mask.numpy() * 255).save(buffer, format="PNG")
        percent = mask.sum().item() * 100 / mask.numel()
        result.update(answer=f"Neural mask: {percent:.2f}% of pixels exceed the 0.5 sigmoid threshold. {label}; this is not verified change.",
                      confidence=torch.where(mask.bool(), scores[0, 0], 1 - scores[0, 0]).mean().item(),
                      change_percentage=percent, changed_pixels=mask.sum().item(), analyzed_pixels=mask.numel(),
                      mask_shape=list(mask.shape), mask_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
                      logit_samples=logits[0, 0, :2, :4].tolist())
    else:
        index = scores[0].argmax().item()
        result.update(confidence=scores[0, index].item(), predicted_class=index, logits=logits[0].tolist())
        if number == 1:
            answers = {index: answer for answer, index in vocabulary.answer_to_id.items()}
            top_scores, top_indices = scores[0].topk(min(5, len(answers)))
            result.update(answer=answers[index], top_answers=[{"answer": answers[i.item()], "score": s.item()} for s, i in zip(top_scores, top_indices)])
        else:
            result.update(answer=f"Fusion network selected class {index}; uncalibrated softmax score {scores[0, index].item():.6f}. Class semantics and alignment are unverified.",
                          class_scores=scores[0].tolist(), score_semantics="uncalibrated class softmax scores")
            result["evidence"].append("SAR uses the existing two display-channel proxy, not calibrated VV/VH measurements.")
    result["latency_ms"] = round((perf_counter() - started) * 1000, 2)
    return result
