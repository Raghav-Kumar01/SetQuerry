"""Explicit, image-dependent demo computations. Never used in production mode."""

import base64
import json
from functools import lru_cache
from time import perf_counter
from threading import Lock

import cv2
import numpy as np
from PIL import Image

from .assets import ROOT, require_image

_MODEL_LOCK = Lock()


def _rgb(path: str, size: int = 256) -> np.ndarray:
    with Image.open(require_image(path)) as source:
        return np.asarray(source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)).copy()


def _response(task, query, started, label, answer, **values):
    return {"task": task, "query": query, "answer": answer, "model_mode": "demo",
            "inference_kind": label.lower(), "label": label, "is_trained": False, "is_mock": False,
            "notice": f"DEMO / {label}: computed from uploaded inputs. Not a trained prediction; accuracy is not established.",
            "confidence": None, "confidence_semantics": "not a calibrated probability",
            "model": task, "model_version": f"demo-{label.lower()}-v1",
            "execution_trace": ["decoded_uploaded_images", f"computed_{task}_{label.lower()}"],
            "latency_ms": round((perf_counter() - started) * 1000, 2), **values}


@lru_cache(maxsize=1)
def _untrained_vqa():
    import torch
    from ml.data.model1_dataset import Model1Vocabulary
    from ml.models.single_image_model import SingleImageVQAModel

    with (ROOT / "ml/data/model1_vocab_cache.json").open(encoding="utf-8") as stream:
        data = json.load(stream)
    vocabulary = Model1Vocabulary(**data)
    # A separate cached model prevents demo weights from entering production caches.
    with _MODEL_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        model = SingleImageVQAModel(vocabulary.question_size, vocabulary.answer_size).eval()
    return model, vocabulary, {index: answer for answer, index in vocabulary.answer_to_id.items()}


def single(image: str, query: str) -> dict:
    import torch
    from ml.data.image_utils import load_image_as_tensor

    started = perf_counter()
    tensor = load_image_as_tensor(require_image(image)).unsqueeze(0)
    if not 3 <= len(query.strip()) <= 500:
        raise ValueError("Question must contain 3 to 500 characters")
    model, vocabulary, answers = _untrained_vqa()
    tokens = torch.tensor([vocabulary.encode_question(query)], dtype=torch.long)
    with torch.inference_mode():
        logits = model(tensor, tokens)
        probabilities = torch.softmax(logits, dim=-1)[0]
        scores, indices = probabilities.topk(5)
    best = indices[0].item()
    return _response("single_image", query, started, "UNTRAINED", answers[best],
                     confidence=float(scores[0]), confidence_semantics="untrained answer softmax score",
                     evidence=["Existing ResNet18 + GRU architecture, seed 2026, random weights.",
                               f"Decoded answer class {best} from {len(answers)} vocabulary entries."],
                     top_answers=[{"answer": answers[i.item()], "score": float(s)} for s, i in zip(scores, indices)],
                     diagnostics={"logits_l2": float(torch.linalg.vector_norm(logits)), "seed": 2026})


def change(before: str, after: str, query: str) -> dict:
    started = perf_counter()
    first, second = _rgb(before), _rgb(after)
    difference = cv2.absdiff(first, second).max(axis=2)
    _, mask = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    changed = int(cv2.countNonZero(mask))
    percentage = 100.0 * changed / mask.size
    success, png = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError("Could not encode change mask")
    return _response("change_analysis", query, started, "BASELINE",
                     f"Pixel-difference baseline: {percentage:.2f}% of resized pixels differ by more than 25/255 in at least one color channel.",
                     evidence=["OpenCV absolute RGB difference at 256 x 256; threshold 25/255.",
                               "Assumes images depict the same registered area; lighting and motion can also trigger changes."],
                     change_percentage=percentage, changed_pixels=changed, analyzed_pixels=int(mask.size),
                     mask_shape=list(mask.shape), mask_base64=base64.b64encode(png).decode("ascii"),
                     mean_absolute_difference=float(difference.mean()))


def _structure(path: str, sar: bool) -> np.ndarray:
    # Preserve numeric grayscale SAR rasters; RGB input is treated as display imagery.
    with Image.open(require_image(path)) as source:
        gray = np.asarray(source.convert("F"), dtype=np.float32)
    if not np.isfinite(gray).all():
        raise ValueError("Image contains non-finite pixel values")
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    low, high = np.percentile(gray, (1, 99))
    if high <= low:
        low, high = float(gray.min()), float(gray.max())
    normalized = np.clip((gray - low) / max(float(high - low), 1e-8), 0, 1)
    if sar:
        normalized = np.log1p(9 * normalized) / np.log(10)
    smooth = cv2.GaussianBlur(normalized.astype(np.float32), (5, 5), 1.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy).reshape(-1).astype(np.float64)


def crossmodal(optical: str, sar: str, query: str) -> dict:
    started = perf_counter()
    first, second = _structure(optical, False), _structure(sar, True)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    score = float(np.clip(np.dot(first, second) / denominator, 0, 1)) if denominator > 1e-12 else None
    answer = (f"Structural similarity baseline: {score:.4f} on a 0–1 scale. This compares edge patterns, not calibrated alignment probability."
              if score is not None else "Structural similarity is undefined because at least one image has no measurable edges.")
    return _response("optical_sar", query, started, "BASELINE", answer,
                     similarity_score=score, score_semantics="cosine similarity of spatial gradient magnitudes",
                     evidence=["Grayscale percentile scaling, SAR log compression, Gaussian smoothing, Sobel gradients.",
                               "Assumes common coverage and orientation. No registration, sensor calibration, or alignment classification is performed."],
                     diagnostics={"optical_edge_energy": float(np.linalg.norm(first)), "sar_edge_energy": float(np.linalg.norm(second))})
