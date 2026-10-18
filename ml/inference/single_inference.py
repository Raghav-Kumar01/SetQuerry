"""Single satellite-image inference adapter with singleton model caching."""

import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any
from .assets import use_demo, trained_metadata, model_mode

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _get_cached_subprocess_runner():
    """Cache environment path resolution once."""
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    script = ROOT / "ml" / "inference" / "run_single.py"
    return str(venv_py), str(script)


def _call_single_subprocess(image: Any, query: str) -> dict[str, Any]:
    venv_py, script = _get_cached_subprocess_runner()
    cmd = [venv_py, "-u", script, "--image", str(image) if image else "", "--query", query]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"run_single failed: {res.stderr}")
    return json.loads(res.stdout.strip())


def predict_single(image: Any, query: str) -> dict[str, Any]:
    """Analyze one image using cached Model 1 architecture & baseline checkpoint."""
    if model_mode() == "architecture":
        from .architecture import predict
        return predict(1, [str(image or "")], query)
    started_at = perf_counter()
    if use_demo(1):
        from .demo import single
        return single(str(image or ""), query)

    from ml.inference.run_single import run_single
    data = run_single(str(image) if image else "", query)

    pred_answer = data.get("answer", "unclassified")
    confidence = data.get("confidence", 0.0)
    pred_idx = data.get("pred_idx", 0)
    classes_count = data.get("classes_count", 6359)
    model_version = data.get("model_version", "v1.0.0-model1-baseline")

    return {
        **trained_metadata(),
        "task": "single_image",
        "query": query,
        "answer": pred_answer,
        "confidence": confidence,
        "confidence_semantics": data["confidence_semantics"],
        "evidence": [f"VQA prediction index {pred_idx} (classes: {classes_count})"],
        "model": "single_image_model",
        "model_version": model_version,
        "execution_trace": [
            "validated_request",
            "selected_single_image",
            f"input_image_{image}",
            "cached_model1_singleton_invoked",
            f"predicted_answer_{pred_answer}",
        ],
        "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        "is_mock": False,
    }
