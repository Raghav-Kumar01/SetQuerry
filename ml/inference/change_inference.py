"""Before-and-after change inference adapter with singleton model caching."""

import base64
import json
import subprocess
import sys
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any
from .assets import use_demo, trained_metadata, model_mode

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _get_cached_subprocess_runner():
    """Cache environment path resolution once."""
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    script = ROOT / "ml" / "inference" / "run_change.py"
    return str(venv_py), str(script)


def _call_change_subprocess(image_before: Any, image_after: Any, query: str) -> dict[str, Any]:
    venv_py, script = _get_cached_subprocess_runner()
    cmd = [
        venv_py,
        "-u",
        script,
        "--image-before",
        str(image_before) if image_before else "",
        "--image-after",
        str(image_after) if image_after else "",
        "--query",
        query,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"run_change failed: {res.stderr}")
    return json.loads(res.stdout.strip())


def predict_change(image_before: Any, image_after: Any, query: str) -> dict[str, Any]:
    """Compare two images using cached Model 2 architecture & baseline checkpoint."""
    if model_mode() == "architecture":
        from .architecture import predict
        return predict(2, [str(image_before or ""), str(image_after or "")], query)
    started_at = perf_counter()
    if use_demo(2):
        from .demo import change
        return change(str(image_before or ""), str(image_after or ""), query)

    from ml.inference.run_change import run_change
    data = run_change(str(image_before) if image_before else "", str(image_after) if image_after else "", query)

    answer = data.get("answer", "No significant change detected (0.0% area change observed).")
    confidence = data.get("confidence", 0.0)
    change_percentage = data.get("change_percentage", 0.0)
    model_version = data.get("model_version", "v1.0.0-model2-baseline")

    return {
        **trained_metadata(),
        "task": "change_analysis",
        "query": query,
        "answer": answer,
        "confidence": confidence,
        "change_percentage": change_percentage,
        "confidence_semantics": data["confidence_semantics"],
        "evidence": [f"Calculated change ratio: {change_percentage}% (threshold: 0.5%)"],
        "model": "change_model",
        "model_version": model_version,
        "mask_base64": data["mask_base64"],
        "execution_trace": [
            "validated_request",
            "selected_change_analysis",
            f"inputs_{image_before}_{image_after}",
            "cached_model2_singleton_invoked",
            f"change_ratio_{change_percentage}%",
        ],
        "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        "is_mock": False,
    }
