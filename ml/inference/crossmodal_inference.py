"""Optical-SAR cross-modal inference adapter with singleton model caching."""

import json
import subprocess
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
    script = ROOT / "ml" / "inference" / "run_crossmodal.py"
    return str(venv_py), str(script)


def _call_crossmodal_subprocess(optical_image: Any, sar_image: Any, query: str) -> dict[str, Any]:
    venv_py, script = _get_cached_subprocess_runner()
    cmd = [
        venv_py,
        "-u",
        script,
        "--optical",
        str(optical_image) if optical_image else "",
        "--sar",
        str(sar_image) if sar_image else "",
        "--query",
        query,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"run_crossmodal failed: {res.stderr}")
    return json.loads(res.stdout.strip())


def predict_crossmodal(optical_image: Any, sar_image: Any, query: str) -> dict[str, Any]:
    """Fuse optical and SAR inputs using cached Model 3 architecture & calibrated checkpoint."""
    if model_mode() == "architecture":
        from .architecture import predict
        return predict(3, [str(optical_image or ""), str(sar_image or "")], query)
    started_at = perf_counter()
    if use_demo(3):
        from .demo import crossmodal
        return crossmodal(str(optical_image or ""), str(sar_image or ""), query)

    from ml.inference.run_crossmodal import run_crossmodal
    data = run_crossmodal(str(optical_image) if optical_image else "", str(sar_image) if sar_image else "", query)

    answer = data.get("answer", "Cross-modal analysis complete.")
    confidence = data.get("confidence", 0.0)
    decision = data.get("decision", "analyzed")
    align_prob = data.get("alignment_probability", 0.0)
    src_opt = data.get("src_optical", str(optical_image))
    src_sar = data.get("src_sar", str(sar_image))
    model_version = data.get("model_version", "v1.0.0-model3-baseline")

    return {
        **trained_metadata(),
        "task": "optical_sar",
        "query": query,
        "answer": answer,
        "confidence": confidence,
        "confidence_semantics": data["confidence_semantics"],
        "evidence": [
            f"Cross-modal alignment probability: {align_prob}",
            f"Modality decision: {decision}",
            f"Resolved optical: {src_opt}",
            f"Resolved SAR: {src_sar}",
        ],
        "model": "optical_sar_model",
        "model_version": model_version,
        "execution_trace": [
            "validated_request",
            "selected_optical_sar",
            f"input_optical_{optical_image}",
            f"input_sar_{sar_image}",
            "cached_model3_singleton_invoked",
            f"predicted_decision_{decision}",
        ],
        "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        "is_mock": False,
    }
