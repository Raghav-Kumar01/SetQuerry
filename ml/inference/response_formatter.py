"""Shared response shape for model adapters."""

from time import perf_counter
from typing import Any


def mock_response(task: str, query: str, model: str, started_at: float, inputs: list[str]) -> dict[str, Any]:
    """Create an honest, consistent response until trained models are available."""
    return {
        "task": task,
        "query": query,
        "answer": "Mock inference response — no trained model is connected yet.",
        "confidence": 0.0,
        "evidence": [],
        "model": model,
        "model_version": "untrained-baseline",
        "execution_trace": [
            "validated_request",
            f"selected_{task}",
            f"received_{len(inputs)}_image_input(s)",
            "returned_mock_inference",
        ],
        "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        "is_mock": True,
    }
