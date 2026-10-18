"""Application service that connects routing to ML adapters."""

from typing import Any

from ml.inference import predict_change, predict_crossmodal, predict_single

from .router import select_track
from .schemas import AnalyzeRequest


def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """Route a validated request and invoke exactly one model adapter."""
    track = select_track(request)
    if track == "change_analysis":
        return predict_change(request.image, request.image_after, request.query)
    if track == "optical_sar":
        return predict_crossmodal(request.image, request.sar_image, request.query)
    return predict_single(request.image, request.query)
