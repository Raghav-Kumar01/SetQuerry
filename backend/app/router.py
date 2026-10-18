"""Routing logic for the three supported ML tracks."""

from typing import Literal

from .schemas import AnalyzeRequest

Track = Literal["single_image", "change_analysis", "optical_sar"]


def select_track(request: AnalyzeRequest) -> Track:
    """Choose a track from supplied modalities; the router is not an ML model."""
    if request.sar_image:
        return "optical_sar"
    if request.image_after:
        return "change_analysis"
    return "single_image"
