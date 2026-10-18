"""Public inference adapters for SetQuerry's three ML tracks."""

from .change_inference import predict_change
from .crossmodal_inference import predict_crossmodal
from .single_inference import predict_single

__all__ = ["predict_single", "predict_change", "predict_crossmodal"]
