"""HTTP transport, upload validation and honest inference status."""

import io
import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from ml.inference import predict_change, predict_crossmodal, predict_single
from ml.inference.assets import ModelUnavailableError, checkpoint_path, model_mode, use_demo
from .model_registry import MODELS
from .schemas import AnalyzeRequest
from .services import analyze
from .validators import validate_request

router = APIRouter(tags=["analysis"])
logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


def is_sar_filename(filename: str) -> bool:
    """A routing hint, never proof of a sensor or alignment."""
    name = Path(filename).stem.lower()
    return name in {"sar", "s1"} or name.startswith(("sar_", "s1_")) or "_sar" in name or "radar" in name


async def _validated_bytes(upload: UploadFile) -> bytes:
    name = upload.filename or ""
    if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(422, f"Unsupported image extension: {name!r}")
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(422, "Image is empty or exceeds 20 MB")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(422, f"Invalid image: {name!r}") from error
    return content


async def _run_uploads(uploads: list[UploadFile], task: str, query: str) -> dict[str, Any]:
    query = query.strip()
    if not 3 <= len(query) <= 500:
        raise HTTPException(422, "Question must contain 3 to 500 characters")
    if len(uploads) != (1 if task == "single_image" else 2):
        raise HTTPException(422, "Incorrect number of images for selected model")
    contents = [await _validated_bytes(upload) for upload in uploads]
    try:
        with tempfile.TemporaryDirectory(prefix="setquery-") as directory:
            paths = []
            for index, (upload, content) in enumerate(zip(uploads, contents)):
                path = Path(directory) / f"image_{index}{Path(upload.filename or '').suffix.lower()}"
                path.write_bytes(content)
                paths.append(str(path))
            if task == "single_image":
                return await run_in_threadpool(predict_single, paths[0], query)
            if task == "change_analysis":
                return await run_in_threadpool(predict_change, paths[0], paths[1], query)
            return await run_in_threadpool(predict_crossmodal, paths[0], paths[1], query)
    except ModelUnavailableError as error:
        raise HTTPException(503, str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        logger.exception("Inference failed")
        raise HTTPException(500, "Inference failed; check server logs") from error


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "setquery", "model_mode": model_mode(), "models": models()["models"]}


@router.get("/models")
def models() -> dict[str, Any]:
    return {"model_mode": model_mode(), "models": [{**item, "checkpoint_present": checkpoint_path(index).is_file(),
                         "available": checkpoint_path(index).is_file() or use_demo(index) or model_mode() == "architecture",
                         "status": "architecture_ready_unverified" if model_mode() == "architecture" else ("untrained_demo" if index == 1 else "baseline_demo") if use_demo(index) else
                         ("checkpoint_present_unverified" if checkpoint_path(index).is_file() else "missing_checkpoint")}
                       for index, item in enumerate(MODELS, start=1)]}


@router.post("/detect-model")
async def detect_model(request: Request) -> dict[str, str]:
    form = await request.form()
    files = [value for _, value in form.multi_items() if hasattr(value, "filename") and hasattr(value, "read")]
    if len(files) not in (1, 2):
        raise HTTPException(422, "Upload one image, or exactly two ordered images")
    for file in files:
        await _validated_bytes(file)
    selected = str(form.get("model", "auto"))
    choices = {
        "single_image": (1, "Single Satellite Image", "Single-Image Understanding", "/analyze/single"),
        "change_analysis": (2, "Before + After Image Pair", "Bi-Temporal Change Analysis", "/analyze/change"),
        "optical_sar": (2, "Optical + SAR Pair", "Optical-SAR Analysis", "/analyze/optical-sar"),
    }
    if selected != "auto":
        if selected not in choices:
            raise HTTPException(422, "Unknown model selection")
        count, input_type, name, route = choices[selected]
        if len(files) != count:
            raise HTTPException(422, f"{name} requires {count} image(s)")
        return {"inputType": input_type, "selectedModel": name, "route": route, "selection": "manual"}
    if len(files) == 1:
        return {"inputType": "Single Satellite Image", "selectedModel": "Single-Image Understanding", "route": "/analyze/single"}
    if any(is_sar_filename(file.filename or "") for file in files):
        return {"inputType": "Optical + SAR Pair", "selectedModel": "Optical-SAR Analysis", "route": "/analyze/optical-sar"}
    return {"inputType": "Before + After Image Pair", "selectedModel": "Bi-Temporal Change Analysis", "route": "/analyze/change"}


@router.post("/analyze/single")
async def analyze_single(image: UploadFile = File(...), query: str = Form("What is in this satellite image?")) -> dict[str, Any]:
    pred = await _run_uploads([image], "single_image", query)
    return {**pred, "annotations": [], "sceneSummary": pred["answer"],
            "detectedFeatures": [{"name": pred["answer"], "confidence": pred["confidence"]}],
            "observations": pred["evidence"], "overallConfidence": pred["confidence"], "aiInsight": pred["answer"]}


@router.post("/analyze/change")
async def analyze_change(image_before: UploadFile = File(...), image_after: UploadFile = File(...)) -> dict[str, Any]:
    pred = await _run_uploads([image_before, image_after], "change_analysis", "Detect changes")
    percent = pred["change_percentage"]
    return {**pred, "metadata": {"beforeDate": "Before", "afterDate": "After"},
            "changeClassification": [{"className": "Pixel-difference mask" if pred.get("label") == "BASELINE" else "Predicted change mask", "area": f"{percent:.2f}%", "severity": "unverified"}],
            "affectedArea": {"totalChanged": f"{percent:.2f}%", "percentChanged": f"{percent:.2f}%", "totalAnalyzed": "100%"},
            "changeSummary": pred["answer"], "aiInsight": pred["answer"], "maskBase64": pred["mask_base64"]}


@router.post("/analyze/optical-sar")
async def analyze_optical_sar(image_optical: UploadFile = File(...), image_sar: UploadFile = File(...)) -> dict[str, Any]:
    pred = await _run_uploads([image_optical, image_sar], "optical_sar", "Evaluate optical-SAR alignment")
    return {**pred, "metadata": {"optical": {"sensor": "Optical", "bands": "RGB"},
                                  "sar": {"sensor": "SAR input", "polarization": "Unverified"}},
            "detectedFeatures": [], "agreement": pred["answer"], "differences": "",
            "analysisResult": pred["answer"], "aiInsight": pred["answer"]}


@router.post("/analyze")
async def analyze_unified(request: Request) -> dict[str, Any]:
    if "multipart/form-data" in request.headers.get("content-type", ""):
        form = await request.form()
        files = [value for _, value in form.multi_items() if hasattr(value, "filename") and hasattr(value, "read")]
        if len(files) not in (1, 2):
            raise HTTPException(422, "Upload one image, or exactly two ordered images")
        query = str(form.get("query") or "Analyze images")
        if len(files) == 1:
            return await _run_uploads(files, "single_image", query)
        if any(is_sar_filename(file.filename or "") for file in files):
            optical, sar = files
            if is_sar_filename(optical.filename or "") and not is_sar_filename(sar.filename or ""):
                optical, sar = sar, optical
            return await _run_uploads([optical, sar], "optical_sar", query)
        return await _run_uploads(files, "change_analysis", query)
    try:
        body = AnalyzeRequest.model_validate(await request.json())
        return analyze(validate_request(body))
    except ModelUnavailableError as error:
        raise HTTPException(503, str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


@router.post("/api/v1/analyze")
def analyze_versioned(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        return analyze(validate_request(request))
    except ModelUnavailableError as error:
        raise HTTPException(503, str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(422, str(error)) from error
