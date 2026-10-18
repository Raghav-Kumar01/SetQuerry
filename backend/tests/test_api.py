"""API contract tests that do not claim absent checkpoints are trained models."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app

client = TestClient(app)


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_distinguishes_api_from_models():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert len(body["models"]) == 3
    assert all("available" in model and "status" in model for model in body["models"])


def test_upload_validation_and_routing():
    good = png_bytes()
    one = client.post("/api/detect-model", files={"image": ("optical.png", good, "image/png")})
    assert one.status_code == 200
    assert one.json()["route"] == "/analyze/single"
    pair = client.post("/api/detect-model", files={"before": ("optical.png", good, "image/png"), "after": ("sar_1.png", good, "image/png")})
    assert pair.status_code == 200
    assert pair.json()["route"] == "/analyze/optical-sar"
    bad = client.post("/api/detect-model", files={"image": ("fake.png", b"not an image", "image/png")})
    assert bad.status_code == 422


def test_missing_reference_is_not_inferred():
    response = client.post("/api/v1/analyze", json={"query": "What is visible?", "image": "missing-file.png"})
    assert response.status_code == 422
    assert "missing" in response.json()["detail"].lower()


def test_valid_upload_requires_checkpoint():
    response = client.post("/api/analyze/single", files={"image": ("scene.png", png_bytes(), "image/png")})
    model_available = client.get("/api/models").json()["models"][0]["available"]
    if not model_available:
        assert response.status_code == 503
        assert "checkpoint" in response.json()["detail"].lower()
    else:
        assert response.status_code == 200


def test_pair_routes_do_not_invent_predictions():
    image = png_bytes()
    cases = [
        ("change", {"image_before": ("before.png", image, "image/png"), "image_after": ("after.png", image, "image/png")}, 1),
        ("optical-sar", {"image_optical": ("optical.png", image, "image/png"), "image_sar": ("sar.png", image, "image/png")}, 2),
    ]
    states = client.get("/api/models").json()["models"]
    for route, files, index in cases:
        if states[index]["available"]:
            continue
        response = client.post(f"/api/analyze/{route}", files=files)
        assert response.status_code == 503
        assert "checkpoint" in response.json()["detail"].lower()


def test_ambiguous_modalities_rejected():
    response = client.post("/api/v1/analyze", json={"query": "Compare inputs", "image": "optical.png", "image_after": "after.png", "sar_image": "sar.tif"})
    assert response.status_code == 422
