"""Integration contract: routing uses real upload bytes and never returns a mock."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


def test_unified_upload_rejects_invalid_image():
    response = TestClient(app).post("/api/analyze", files={"file_0": ("optical.png", b"invalid", "image/png")})
    assert response.status_code == 422


def test_unified_upload_routes_ordered_pair(monkeypatch):
    def fake_predict(before, after, query):
        with Image.open(before) as image:
            assert image.getpixel((0, 0)) == (255, 0, 0)
        with Image.open(after) as image:
            assert image.getpixel((0, 0)) == (0, 0, 255)
        return {"task": "change_analysis", "is_mock": False}

    monkeypatch.setattr("backend.app.api.predict_change", fake_predict)
    def png(color):
        stream = io.BytesIO()
        Image.new("RGB", (2, 2), color).save(stream, format="PNG")
        return stream.getvalue()
    response = TestClient(app).post("/api/analyze", files=[
        ("file_0", ("before.png", png((255, 0, 0)), "image/png")),
        ("file_1", ("after.png", png((0, 0, 255)), "image/png")),
    ])
    assert response.status_code == 200
    assert response.json()["task"] == "change_analysis"
