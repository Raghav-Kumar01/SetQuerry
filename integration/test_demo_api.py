"""Real HTTP tests: launch Uvicorn, upload generated images, inspect computed outputs."""

import base64
import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture(scope="module")
def images():
    empty = np.zeros((128, 128, 3), dtype=np.uint8)
    changed = empty.copy()
    changed[:64, :64] = 255
    shifted = empty.copy()
    shifted[64:, 64:] = 255
    return {"empty": png(empty), "changed": png(changed), "shifted": png(shifted)}


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    yield from start_api(tmp_path_factory, "demo")


def start_api(tmp_path_factory, mode):
    directory = tmp_path_factory.mktemp(mode + "-server")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**os.environ, "MODEL_MODE": mode}
    for number in (1, 2, 3):
        env[f"SETQUERY_MODEL{number}_CHECKPOINT"] = str(directory / f"absent-{number}.pt")
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", str(port)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as client:
                for _ in range(100):
                    try:
                        if client.get("/api/health").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    if process.poll() is not None:
                        pytest.fail((directory / "server.log").read_text())
                    time.sleep(0.1)
                else:
                    pytest.fail("Demo API did not start")
                yield client
        finally:
            process.terminate()
            process.wait(timeout=15)


def post(api, route, **files):
    response = api.post(f"/api/analyze/{route}", files={key: (f"{key}.png", value, "image/png") for key, value in files.items()},
                        data={"query": "Is there a road?"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["model_mode"] == "demo" and result["is_trained"] is False
    assert result["is_mock"] is False
    return result


def test_vqa_computes_input_dependent_logits_and_decodes(api, images):
    first = post(api, "single", image=images["empty"])
    second = post(api, "single", image=images["changed"])
    repeated = post(api, "single", image=images["changed"])
    assert first["label"] == "UNTRAINED"
    assert first["diagnostics"]["logits_l2"] != second["diagnostics"]["logits_l2"]
    assert second["top_answers"] == repeated["top_answers"]
    assert second["answer"] == second["top_answers"][0]["answer"]
    assert second["annotations"] == []
    print("VQA:", first["answer"], second["answer"], "logit norms", first["diagnostics"]["logits_l2"], second["diagnostics"]["logits_l2"])


def test_change_mask_matches_computed_percentage(api, images):
    same = post(api, "change", image_before=images["empty"], image_after=images["empty"])
    changed = post(api, "change", image_before=images["empty"], image_after=images["changed"])
    assert same["change_percentage"] == 0
    assert 24 <= changed["change_percentage"] <= 27
    mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(changed["mask_base64"]))))
    assert mask.shape == (256, 256)
    assert set(np.unique(mask)) <= {0, 255}
    assert np.count_nonzero(mask) == changed["changed_pixels"]
    assert np.count_nonzero(mask) * 100 / mask.size == changed["change_percentage"]
    assert changed["confidence"] is None and changed["label"] == "BASELINE"
    print("CHANGE:", same["change_percentage"], changed["change_percentage"], changed["changed_pixels"])


def test_similarity_uses_structure_and_handles_blank_inputs(api, images):
    same = post(api, "optical-sar", image_optical=images["changed"], image_sar=images["changed"])
    shifted = post(api, "optical-sar", image_optical=images["changed"], image_sar=images["shifted"])
    blank = post(api, "optical-sar", image_optical=images["empty"], image_sar=images["empty"])
    repeated = post(api, "optical-sar", image_optical=images["changed"], image_sar=images["changed"])
    assert same["similarity_score"] > shifted["similarity_score"]
    assert same["similarity_score"] == repeated["similarity_score"]
    assert blank["similarity_score"] is None
    assert same["confidence"] is None and "alignment_probability" not in same
    print("SAR:", same["similarity_score"], shifted["similarity_score"], blank["similarity_score"])


@pytest.mark.parametrize("route,fields", [("single", ["image"]), ("change", ["image_before", "image_after"]), ("optical-sar", ["image_optical", "image_sar"])])
def test_invalid_and_missing_uploads_are_rejected(api, route, fields):
    assert api.post(f"/api/analyze/{route}").status_code == 422
    assert api.post(f"/api/analyze/{route}", files={name: ("invalid.png", b"invalid", "image/png") for name in fields}).status_code == 422


def test_explicit_selection_and_missing_path(api, images):
    files = [("file_0", ("neutral.png", images["changed"], "image/png")), ("file_1", ("neutral2.png", images["shifted"], "image/png"))]
    response = api.post("/api/detect-model", files=files, data={"model": "optical_sar"})
    assert response.status_code == 200 and response.json()["selection"] == "manual"
    assert response.json()["route"] == "/analyze/optical-sar"
    assert api.post("/api/detect-model", files=files, data={"model": "single_image"}).status_code == 422
    assert api.post("/api/v1/analyze", json={"image": "does-not-exist.png", "query": "What is there?"}).status_code == 422


def test_production_never_reuses_demo_weights(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import app
    for number in (1, 2, 3):
        monkeypatch.setenv(f"SETQUERY_MODEL{number}_CHECKPOINT", str(tmp_path / f"absent-{number}.pt"))
    valid = png(np.zeros((8, 8, 3), dtype=np.uint8))
    with TestClient(app) as client:
        monkeypatch.setenv("MODEL_MODE", "demo")
        warm = client.post('/api/analyze/single', files={"image": ("scene.png", valid, "image/png")})
        assert warm.status_code == 200 and warm.json()["label"] == "UNTRAINED"
        monkeypatch.setenv("MODEL_MODE", "production")
        for route, fields in [("single", ["image"]), ("change", ["image_before", "image_after"]), ("optical-sar", ["image_optical", "image_sar"])]:
            response = client.post(f"/api/analyze/{route}", files={field: ("scene.png", valid, "image/png") for field in fields})
            assert response.status_code == 503, response.text
