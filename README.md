# SetQuery

SetQuery is a remote-sensing analysis prototype with a React frontend, FastAPI backend, and three existing PyTorch neural networks. Upload images, select an analysis task, and view computed answers, masks, or class scores.

**Current status:** all three architectures execute end to end. No trained checkpoints are included. Random-weight outputs are untrained; accuracy is not established. The clean result screens retain a small grey **Untrained model** label. There are no large warning banners.

## Features and architecture

- Explicit model selection, image uploads, input validation, loading/error recovery, and result views.
- Model 1: **ResNet18 + GRU Single Image VQA** (`SingleImageVQAModel`): RGB image and vocabulary tokens -> 6,359 answer logits -> vocabulary-decoded answer.
- Model 2: **CNN-based Bi-Temporal Change Detection** (`ChangeDetectionModel`): shared before/after encoder -> absolute feature difference -> decoder -> pixel logits -> binary mask.
- Model 3: **Optical-SAR Cross-Modal Fusion CNN** (`OpticalSARFusionModel`): separate optical/SAR encoders -> concatenated features -> class logits and uncalibrated scores.
- CPU operation supported; architecture mode automatically selects CUDA if PyTorch reports it available.

```text
React upload + model selector
  -> multipart HTTP -> FastAPI validation/routing
  -> existing ML adapter -> mode + checkpoint selection
  -> image preprocessing -> neural forward / demo baseline
  -> JSON answer, mask, or scores -> existing result screen
```

The router is orchestration, not a fourth model. Architecture inference uses `model.eval()` and `torch.inference_mode()`; it does not train or update weights.

## Modes

| MODEL_MODE | Behavior |
| --- | --- |
| `architecture` | Runs all three existing neural networks. Strictly loads compatible checkpoints when available; otherwise uses random weights. Random outputs are marked untrained. Loaded checkpoint training provenance is unverified. |
| `demo` | Runs untrained neural VQA, an OpenCV pixel-difference baseline, and an optical/SAR structural-similarity baseline. |
| `production` | Default mode. Requires compatible checkpoints; missing required checkpoints cause HTTP 503. Never falls back to architecture/demo random weights. |

A successfully loaded checkpoint is not proof of accuracy. Incompatible files fail explicitly. Optional checkpoint paths: `SETQUERY_MODEL1_CHECKPOINT`, `SETQUERY_MODEL2_CHECKPOINT`, `SETQUERY_MODEL3_CHECKPOINT`; defaults: `ml/checkpoints/modelN_baseline.pt`. Architecture mode needs no checkpoint or dataset download. The Model 1 vocabulary is included.

## Technology and requirements

- Windows 64-bit; **Python 3.14 (64-bit)** for the verified dependency snapshot; Git; **Node.js 24** with npm. Frontend dependency engines require Node >=20.
- FastAPI, Uvicorn, PyTorch, torchvision, NumPy, Pillow, OpenCV, pytest/httpx.
- React, Vite, React Router, CSS, Playwright for browser checks.
- The verified NumPy version requires Python >=3.12 and Pillow >=3.11. Python 3.10 is therefore **not compatible with this pinned snapshot**; no Python 3.10 setup is claimed.
- No NVIDIA GPU or CUDA toolkit is required. The tested Windows PyTorch wheel runs on CPU. CUDA execution is conditional and was not verified on this CPU environment.
- `requirements.txt` contains runtime/test requirements; `constraints-windows-py314.txt` records the tested versions. `frontend/package-lock.json` supports reproducible `npm ci` installation.

## Clone and install — Windows CMD

Install Python 3.14, Node.js 24, and Git first. Verify `py -3.14 --version`, `node --version`, and `git --version`.

```cmd
git clone --branch handover/windows-architecture-2026-10-18 --single-branch https://github.com/Raghav-Kumar01/SetQuerry.git
cd SetQuerry
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints-windows-py314.txt
cd frontend
npm ci
cd ..
```

Use the root requirements for running the integrated app. `ml/requirements-ml.txt` includes optional research packages and is not needed for architecture inference. No training commands are part of setup.

### Terminal 1 — backend (repository root)

```cmd
set MODEL_MODE=architecture
set SETQUERY_CORS_ORIGINS=http://127.0.0.1:5174,http://localhost:5174
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8765
```

Health: http://127.0.0.1:8765/api/health. API documentation: http://127.0.0.1:8765/docs.

### Terminal 2 — frontend (repository root)

```cmd
cd frontend
set VITE_API_BASE_URL=http://127.0.0.1:8765/api
npm run dev -- --host 127.0.0.1 --port 5174 --strictPort
```

Open **http://127.0.0.1:5174**. Keep both terminals open. Stop a server with Ctrl+C. Environment settings apply to their terminal; repeat them when opening a new terminal. Backend `.env.example` is documentation, not automatically loaded. Frontend may alternatively copy `frontend/.env.example` to `frontend/.env`; restart Vite after changes.

## Use the application

1. Select the desired model in the upload panel.
2. Model 1: upload one image and enter a question (3–500 characters).
3. Model 2: upload before, then after. Model 3: upload optical, then SAR.
4. Use **Swap image order** if needed. Explicit model selection respects that order.
5. Click **Analyze Imagery**. View the computed answer, binary change mask, or fusion class score.

Accepted extensions include PNG, JPG/JPEG, TIFF, WebP, and BMP; each upload must decode as an image and be <=20 MB. Filename hints are used only by optional automatic model selection, not as proof of sensor identity.

## Repository structure

```text
frontend/          React UI, API client, browser tests, Vite configuration
backend/app/       FastAPI routes, validation, model routing
backend/tests/     API contract tests
ml/models/         Existing three neural network definitions
ml/inference/      Mode selection, neural forward execution, demo baselines
ml/data/           Preprocessing and included VQA vocabulary
integration/       Live HTTP tests and generated test-image helper
constraints-windows-py314.txt  Tested Python dependency versions
data_pipeline/     Existing data preparation code and small metadata/manifests
docs/              API contracts and engineering evidence
```

Source dataset images, uploaded user images, checkpoints, virtualenvs, node_modules, secrets, and generated outputs are not included in this handover. Small manifest/metadata files and the decoding vocabulary are source assets, not bundled image datasets. Historical report commands describing a developer's machine are evidence, not setup prerequisites; use this README for portable setup.

## Verification commands

From the root:

```cmd
.venv\Scripts\python.exe -m pytest -q backend/tests integration
.venv\Scripts\python.exe -m compileall -q backend ml integration
```

From `frontend` (browser tests require installed Google Chrome):

```cmd
set SETQUERY_TEST_MODE=architecture
npm run test:e2e
set SETQUERY_TEST_MODE=demo
npm run test:e2e
npm run build
```

Browser tests start isolated servers on 8766/5175 and do not reuse the app's 8765/5174 servers. Tests generate local fixtures, not training data. Do not run the entire ML test directory indiscriminately: some legacy scripts require datasets or execute training loops. See [handover verification](docs/handover.md) for measured results and boundaries.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `py -3.14` not found | Install 64-bit Python 3.14; reopen CMD and check `py -0p`. |
| Missing Python module | From root rerun `.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints-windows-py314.txt`; use that same Python to launch Uvicorn. |
| `npm` unavailable | Install Node.js 24 and reopen CMD. |
| Missing frontend package | Run `npm ci` in `frontend`. |
| Network/CORS error | Verify both servers run, frontend API URL points to 8765/api, and backend origins include 5174. Restart after changing variables. |
| HTTP 503 | Production is the default. Set `MODEL_MODE=architecture` in the backend terminal and restart to run without checkpoints. |
| Port already in use | Stop your old server with Ctrl+C. If choosing different ports, update both API URL and CORS origin consistently. |
| Blank/all-black change mask | Random weights can produce every sigmoid score below 0.5. This is a valid computed output, not evidence of no real-world change. |
| Nonsensical VQA answer | Expected with random weights; vocabulary decoding works but accuracy is unestablished. |
| Chrome missing in browser tests | Install Google Chrome; normal app use can use any modern browser. |

Do not commit `.env`, keys, uploads, or dependencies. No API key is required for local inference.

## Limitations and roadmap

No trained checkpoints, benchmark accuracy, calibration, or geospatial correctness is established. Model 3 uses the existing two display-channel SAR proxy, not calibrated VV/VH measurements, and does not assert class semantics from random weights. CPU execution is verified; CUDA selection needs hardware-specific verification. Tailwind configuration and test-client deprecation warnings remain; npm dependency advisories need separate review before deployment.

Next steps: document verified sensor/label contracts, review dependency advisories, validate compatible checkpoint provenance, and evaluate on independent labeled data. Training is a separate explicitly authorized phase. This handover does not perform it.

## Team contribution workflow

The handover branch is a clean source snapshot because inherited history contained personal uploads. The original development checkout/history and friend's `origin` remote were preserved locally. No remote history was rewritten.

```cmd
git switch -c feature/your-change
```

Make focused changes in your area (ML, data, backend, frontend); preserve shared API contracts and UI design. Run the relevant tests above, review `git diff`, stage explicit files, and open a pull request targeting the handover branch until maintainers choose a default integration branch. Never force-push shared branches. Never add checkpoints/datasets or report random-weight output as trained inference. See [team workflow](docs/team_workflow.md).
