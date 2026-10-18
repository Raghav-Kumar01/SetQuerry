# Functional demo integration — 2026-10-18

Phase: explicit demo computations and complete frontend integration. Status: COMPLETE for the requested demo workflow. All three routes execute image-dependent computations and render their results. Trained model quality, physical sensor alignment, and deployment readiness remain unvalidated. Overall project completion is not measurable without a fixed release scope; demo functionality is complete, production ML acceptance remains outstanding.

## What was built

- UNTRAINED Model 1: the existing ResNet18/GRU VQA network with deterministic random initialization, real image/question tensors, and vocabulary decoding.
- BASELINE Model 2: OpenCV pixel differences, a binary PNG mask, changed pixel count and area percentage.
- BASELINE Model 3: grayscale structure extraction and spatial gradient cosine similarity. No calibrated probability or claimed alignment decision.
- Explicit model selection, ordered pair previews and swapping, question submission, visible provenance, loading and error recovery.
- Production remains the default. Missing checkpoints return 503, including after a demo model has been cached.

No training or fine-tuning was performed. Existing model architectures and user assets were retained. No Git push was performed.

## How the algorithms work

In simple terms, the VQA network looks at the image and question but has never learned their meaning. It still computes an answer, which can be nonsense. The change baseline compares the color of corresponding pixels. The optical/SAR baseline compares where edges occur in the two images; similar edges alone cannot prove they are correctly aligned sensor captures.

### Model 1

The input is one image resized to 512 x 512 and normalized by the existing image loader, plus 32 question token IDs from the existing vocabulary. The ResNet18 image branch and GRU text branch feed the existing answer classifier. Random seed 2026 creates repeatable initial weights in a separate cache; eval/inference mode disables dropout and gradient recording. No gradients or optimizer steps are executed.

For logits z, softmax is p_i = exp(z_i) / sum_j exp(z_j). Each z_i is the score for vocabulary answer i; p_i is its normalized output score. With logits [0,1], the scores are approximately [0.269,0.731]. The answer with highest score is decoded, but these untrained scores give no evidence of correctness. Code: `ml/inference/demo.py::single`.

The cache is separate from the trained loaders so demo weights cannot satisfy a production checkpoint requirement. The existing architecture was selected to exercise the exact neural interface without inventing a trained model. CPU inference is dominated by the ResNet image branch; no latency service objective is claimed.

### Model 2

Both images are decoded to RGB and resized to 256 x 256. At pixel i, d_i = max_c |A_ic - B_ic|, where A/B are before/after colors and c indexes RGB channels. The mask is M_i = 1 if d_i > 25, otherwise 0. Changed percentage is 100 * sum_i M_i / N, where N is the 65,536 resized pixels. If two pixels of four exceed the threshold, the result is 50%.

The actual mask is PNG-encoded and returned. The percentage and mask use exactly the same pixels. This O(N) deterministic baseline needs no checkpoint. It assumes common coverage/registration; lighting, shadows, resizing, and camera movement can trigger change. A learned detector or registered radiometric comparison could replace it once trained assets and independent evaluation justify that step. Code: `demo.py::change`.

### Model 3

The inputs are decoded as floating grayscale images (numeric grayscale values are preserved), resized to 256 x 256, and independently scaled by the 1st and 99th intensity percentiles. SAR display intensities additionally use log1p(9*x)/log(10). Gaussian smoothing (5 x 5, sigma 1) reduces local noise. Sobel derivatives yield gradient magnitudes at each pixel.

The score is s = (a dot b) / (||a|| * ||b||), where a and b are flattened nonnegative gradient magnitudes. It lies in [0,1]. Example: [1,0] versus [1,0] gives 1; [1,0] versus [0,1] gives 0. If either vector has zero energy, the diagnostic is undefined and returned as null, never as a fabricated alignment score.

This O(N) method compares spatial structure while reducing dependence on absolute brightness. It does not calibrate VV/VH radar measurements, estimate geometric transforms, perform registration, or identify alignment/misalignment. RGB SAR inputs are display imagery. Real sensor validation may require a different preprocessing contract and a trained cross-modal model. Code: `demo.py::_structure` and `crossmodal`.

## Important file responsibilities

| File | Input and output | Design / interview explanation |
| --- | --- | --- |
| `ml/inference/assets.py` | Process settings and checkpoint paths -> mode decision | Controls explicit demo fallback while preserving strict production behavior. |
| `ml/inference/demo.py` | Valid paths and question -> computed JSON-ready result | Keeps untrained/baseline algorithms separate from trained runners and records their provenance. |
| `ml/inference/*_inference.py` | Adapter request -> demo or checkpoint result | Selects a computation once, behind the existing backend interface. |
| `backend/app/api.py` | Upload bytes/model choice -> validated inference response | Validates user input and preserves explicit pair order; emits frontend-compatible fields. |
| `frontend/src/services/analysisService.js` | Files/question -> multipart requests | Every demo result comes from the server; canned frontend data is unused. |
| `frontend/src/components/UploadWorkspace/UploadWorkspace.jsx` | User selection/files -> chosen result route | Lets users choose each model and correct pair order before analysis. |
| `frontend/src/components/InferenceNotice.jsx` | Provenance metadata -> visible banner | Explains UNTRAINED or BASELINE output before the result is interpreted. |
| `integration/test_demo_api.py` | Generated valid/invalid images -> live HTTP assertions | Starts an actual Uvicorn server and verifies computation and rejection behavior. |
| `frontend/e2e/demo.spec.js` | Real browser uploads -> rendered result assertions/screenshots | Verifies the complete application, including loading, errors, mask decoding and question resubmission. |

## Files changed for this demo phase

Created:
- `ml/inference/demo.py`
- `frontend/src/components/InferenceNotice.jsx`
- `integration/test_demo_api.py`
- `integration/create_demo_inputs.py`
- `frontend/playwright.config.js`
- `frontend/e2e/demo.spec.js`
- `docs/demo_inference.md`

Modified:
- `.env.example`, `.gitignore`, `README.md`, `requirements.txt`, `docs/api_contract.md`
- `ml/requirements-ml.txt`, `ml/inference/assets.py`, `ml/inference/single_inference.py`, `ml/inference/change_inference.py`, `ml/inference/crossmodal_inference.py`
- `backend/app/api.py`, `backend/app/schemas.py`
- `frontend/package.json`, `frontend/package-lock.json`, `frontend/index.html`, `frontend/src/index.css`
- `frontend/src/services/analysisService.js`
- `frontend/src/components/UploadWorkspace/UploadWorkspace.jsx`, `frontend/src/components/Navbar/Navbar.jsx`
- `frontend/src/pages/Analysis/SingleAnalysis.jsx`, `frontend/src/pages/Analysis/ChangeAnalysisWorkspace.jsx`, `frontend/src/pages/Analysis/OpticalSarAnalysis.jsx`

Earlier repair changes are also still uncommitted on the branch; this list describes the additional demo work.

## Verification and observed results

The test images are generated fixtures under temporary test directories or ignored `output/playwright/fixtures`, never training data. HTTP test logs showed:

- VQA: black image -> `1077m2`; changed quarter image -> `3370m2`. Logit L2 norms changed from 47.8415412902832 to 52.162601470947266. Repeated input produced the same top answers. These decoded values are untrained outputs, not correct image descriptions.
- Change: identical images -> 0%; changed quarter image -> 25.390625%, 16,640 changed pixels of 65,536. Resizing interpolation accounts for the boundary pixels beyond exactly 25%.
- Structure: matching patterns -> 1.0; displaced patterns -> 0.00777390545347254; blank pair -> null.
- Live API tests cover all three routes, invalid uploads, absent fields, nonexistent paths, explicit selection, and strict production rejection.
- Browser tests use real backend responses. A request gate briefly holds the real request only to observe loading; responses are never fabricated.

Screenshots are saved in `output/playwright/single.png`, `change.png`, and `optical-sar.png`. Tests are reproducible with the commands in README.

### Final regression verification — 2026-10-18

- **PASS:** backend, integration, and selected non-training ML tests: 22 passed in 7.81 seconds. The only pytest warning concerns deprecated Starlette/httpx TestClient integration.
- **PASS:** browser integration: 5 passed in 16.8 seconds. All three analysis screens completed real uploads and rendered computed responses; loading states, follow-up VQA submission, invalid-image retry, and missing-input recovery passed.
- **PASS:** frontend production build: Vite compiled 1,953 modules in 7.76 seconds. The existing Tailwind content-configuration warning remains nonblocking.
- **PASS:** production isolation: after warming the untrained VQA cache in demo mode, switching to production returned HTTP 503 for all three routes with absent checkpoints. Production did not reuse the demo model.
- No new failures required implementation changes. No training, fine-tuning, or GitHub push was performed.

These timings describe local test/build runs, not inference benchmarks. Browser checks used the development server; production bundle compilation was checked separately. Functional demo integration is complete; predictive accuracy is unvalidated.

## Current pipeline

```text
Choose model + upload image(s) + optional question
    -> backend validates bytes and file count
    -> mode/asset selection
       -> demo + missing checkpoint: untrained VQA / pixel difference / gradient similarity
       -> checkpoint present: existing checkpoint inference
       -> production + missing checkpoint: 503
    -> JSON with provenance + result/mask/diagnostic
    -> frontend result and visible UNTRAINED/BASELINE notice
```

## Revision questions

- Why is an untrained neural output still a real computation? It comes from input-dependent tensor operations and vocabulary decoding, but its weights have not learned meaningful predictions.
- Why can similarity 1.0 fail to prove alignment? Matching gradient patterns do not establish sensor provenance, georeferencing, or physical correspondence.
- Why does the baseline omit confidence? The method calculates pixel differences or similarity, without a calibrated probability model.
- Why use a fixed random seed? It makes the untrained network repeatable for functional tests while preserving image/question-dependent operations.
- Why keep separate caches? Demo initialization must never satisfy production's requirement for trained weights.

## Remaining work and Git checkpoint

No checkpoint is required to use demo mode. Production still needs compatible trained assets and validation; no accuracy, calibration, physical alignment, or geospatial acceptance is established. Existing dependency warnings and npm audit advisories should be reviewed before deployment. Test browser setup uses installed Chrome. Full training smoke tests remain outside this run because they execute optimization steps.

Next recommended phase: independently validate compatible trained assets and preprocessing against labeled data. Do not automatically train or fine-tune.

Branch: `repair/end-to-end-integration-2026-10-18`. Suggested commit message: `Add explicit computed demo inference and browser integration tests`. Commit only as a demo-capable integration checkpoint, not as validated production ML. No push is authorized.
