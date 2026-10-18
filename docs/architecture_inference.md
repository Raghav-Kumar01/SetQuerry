# Phase completion report: neural architecture integration

Date: 2026-10-18. Status: COMPLETE for forward inference and frontend integration.
Requested phase completion: 100%. Overall production ML completion is not estimated: trained assets and accuracy validation remain outside scope.

## What was built

An explicit `MODEL_MODE=architecture` path runs existing classes without modifying any files in `ml/models/`. Demo mode retains its original neural VQA and two image baselines. Production remains strict. Model selection and visual layout are preserved; the notice now identifies architecture mode.

No trained checkpoints were found at the configured/default paths. Actual verification used random weights with seed 2026. The loader uses CUDA when available, otherwise CPU. This environment is PyTorch 2.14.1+cpu; CUDA is unavailable, so GPU execution is untested.

## Exact forward results

Generated functional fixtures, not training data: black RGB image, top-left white quarter, displaced quarter. Question: `Is there a road?`.

| Model class | Inputs | Output | Observed output | Weights / UI |
| --- | --- | --- | --- | --- |
| SingleImageVQAModel | image [1,3,512,512], token IDs [1,32] | logits [1,6359] | class 2670, decoded `3370m2`, top softmax 0.001291431486606598; logits min -2.198524, max 2.331311 | Random; initialization PASS; UI PASS |
| ChangeDetectionModel | before and after each [1,3,256,256] | logits [1,1,256,256] | logits min -0.2783363461, max -0.1591789275; thresholded mask 0/65536 changed pixels, 0% | Random; initialization PASS; UI PASS |
| OpticalSARFusionModel | optical [1,3,128,128], SAR [1,2,128,128] | logits [1,2] | logits [-0.0371783450, 0.0878995210]; softmax [0.4687712491,0.5312287807]; class 1 | Random; initialization PASS; UI PASS |

All three logit hashes changed when image inputs changed and repeated identically on repeated inputs on this CPU. The change mask remains all black on these fixtures because every sigmoid value is below 0.5; this is a genuine untrained result, not a substitute mask. Class indices in Model 3 are reported without asserting semantic alignment.

Full actual JSON outputs: ignored `output/architecture/single.json`, `change.json`, `optical-sar.json`. Browser screenshots: ignored `output/playwright/architecture-single.png`, `architecture-change.png`, `architecture-optical-sar.png`.

## Tests and evidence

- PASS: 29 backend/API/non-training regression tests in 10.13 seconds.
- PASS: 5 architecture browser tests on first run (16.0 s).
- PASS: 5 preserved demo browser tests (16.0 s).
- PASS: 5 architecture browser tests after stopping test servers and starting fresh backend/frontend processes (15.1 s). Backend PID changed from 45880 to 39552.
- PASS: production returns 503 for every missing-checkpoint route after all architecture models were warmed in the same process.
- PASS: strict compatible checkpoint loading for all three classes, using explicitly untrained serialized parameter fixtures. These tests do not establish that trained checkpoints exist. Incompatible checkpoint fixtures raise explicit errors without a baseline fallback.
- PASS: invalid and missing image rejection; loading indicators; real uploads; answer/mask/score rendering; VQA follow-up; error retry and missing-input recovery.
- PASS: frontend production build, 1953 modules, 6.75 seconds. Browser tests use Vite dev servers; production bundle compilation is verified separately.
- Existing checkpoint serialization regression creates an SGD object for save/load only; no optimizer step or training loop was run. No fine-tuning, dataset download, or push occurred.
- Nonblocking warnings: Tailwind content configuration and deprecated Starlette/httpx test-client integration.

## Files created

### ml/inference/architecture.py
Purpose: forward-only orchestration around the three existing model classes.
Input: model number, validated image paths, question, mode/checkpoint configuration.
Output: JSON-compatible result with model output, provenance, mask/scores, and tensor diagnostics.
Important functions: `_load` strictly loads parameters or initializes deterministic random weights; `predict` preprocesses real images, invokes eval/inference_mode, and decodes outputs.
Connection: called by existing inference adapters only in architecture mode; shares current image resolvers and vocabulary.
Design: a separate cache keyed by model, checkpoint path/file stamp, and device prevents random models from entering production caches. Existing architecture definitions are reused unchanged.
Interview explanation: This file connects uploaded images to our existing neural networks and reports exactly how their outputs were produced.

### integration/test_architecture_api.py
Purpose: validate genuine input-dependent network execution through live Uvicorn HTTP, strict checkpoint loading, and production cache isolation.
Input: local generated valid and invalid images plus explicitly random checkpoint fixtures.
Output: assertions and saved JSON evidence; no learned artifact is produced.
Interview explanation: This proves model wiring and provenance, not model accuracy.

### docs/architecture_inference.md
Purpose: reproducible phase report and learning reference.

## Files modified in this phase

- `ml/inference/assets.py`: accept architecture as an explicit mode; default production unchanged.
- `ml/inference/single_inference.py`, `change_inference.py`, `crossmodal_inference.py`: route architecture mode to neural execution before existing demo/production branches.
- `backend/app/api.py`: advertise architecture availability independently of missing checkpoints; status is explicitly unverified.
- `frontend/src/components/InferenceNotice.jsx`: display ARCHITECTURE / UNTRAINED MODEL.
- `frontend/playwright.config.js`: allow mode selection through SETQUERY_TEST_MODE.
- `frontend/e2e/demo.spec.js`: exercise both modes and preserve actual backend responses.
- `integration/test_demo_api.py`: share the live-server harness with architecture tests.
- `.env.example`, `README.md`, `docs/api_contract.md`: document mode selection and additive output fields.

Earlier uncommitted repair/demo changes remain on the branch and are not newly attributed to this phase.

## Learn the implementation

Easy explanation: the neural networks are three machines with different input slots. Architecture mode plugs real images into those machines. Random weights let the machines execute, but do not teach them to recognize correct answers. Eval mode switches off training behavior; inference_mode avoids building a gradient graph.

Technical explanation:
- Model 1 uses the existing ResNet18 (weights=None), a GRU over vocabulary token embeddings, concatenation, and answer classifier. RGB images use existing 512x512 ImageNet normalization; questions use 32 token IDs. A softmax and vocabulary lookup decode the predicted answer.
- Model 2 uses one shared CNN encoder on each image, absolute feature difference, and a transposed-convolution decoder. This learned-feature difference is part of the existing neural architecture, not the OpenCV baseline. RGB tensors use existing normalization at 256x256.
- Model 3 uses separate CNN encoders, pooled 64-element features each, concatenation to 128 elements, and an MLP. It retains the existing two-class initialization. Compatible checkpoint classifier dimensions determine the class count without changing layer types. The existing upload adapter scales RGB to [0,1] at 128x128 and selects the first two SAR display channels. Those channels are not verified calibrated VV/VH data.

For Model 2, p = 1/(1+exp(-z)), where z is a pixel logit and p is its sigmoid score. The binary mask m = 1 if p > 0.5, otherwise 0. With z=-0.23, p is about 0.443, so the pixel is black. Changed percentage = 100 * sum(m) / N, where N is the number of pixels. For these random weights sum(m)=0 and N=65536, hence 0%. Code: `predict`, number==2 branch. This fixed threshold preserves honest output rather than forcing an interesting mask.

For Models 1 and 3, s_i = exp(z_i) / sum_j exp(z_j), where z_i is class i's logit and s_i is its normalized class score. For Model 3's observed logits, scores are about [0.46877,0.53123], so class 1 is selected. Normalization does not establish calibration or correctness.

Why this design: preserves trained-path and demo behavior, avoids new architectures/dependencies, and reuses preprocessing contracts. An alternative would refactor every existing runner into one loader, but that would expand the change surface. A baseline substitute would not satisfy this phase. Forward costs scale with image pixels and existing convolution widths; no speed claims are made from test-suite durations. Cached models avoid repeated initialization; model parameters remain in device memory. CUDA selection exists, but real GPU validation is still needed on a CUDA installation.

## Revision questions

- Does eval mode train a model? No; it disables training-dependent behavior such as dropout and batch-statistic updates.
- Does inference_mode mean the result is correct? No; it disables autograd bookkeeping.
- Why can different inputs yield the same binary mask? Different continuous logits can remain on the same side of a fixed threshold.
- Does a compatible checkpoint prove trained weights? No; even random parameters can be serialized. The new mode reports loaded checkpoint provenance as unverified.
- Why not interpret class 1 as alignment? Random weights and an unverified label mapping cannot establish that semantic claim.

## Current pipeline

```text
Upload + selected model + question
 -> validated actual image bytes
 -> existing adapter
    -> architecture: existing class + random/strict checkpoint loader
       -> existing tensor preprocessing -> eval + inference_mode
       -> vocabulary answer / neural binary mask / class logits and scores
    -> demo: preserved previous computations
    -> production: strict required checkpoint inference
 -> JSON with provenance -> existing frontend with visible notice
```

## Run in Windows CMD

Backend (separate terminal):
```cmd
cd /d F:\CodingWorld\SETqueery\SetQuery-latest
set MODEL_MODE=architecture
set SETQUERY_CORS_ORIGINS=http://127.0.0.1:5174,http://localhost:5174
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8765
```
Frontend:
```cmd
cd /d F:\CodingWorld\SETqueery\SetQuery-latest\frontend
set VITE_API_BASE_URL=http://127.0.0.1:8765/api
npm run dev -- --host 127.0.0.1 --port 5174 --strictPort
```
Open http://127.0.0.1:5174. Stop existing processes on these ports before launching replacements. Both local servers were restarted with this configuration during this phase.

## Remaining work / next phase / Git

No functional blocker for architecture mode. Missing trained assets, sensor-specific calibrated preprocessing, semantic class mappings, accuracy/calibration validation, and actual CUDA verification remain. No scientifically meaningful predictive-performance result exists yet. A sensible next phase is to validate asset and sensor/label contracts; it was not implemented here.

Branch: repair/end-to-end-integration-2026-10-18.
Suggested commit: Add explicit neural architecture inference mode and integration coverage.
The verified changes are suitable for a local integration checkpoint; review all existing uncommitted repair changes before staging. No commit or push was performed.
