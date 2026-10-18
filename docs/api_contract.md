# API contract

`POST /api/v1/analyze` accepts JSON `query`, `image`, and optionally either `image_after` or `sar_image`. The paths must point to readable images on the server. `image_after` selects change analysis; `sar_image` selects optical-SAR; neither selects single-image analysis.

The frontend uses multipart routes under `/api`: `POST /detect-model` accepts one or two image parts; `POST /analyze/single` takes `image` and optional `query`; `POST /analyze/change` takes `image_before` and `image_after` in temporal order; `POST /analyze/optical-sar` takes `image_optical` and `image_sar`. `POST /analyze` also accepts one or two uploaded files. Two-file automatic routing uses SAR filename hints only. Explicit routes should be used when filenames are ambiguous.

All uploads are checked for supported extension, nonempty content, size at most 20 MB, and decodable image bytes. Images are written to a temporary directory during inference and removed afterward. HTTP 422 means invalid input or missing image; HTTP 503 means a required checkpoint is unavailable. `GET /health` reports the API state separately from each model's checkpoint state, and `GET /models` reports whether a checkpoint file exists. A present checkpoint has status `checkpoint_present_unverified` until compatibility and quality are checked.

Successful inference responses retain `task`, `query`, `answer`, `confidence`, `evidence`, model metadata, trace, latency, and `is_mock`. `confidence_semantics` describes the raw, uncalibrated model score. Change detection also returns a model-derived binary PNG mask in `mask_base64`. These scores are not accuracy estimates.

## Explicit demo computations

Set `MODEL_MODE=demo` on the backend process. When a checkpoint is absent, adapters execute an untrained network (Model 1) or deterministic OpenCV baselines (Models 2 and 3). Existing checkpoints are preferred; incompatible/corrupt checkpoints raise errors instead of silently selecting a baseline. Default mode is `production`, which requires checkpoints.

Responses add `model_mode`, `inference_kind`, `label`, `is_trained`, and `notice`. `is_mock=false` means actual computation took place; it does **not** mean trained or accurate. Model 1's `confidence` is an untrained softmax score. For baselines, `confidence=null`; Model 2 reports `change_percentage`, `changed_pixels`, `analyzed_pixels`, and `mask_base64` (also `maskBase64` on the frontend route). Model 3 reports `similarity_score` in [0,1], or null when either image has no measurable edges. A similarity score is not a probability. No baseline emits `alignment_probability`.

`POST /detect-model` accepts a `model` form field: `single_image`, `change_analysis`, `optical_sar`, or `auto`. Explicit selection validates the required file count and returns `selection=manual`. Explicit optical/SAR uploads use the first image as optical and the second as SAR, regardless of filenames. `GET /health` and `/models` include `model_mode`; per-model status can be `untrained_demo` or `baseline_demo`, with `checkpoint_present=false` and `available=true`.


## Architecture mode

`MODEL_MODE=architecture` runs the existing neural networks. Responses add `weights_source` (`random` or `checkpoint`), `device`, and `diagnostics` containing input/output shapes, initialization status, eval/gradient state and logit statistics. Random weights have `is_trained: false` and `label: "UNTRAINED MODEL"`. Strictly compatible checkpoint files have `is_trained: null` because their training provenance is not established by deserialization. Incompatible checkpoint files fail explicitly without fallback.

Model 1 returns vocabulary-decoded top answers and raw logits. Model 2 returns a PNG binary mask from sigmoid(logits) > 0.5, changed pixel count and percentage. Model 3 returns raw logits, `predicted_class`, and uncalibrated `class_scores`; no calibrated alignment probability is asserted. Existing response fields remain available to the frontend.
