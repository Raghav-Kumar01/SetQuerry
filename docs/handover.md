# Handover verification — 2026-10-18

## Scope and Git safety

Destination: https://github.com/Raghav-Kumar01/SetQuerry.git
Branch: handover/windows-architecture-2026-10-18

The destination was empty (no branch refs; GitHub commits API returned empty-repository status). Authenticated account Raghav-Kumar01 has push permission. The original development source history contains personal uploaded images. A separate clean snapshot was prepared rather than publishing those historical files. No existing branch was overwritten or force-pushed. The original checkout, uncommitted source work, branch history, and friend's origin remote remain intact; raghav was added separately.

## Included and excluded

Included: frontend/backend/ML source, unchanged architecture definitions, current clean UI, inference adapters and tests, VQA vocabulary, dependency manifests/lockfile, portable setup examples, CI, documentation, small metadata/manifests, and required frontend media. The existing intro video (37,218,705 bytes) is retained to preserve the UI.

Excluded: seven tracked personal uploads from the original tree/history, generated Model 3 full manifest (10,114,735 bytes; not needed for inference), raw/processed image datasets, all checkpoints, .venv, node_modules, .env files and secrets, generated results, build folders, caches/logs and temporary uploads. Files remain in the original local checkout; they were not deleted. The snapshot has a fresh Git root, so excluded historical blobs cannot be reached through earlier commits.

A pattern scan for private keys, common GitHub/AWS/API token formats, and hardcoded secret assignments found no matches in the export. Staged file inventory and sizes were reviewed; heuristic scans cannot guarantee absence of every possible secret. Models and live inference paths have no laptop-specific F:/G: requirement. Historical reports are retained as evidence; use root README for portable installation.

## Environment and installation evidence

Windows, Python 3.14, Node v24.15.0, PyTorch 2.14.1+cpu; no CUDA. A separate empty virtualenv was created, dependencies installed, and constrained to the verified versions in constraints-windows-py314.txt. pip check passed. A fresh npm ci installed 165 packages in the export. No existing source virtualenv/node_modules were copied into the published files. For test execution only, the export's ignored .venv points to the newly created test environment.

Python 3.10 was not chosen because the verified NumPy pin requires >=3.12 and Pillow >=3.11. The README uses Python 3.14 to match the verified installation, rather than claiming unsupported compatibility.

## Verification

- Backend import/compile: PASS.
- Exported copy backend and integration: 23 passed in 12.38 seconds.
- Architecture browser flows: 5 passed in 21.1 seconds, using real uploads/backend responses.
- Frontend production build: PASS, 1953 modules, 8.32 seconds.
- Production enforcement: covered in the passing API suite, including switching from initialized random-weight architecture models to production and expecting 503 for all missing checkpoints.
- Compatible/incompatible checkpoint loading: covered using explicitly random serialized test fixtures; no trained assets claimed.
- Image-dependent forward outputs, invalid/missing input rejection, actual mask rendering, VQA resubmission, loading/error recovery, no large warning boxes, and neutral-grey provenance labels: covered by tests.
- Demo browser flows: 5 passed in 18.0 seconds.

No training, fine-tuning, optimizer step, model redesign, or UI redesign was performed for this handover. Windows local results do not constitute CI or deployed-service results. GPU behavior and real prediction accuracy remain unverified.

## Known limitations

No trained checkpoint or predictive accuracy is established. SAR uses the existing display-channel proxy. Random masks may be entirely black. npm ci reports 9 advisories (3 moderate, 6 high); no automatic force-upgrade was applied. Tailwind content and Starlette/httpx deprecation warnings remain. Docker/legacy research instructions are not the verified setup path; use the root Windows instructions.

## Suggested next phase

Review dependencies and documented sensor/label contracts before deployment. Independently validate compatible trained assets if supplied. Training requires a separate decision. Development continues through focused feature branches and pull requests targeting the handover branch.
