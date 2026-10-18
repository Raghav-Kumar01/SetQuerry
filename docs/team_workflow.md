# Team contribution workflow

Use the root README for verified Windows installation. Clone the handover branch, then create a focused `feature/...` branch. Preserve the current module boundaries: ML architecture/inference, data preparation, backend transport/contracts, and frontend design. Discuss contract changes before implementation.

Before a pull request:

1. Run `.venv\Scripts\python.exe -m pytest -q backend/tests integration` for API/ML integration changes.
2. Run the architecture/demo browser suites when upload/results flows change, plus `npm run build`.
3. Review `git diff` and stage explicit files. Exclude datasets, checkpoints, uploads, .env, keys, and generated outputs.
4. Describe behavior changes, tests, and limitations. Target `handover/windows-architecture-2026-10-18` until maintainers designate an integration/default branch.
5. Never force-push shared branches or describe untrained output as trained inference.

The initial handover is a clean snapshot because inherited source history contained personal uploads. Original history remains preserved in the developer checkout; do not merge or push that history without reviewing its private assets first. Existing CODEOWNERS is retained as historical module ownership; maintainers should configure collaborator access/review rules explicitly.
