"""FastAPI application entry point."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router

app = FastAPI(title="SetQuerry API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "SETQUERY_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route mounting: root / (e.g. /detect-model), /api (used by frontend analysisService.js), and /api/v1 (versioned)
app.include_router(router)
app.include_router(router, prefix="/api")
app.include_router(router, prefix="/api/v1")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "SetQuerry", "status": "online", "docs": "/docs"}
