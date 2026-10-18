"""Request and response schemas for the analysis API."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AnalyzeRequest(BaseModel):
    """Image references required for one of the three analysis tracks."""

    query: str = Field(min_length=3, max_length=500, examples=["What changed?"])
    image: str | None = Field(default=None, min_length=1, description="Primary or optical image reference")
    image_after: str | None = Field(default=None, min_length=1, description="After image reference")
    sar_image: str | None = Field(default=None, min_length=1, description="SAR image reference")

    @model_validator(mode="after")
    def require_valid_image_combination(self) -> "AnalyzeRequest":
        if not self.image:
            raise ValueError("image is required as the primary or optical image")
        if self.image_after and self.sar_image:
            raise ValueError("provide either image_after or sar_image, not both")
        return self


class AnalysisResponse(BaseModel):
    task: Literal["single_image", "change_analysis", "optical_sar"]
    query: str
    answer: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[str]
    model: str
    model_version: str
    execution_trace: list[str]
    latency_ms: float = Field(ge=0.0)
    is_mock: bool
