"""Small validation helpers used before routing requests."""

from .schemas import AnalyzeRequest


def validate_request(request: AnalyzeRequest) -> AnalyzeRequest:
    """Normalize user text while preserving Pydantic's field validation."""
    request.query = request.query.strip()
    if len(request.query) < 3:
        raise ValueError("query must contain at least 3 non-space characters")
    return request
