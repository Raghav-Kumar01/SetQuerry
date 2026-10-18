"""Small starter utility for validating a source data directory."""

from pathlib import Path


def list_supported_files(source: str | Path) -> list[Path]:
    """Return common raster files without modifying the source directory."""
    root = Path(source)
    extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    return sorted(path for path in root.glob("**/*") if path.suffix.lower() in extensions)
