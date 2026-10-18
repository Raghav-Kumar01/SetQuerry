"""Create deterministic UI fixtures, separate from any training or evaluation data."""
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    output = Path(__file__).resolve().parents[1] / "output" / "playwright" / "fixtures"
    output.mkdir(parents=True, exist_ok=True)
    before = np.zeros((128, 128, 3), dtype=np.uint8)
    after = before.copy()
    after[:64, :64] = 255
    for name, array in (("before", before), ("after", after), ("optical", after), ("sar", after)):
        Image.fromarray(array).save(output / f"{name}.png")
    (output / "invalid.png").write_bytes(b"invalid image")
    print(output)


if __name__ == "__main__":
    main()
