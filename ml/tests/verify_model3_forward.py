"""Verification script for Model 3 (Optical-SAR Cross-Modal Fusion).

Asserts that:
1. Aligned optical-SAR pairs produce > 80% alignment confidence.
2. Misaligned optical-SAR pairs produce < 20% alignment confidence.
"""

import sys
from pathlib import Path
import torch

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.inference.run_crossmodal import run_crossmodal


def main():
    print("=" * 60)
    print("VERIFYING MODEL 3 FORWARD PASS (OPTICAL-SAR CROSS-MODAL)")
    print("=" * 60)

    # 1. Aligned pair verification
    print("\n--- Test 1: Aligned Pair (opt_1.png vs sar_1.png) ---")
    res_aligned = run_crossmodal("opt_1.png", "sar_1.png", "Verify optical-SAR alignment")
    print(f"Decision: {res_aligned['decision']}")
    print(f"Answer: {res_aligned['answer']}")
    print(f"Alignment Probability: {res_aligned['alignment_probability'] * 100:.2f}%")
    print(f"Confidence: {res_aligned['confidence']}")

    assert res_aligned["alignment_probability"] >= 0.80, (
        f"Expected aligned probability >= 0.80, got {res_aligned['alignment_probability']}"
    )
    print("PASS: Aligned pair satisfied > 80% threshold.")

    # 2. Misaligned pair verification
    print("\n--- Test 2: Misaligned Pair (opt_1.png vs sar_2.png) ---")
    res_misaligned = run_crossmodal("opt_1.png", "sar_2.png", "Verify optical-SAR misalignment")
    print(f"Decision: {res_misaligned['decision']}")
    print(f"Answer: {res_misaligned['answer']}")
    print(f"Alignment Probability: {res_misaligned['alignment_probability'] * 100:.2f}%")
    print(f"Confidence: {res_misaligned['confidence']}")

    assert res_misaligned["alignment_probability"] <= 0.20, (
        f"Expected misaligned alignment probability <= 0.20, got {res_misaligned['alignment_probability']}"
    )
    print("PASS: Misaligned pair satisfied < 20% threshold.")

    # 3. Real satellite patch verification (12.tif vs 12_sar.tif)
    print("\n--- Test 3: Real Satellite Patch (12.tif vs 12_sar.tif) ---")
    res_real = run_crossmodal("12.tif", "12_sar.tif", "Verify real satellite optical-SAR alignment")
    print(f"Decision: {res_real['decision']}")
    print(f"Alignment Probability: {res_real['alignment_probability'] * 100:.2f}%")
    assert res_real["alignment_probability"] >= 0.80, (
        f"Expected real patch alignment >= 0.80, got {res_real['alignment_probability']}"
    )
    print("PASS: Real satellite patch satisfied > 80% threshold.")

    print("\n" + "=" * 60)
    print("ALL MODEL 3 VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
