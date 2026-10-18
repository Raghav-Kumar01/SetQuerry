# Model 3 Validation Summary

## Manifest
- File: `model3_final_manifest.csv`
- Rows: 75,724
- Columns: pair_id, optical_path, sar_path, split, season, scene_id
- SHA-256: `cd81140c20788a16b13912404092247999c83c4a75b4129c96e1709c84256178`
- File size: 10,114,735 bytes

## Split
| Split | Scenes | Pairs | Percentage |
|---|---:|---:|---:|
| train | 49 | 52,814 | 69.7454% |
| val | 10 | 10,236 | 13.5175% |
| test | 11 | 12,674 | 16.7371% |

## Checks
- Required columns: PASS
- Missing values: PASS (0)
- Duplicate `pair_id`: PASS (0)
- Unique scenes: 70
- Season: PASS (`spring`)
- Optical/SAR project-relative path structure: PASS
- SAR/optical path pairing structure: PASS
- Train/val scene overlap: 0
- Train/test scene overlap: 0
- Val/test scene overlap: 0

## Approved Model 3 split
The finalized scene-isolated split uses seed 42:
- Train: 49 scenes / 52,814 pairs
- Validation: 10 scenes / 10,236 pairs
- Test: 11 scenes / 12,674 pairs

The complete SEN1-2 image dataset is intentionally **not included** in this handoff.
