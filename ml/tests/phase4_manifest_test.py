from pathlib import Path

from ml.data.manifest_reader import ManifestReader


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


MODEL1_MANIFEST = (
    PROJECT_ROOT
    / "data_pipeline"
    / "manifests"
    / "model1"
    / "model1_final_manifest.csv"
)


MODEL2_MANIFEST = (
    PROJECT_ROOT
    / "data_pipeline"
    / "manifests"
    / "model2"
    / "model2_final_manifest.csv"
)


MODEL3_MANIFEST = (
    PROJECT_ROOT
    / "data_pipeline"
    / "manifests"
    / "model3_final_manifest.csv"
)


def verify_manifest(
    model_name,
    manifest_path,
    required_columns,
    expected_counts
):
    """
    Verify one prepared manifest.

    READ ONLY.
    """

    print()
    print("-" * 60)
    print(model_name)
    print("-" * 60)

    reader = ManifestReader(
        manifest_path
    )

    columns = reader.get_columns()

    print(
        "Manifest:",
        manifest_path
    )

    print(
        "Columns:",
        columns
    )

    missing_columns = (
        required_columns
        -
        set(columns)
    )

    assert not missing_columns, (
        f"{model_name}: "
        f"missing columns "
        f"{missing_columns}"
    )

    summary = (
        reader.summarize_splits()
    )

    print(
        "Train:",
        summary["train"]
    )

    print(
        "Validation:",
        summary["val"]
    )

    print(
        "Test:",
        summary["test"]
    )

    print(
        "Other:",
        summary["other"]
    )

    print(
        "Total:",
        summary["total"]
    )

    assert (
        summary["train"]
        ==
        expected_counts["train"]
    )

    assert (
        summary["val"]
        ==
        expected_counts["val"]
    )

    assert (
        summary["test"]
        ==
        expected_counts["test"]
    )

    assert (
        summary["total"]
        ==
        expected_counts["total"]
    )

    assert summary["other"] == 0

    first_row = reader.first_row()

    assert first_row is not None

    print(
        "First sample ID:",
        first_row.get(
            "sample_id",
            first_row.get(
                "pair_id",
                "N/A"
            )
        )
    )

    print(
        f"{model_name} MANIFEST PASSED"
    )


def main():

    print("=" * 60)
    print("SETQUERY PHASE 4")
    print("ALL REAL MANIFESTS VERIFICATION")
    print("=" * 60)

    # ------------------------------------------
    # MODEL 1
    # ------------------------------------------

    verify_manifest(

        model_name="MODEL 1 - RSVQA-HR",

        manifest_path=MODEL1_MANIFEST,

        required_columns={
            "sample_id",
            "image_path",
            "question",
            "answer",
            "split"
        },

        expected_counts={
            "train": 625340,
            "val": 102843,
            "test": 222684,
            "total": 950867
        }
    )

    # ------------------------------------------
    # MODEL 2
    # ------------------------------------------

    verify_manifest(

        model_name="MODEL 2 - LEVIR-CD",

        manifest_path=MODEL2_MANIFEST,

        required_columns={
            "pair_id",
            "before_image_path",
            "after_image_path",
            "mask_path",
            "split"
        },

        expected_counts={
            "train": 445,
            "val": 64,
            "test": 128,
            "total": 637
        }
    )

    # ------------------------------------------
    # MODEL 3
    # ------------------------------------------

    verify_manifest(

        model_name="MODEL 3 - SEN1-2",

        manifest_path=MODEL3_MANIFEST,

        required_columns={
            "pair_id",
            "optical_path",
            "sar_path",
            "split",
            "season",
            "scene_id"
        },

        expected_counts={
            "train": 52814,
            "val": 10236,
            "test": 12674,
            "total": 75724
        }
    )

    print()
    print("=" * 60)

    print(
        "PHASE 4 MANIFEST VERIFICATION PASSED"
    )

    print(
        "ALL THREE PREPARED DATA CONTRACTS ARE VALID"
    )

    print(
        "NO DATA FILES WERE MODIFIED"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()