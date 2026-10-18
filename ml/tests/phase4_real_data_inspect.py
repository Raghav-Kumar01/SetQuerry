from pathlib import Path
import argparse

from PIL import Image


def inspect_image(label, image_path):
    image_path = Path(image_path)

    print()
    print("-" * 60)
    print(label)
    print("-" * 60)

    print("Path:", image_path)
    print("Exists:", image_path.exists())

    if not image_path.exists():
        print("FILE NOT FOUND")
        return

    with Image.open(image_path) as image:
        print("PIL mode:", image.mode)
        print("Size:", image.size)
        print("Format:", image.format)

        try:
            print(
                "Number of bands:",
                len(image.getbands())
            )

            print(
                "Bands:",
                image.getbands()
            )

        except Exception:
            print(
                "Band information unavailable"
            )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Inspect one real sample "
            "from each SetQuerry dataset."
        )
    )

    parser.add_argument(
        "--model1-root",
        type=str,
        required=True
    )

    parser.add_argument(
        "--model2-root",
        type=str,
        required=True
    )

    parser.add_argument(
        "--model3-root",
        type=str,
        required=True
    )

    args = parser.parse_args()

    model1_root = Path(
        args.model1_root
    )

    model2_root = Path(
        args.model2_root
    )

    model3_root = Path(
        args.model3_root
    )

    print("=" * 60)
    print("SETQUERY PHASE 4")
    print("REAL DATA IMAGE INSPECTION")
    print("=" * 60)

    # --------------------------------------
    # MODEL 1
    # --------------------------------------

    inspect_image(
        "MODEL 1 - RSVQA-HR IMAGE",
        model1_root
        / "images"
        / "12.tif"
    )

    # --------------------------------------
    # MODEL 2
    # --------------------------------------

    inspect_image(
        "MODEL 2 - BEFORE IMAGE",
        model2_root
        / "train"
        / "A"
        / "train_1.png"
    )

    inspect_image(
        "MODEL 2 - AFTER IMAGE",
        model2_root
        / "train"
        / "B"
        / "train_1.png"
    )

    inspect_image(
        "MODEL 2 - CHANGE MASK",
        model2_root
        / "train"
        / "label"
        / "train_1.png"
    )

    # --------------------------------------
    # MODEL 3
    # --------------------------------------

    inspect_image(
        "MODEL 3 - OPTICAL IMAGE",
        model3_root
        / "ROIs1158_spring"
        / "s2_0"
        / "ROIs1158_spring_s2_0_p1.png"
    )

    inspect_image(
        "MODEL 3 - SAR IMAGE",
        model3_root
        / "ROIs1158_spring"
        / "s1_0"
        / "ROIs1158_spring_s1_0_p1.png"
    )

    print()
    print("=" * 60)
    print("REAL DATA INSPECTION FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()