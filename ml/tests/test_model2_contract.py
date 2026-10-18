"""Model 2 target semantics and archive-free dataset contract."""

import csv

from PIL import Image

from ml.data.model2_dataset import Model2Dataset


def test_model2_reads_ordered_images_and_binary_mask(tmp_path):
    for folder, color in (("A", (255, 0, 0)), ("B", (0, 0, 255))):
        target = tmp_path / "train" / folder
        target.mkdir(parents=True)
        Image.new("RGB", (4, 4), color).save(target / "pair.png")
    labels = tmp_path / "train" / "label"
    labels.mkdir(parents=True)
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((0, 0), 255)
    mask.save(labels / "pair.png")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pair_id", "before_image_path", "after_image_path", "mask_path", "split"])
        writer.writeheader()
        writer.writerow({"pair_id": "pair", "before_image_path": "train/A/pair.png", "after_image_path": "train/B/pair.png", "mask_path": "train/label/pair.png", "split": "train"})
    sample = Model2Dataset(manifest, tmp_path, "train")[0]
    assert sample["before"].shape == sample["after"].shape == (3, 512, 512)
    assert sample["mask"].shape == (1, 512, 512)
    assert set(sample["mask"].unique().tolist()) == {0.0, 1.0}
    assert sample["before"][0, 0, 0] > sample["after"][0, 0, 0]
