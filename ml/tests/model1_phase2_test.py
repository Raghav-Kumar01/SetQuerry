import csv
import tempfile
from pathlib import Path

import torch
from PIL import Image

from ml.data.model1_dataset import Model1Dataset, Model1Vocabulary
from ml.models.single_image_model import SingleImageVQAModel
from ml.training.checkpoint import load_checkpoint, save_checkpoint
from ml.training.model1_checkpoint import (
    build_model1_checkpoint_metadata,
    model1_constructor_config,
)


FIELDS = [
    "dataset_name",
    "image_id",
    "image_path",
    "question",
    "answer",
    "caption",
    "split",
    "source",
    "location",
    "sensor",
    "acquisition_date",
]


def write_manifest(path, split, image_name, answer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({
            "dataset_name": "synthetic",
            "image_id": image_name,
            "image_path": f"images/{image_name}",
            "question": "what is here",
            "answer": answer,
            "caption": "",
            "split": split,
            "source": "synthetic",
            "location": "",
            "sensor": "",
            "acquisition_date": "",
        })


def test_dataset_path_contract():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
        root = Path(temporary_directory)
        for split, image_name, answer in [
            ("train", "example.tif", "yes"),
            ("val", "val.tif", "yes"),
            ("test", "test.tif", "yes"),
        ]:
            image_directory = root / "single" / split / "images"
            image_directory.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), color=(10, 20, 30)).save(
                image_directory / image_name
            )
            write_manifest(
                root / "single" / split / "manifest.csv",
                split,
                image_name,
                answer,
            )

        vocabulary = Model1Vocabulary.from_manifest(
            root / "single" / "train" / "manifest.csv"
        )
        for split, image_name in [("train", "example.tif"),
                                  ("val", "val.tif"),
                                  ("test", "test.tif")]:
            dataset = Model1Dataset(
                manifest_path=root / "single" / split / "manifest.csv",
                dataset_root=root,
                split=split,
                vocabulary=vocabulary,
                max_samples=1,
            )
            sample = dataset[0]
            # The loader contract resizes every source image to the model's 512 square input.
            assert sample["image"].shape == (3, 512, 512)
            assert sample["image_id"] == image_name


def test_model1_checkpoint_round_trip():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
        root = Path(temporary_directory)
        vocabulary = Model1Vocabulary(
            question_to_id={"<pad>": 0, "<unk>": 1, "what": 2},
            answer_to_id={f"answer_{index}": index for index in range(6359)},
            max_question_length=32,
        )
        model = SingleImageVQAModel(
            vocab_size=vocabulary.question_size,
            num_answers=vocabulary.answer_size,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        metadata = build_model1_checkpoint_metadata(vocabulary, model)
        checkpoint_path = root / "model1.pt"

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=2,
            metric=0.25,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
        )

        restored_model = SingleImageVQAModel(
            **model1_constructor_config(metadata)
        )
        restored_optimizer = torch.optim.AdamW(
            restored_model.parameters(), lr=0.001
        )
        epoch, metric, restored_metadata = load_checkpoint(
            model=restored_model,
            optimizer=restored_optimizer,
            checkpoint_path=checkpoint_path,
            device="cpu",
            return_metadata=True,
        )

        assert epoch == 2
        assert metric == 0.25
        assert restored_metadata["answer_to_id"] == metadata["answer_to_id"]
        assert restored_metadata["id_to_answer"] == metadata["id_to_answer"]
        assert restored_metadata["question_to_id"] == metadata["question_to_id"]
        assert restored_metadata["max_question_length"] == 32
        restored_model.eval()
        with torch.no_grad():
            logits = restored_model(
                torch.randn(1, 3, 32, 32),
                torch.ones(1, 32, dtype=torch.long),
            )
        assert logits.shape == (1, 6359)


if __name__ == "__main__":
    test_dataset_path_contract()
    test_model1_checkpoint_round_trip()
    print("MODEL 1 PHASE 2 TESTS PASSED")
