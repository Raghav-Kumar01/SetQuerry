from pathlib import Path
import csv

import torch
from torch.utils.data import Dataset

from ml.data.image_utils import load_image_as_tensor


class Model1Vocabulary:
    """Deterministic Model 1 vocabulary built from training annotations."""

    PAD_TOKEN = "<pad>"
    UNKNOWN_TOKEN = "<unk>"

    def __init__(
        self,
        question_to_id,
        answer_to_id,
        max_question_length=32
    ):
        self.question_to_id = dict(question_to_id)
        self.answer_to_id = dict(answer_to_id)
        self.max_question_length = max_question_length

    @classmethod
    def from_manifest(
        cls,
        manifest_path,
        max_question_length=32
    ):
        question_tokens = set()
        answers = set()

        with Path(manifest_path).open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row.get("split") != "train":
                    continue

                question_tokens.update(
                    cls.tokenize(row["question"])
                )
                answers.add(row["answer"])

        question_to_id = {
            cls.PAD_TOKEN: 0,
            cls.UNKNOWN_TOKEN: 1,
        }

        for token in sorted(question_tokens):
            question_to_id[token] = len(question_to_id)

        answer_to_id = {
            answer: index
            for index, answer in enumerate(sorted(answers))
        }

        return cls(
            question_to_id=question_to_id,
            answer_to_id=answer_to_id,
            max_question_length=max_question_length
        )

    @staticmethod
    def tokenize(question):
        return question.lower().split()

    def encode_question(self, question):
        tokens = self.tokenize(question)
        tokens = tokens[:self.max_question_length]
        unknown_id = self.question_to_id[self.UNKNOWN_TOKEN]

        encoded = [
            self.question_to_id.get(token, unknown_id)
            for token in tokens
        ]

        encoded.extend(
            [
                self.question_to_id[self.PAD_TOKEN]
                for _ in range(
                    self.max_question_length - len(encoded)
                )
            ]
        )

        return encoded

    def encode_answer(self, answer):
        if answer not in self.answer_to_id:
            raise ValueError(
                "Answer is absent from the training vocabulary: "
                f"{answer!r}"
            )

        return self.answer_to_id[answer]

    @property
    def question_size(self):
        return len(self.question_to_id)

    @property
    def answer_size(self):
        return len(self.answer_to_id)


class Model1Dataset(Dataset):
    """RSVQA-HR Dataset with explicit question and answer encoding."""

    DATASET_SUBDIRECTORY = "single"

    def __init__(
        self,
        manifest_path,
        dataset_root,
        split,
        max_samples=None,
        vocabulary=None,
        max_question_length=32
    ):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.split = split

        if vocabulary is None:
            if split != "train":
                raise ValueError(
                    "Validation/test datasets require the training "
                    "vocabulary explicitly"
                )

            vocabulary = Model1Vocabulary.from_manifest(
                self.manifest_path,
                max_question_length=max_question_length
            )

        self.vocabulary = vocabulary
        self.samples = []

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:
            reader = csv.DictReader(file)

            for row_number, row in enumerate(reader):
                if row["split"] != split:
                    continue

                sample = dict(row)
                sample.setdefault(
                    "sample_id",
                    f"{row.get('image_id', '')}:{row_number}"
                )
                self.samples.append(sample)

                if (
                    max_samples is not None
                    and len(self.samples) >= max_samples
                ):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = (
            self.dataset_root
            / self.DATASET_SUBDIRECTORY
            / sample["split"]
            / sample["image_path"]
        )

        image = load_image_as_tensor(image_path)
        question_ids = self.vocabulary.encode_question(
            sample["question"]
        )

        answer_known = sample["answer"] in self.vocabulary.answer_to_id
        if answer_known:
            target = self.vocabulary.encode_answer(sample["answer"])
        elif self.split == "train":
            # Training must never contain an answer outside its vocabulary.
            target = self.vocabulary.encode_answer(sample["answer"])
        else:
            # -1 is an evaluation-only sentinel, not an answer class.
            target = -1

        return {
            "sample_id": sample["sample_id"],
            "image_id": sample.get("image_id", ""),
            "image": image,
            "question": sample["question"],
            "question_ids": torch.tensor(
                question_ids,
                dtype=torch.long
            ),
            "answer": sample["answer"],
            "answer_known": answer_known,
            "target": torch.tensor(
                target,
                dtype=torch.long
            ),
        }
