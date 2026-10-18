import argparse

import torch

from ml.data.dataloader import create_dataloader
from ml.data.model1_dataset import (
    Model1Dataset,
    Model1Vocabulary,
)
from ml.models.single_image_model import SingleImageVQAModel
from ml.training.losses import get_classification_loss
from ml.training.trainer import unpack_classification_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()

    vocabulary = Model1Vocabulary.from_manifest(
        args.manifest
    )

    dataset = Model1Dataset(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        split="train",
        max_samples=1,
        vocabulary=vocabulary,
    )

    loader = create_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    batch = next(iter(loader))
    image, question_ids, target = (
        unpack_classification_batch(batch)
    )

    model = SingleImageVQAModel(
        vocab_size=vocabulary.question_size,
        num_answers=vocabulary.answer_size,
    )
    criterion = get_classification_loss()

    with torch.no_grad():
        logits = model(image, question_ids)
        loss = criterion(logits, target)

    assert tuple(image.shape) == (1, 3, 512, 512)
    assert question_ids.dtype == torch.long
    assert target.dtype == torch.long
    assert tuple(logits.shape) == (
        1,
        vocabulary.answer_size,
    )
    assert loss.ndim == 0

    print("MODEL 1 REAL DATA TEST PASSED")
    print("question_vocab_size=", vocabulary.question_size)
    print("answer_vocab_size=", vocabulary.answer_size)
    print("image_shape=", tuple(image.shape))
    print("question_shape=", tuple(question_ids.shape))
    print("target_shape=", tuple(target.shape))
    print("logits_shape=", tuple(logits.shape))
    print("loss=", float(loss))


if __name__ == "__main__":
    main()
