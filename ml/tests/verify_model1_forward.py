import sys
import json
from pathlib import Path
import torch

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.image_utils import load_image_as_tensor
from ml.data.model1_dataset import Model1Vocabulary
from ml.models.single_image_model import SingleImageVQAModel
from ml.training.checkpoint import load_checkpoint
from ml.inference.assets import dataset_root, require_checkpoint


def load_model1_vocabulary() -> Model1Vocabulary:
    cache_path = ROOT / "ml" / "data" / "model1_vocab_cache.json"
    if cache_path.is_file():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Model1Vocabulary(
            question_to_id=data["question_to_id"],
            answer_to_id=data["answer_to_id"],
            max_question_length=data.get("max_question_length", 32),
        )
    manifest_path = dataset_root(1) / "single" / "train" / "manifest.csv"
    return Model1Vocabulary.from_manifest(manifest_path)


def main():
    print("=" * 60)
    print("VERIFYING MODEL 1 FORWARD PASS (SINGLE IMAGE VQA)")
    print("=" * 60)

    # 1. Load Vocabulary
    vocab = load_model1_vocabulary()
    print(f"Loaded Vocabulary: {vocab.question_size} question tokens, {vocab.answer_size} answer classes.")

    # 2. Instantiate Model and Load Checkpoint
    model = SingleImageVQAModel(
        vocab_size=vocab.question_size,
        num_answers=vocab.answer_size,
    )
    checkpoint_path = require_checkpoint(1)
    print(f"Loading checkpoint weights from {checkpoint_path}...")
    load_checkpoint(model, optimizer=None, checkpoint_path=checkpoint_path, device=torch.device("cpu"))
    model.eval()

    # 3. Load 12.tif
    image_path = dataset_root(1) / "single" / "train" / "images" / "12.tif"
    if not image_path.exists():
        raise FileNotFoundError(f"Target image not found at: {image_path}")

    image_tensor = load_image_as_tensor(image_path)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    print(f"Loaded Image Tensor: shape={image_tensor.shape}, dtype={image_tensor.dtype}")

    # 4. Tokenize Question
    question = "Is a nature reserve present?"
    question_tokens = vocab.encode_question(question)
    question_tensor = torch.tensor([question_tokens], dtype=torch.long)
    print(f"Question: '{question}' -> encoded tokens shape={question_tensor.shape}")

    # 5. Forward Pass
    with torch.no_grad():
        logits = model(image_tensor, question_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()

    # 6. Map to Answer String
    id_to_answer = {idx: ans for ans, idx in vocab.answer_to_id.items()}
    pred_answer = id_to_answer.get(pred_idx, "unknown")

    print(f"Logits shape: {logits.shape}")
    print(f"Predicted index: {pred_idx}")
    print(f"Predicted Answer: '{pred_answer}'")
    print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")

    # 7. Assertions
    assert logits is not None and logits.numel() > 0, "Logits tensor is empty!"
    assert logits.shape == (1, vocab.answer_size), f"Expected shape (1, {vocab.answer_size}), got {logits.shape}"
    assert isinstance(pred_answer, str) and len(pred_answer) > 0, "Prediction string is invalid or empty!"

    print("\nMODEL 1 FORWARD PASS: PASS")
    print("=" * 60)
    return logits, pred_answer, confidence


if __name__ == "__main__":
    main()
