import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from ml.data.image_utils import load_image_as_tensor
from ml.data.model1_dataset import Model1Vocabulary
from ml.models.single_image_model import SingleImageVQAModel
from ml.training.checkpoint import load_checkpoint
from ml.inference.assets import ModelUnavailableError, dataset_root, require_checkpoint, require_image
_MODEL1_SINGLETON = None
_VOCAB_SINGLETON = None
_ID_TO_ANSWER_SINGLETON = None
def get_cached_model_and_vocab():
    global _MODEL1_SINGLETON, _VOCAB_SINGLETON, _ID_TO_ANSWER_SINGLETON
    if _MODEL1_SINGLETON is not None:
        return _MODEL1_SINGLETON, _VOCAB_SINGLETON, _ID_TO_ANSWER_SINGLETON

    checkpoint_path = require_checkpoint(1)
    cache_path = ROOT / "ml" / "data" / "model1_vocab_cache.json"
    if cache_path.is_file():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _VOCAB_SINGLETON = Model1Vocabulary(
            question_to_id=data["question_to_id"],
            answer_to_id=data["answer_to_id"],
            max_question_length=data.get("max_question_length", 32),
        )
    else:
        raise ModelUnavailableError(f"Model 1 vocabulary is missing: {cache_path}")
    _ID_TO_ANSWER_SINGLETON = {idx: ans for ans, idx in _VOCAB_SINGLETON.answer_to_id.items()}
    model = SingleImageVQAModel(
        vocab_size=_VOCAB_SINGLETON.question_size,
        num_answers=_VOCAB_SINGLETON.answer_size,
    )
    load_checkpoint(model, optimizer=None, checkpoint_path=checkpoint_path, device=torch.device("cpu"))
    model.eval()
    _MODEL1_SINGLETON = model
    return _MODEL1_SINGLETON, _VOCAB_SINGLETON, _ID_TO_ANSWER_SINGLETON
def resolve_single_image(image_input: str) -> tuple[torch.Tensor, str]:
    """Dynamically resolves any valid image path or ID from disk or RSVQA dataset."""
    img_p = Path(image_input)
    # 1. Direct path on local filesystem
    if img_p.is_file():
        t = load_image_as_tensor(img_p)
        if t.ndim == 3:
            t = t.unsqueeze(0)
        return t, str(img_p)
    # 2. Check RSVQA-HR dataset directories on Google Drive across splits
    base_rsvqa = dataset_root(1) / "single"
    candidates = [img_p.name]
    if not img_p.suffix:
        candidates.append(f"{img_p.name}.tif")
    if base_rsvqa.exists():
        for split in ("train", "val", "test"):
            split_dir = base_rsvqa / split / "images"
            for c in candidates:
                cand_p = split_dir / c
                if cand_p.is_file():
                    t = load_image_as_tensor(cand_p)
                    if t.ndim == 3:
                        t = t.unsqueeze(0)
                    return t, str(cand_p)
    raise FileNotFoundError(f"Model 1 image is missing: {image_input!r}")
def run_single(image_input: str, query: str) -> dict:
    image_tensor, resolved_source = resolve_single_image(image_input)
    model, vocab, id_to_answer = get_cached_model_and_vocab()
    question_tokens = vocab.encode_question(query)
    question_tensor = torch.tensor([question_tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(image_tensor, question_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = round(float(probs[0, pred_idx].item()), 4)
    pred_answer = id_to_answer.get(pred_idx, "unclassified")
    return {
        "answer": pred_answer,
        "confidence": confidence,
        "confidence_semantics": "uncalibrated answer softmax probability",
        "pred_idx": pred_idx,
        "classes_count": len(id_to_answer),
        "resolved_source": resolved_source,
        "model_version": "v1.0.0-model1-baseline",
    }
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="")
    parser.add_argument("--query", default="Is a nature reserve present?")
    args = parser.parse_args()
    result = run_single(args.image, args.query)
    print(json.dumps(result))
if __name__ == "__main__":
    main()
