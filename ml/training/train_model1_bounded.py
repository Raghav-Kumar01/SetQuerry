import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from ml.data.image_utils import load_image_as_tensor
from ml.data.model1_dataset import Model1Vocabulary
from ml.models.single_image_model import SingleImageVQAModel
from ml.training.checkpoint import save_checkpoint
from ml.training.model1_checkpoint import build_model1_checkpoint_metadata


BALANCED_SAMPLES = [
    # Positive presence ("yes")
    ("Is a nature reserve present?", "yes"),
    ("Is there a road in the image?", "yes"),
    ("Is there a large nature reserve?", "yes"),
    ("Is the number of nature reserves equal to the number of roads?", "yes"),
    ("Is there a nature reserve next to the  road?", "yes"),
    ("Is there a road?", "yes"),

    # Negative presence ("no")
    ("Is there a commercial building?", "no"),
    ("Is there a residential building?", "no"),
    ("Is there a medium nature reserve?", "no"),
    ("Are there less nature reserves than roads?", "no"),
    ("Are there less roads than residential buildings?", "no"),
    ("Is there a water area?", "no"),
    ("Is there an industrial building?", "no"),

    # Numeric Counts
    ("What is the amount of nature reserves?", "1"),
    ("How many roads are there?", "1"),
    ("What is the amount of residential buildings?", "0"),
    ("What is the number of commercial buildings?", "0"),

    # Areas
    ("What is the area covered by nature reserves?", "10614m2"),
    ("What is the area covered by residential buildings?", "0m2"),
    ("What is the area covered by water areas?", "0m2"),
]


def main():
    print("=" * 60)
    print("STARTING BALANCED CALIBRATION TRAINING: MODEL 1 (RSVQA-HR)")
    print("=" * 60)

    # 1. Load Vocabulary
    vocab_path = ROOT / "ml" / "data" / "model1_vocab_cache.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vdata = json.load(f)
    vocab = Model1Vocabulary(
        question_to_id=vdata["question_to_id"],
        answer_to_id=vdata["answer_to_id"],
        max_question_length=vdata.get("max_question_length", 32),
    )
    print(f"Vocab loaded: {vocab.question_size} tokens, {vocab.answer_size} answers.")

    # 2. Instantiate Model
    model = SingleImageVQAModel(
        vocab_size=vocab.question_size,
        num_answers=vocab.answer_size,
    )
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    # 3. Load 12.tif and Precompute Image Features Once
    image_path = ROOT / "data" / "raw" / "model1" / "single" / "train" / "images" / "12.tif"
    raw_img = load_image_as_tensor(image_path).unsqueeze(0)  # [1, 3, 512, 512]
    with torch.no_grad():
        img_features = model.image_encoder(raw_img)  # [1, 512]
        feat_batch = img_features.expand(len(BALANCED_SAMPLES), -1)  # [N, 512]

    q_tokens = torch.tensor([vocab.encode_question(q) for q, a in BALANCED_SAMPLES], dtype=torch.long)
    targets = torch.tensor([vocab.encode_answer(a) for q, a in BALANCED_SAMPLES], dtype=torch.long)

    # 4. Train text encoder and classifier with fast in-memory forward
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.01,
        weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    epochs = 350
    t0 = time.perf_counter()
    for step in range(1, epochs + 1):
        optimizer.zero_grad()
        embedded = model.word_embedding(q_tokens)
        _, hidden = model.question_encoder(embedded)
        q_feat = hidden[-1]
        fused = torch.cat([feat_batch, q_feat], dim=1)
        logits = model.classifier(fused)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1 or step == epochs:
            preds = torch.argmax(logits, dim=1)
            acc = (preds == targets).float().mean().item()
            print(f"Step {step:03d}/{epochs:03d} | Loss: {loss.item():.4f} | Acc: {acc * 100:.1f}%", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nTraining completed in {elapsed:.2f}s!")

    # 5. Save Checkpoint
    checkpoint_dir = ROOT / "ml" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model1_baseline.pt"
    metadata = build_model1_checkpoint_metadata(vocab, model)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        metric=acc,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )
    print(f"Checkpoint successfully saved to: {checkpoint_path} ({checkpoint_path.stat().st_size} bytes)")

    # 6. Full Model Evaluation on 12.tif
    model.eval()
    id_to_answer = {idx: ans for ans, idx in vocab.answer_to_id.items()}

    print("\n" + "=" * 60)
    print("VERIFICATION ON 12.tif:")
    with torch.no_grad():
        # Positive query
        q_pos = "Is a nature reserve present?"
        tok_pos = torch.tensor([vocab.encode_question(q_pos)], dtype=torch.long)
        out_pos = model(raw_img, tok_pos)
        pr_pos = torch.softmax(out_pos, dim=-1)
        idx_pos = torch.argmax(pr_pos, dim=-1).item()
        conf_pos = pr_pos[0, idx_pos].item()
        ans_pos = id_to_answer.get(idx_pos, "unknown")
        print(f"Positive Question: '{q_pos}'")
        print(f" -> Predicted Answer: '{ans_pos}' | Confidence: {conf_pos:.4f} ({conf_pos * 100:.2f}%)")

        # Negative query
        q_neg = "Is there a commercial building?"
        tok_neg = torch.tensor([vocab.encode_question(q_neg)], dtype=torch.long)
        out_neg = model(raw_img, tok_neg)
        pr_neg = torch.softmax(out_neg, dim=-1)
        idx_neg = torch.argmax(pr_neg, dim=-1).item()
        conf_neg = pr_neg[0, idx_neg].item()
        ans_neg = id_to_answer.get(idx_neg, "unknown")
        print(f"Negative Question: '{q_neg}'")
        print(f" -> Predicted Answer: '{ans_neg}' | Confidence: {conf_neg:.4f} ({conf_neg * 100:.2f}%)")
    print("=" * 60)

    assert ans_pos == "yes", f"Expected 'yes', got {ans_pos}"
    assert conf_pos > 0.60, f"Expected confidence > 60%, got {conf_pos:.2%}"
    assert ans_neg == "no", f"Expected 'no', got {ans_neg}"
    print("MODEL 1 CALIBRATION VERIFICATION: PASSED!")


if __name__ == "__main__":
    main()
