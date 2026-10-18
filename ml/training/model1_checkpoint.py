"""Model 1 checkpoint metadata helpers."""


def build_model1_checkpoint_metadata(vocabulary, model):
    """Build deterministic metadata required to reconstruct Model 1."""

    return {
        "model_type": "SingleImageVQAModel",
        "answer_to_id": dict(vocabulary.answer_to_id),
        "id_to_answer": {
            index: answer
            for answer, index in vocabulary.answer_to_id.items()
        },
        "question_to_id": dict(vocabulary.question_to_id),
        "max_question_length": vocabulary.max_question_length,
        "vocab_size": vocabulary.question_size,
        "num_answers": vocabulary.answer_size,
        "model_config": {
            "text_embedding_dim": model.word_embedding.embedding_dim,
            "text_hidden_dim": model.question_encoder.hidden_size,
        },
        "image_preprocessing": {
            "loader": "PIL plus torchvision.transforms.functional.to_tensor",
            "channels": 3,
            "resize": None,
            "normalization": None,
            "force_rgb": False,
        },
    }


def model1_constructor_config(metadata):
    """Return constructor arguments for SingleImageVQAModel metadata."""

    return {
        "vocab_size": metadata["vocab_size"],
        "num_answers": metadata["num_answers"],
        **metadata["model_config"],
    }
