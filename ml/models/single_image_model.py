import torch
import torch.nn as nn

from torchvision.models import resnet18


class SingleImageVQAModel(nn.Module):
    """
    Model 1 for SetQuerry.

    Inputs:
        image
        question token IDs

    Output:
        answer scores
    """

    def __init__(
        self,
        vocab_size,
        num_answers,
        text_embedding_dim=128,
        text_hidden_dim=128
    ):
        super().__init__()

        # ----------------------------------------
        # IMAGE ENCODER
        # ----------------------------------------

        self.image_encoder = resnet18(weights=None)

        image_feature_dim = self.image_encoder.fc.in_features

        self.image_encoder.fc = nn.Identity()

        # ----------------------------------------
        # TEXT ENCODER
        # ----------------------------------------

        self.word_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=text_embedding_dim,
            padding_idx=0
        )

        self.question_encoder = nn.GRU(
            input_size=text_embedding_dim,
            hidden_size=text_hidden_dim,
            batch_first=True
        )

        # ----------------------------------------
        # IMAGE + TEXT FUSION
        # ----------------------------------------

        fusion_dim = (
            image_feature_dim
            +
            text_hidden_dim
        )

        # ----------------------------------------
        # ANSWER CLASSIFIER
        # ----------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_answers)
        )

    def forward(
        self,
        image,
        question_tokens
    ):

        # Extract image features
        image_features = self.image_encoder(
            image
        )

        # Convert word IDs into embeddings
        embedded_words = self.word_embedding(
            question_tokens
        )

        # Read question sequence
        _, hidden_state = self.question_encoder(
            embedded_words
        )

        question_features = hidden_state[-1]

        # Join image and question knowledge
        fused_features = torch.cat(
            [
                image_features,
                question_features
            ],
            dim=1
        )

        # Generate answer scores
        output = self.classifier(
            fused_features
        )

        return output