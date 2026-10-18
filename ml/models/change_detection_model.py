import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """
    Shared encoder used for both
    before and after images.
    """

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

    def forward(self, image):

        return self.features(image)


class ChangeDetectionModel(nn.Module):
    """
    Model 2 for SetQuerry.

    Inputs:
        before image
        after image

    Output:
        pixel-level change map
    """

    def __init__(self):

        super().__init__()

        # Same encoder is used
        # for before and after images.
        self.encoder = FeatureEncoder()

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=2,
                stride=2
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=2,
                stride=2
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=16,
                out_channels=1,
                kernel_size=1
            )
        )

    def forward(
        self,
        before_image,
        after_image
    ):

        before_features = self.encoder(
            before_image
        )

        after_features = self.encoder(
            after_image
        )

        difference = torch.abs(
            before_features
            -
            after_features
        )

        change_logits = self.decoder(
            difference
        )

        return change_logits