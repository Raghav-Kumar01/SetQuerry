import torch
import torch.nn as nn


class SensorEncoder(nn.Module):
    """
    Small CNN encoder for one sensor.
    """

    def __init__(
        self,
        input_channels
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

    def forward(self, image):

        features = self.encoder(image)

        features = torch.flatten(
            features,
            start_dim=1
        )

        return features


class OpticalSARFusionModel(nn.Module):
    """
    Model 3 for SetQuerry.

    Inputs:
        optical image
        SAR image

    Output:
        scene / land-cover scores
    """

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        # RGB optical branch
        self.optical_encoder = SensorEncoder(
            input_channels=3
        )

        # VV + VH SAR branch
        self.sar_encoder = SensorEncoder(
            input_channels=2
        )

        # 64 optical features
        # +
        # 64 SAR features
        # =
        # 128 combined features

        self.fusion_classifier = nn.Sequential(

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(
        self,
        optical_image,
        sar_image
    ):

        optical_features = self.optical_encoder(
            optical_image
        )

        sar_features = self.sar_encoder(
            sar_image
        )

        fused_features = torch.cat(
            [
                optical_features,
                sar_features
            ],
            dim=1
        )

        output = self.fusion_classifier(
            fused_features
        )

        return output