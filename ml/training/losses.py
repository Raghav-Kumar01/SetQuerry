import torch.nn as nn


def get_classification_loss():
    """
    Loss for classification tasks.

    Used by:
    Model 1 - answer classification
    Model 3 - scene / land-cover classification
    """

    return nn.CrossEntropyLoss()


def get_segmentation_loss():
    """
    Loss for binary change detection.

    Used by:
    Model 2 - change mask prediction
    """

    return nn.BCEWithLogitsLoss()