"""CPU fixtures for isolated training infrastructure tests."""

import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model():
    return torch.nn.Linear(2, 1)


@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)
