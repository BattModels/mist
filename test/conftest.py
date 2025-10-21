import random

import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seed():
    """Fix random seed for reproducibility"""
    seed = 41
    random.seed(seed)
    torch.manual_seed(seed)
