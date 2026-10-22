"""Position IDs shared by training and standalone inference models."""

import torch


def create_position_ids(
    input_ids: torch.Tensor, token_padding_idx: int, position_padding_idx: int
) -> torch.Tensor:
    """Create positions without changing a pretrained positional offset."""
    mask = input_ids.ne(token_padding_idx).to(dtype=torch.long)
    return mask.cumsum(dim=-1) * mask + position_padding_idx
