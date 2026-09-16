from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


def pool_semantic_views(
    image_features: torch.Tensor,
    pixel_values: torch.Tensor,
    num_views: int = 3,
) -> torch.Tensor:
    """Average Idefics image tiles back into the user-supplied semantic views."""
    encoded_images = image_features.mean(dim=-2)
    real_tiles = pixel_values.ne(0).any(dim=(-1, -2, -3))
    counts = real_tiles.sum(dim=1).tolist()
    samples = []
    cursor = 0
    for count in counts:
        if count == 0 or count % num_views:
            raise ValueError(
                f"Encoded tile count {count} is not divisible by {num_views} semantic views"
            )
        sample_tiles = encoded_images[cursor : cursor + count]
        tiles_per_view = count // num_views
        samples.append(sample_tiles.reshape(num_views, tiles_per_view, -1).mean(dim=1))
        cursor += count
    if cursor != encoded_images.shape[0]:
        raise ValueError("Image feature count does not match non-padding pixel tiles")
    return torch.stack(samples)


class PixelShuffleTokens(nn.Module):
    """Space-to-depth token compression (paper factor r reduces H/W by r)."""

    def __init__(self, factor: int = 4) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError("factor must be positive")
        self.factor = factor

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, count, channels = tokens.shape
        if count != height * width:
            raise ValueError("token count must equal height * width")
        if height % self.factor or width % self.factor:
            raise ValueError("height and width must be divisible by factor")
        feature = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        compressed = functional.pixel_unshuffle(feature, self.factor)
        return compressed.flatten(2).transpose(1, 2)


class DecoupledContrastiveLoss(nn.Module):
    """Symmetric DCL: positives are excluded from the negative log-sum-exp."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError("DCL inputs must have matching [batch, features] shape")
        if first.shape[0] < 2:
            return first.sum() * 0.0
        # Contrastive logits are deliberately FP32 even under BF16/FP16 model
        # execution; PEFT may expose the two modalities at different dtypes.
        first = functional.normalize(first.float(), dim=-1)
        second = functional.normalize(second.float(), dim=-1)
        logits = first @ second.T / self.temperature
        positive = logits.diagonal()
        mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        negative_forward = logits.masked_fill(mask, float("-inf")).logsumexp(dim=1)
        negative_backward = logits.T.masked_fill(mask, float("-inf")).logsumexp(dim=1)
        return 0.5 * ((-positive + negative_forward).mean() + (-positive + negative_backward).mean())


class InfoNCELoss(nn.Module):
    """Symmetric in-batch image-text InfoNCE loss."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if images.shape != text.shape or images.ndim != 2:
            raise ValueError("InfoNCE inputs must have matching [batch, features] shape")
        if images.shape[0] < 2:
            return images.sum() * 0.0
        images = functional.normalize(images.float(), dim=-1)
        text = functional.normalize(text.float(), dim=-1)
        logits = images @ text.T / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (
            functional.cross_entropy(logits, targets)
            + functional.cross_entropy(logits.T, targets)
        )
