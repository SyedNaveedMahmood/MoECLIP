"""Region pooling and registration-tolerant thermal context for RGB routing.

The functions in this module never consume anomaly masks.  RGB SLIC labels are
reduced to the CLIP patch grid, RGB patch features are pooled per region, and
each region may attend over the *entire* thermal grid.  Matching RGB/thermal
token indices are therefore not assumed to be spatially registered.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RegionPoolOutput:
    """Padded region statistics for a batch of RGB patch tokens."""

    region_features: torch.Tensor
    region_coordinates: torch.Tensor
    valid_regions: torch.Tensor
    region_counts: torch.Tensor
    patch_region_indices: torch.Tensor
    grid_size: Tuple[int, int]


@dataclass(frozen=True)
class RegionContextOutput:
    """Region and patch conditioning features plus interpretable attention."""

    region_context: torch.Tensor
    patch_context: torch.Tensor
    thermal_region_features: torch.Tensor
    thermal_attention: torch.Tensor
    pool: RegionPoolOutput


def _validate_grid_size(grid_size: Tuple[int, int]) -> Tuple[int, int]:
    if len(grid_size) != 2:
        raise ValueError("grid_size must contain (height, width)")
    height, width = int(grid_size[0]), int(grid_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("grid dimensions must be positive")
    return height, width


def _normalized_grid_coordinates(
    grid_size: Tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return patch-center ``(x, y)`` coordinates in ``[-1, 1]``."""

    height, width = _validate_grid_size(grid_size)
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    y = y.mul(2.0).sub(1.0)
    x = x.mul(2.0).sub(1.0)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(height * width, 2)


def pixel_regions_to_patch_regions(
    region_map: torch.Tensor,
    grid_size: Tuple[int, int],
) -> torch.Tensor:
    """Assign each patch the modal RGB SLIC label among its source pixels.

    Args:
        region_map: Integer SLIC labels with shape ``[B,H,W]``.
        grid_size: Target RGB patch grid ``(Hp,Wp)``.

    Returns:
        Integer region labels with shape ``[B,Hp*Wp]``. Ties follow
        ``torch.mode``'s deterministic smallest-label convention.
    """

    if region_map.ndim != 3:
        raise ValueError(
            f"region_map must have shape [B,H,W], got {tuple(region_map.shape)}"
        )
    if torch.is_floating_point(region_map) or region_map.dtype == torch.bool:
        raise TypeError("region_map must contain integer labels")
    if region_map.numel() and int(region_map.min()) < 0:
        raise ValueError("region_map labels must be non-negative")

    grid_height, grid_width = _validate_grid_size(grid_size)
    pixel_height, pixel_width = int(region_map.shape[1]), int(region_map.shape[2])
    if pixel_height % grid_height or pixel_width % grid_width:
        raise ValueError(
            f"region_map size {(pixel_height, pixel_width)} is not evenly divisible "
            f"by patch grid {(grid_height, grid_width)}"
        )
    patch_height = pixel_height // grid_height
    patch_width = pixel_width // grid_width
    windows = region_map.unfold(1, patch_height, patch_height).unfold(
        2, patch_width, patch_width
    )
    windows = windows.contiguous().view(
        region_map.shape[0], grid_height * grid_width, patch_height * patch_width
    )
    return windows.mode(dim=-1).values


def identity_patch_regions(
    batch_size: int,
    grid_size: Tuple[int, int],
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Give each RGB patch its own region for patch-conditioned ablations."""

    height, width = _validate_grid_size(grid_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    labels = torch.arange(height * width, device=device, dtype=torch.long)
    return labels.unsqueeze(0).expand(batch_size, -1)


def pool_patch_regions(
    patch_tokens: torch.Tensor,
    patch_region_ids: torch.Tensor,
    grid_size: Tuple[int, int],
) -> RegionPoolOutput:
    """Mean-pool RGB patches and coordinates independently within each image."""

    if patch_tokens.ndim != 3:
        raise ValueError("patch_tokens must have shape [B,N,D]")
    if patch_region_ids.ndim != 2:
        raise ValueError("patch_region_ids must have shape [B,N]")
    if torch.is_floating_point(patch_region_ids) or patch_region_ids.dtype == torch.bool:
        raise TypeError("patch_region_ids must contain integer labels")
    batch_size, token_count, feature_dim = patch_tokens.shape
    grid_height, grid_width = _validate_grid_size(grid_size)
    if token_count != grid_height * grid_width:
        raise ValueError(
            f"token count {token_count} does not match grid {grid_size}"
        )
    if patch_region_ids.shape != (batch_size, token_count):
        raise ValueError("patch tokens and region IDs have incompatible shapes")
    if patch_region_ids.device != patch_tokens.device:
        raise ValueError("patch tokens and region IDs must be on the same device")
    if patch_region_ids.numel() and int(patch_region_ids.min()) < 0:
        raise ValueError("patch region labels must be non-negative")

    patch_coordinates = _normalized_grid_coordinates(
        grid_size, device=patch_tokens.device, dtype=patch_tokens.dtype
    )
    feature_rows = []
    coordinate_rows = []
    count_rows = []
    inverse_rows = []
    for batch_index in range(batch_size):
        _, inverse = torch.unique(
            patch_region_ids[batch_index], sorted=True, return_inverse=True
        )
        region_count = int(inverse.max()) + 1
        counts = torch.bincount(inverse, minlength=region_count)
        feature_sums = patch_tokens.new_zeros((region_count, feature_dim))
        feature_sums.index_add_(0, inverse, patch_tokens[batch_index])
        coordinate_sums = patch_tokens.new_zeros((region_count, 2))
        coordinate_sums.index_add_(0, inverse, patch_coordinates)
        denominator = counts.to(dtype=patch_tokens.dtype).unsqueeze(-1)
        feature_rows.append(feature_sums / denominator)
        coordinate_rows.append(coordinate_sums / denominator)
        count_rows.append(counts)
        inverse_rows.append(inverse)

    max_regions = max(row.shape[0] for row in feature_rows)
    region_features = patch_tokens.new_zeros(
        (batch_size, max_regions, feature_dim)
    )
    region_coordinates = patch_tokens.new_zeros((batch_size, max_regions, 2))
    region_counts = torch.zeros(
        (batch_size, max_regions), dtype=torch.long, device=patch_tokens.device
    )
    valid_regions = torch.zeros(
        (batch_size, max_regions), dtype=torch.bool, device=patch_tokens.device
    )
    for batch_index, features in enumerate(feature_rows):
        count = features.shape[0]
        region_features[batch_index, :count] = features
        region_coordinates[batch_index, :count] = coordinate_rows[batch_index]
        region_counts[batch_index, :count] = count_rows[batch_index]
        valid_regions[batch_index, :count] = True

    return RegionPoolOutput(
        region_features=region_features,
        region_coordinates=region_coordinates,
        valid_regions=valid_regions,
        region_counts=region_counts,
        patch_region_indices=torch.stack(inverse_rows),
        grid_size=(grid_height, grid_width),
    )


class RegionThermalAttention(nn.Module):
    """Let RGB regions attend to all thermal patches with an optional soft prior."""

    def __init__(
        self,
        rgb_dim: int,
        thermal_dim: int,
        attention_dim: int = 256,
        num_heads: int = 4,
        coordinate_bias_strength: float = 1.0,
        coordinate_bias_sigma: float = 0.75,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(rgb_dim, thermal_dim, attention_dim, num_heads) <= 0:
            raise ValueError("feature dimensions and num_heads must be positive")
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        if coordinate_bias_strength < 0.0:
            raise ValueError("coordinate_bias_strength must be non-negative")
        if coordinate_bias_sigma <= 0.0:
            raise ValueError("coordinate_bias_sigma must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0,1)")

        self.attention_dim = int(attention_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.attention_dim // self.num_heads
        self.coordinate_bias_strength = float(coordinate_bias_strength)
        self.coordinate_bias_sigma = float(coordinate_bias_sigma)
        self.dropout = float(dropout)
        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.thermal_norm = nn.LayerNorm(thermal_dim)
        self.query = nn.Linear(rgb_dim, self.attention_dim, bias=False)
        self.key = nn.Linear(thermal_dim, self.attention_dim, bias=False)
        self.value = nn.Linear(thermal_dim, self.attention_dim, bias=False)
        self.output = nn.Linear(self.attention_dim, self.attention_dim, bias=False)

    def forward(
        self,
        rgb_regions: torch.Tensor,
        region_coordinates: torch.Tensor,
        valid_regions: torch.Tensor,
        thermal_tokens: torch.Tensor,
        thermal_grid_size: Tuple[int, int],
        thermal_available: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rgb_regions.ndim != 3 or thermal_tokens.ndim != 3:
            raise ValueError("RGB regions and thermal tokens must be rank-three")
        batch_size, region_count, _ = rgb_regions.shape
        if region_coordinates.shape != (batch_size, region_count, 2):
            raise ValueError("region coordinate shape does not match RGB regions")
        if valid_regions.shape != (batch_size, region_count):
            raise ValueError("valid-region mask shape does not match RGB regions")
        if thermal_tokens.shape[0] != batch_size:
            raise ValueError("RGB and thermal batch sizes differ")
        thermal_height, thermal_width = _validate_grid_size(thermal_grid_size)
        if thermal_tokens.shape[1] != thermal_height * thermal_width:
            raise ValueError("thermal token count does not match its grid")

        work_dtype = self.query.weight.dtype
        rgb_regions = rgb_regions.to(dtype=work_dtype)
        thermal_tokens = thermal_tokens.to(dtype=work_dtype)
        q = self.query(self.rgb_norm(rgb_regions))
        k = self.key(self.thermal_norm(thermal_tokens))
        v = self.value(self.thermal_norm(thermal_tokens))
        q = q.view(batch_size, region_count, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.coordinate_bias_strength > 0.0:
            thermal_coordinates = _normalized_grid_coordinates(
                (thermal_height, thermal_width),
                device=thermal_tokens.device,
                dtype=work_dtype,
            )
            offset = region_coordinates.to(dtype=work_dtype).unsqueeze(2)
            offset = offset - thermal_coordinates.view(1, 1, -1, 2)
            squared_distance = offset.square().sum(dim=-1)
            spatial_bias = -self.coordinate_bias_strength * squared_distance
            spatial_bias = spatial_bias / (2.0 * self.coordinate_bias_sigma**2)
            logits = logits + spatial_bias.unsqueeze(1)

        attention = F.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
        dropped_attention = F.dropout(
            attention, p=self.dropout, training=self.training
        )
        summary = torch.matmul(dropped_attention, v)
        summary = summary.transpose(1, 2).reshape(
            batch_size, region_count, self.attention_dim
        )
        summary = self.output(summary)
        mean_attention = attention.mean(dim=1)

        active = valid_regions
        if thermal_available is not None:
            if thermal_available.shape != (batch_size,):
                raise ValueError("thermal_available must have shape [B]")
            active = active & thermal_available.to(
                device=valid_regions.device, dtype=torch.bool
            ).unsqueeze(1)
        active_float = active.to(dtype=summary.dtype).unsqueeze(-1)
        summary = summary * active_float
        mean_attention = mean_attention * active_float
        return summary, mean_attention


class RegionContextEncoder(nn.Module):
    """Build multimodal region context and broadcast it back to RGB patches."""

    def __init__(
        self,
        rgb_dim: int,
        thermal_dim: int,
        context_dim: int = 256,
        num_heads: int = 4,
        coordinate_bias_strength: float = 1.0,
        coordinate_bias_sigma: float = 0.75,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        self.context_dim = int(context_dim)
        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.rgb_projection = nn.Linear(rgb_dim, self.context_dim, bias=False)
        self.thermal_attention = RegionThermalAttention(
            rgb_dim=rgb_dim,
            thermal_dim=thermal_dim,
            attention_dim=self.context_dim,
            num_heads=num_heads,
            coordinate_bias_strength=coordinate_bias_strength,
            coordinate_bias_sigma=coordinate_bias_sigma,
            dropout=dropout,
        )
        self.context_norm = nn.LayerNorm(self.context_dim * 4)
        self.context_mlp = nn.Sequential(
            nn.Linear(self.context_dim * 4, self.context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.context_dim, self.context_dim),
        )

    def forward(
        self,
        rgb_patch_tokens: torch.Tensor,
        patch_region_ids: torch.Tensor,
        rgb_grid_size: Tuple[int, int],
        thermal_tokens: Optional[torch.Tensor] = None,
        thermal_grid_size: Optional[Tuple[int, int]] = None,
        thermal_available: Optional[torch.Tensor] = None,
    ) -> RegionContextOutput:
        pool = pool_patch_regions(
            rgb_patch_tokens, patch_region_ids, rgb_grid_size
        )
        work_dtype = self.rgb_projection.weight.dtype
        rgb_regions = pool.region_features.to(dtype=work_dtype)
        projected_rgb = self.rgb_projection(self.rgb_norm(rgb_regions))

        if thermal_tokens is None:
            projected_thermal = torch.zeros_like(projected_rgb)
            attention = projected_rgb.new_zeros(
                (projected_rgb.shape[0], projected_rgb.shape[1], 0)
            )
        else:
            if thermal_grid_size is None:
                raise ValueError("thermal_grid_size is required with thermal tokens")
            projected_thermal, attention = self.thermal_attention(
                pool.region_features,
                pool.region_coordinates,
                pool.valid_regions,
                thermal_tokens,
                thermal_grid_size,
                thermal_available,
            )

        combined = torch.cat(
            (
                projected_rgb,
                projected_thermal,
                projected_rgb - projected_thermal,
                projected_rgb * projected_thermal,
            ),
            dim=-1,
        )
        region_context = self.context_mlp(self.context_norm(combined))
        region_context = region_context * pool.valid_regions.to(
            dtype=region_context.dtype
        ).unsqueeze(-1)
        gather_index = pool.patch_region_indices.unsqueeze(-1).expand(
            -1, -1, self.context_dim
        )
        patch_context = torch.gather(region_context, 1, gather_index)
        return RegionContextOutput(
            region_context=region_context,
            patch_context=patch_context,
            thermal_region_features=projected_thermal,
            thermal_attention=attention,
            pool=pool,
        )


__all__ = [
    "RegionContextEncoder",
    "RegionContextOutput",
    "RegionPoolOutput",
    "RegionThermalAttention",
    "identity_patch_regions",
    "pixel_regions_to_patch_regions",
    "pool_patch_regions",
]
