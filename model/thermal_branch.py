"""Thermal feature encoders.

``ThermalEncoder`` is the v1 MulSen-AD branch.  It consumes the loader's
single-channel thermal tensor and returns spatial patch tokens used only as
router conditioning.  It deliberately has no CLS token and makes no claim that
its patch index is registered to the RGB CLIP patch with the same index.

The discarded exploratory TwinCLIP transformer is intentionally absent: it
created a second expert stream and an unregistered lazy positional tensor,
neither of which belongs to the region-conditioned v1 architecture.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class ThermalEncoderOutput:
    """Outputs from the lightweight thermal branch.

    Attributes:
        tokens: Final patch tokens with shape ``[B, Ht*Wt, output_dim]``.
        taps: One normalized token tensor after every spatial block.
        grid_size: Thermal token grid ``(Ht, Wt)``.
    """

    tokens: torch.Tensor
    taps: Tuple[torch.Tensor, ...]
    grid_size: Tuple[int, int]


class ThermalSpatialBlock(nn.Module):
    """A modest ConvNeXt-style local block on the thermal patch grid."""

    def __init__(
        self,
        width: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-4,
    ) -> None:
        super().__init__()
        hidden = int(round(width * mlp_ratio))
        if hidden <= 0:
            raise ValueError("mlp_ratio produces an empty hidden dimension")
        self.depthwise = nn.Conv2d(
            width, width, kernel_size=3, padding=1, groups=width, bias=True
        )
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )
        self.layer_scale = nn.Parameter(
            torch.full((width,), float(layer_scale_init), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x * self.layer_scale.to(dtype=x.dtype)
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ThermalEncoder(nn.Module):
    """One-channel, patch-grid-preserving thermal conditioning encoder.

    The default 14-pixel stem maps a 518x518 input to the same 37x37 grid used
    by CLIP ViT-L/14.  Matching grid dimensions are useful for common tensor
    shapes but do not imply spatial registration; the region router will use a
    learned correspondence module instead of same-index fusion.
    """

    def __init__(
        self,
        width: int = 256,
        output_dim: int = 256,
        depth: int = 4,
        patch_size: int = 14,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-4,
    ) -> None:
        super().__init__()
        if width <= 0 or output_dim <= 0:
            raise ValueError("width and output_dim must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if layer_scale_init < 0.0:
            raise ValueError("layer_scale_init must be non-negative")

        self.width = int(width)
        self.output_dim = int(output_dim)
        self.depth = int(depth)
        self.patch_size = int(patch_size)
        self.patch_embed = nn.Conv2d(
            1,
            self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.stem_norm = nn.LayerNorm(self.width)
        self.blocks = nn.ModuleList(
            ThermalSpatialBlock(
                self.width,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(self.depth)
        )
        self.tap_norms = nn.ModuleList(
            nn.LayerNorm(self.width) for _ in range(self.depth)
        )
        self.tap_projections = nn.ModuleList(
            nn.Identity()
            if self.output_dim == self.width
            else nn.Linear(self.width, self.output_dim, bias=False)
            for _ in range(self.depth)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.flatten(2).transpose(1, 2)

    def forward(self, thermal: torch.Tensor) -> ThermalEncoderOutput:
        if thermal.ndim != 4:
            raise ValueError(
                f"thermal input must have shape [B,1,H,W], got {tuple(thermal.shape)}"
            )
        if thermal.shape[1] != 1:
            raise ValueError(
                f"thermal input must be one-channel, got {thermal.shape[1]} channels"
            )
        if not torch.is_floating_point(thermal):
            raise TypeError("thermal input must be floating point and normalized by the loader")
        height, width = int(thermal.shape[-2]), int(thermal.shape[-1])
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"thermal spatial size {(height, width)} must be divisible by "
                f"patch_size={self.patch_size}"
            )

        x = thermal.to(dtype=self.patch_embed.weight.dtype)
        x = self.patch_embed(x)
        grid_size = (int(x.shape[-2]), int(x.shape[-1]))
        x = self._to_tokens(x)
        x = self.stem_norm(x)
        x = x.transpose(1, 2).reshape(
            thermal.shape[0], self.width, grid_size[0], grid_size[1]
        )

        taps = []
        for block, norm, projection in zip(
            self.blocks, self.tap_norms, self.tap_projections
        ):
            x = block(x)
            tokens = projection(norm(self._to_tokens(x)))
            taps.append(tokens)
        return ThermalEncoderOutput(
            tokens=taps[-1], taps=tuple(taps), grid_size=grid_size
        )
