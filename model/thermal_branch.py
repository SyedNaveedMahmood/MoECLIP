"""Thermal feature encoders.

``ThermalEncoder`` is the v1 MulSen-AD branch.  It consumes the loader's
single-channel thermal tensor and returns spatial patch tokens used only as
router conditioning.  It deliberately has no CLS token and makes no claim that
its patch index is registered to the RGB CLIP patch with the same index.

``ThermalTransformer`` below is retained temporarily for the exploratory
MoE-TwinCLIP path.  New RGB-thermal work must not instantiate it; it will be
removed together with that path during router integration.
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


class ThermalBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x


class ThermalTransformer(nn.Module):
    """Patchify + shallow transformer for thermal images.

    Returns the token state after every block ("taps"). The caller maps taps
    onto the frozen CLIP MoE layer positions proportionally.
    """

    def __init__(
        self,
        width: int = 1024,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 2.0,
        patch_size: int = 14,
        in_chans: int = 3,
        pos_embed_scale: float = 0.02,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_chans, width, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        # positional embedding created lazily on first forward -> supports any input size
        self.pos_embed = None
        self.pos_embed_scale = pos_embed_scale
        self.blocks = nn.ModuleList(
            [ThermalBlock(width, heads, mlp_ratio) for _ in range(depth)]
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _ensure_pos_embed(self, n_tokens: int, device, dtype) -> torch.Tensor:
        if self.pos_embed is None or self.pos_embed.shape[1] != n_tokens or self.pos_embed.device != device:
            self.pos_embed = (
                torch.randn(1, n_tokens, self.width, device=device) * self.pos_embed_scale
            ).to(dtype)
        return self.pos_embed.to(dtype)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify + cls token + positional embedding. Accepts 1ch or 3ch."""
        x = x.to(self.patch_embed.weight.dtype)
        if x.shape[1] == 1:
            x = x.repeat(1, self.patch_embed.in_channels, 1, 1)
        x = self.patch_embed(x)                       # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)              # (B, N, C)
        cls = self.cls_token.expand(x.shape[0], -1, -1).to(x.dtype)
        x = torch.cat([cls, x], dim=1)                # (B, N+1, C)
        x = x + self._ensure_pos_embed(x.shape[1], x.device, x.dtype)
        return x

    def step(self, t: torch.Tensor) -> torch.Tensor:
        """Advance the thermal stream by the next transformer block."""
        blk = self.blocks[self._step_idx % self.depth]
        self._step_idx += 1
        return blk(t)

    @property
    def _step_idx(self) -> int:
        return getattr(self, "__step_idx", 0)

    @_step_idx.setter
    def _step_idx(self, v: int):
        self.__step_idx = v

    def reset_step(self):
        """Reset the stepping cursor before a new forward pass."""
        self._step_idx = 0

    def forward(self, x: torch.Tensor):
        """Full forward; returns token state after every block ("taps")."""
        x = self.embed(x)
        taps = []
        for blk in self.blocks:
            x = blk(x)
            taps.append(x)
        return taps                                   # list of depth tensors (B, N+1, C)
