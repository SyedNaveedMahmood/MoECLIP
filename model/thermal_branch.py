"""Lightweight thermal encoder for MoE-TwinCLIP.

A shallow transformer (default 4 blocks, width=1024) that encodes registered
thermal images into patch tokens matching the CLIP ViT-L token dimension, so
its tap outputs can condition the RGB stream's MoE routers (cross-modal
routing) and be routed through the same shared experts themselves.
"""
import torch
from torch import nn


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
