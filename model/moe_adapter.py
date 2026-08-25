import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import SimpleProj, ConvAdapterProj
from .config import LoraConfig, MixLoraConfig
from .region_context import (
    RegionContextEncoder,
    identity_patch_regions,
    pixel_regions_to_patch_regions,
)
from .thermal_branch import ThermalEncoder, ThermalEncoderOutput
from typing import Dict, Optional, Tuple, List
import math

def etf_loss(
    expert_outputs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:

    num_tokens, num_experts, _ = expert_outputs.shape

    if num_experts <= 1:
        return torch.tensor(0.0, device=expert_outputs.device)


    # LoRA B is intentionally zero-initialized. In fp16, differentiating
    # normalization at that point with eps=1e-6 overflows (1 / eps exceeds the
    # finite fp16 range) even though the forward loss is finite. ETF is a small
    # auxiliary loss, so keep this numerically sensitive reduction in fp32.
    working_outputs = expert_outputs.float()
    norm_outputs = F.normalize(working_outputs, p=2, dim=-1, eps=eps)
    
    gram_matrix = torch.bmm(norm_outputs, norm_outputs.transpose(1, 2))

    target_val = -1.0 / (num_experts - 1)
    target_matrix = torch.full(
        (num_experts, num_experts), 
        target_val, 
        device=expert_outputs.device, 
        dtype=working_outputs.dtype
    )

    target_matrix.fill_diagonal_(1.0)

    loss = F.mse_loss(gram_matrix, target_matrix.unsqueeze(0).expand_as(gram_matrix))
    
    return loss


def match_adapter_token_norm(
    adapter_output: torch.Tensor,
    reference: torch.Tensor,
    min_adapter_norm: float = 1.0,
) -> torch.Tensor:
    """Match large adapter outputs to RGB-token norms without a zero singularity.

    The released expression divided by ``adapter_norm + 1e-6``. Since every
    LoRA B matrix starts at zero, its first backward pass amplified gradients by
    roughly ``reference_norm / 1e-6`` and overflowed under AMP. Below the
    explicit floor we retain a finite linear path; above it, the original
    token-wise norm-matching behavior is recovered.
    """

    if adapter_output.shape != reference.shape:
        raise ValueError("adapter output and reference must have identical shapes")
    if min_adapter_norm <= 0.0:
        raise ValueError("min_adapter_norm must be positive")
    adapter_norm = torch.linalg.vector_norm(
        adapter_output.float(), dim=-1, keepdim=True
    )
    reference_norm = torch.linalg.vector_norm(
        reference.float(), dim=-1, keepdim=True
    )
    scale = reference_norm / adapter_norm.clamp_min(min_adapter_norm)
    return adapter_output * scale.to(dtype=adapter_output.dtype)

class SimpleLoraExpert(nn.Module):
    def __init__(
        self,
        in_features: int, 
        out_features: int,
        config: LoraConfig,
        weight: Tuple[torch.Tensor, torch.Tensor] = (None, None),
        device: str = None,
    ):
        super().__init__()

        self.config_ = config
        self.initializer_ = config.lora_init_

        self.dtype_ = config.dtype_
        self.r_ = config.lora_r_
        self.alpha_ = config.lora_alpha_
        # Allocate on the caller's/default device and let the containing model's
        # ``.to(device)`` decide placement. Hard-coding CUDA made CPU smoke tests
        # impossible and silently ignored the constructor's device argument.
        self.device_ = None if device is None else torch.device(device)
        
        if config.use_rslora_:
            self.scaling_ = self.alpha_ / math.sqrt(self.r_)
        else:
            self.scaling_ = self.alpha_ / self.r_

        if config.lora_dropout_ > 0.0:
            self.dropout_ = nn.Dropout(p=config.lora_dropout_)
        else:
            self.dropout_ = nn.Identity()

        self.lora_A = nn.Linear(in_features, self.r_, bias=False, dtype=self.dtype_, device=self.device_)
        self.lora_B = nn.Linear(self.r_, out_features, bias=False, dtype=self.dtype_, device=self.device_)
        
        self.use_dora_: bool = config.use_dora_
        self.magnitude_vector_: nn.Parameter = None
        self.reset_parameters(weight)

    def reset_parameters(
        self, weight: Tuple[torch.Tensor, torch.Tensor] = (None, None)
    ) -> None:
        assert isinstance(weight, tuple)
        assert len(weight) == 2
        assert ((weight[0] is None) and (weight[1] is None)) or (
            isinstance(weight[0], torch.Tensor)
        )

        if weight[0] is None and weight[1] is None:
            if self.initializer_ == "original":
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            elif self.initializer_ == "gaussian":
                nn.init.normal_(self.lora_A.weight, std=1 / self.r_)
            else:
                raise ValueError(f"Unknown initialization {self.initializer_}")
            nn.init.zeros_(self.lora_B.weight)
        else:
            with torch.no_grad():
                if weight[0] is not None:
                    self.lora_A.weight.copy_(weight[0])
                    self.lora_A.weight.requires_grad = False
                
                if weight[1] is not None:
                    self.lora_B.weight.copy_(weight[1])
                else:
                    nn.init.zeros_(self.lora_B.weight)
    
    def get_lora_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout_(hidden_states))) * self.scaling_

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lora_output = self.get_lora_output(hidden_states)
        return lora_output
    
class BaseIndependentMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        config,
        use_fofs: bool = True,
        router_context_dim: Optional[int] = None,
        num_context_experts: Optional[int] = None,
        base_router_init: str = "zero",
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        if base_router_init not in {"zero", "normal"}:
            raise ValueError("base_router_init must be 'zero' or 'normal'")
        self.base_router_init = base_router_init
        
        self.gate = nn.Linear(d_model, config.num_experts_, bias=False)
        
        fixed_A_weights = None
        if use_fofs:
            fixed_A_weights = self._create_fofs_A_matrices()

        self.experts = nn.ModuleList()
        for i in range(config.num_experts_):
            lora_A_weight = fixed_A_weights[i] if use_fofs else None
            expert = SimpleLoraExpert(d_model, d_model, config, weight=(lora_A_weight, None))
            self.experts.append(expert)
        
        # V1 region-guided routing adds a context-dependent *logit residual*.
        # Experts still receive only ``hidden_states`` below.  Applying context
        # to all experts is the default; a fixed subset is available solely as
        # a configurable ablation and carries no hard-coded modality meaning.
        self.router_context_dim = router_context_dim
        if router_context_dim is not None:
            if router_context_dim <= 0:
                raise ValueError("router_context_dim must be positive")
            if num_context_experts is None:
                num_context_experts = config.num_experts_
            if not 0 <= num_context_experts <= config.num_experts_:
                raise ValueError(
                    "num_context_experts must be between zero and num_experts"
                )
            self.context_gate = nn.Linear(
                router_context_dim, config.num_experts_, bias=False
            )
            context_mask = torch.zeros(config.num_experts_, dtype=torch.float32)
            context_mask[:num_context_experts] = 1.0
            self.register_buffer("context_expert_mask", context_mask)
        
        self.jitter_noise = getattr(config, "jitter_noise_", 0.0)
        self.init_custom_weights()
        
    def _create_fofs_A_matrices(self) -> list:
        num_experts = self.config.num_experts_
        in_features = self.d_model
        rank = self.config.lora_r_
        
        base_chunk_size = in_features // num_experts
        remainder = in_features % num_experts
        
        fixed_A_matrices = []
        current_start_idx = 0
        
        print(f"Creating fofs LoRA A matrices for d_model={in_features}, num_experts={num_experts}")

        for i in range(num_experts):
            chunk_size = base_chunk_size + 1 if i < remainder else base_chunk_size
            
            start_idx = current_start_idx
            end_idx = start_idx + chunk_size
            
            print(f"  - Expert {i+1}: features from {start_idx} to {end_idx-1} (size: {chunk_size})")

            temp_matrix = torch.randn(chunk_size, rank, device=self.gate.weight.device, dtype=self.config.dtype_)
            q, _ = torch.linalg.qr(temp_matrix)
            
            q_ortho = q.T

            A_matrix = torch.zeros(rank, in_features, device=self.gate.weight.device, dtype=self.config.dtype_)
            A_matrix[:, start_idx:end_idx] = q_ortho
            
            fixed_A_matrices.append(A_matrix)
            
            current_start_idx = end_idx
            
        return fixed_A_matrices

    def init_custom_weights(self):
        if self.base_router_init == "zero":
            nn.init.zeros_(self.gate.weight)
        else:
            nn.init.normal_(
                self.gate.weight, std=float(self.config.router_init_range_)
            )
        if hasattr(self, "context_gate"):
            nn.init.zeros_(self.context_gate.weight)

    def compute_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute base RGB logits plus an optional region-context residual."""

        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                f"hidden_states width must be {self.d_model}, got "
                f"{hidden_states.shape[-1]}"
            )
        hidden_states_flat = hidden_states.reshape(-1, self.d_model)
        router_logits = self.gate(hidden_states_flat)

        if router_context is not None:
            if not hasattr(self, "context_gate"):
                raise ValueError(
                    "router context was supplied but router_context_dim is disabled"
                )
            expected_shape = hidden_states.shape[:-1] + (self.router_context_dim,)
            if router_context.shape != expected_shape:
                raise ValueError(
                    f"router context must have shape {expected_shape}, got "
                    f"{tuple(router_context.shape)}"
                )
            context_flat = router_context.reshape(-1, self.router_context_dim)
            context_flat = context_flat.to(dtype=self.context_gate.weight.dtype)
            residual = self.context_gate(context_flat)
            residual = residual * self.context_expert_mask.to(residual.dtype)
            router_logits = router_logits + residual.to(router_logits.dtype)
        return router_logits
        
    def _vit_forward(self, expert_mask: torch.Tensor, hidden_states: torch.Tensor) -> Dict[int, torch.Tensor]:
        expert_outputs_dict = {}
        for expert_idx in range(self.config.num_experts_):
            _, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            
            current_hidden_states = hidden_states[top_x]
            expert_output = self.experts[expert_idx](current_hidden_states)
            expert_outputs_dict[expert_idx] = expert_output

        return expert_outputs_dict

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        if self.jitter_noise > 0 and self.training:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        
        input_dtype = hidden_states.dtype
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)

        router_logits = self.compute_router_logits(
            hidden_states, router_context=router_context
        )

        router_probs = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(router_probs, self.config.top_k_, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(input_dtype)

        final_hidden_states = torch.zeros_like(hidden_states_flat)
        load_balance_loss = torch.tensor(0.0, device=hidden_states.device)
        all_expert_outputs = None

        if self.training:
            alpha = getattr(self.config, "router_aux_loss_coef_", 0.0)
            if alpha > 0:
                gate_sum = router_probs.sum(dim=0)
                mean, std = gate_sum.mean(), gate_sum.std()
                cv_squared = (std / (mean + 1e-6)).pow(2)
                load_balance_loss = cv_squared

            all_expert_outputs = torch.stack(
                [expert(hidden_states_flat) for expert in self.experts], 
                dim=1 
            )
            
            token_indices = torch.arange(hidden_states_flat.shape[0], device=hidden_states.device).unsqueeze(1)
            activated_outputs = all_expert_outputs[token_indices, selected_experts]

            weighted_outputs = activated_outputs * routing_weights.unsqueeze(-1)
            final_hidden_states = weighted_outputs.sum(dim=1)

        else:
            expert_mask = F.one_hot(selected_experts, num_classes=self.config.num_experts_).permute(2, 1, 0)
            expert_outputs_dict = self._vit_forward(expert_mask, hidden_states_flat)
            per_token_expert_outputs = torch.zeros(
                (hidden_states_flat.shape[0], self.config.top_k_, hidden_dim),
                dtype=input_dtype, device=hidden_states.device
            )
            for expert_idx, output_tensor in expert_outputs_dict.items():
                top_k_idx, top_x = torch.where(expert_mask[expert_idx])
                per_token_expert_outputs.index_put_((top_x, top_k_idx), output_tensor)
            
            weighted_outputs = per_token_expert_outputs * routing_weights.unsqueeze(-1)
            final_hidden_states = weighted_outputs.sum(dim=1)
            all_expert_outputs, selected_experts = None, None

        return (
            final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), 
            load_balance_loss, 
            all_expert_outputs,
            selected_experts
        )

class MoECLIP(nn.Module):
    def __init__(
        self,
        clip_model,
        use_paa: bool = True,
        use_segment_paa: bool = False,
        seg_proj_sharing_strategy: str = "shared",
        image_adapt_weight: float = 0.1,
        levels: list = [6, 12, 18, 24],
        moe_r: int = 8,
        moe_lora_alpha: int = 16,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        router_init: str = "zero",
        use_fofs: bool = True,
        moe_layers: Optional[List[int]] = None,
        relu: bool = True,
        # --- Region-guided RGB-thermal routing options ---
        use_thermal: bool = False,
        use_region_routing: bool = False,
        thermal_depth: int = 4,
        thermal_width: int = 256,
        region_context_dim: int = 256,
        region_attention_heads: int = 4,
        region_coordinate_bias: float = 1.0,
        region_coordinate_sigma: float = 0.75,
        num_context_experts: Optional[int] = None,
        num_shared_experts: Optional[int] = None,
        modality_dropout: float = 0.2,
        stable_adapter_norm: bool = False,
        adapter_norm_floor: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.i_w = image_adapt_weight
        self.levels = levels
        self.moe_layers = moe_layers if moe_layers is not None else [5, 11, 17, 23]
        transformer_depth = len(self.image_encoder.transformer.resblocks)
        if not self.moe_layers or len(set(self.moe_layers)) != len(self.moe_layers):
            raise ValueError("moe_layers must be a non-empty list of unique indices")
        if any(index < 0 or index >= transformer_depth for index in self.moe_layers):
            raise ValueError(
                f"moe_layers must be within the {transformer_depth}-block visual encoder"
            )
        if any(level <= 0 or level > transformer_depth for level in self.levels):
            raise ValueError(
                f"levels must be within 1..{transformer_depth}"
            )
        self.use_thermal = bool(use_thermal)
        self.use_region_routing = bool(use_region_routing)
        self.use_context_routing = self.use_thermal or self.use_region_routing
        if use_segment_paa and not use_paa:
            raise ValueError("use_segment_paa requires use_paa=True")
        if use_segment_paa and not self.use_region_routing:
            raise ValueError("use_segment_paa requires region-guided routing")
        self.use_segment_paa = bool(use_segment_paa)
        if not 0.0 <= modality_dropout <= 1.0:
            raise ValueError("modality_dropout must be in [0,1]")
        self.modality_dropout = float(modality_dropout)
        if not math.isfinite(adapter_norm_floor) or adapter_norm_floor <= 0.0:
            raise ValueError("adapter_norm_floor must be finite and positive")
        self.stable_adapter_norm = bool(stable_adapter_norm)
        self.adapter_norm_floor = float(adapter_norm_floor)
        if num_shared_experts is not None:
            if (
                num_context_experts is not None
                and num_context_experts != num_shared_experts
            ):
                raise ValueError(
                    "num_shared_experts is a legacy alias and conflicts with "
                    "num_context_experts"
                )
            num_context_experts = num_shared_experts
        
        self.use_paa = use_paa
        
        self.seg_proj_sharing_strategy = seg_proj_sharing_strategy
        if self.use_paa:
            assert seg_proj_sharing_strategy in ["separate", "shared"], \
                "seg_proj_sharing_strategy must be 'separate' or 'shared' when using paa"
            
            if self.seg_proj_sharing_strategy == "separate":
                num_seg_projs = len(levels) * 3
            else:
                num_seg_projs = len(levels)
            
            self._create_gaussian_kernels()
        else:
            num_seg_projs = len(levels)
        
        d_model = int(self.image_encoder.conv1.out_channels)
        patch_stride = self.image_encoder.conv1.stride
        patch_kernel = self.image_encoder.conv1.kernel_size
        if patch_stride[0] != patch_stride[1] or patch_kernel != patch_stride:
            raise ValueError("region routing requires a square non-overlapping patch stem")
        self.visual_patch_size = int(patch_stride[0])
        
        moe_config = MixLoraConfig.from_config(
            {
                "bias": "none",
                "peft_type": "MIXLORA",
                "r": moe_r,
                "lora_alpha": moe_lora_alpha,
                "lora_dropout": 0.05,
                "target_modules": ["c_fc", "c_proj"],
                "routing_strategy": "mixlora",
                "num_experts": moe_num_experts,
                "num_lora_experts": moe_num_experts,
                "top_k": moe_top_k,
                "act_fn": "silu",
                "base_model_name_or_path": "CLIP_VIT",
                "task_type": "VISION",
            }
        )


        moe_adapters = nn.ModuleList([
            BaseIndependentMoE(
                d_model=d_model,
                config=moe_config,
                use_fofs=use_fofs,
                router_context_dim=(
                    region_context_dim if self.use_context_routing else None
                ),
                num_context_experts=num_context_experts,
                base_router_init=router_init,
            )
            for _ in self.moe_layers
        ])

        seg_proj = nn.ModuleList(
            [SimpleProj(d_model, 768, relu) for _ in range(num_seg_projs)]
        )
        
        
        det_proj = ConvAdapterProj(d_model, 768)
        image_adapter = {
            "seg_proj": seg_proj,
            "det_proj": det_proj,
            "moe_adapters": moe_adapters,
        }
        if self.use_context_routing:
            image_adapter["region_contexts"] = nn.ModuleList(
                RegionContextEncoder(
                    rgb_dim=d_model,
                    thermal_dim=(thermal_width if self.use_thermal else None),
                    context_dim=region_context_dim,
                    num_heads=region_attention_heads,
                    coordinate_bias_strength=region_coordinate_bias,
                    coordinate_bias_sigma=region_coordinate_sigma,
                )
                for _ in self.moe_layers
            )
        self.image_adapter = nn.ModuleDict(image_adapter)
        self.text_adapter = nn.ModuleList(
            [SimpleProj(768, 768, relu=True)]
        )

        if self.use_thermal:
            self.thermal_branch = ThermalEncoder(
                width=thermal_width,
                output_dim=thermal_width,
                depth=thermal_depth,
                patch_size=self.visual_patch_size,
            )
        self.last_thermal_available = None
        
    @staticmethod
    def _gaussian_kernel(size: int, sigma: float = 2.0) -> torch.Tensor:
        x_coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        y_coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        x, y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    def _create_gaussian_kernels(self):
        kernel_3x3 = self._gaussian_kernel(3)
        kernel_5x5 = self._gaussian_kernel(5)
        self.register_buffer('gaussian_kernel_3', kernel_3x3)
        self.register_buffer('gaussian_kernel_5', kernel_5x5)

    def _aggregate_neighbor(self, x: torch.Tensor, r: int) -> torch.Tensor:
        if r == 1:
            return x
        
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        
        b, l, c = patch_tokens.shape
        h = w = int(math.sqrt(l))
        
        patch_tokens = patch_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        padding = r // 2
        unfolded_patches = F.unfold(patch_tokens, kernel_size=r, padding=padding, stride=1)
        
        unfolded_patches = unfolded_patches.permute(0, 2, 1)
        unfolded_patches = unfolded_patches.reshape(b * l, c, r * r).permute(0, 2, 1)


        aggregated_features = torch.mean(unfolded_patches, dim=1)
            
        aggregated_patches = aggregated_features.reshape(b, l, c)
        
        return torch.cat([cls_token, aggregated_patches], dim=1)

    def _aggregate_neighbor_by_segment(
        self,
        x: torch.Tensor,
        patch_region_ids: torch.Tensor,
        r: int,
    ) -> torch.Tensor:
        """Average only spatial neighbors sharing the center patch's region."""

        if r not in {1, 3, 5}:
            raise ValueError("segment-aware PAA supports scales 1, 3, and 5")
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        batch_size, patch_count, channels = patch_tokens.shape
        side = int(round(math.sqrt(patch_count)))
        if side * side != patch_count:
            raise ValueError("segment-aware PAA requires a square patch grid")
        if patch_region_ids.shape != (batch_size, patch_count):
            raise ValueError(
                "patch_region_ids must match the batch and patch-token layout"
            )
        if r == 1:
            return x

        padding = r // 2
        patch_grid = patch_tokens.permute(0, 2, 1).reshape(
            batch_size, channels, side, side
        )
        neighbor_features = F.unfold(
            patch_grid, kernel_size=r, padding=padding, stride=1
        )
        neighbor_features = neighbor_features.reshape(
            batch_size, channels, r * r, patch_count
        ).permute(0, 3, 2, 1)

        region_grid = patch_region_ids.reshape(
            batch_size, 1, side, side
        ).to(dtype=torch.float32)
        neighbor_regions = F.unfold(
            region_grid, kernel_size=r, padding=padding, stride=1
        ).transpose(1, 2)
        valid_neighbors = F.unfold(
            torch.ones_like(region_grid),
            kernel_size=r,
            padding=padding,
            stride=1,
        ).transpose(1, 2).bool()
        same_region = (
            neighbor_regions == patch_region_ids.unsqueeze(-1).to(torch.float32)
        ) & valid_neighbors
        weights = same_region.unsqueeze(-1).to(dtype=neighbor_features.dtype)
        denominator = weights.sum(dim=2).clamp_min(1.0)
        aggregated = (neighbor_features * weights).sum(dim=2) / denominator
        return torch.cat([cls_token, aggregated], dim=1)

    def _aggregate_neighbors(
        self,
        tokens_from_layers: list,
        patch_region_ids: Optional[torch.Tensor] = None,
    ) -> list:
        aggregated_token_list = []
        for token_map in tokens_from_layers:
            for r in [1, 3, 5]:
                permuted_token_map = token_map.permute(1, 0, 2)
                if self.use_segment_paa:
                    if patch_region_ids is None:
                        raise ValueError(
                            "segment-aware PAA requires patch_region_ids"
                        )
                    aggregated_token = self._aggregate_neighbor_by_segment(
                        permuted_token_map, patch_region_ids, r
                    )
                else:
                    aggregated_token = self._aggregate_neighbor(
                        permuted_token_map, r
                    )
                aggregated_token_list.append(aggregated_token.permute(1, 0, 2))
        return aggregated_token_list

    def forward_original(self, x, modality="visual"):
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [
                self.clipmodel.visual._global_pool(t)[1] for t in patch_features
            ]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")

    def train(self, mode: bool = True):
        """Train adapters while keeping the frozen CLIP towers deterministic."""

        super().train(mode)
        self.clipmodel.eval()
        return self

    def _normalize_adapter_output(self, adapter_output, reference):
        if self.stable_adapter_norm:
            return match_adapter_token_norm(
                adapter_output,
                reference,
                min_adapter_norm=self.adapter_norm_floor,
            )
        # Preserve the released MoECLIP computation unless the new MulSen path
        # explicitly opts into the AMP-safe formulation.
        return (
            adapter_output
            * reference.norm(dim=-1, keepdim=True)
            / (adapter_output.norm(dim=-1, keepdim=True) + 1e-6)
        )

    def forward(
        self,
        x,
        thermal=None,
        region_map=None,
        return_align=False,
    ):
        """Run RGB MoECLIP with optional region/thermal router conditioning.

        Thermal tensors only affect router context.  Segmentation and detection
        outputs are always derived from adapted RGB CLIP tokens.
        """

        if self.use_context_routing:
            outputs = self._forward_conditioned(x, thermal, region_map)
        else:
            outputs = self._forward_rgb(x)
        if return_align:
            return (*outputs, None)
        return outputs

    def _sample_thermal_availability(
        self, thermal: torch.Tensor
    ) -> torch.Tensor:
        available = torch.ones(
            thermal.shape[0], dtype=torch.bool, device=thermal.device
        )
        conditioning_is_training = self.image_adapter["region_contexts"].training
        if conditioning_is_training and self.modality_dropout > 0.0:
            available = torch.rand(
                thermal.shape[0], device=thermal.device
            ) >= self.modality_dropout
        self.last_thermal_available = available.detach()
        return available

    @staticmethod
    def _thermal_tap(
        output: ThermalEncoderOutput,
        adapter_index: int,
        adapter_count: int,
    ) -> torch.Tensor:
        if adapter_count == 1:
            tap_index = len(output.taps) - 1
        else:
            tap_index = round(
                adapter_index * (len(output.taps) - 1) / (adapter_count - 1)
            )
        return output.taps[tap_index]

    def _forward_conditioned(self, image, thermal, region_map):
        batch_size = image.shape[0]
        thermal_output = None
        thermal_available = None
        if self.use_thermal and thermal is not None:
            if thermal.shape[0] != batch_size:
                raise ValueError("RGB and thermal batch sizes differ")
            thermal = thermal.to(image.device)
            thermal_available = self._sample_thermal_availability(thermal)
            masked_thermal = thermal * thermal_available.to(
                dtype=thermal.dtype
            ).view(-1, 1, 1, 1)
            thermal_output = self.thermal_branch(masked_thermal)
        elif self.use_thermal:
            self.last_thermal_available = torch.zeros(
                batch_size, dtype=torch.bool, device=image.device
            )

        x = self.image_encoder.conv1(image)
        rgb_grid_size = (int(x.shape[-2]), int(x.shape[-1]))
        patch_count = rgb_grid_size[0] * rgb_grid_size[1]
        if self.use_region_routing:
            if region_map is None:
                raise ValueError(
                    "region_map is required when use_region_routing=True"
                )
            if region_map.shape[0] != batch_size:
                raise ValueError("RGB and region-map batch sizes differ")
            patch_region_ids = pixel_regions_to_patch_regions(
                region_map.to(image.device), rgb_grid_size
            )
        else:
            patch_region_ids = identity_patch_regions(
                batch_size, rgb_grid_size, device=image.device
            )

        x = x.reshape(batch_size, x.shape[1], patch_count).permute(0, 2, 1)
        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    batch_size, 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        if self.image_encoder.positional_embedding.shape[0] != patch_count + 1:
            raise ValueError(
                "visual positional embedding does not match the current patch grid"
            )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        if self.image_encoder.patch_dropout.training:
            raise RuntimeError(
                "frozen CLIP PatchDropout must remain in eval mode for region routing"
            )
        x = self.image_encoder.patch_dropout(x)
        if x.shape[1] != patch_count + 1:
            raise RuntimeError("PatchDropout changed the region-aligned token layout")
        x = self.image_encoder.ln_pre(x).permute(1, 0, 2)

        tokens = []
        total_load_balance_loss = x.new_zeros(())
        total_etf_loss = x.new_zeros(())
        adapter_count = len(self.moe_layers)
        for block_index, block in enumerate(
            self.image_encoder.transformer.resblocks
        ):
            x, _, _ = block(x, attn_mask=None)
            if block_index in self.moe_layers:
                adapter_index = self.moe_layers.index(block_index)
                thermal_tokens = None
                thermal_grid_size = None
                if thermal_output is not None:
                    thermal_tokens = self._thermal_tap(
                        thermal_output, adapter_index, adapter_count
                    )
                    thermal_grid_size = thermal_output.grid_size
                context_output = self.image_adapter["region_contexts"][
                    adapter_index
                ](
                    x[1:].permute(1, 0, 2),
                    patch_region_ids,
                    rgb_grid_size,
                    thermal_tokens,
                    thermal_grid_size,
                    thermal_available,
                )
                class_context = context_output.patch_context.new_zeros(
                    batch_size, 1, context_output.patch_context.shape[-1]
                )
                router_context = torch.cat(
                    (class_context, context_output.patch_context), dim=1
                ).permute(1, 0, 2)
                moe_output, moe_loss, expert_outputs, _ = self.image_adapter[
                    "moe_adapters"
                ][adapter_index](x, router_context=router_context)
                if expert_outputs is not None:
                    total_etf_loss = total_etf_loss + etf_loss(expert_outputs)
                moe_output = self._normalize_adapter_output(moe_output, x)
                x = self.i_w * moe_output + (1.0 - self.i_w) * x
                total_load_balance_loss = total_load_balance_loss + moe_loss
            if block_index + 1 in self.levels:
                tokens.append(x)

        if self.use_paa:
            tokens = self._aggregate_neighbors(
                tokens,
                patch_region_ids=(
                    patch_region_ids if self.use_segment_paa else None
                ),
            )
        return self._readout_rgb(
            tokens, total_load_balance_loss, total_etf_loss
        )

    def _forward_rgb(self, x):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)
        
        tokens = []
        total_load_balance_loss = x.new_zeros(())
        total_etf_loss = x.new_zeros(())

        for i, block in enumerate(self.image_encoder.transformer.resblocks):
            x, attn, _ = block(x, attn_mask=None)
            
            if i in self.moe_layers:
                moe_idx = self.moe_layers.index(i)
                
                moe_output, moe_lb_loss, all_expert_outputs, selected_experts = \
                    self.image_adapter["moe_adapters"][moe_idx](x)
                
                if all_expert_outputs is not None:
                    moe_etf_l = etf_loss(all_expert_outputs)
                    total_etf_loss += moe_etf_l
                moe_output_normalized = self._normalize_adapter_output(moe_output, x)
                x = self.i_w * moe_output_normalized + (1 - self.i_w) * x
                
                total_load_balance_loss += moe_lb_loss
                
            if i + 1 in self.levels:
                if self.use_paa:
                    tokens.append(x)
                else:
                    tokens.append(x)
                
        if self.use_paa:
            tokens = self._aggregate_neighbors(tokens)

        return self._readout_rgb(
            tokens, total_load_balance_loss, total_etf_loss
        )

    def _readout_rgb(
        self,
        tokens,
        total_load_balance_loss,
        total_etf_loss,
    ):
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]
        tokens = [t[:, 1:, :] for t in tokens]
        if self.use_paa and self.seg_proj_sharing_strategy == "shared":
            seg_tokens = [
                self.image_adapter["seg_proj"][i // 3](t)
                for i, t in enumerate(tokens)
            ]
        else:
            seg_tokens = [
                self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
            ]

        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]

        det_token = self.image_adapter["det_proj"](tokens[-3])
        det_token = F.normalize(det_token, dim=-1).mean(1)

        return (
            seg_tokens,
            det_token,
            total_load_balance_loss,
            total_etf_loss,
        )

    def encode_text(self, text, adapt_text=True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(
            cast_dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(12):
            x, attn, _ = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        return x
