import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import SimpleAdapter, SimpleProj, SimpleLoRA, ConvAdapterProj
from .config import LoraConfig, MixLoraConfig
from typing import Dict, Optional, Tuple, List
import math

def etf_loss(
    expert_outputs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:

    num_tokens, num_experts, _ = expert_outputs.shape

    if num_experts <= 1:
        return torch.tensor(0.0, device=expert_outputs.device)


    norm_outputs = F.normalize(expert_outputs, p=2, dim=-1, eps=eps)
    
    gram_matrix = torch.bmm(norm_outputs, norm_outputs.transpose(1, 2))

    target_val = -1.0 / (num_experts - 1)
    target_matrix = torch.full(
        (num_experts, num_experts), 
        target_val, 
        device=expert_outputs.device, 
        dtype=expert_outputs.dtype
    )

    target_matrix.fill_diagonal_(1.0)

    loss = F.mse_loss(gram_matrix, target_matrix.unsqueeze(0).expand_as(gram_matrix))
    
    return loss

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
        self.device_ = torch.device("cuda:0")
        
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
        assert isinstance(weight, Tuple)
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
        num_shared_experts: int = 0,
        num_thermal_experts: int = 0,
        use_cond: bool = False,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        
        self.gate = nn.Linear(d_model, config.num_experts_, bias=False)
        
        fixed_A_weights = None
        if use_fofs:
            fixed_A_weights = self._create_fofs_A_matrices()

        self.experts = nn.ModuleList()
        for i in range(config.num_experts_):
            lora_A_weight = fixed_A_weights[i] if use_fofs else None
            expert = SimpleLoraExpert(d_model, d_model, config, weight=(lora_A_weight, None))
            self.experts.append(expert)
        
        # --- MoE-TwinCLIP: modality-aware expert partitioning ---
        # experts [0, n_rgb)                 -> rgb-private
        # experts [n_rgb, n_rgb+n_shared)    -> shared (both modalities)
        # experts [n_rgb+n_shared, K)        -> thermal-private
        n_total = config.num_experts_
        n_shared = min(max(0, num_shared_experts), n_total)
        n_therm = min(max(0, num_thermal_experts), max(0, n_total - n_shared - 1))
        n_rgb = n_total - n_shared - n_therm
        self.expert_groups = {
            "rgb": list(range(n_rgb)) + list(range(n_rgb, n_rgb + n_shared)),
            "thermal": list(range(n_rgb + n_shared, n_total))
            + list(range(n_rgb, n_rgb + n_shared)),
        }
        self.use_cond = use_cond
        if use_cond:
            # cross-modal conditioning: router sees [own-stream token ; other-stream token]
            self.cond_proj = nn.Linear(d_model * 2, d_model, bias=False)
        
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
        nn.init.zeros_(self.gate.weight)
        
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
        cond: Optional[torch.Tensor] = None,
        expert_group: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        if self.jitter_noise > 0 and self.training:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        
        input_dtype = hidden_states.dtype
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)

        # --- MoE-TwinCLIP: cross-modal conditioned routing + expert-group masking ---
        if self.use_cond and cond is not None:
            cond_flat = cond.reshape(-1, hidden_dim).to(hidden_states_flat.dtype)
            router_in = self.cond_proj(torch.cat([hidden_states_flat, cond_flat], dim=-1))
        else:
            router_in = hidden_states_flat

        router_logits = self.gate(router_in)
        if expert_group is not None:
            group_mask = torch.full_like(router_logits, float("-inf"))
            group_mask[:, self.expert_groups[expert_group]] = 0.0
            router_logits = router_logits + group_mask

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
        seg_proj_sharing_strategy: str = "shared",
        image_adapt_weight: float = 0.1,
        levels: list = [6, 12, 18, 24],
        moe_r: int = 8,
        moe_lora_alpha: int = 16,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        use_fofs: bool = True,
        moe_layers: Optional[List[int]] = None,
        relu: bool = True,
        # --- MoE-TwinCLIP options ---
        use_thermal: bool = False,
        thermal_depth: int = 4,
        num_shared_experts: int = 1,
        modality_dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.i_w = image_adapt_weight
        self.levels = levels
        self.moe_layers = moe_layers if moe_layers is not None else [5, 11, 17, 23]
        
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
        
        d_model = 1024
        
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


        n_therm_experts = 1 if use_thermal else 0
        moe_adapters = nn.ModuleList([
            BaseIndependentMoE(
                d_model=d_model,
                config=moe_config,
                use_fofs=use_fofs,
                num_shared_experts=num_shared_experts if use_thermal else 0,
                num_thermal_experts=n_therm_experts,
                use_cond=use_thermal,
            )
            for _ in self.moe_layers
        ])

        seg_proj = nn.ModuleList(
            [SimpleProj(1024, 768, relu) for _ in range(num_seg_projs)]
        )
        
        
        det_proj = ConvAdapterProj(1024, 768)
        self.image_adapter = nn.ModuleDict(
            {
                "seg_proj": seg_proj,
                "det_proj": det_proj,
                "moe_adapters": moe_adapters,
            }
        )
        self.text_adapter = nn.ModuleList(
            [SimpleProj(768, 768, relu=True)]
        )

        # --- MoE-TwinCLIP: thermal branch + readout + fusion gates ---
        self.use_thermal = use_thermal
        self.modality_dropout = modality_dropout
        if use_thermal:
            from .thermal_branch import ThermalTransformer

            self.thermal_branch = ThermalTransformer(
                width=d_model, depth=thermal_depth, heads=8, mlp_ratio=2.0, patch_size=14
            )
            thermal_seg_proj = nn.ModuleList(
                [SimpleProj(1024, 768, relu) for _ in range(num_seg_projs)]
            )
            thermal_det_proj = ConvAdapterProj(1024, 768)
            self.thermal_adapter = nn.ModuleDict(
                {"seg_proj": thermal_seg_proj, "det_proj": thermal_det_proj}
            )
            # fusion gates (init ~ -2 -> sigmoid ~= 0.12 -> mostly RGB at start)
            self.seg_gate_logits = nn.Parameter(torch.full((num_seg_projs,), -2.0))
            self.det_gate = nn.Sequential(
                nn.Linear(768 * 2, 256), nn.GELU(), nn.Linear(256, 1)
            )
        
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

    def _aggregate_neighbors(self, tokens_from_layers: list) -> list:
        aggregated_token_list = []
        for token_map in tokens_from_layers:
            for r in [1, 3, 5]:
                permuted_token_map = token_map.permute(1, 0, 2)
                aggregated_token = self._aggregate_neighbor(permuted_token_map, r)
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

    def forward(self, x, thermal=None, return_align=False):
        """Dispatch: thermal=None -> original MoECLIP path (unchanged behavior)."""
        if not getattr(self, "use_thermal", False) or thermal is None:
            seg_tokens, det_token, aux_loss, special_loss = self._forward_rgb(x)
            if return_align:
                return seg_tokens, det_token, aux_loss, special_loss, None
            return seg_tokens, det_token, aux_loss, special_loss
        return self._forward_rgb_thermal(x, thermal, return_align)

    def _forward_rgb_thermal(self, x, thermal, return_align=False):
        device = x.device

        # modality dropout: occasionally train RGB-only for graceful degradation
        use_t = True
        if (
            self.training
            and self.modality_dropout > 0
            and torch.rand(1).item() < self.modality_dropout
        ):
            use_t = False
            thermal = torch.zeros_like(x)

        # Interleaved execution: the thermal branch advances one block right
        # before each CLIP MoE layer, so every MoE-updated thermal tap feeds
        # all subsequent thermal computation (nothing is discarded).
        self.thermal_branch.reset_step()
        t_state = self.thermal_branch.embed(thermal.to(device))

        # ---- RGB stream preamble (identical to _forward_rgb) ----
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=device
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
        total_load_balance_loss = torch.tensor(0.0, device=device)
        total_etf_loss = torch.tensor(0.0, device=device)

        for i in range(24):
            x, attn, _ = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)

            if i in self.moe_layers:
                t_j = self.thermal_branch.step(t_state)
                moe_idx = self.moe_layers.index(i)
                moe = self.image_adapter["moe_adapters"][moe_idx]

                # cross-modal conditioned routing with expert-group masking;
                # both streams share the same expert weights.
                moe_out_r, lb_r, eo_r, _ = moe(x, cond=t_j, expert_group="rgb")
                moe_out_t, lb_t, eo_t, _ = moe(t_j, cond=x, expert_group="thermal")

                if eo_r is not None:
                    total_etf_loss += etf_loss(eo_r)
                if eo_t is not None:
                    total_etf_loss += etf_loss(eo_t)

                moe_out_r_n = moe_out_r * x.norm(dim=-1, keepdim=True) / (
                    moe_out_r.norm(dim=-1, keepdim=True) + 1e-6
                )
                moe_out_t_n = moe_out_t * t_j.norm(dim=-1, keepdim=True) / (
                    moe_out_t.norm(dim=-1, keepdim=True) + 1e-6
                )
                x = self.i_w * moe_out_r_n + (1 - self.i_w) * x
                t_state = self.i_w * moe_out_t_n + (1 - self.i_w) * t_j

                total_load_balance_loss += lb_r + lb_t

            if i + 1 in self.levels:
                tokens.append(x)

        if self.use_paa:
            tokens = self._aggregate_neighbors(tokens)

        return self._fuse_readouts(
            x, tokens, t_state, use_t, total_load_balance_loss, total_etf_loss,
            return_align,
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
        total_load_balance_loss = torch.tensor(0.0, device=x.device)
        total_etf_loss = torch.tensor(0.0, device=x.device)

        for i in range(24):
            x, attn, _ = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            
            if i in self.moe_layers:
                moe_idx = self.moe_layers.index(i)
                
                moe_output, moe_lb_loss, all_expert_outputs, selected_experts = \
                    self.image_adapter["moe_adapters"][moe_idx](x)
                
                if self.training and all_expert_outputs is not None:
                    moe_etf_l = etf_loss(all_expert_outputs)
                    total_etf_loss += moe_etf_l
                moe_output_normalized = (
                    moe_output * x.norm(dim=-1, keepdim=True)
                    / (moe_output.norm(dim=-1, keepdim=True) + 1e-6)
                )
                x = self.i_w * moe_output_normalized + (1 - self.i_w) * x
                
                total_load_balance_loss += moe_lb_loss
                
            if i + 1 in self.levels:
                if self.use_paa:
                    tokens.append(x)
                else:
                    tokens.append(x)
                
        if self.use_paa:
            tokens = self._aggregate_neighbors(tokens)

        x = x.permute(1, 0, 2)
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

        total_aux_loss = total_load_balance_loss
        special_loss = total_etf_loss
        
        return seg_tokens, det_token, total_aux_loss, special_loss

    def _fuse_readouts(
        self, x, tokens, t_final, use_t, total_load_balance_loss, total_etf_loss,
        return_align=False,
    ):
        device = x.device

        # ---- RGB readout (identical to _forward_rgb tail) ----
        x = x.permute(1, 0, 2)
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]
        rgb_patch_tokens = [t[:, 1:, :] for t in tokens]

        if self.use_paa and self.seg_proj_sharing_strategy == "shared":
            seg_tokens_rgb = [
                self.image_adapter["seg_proj"][i // 3](t)
                for i, t in enumerate(rgb_patch_tokens)
            ]
        else:
            seg_tokens_rgb = [
                self.image_adapter["seg_proj"][i](t)
                for i, t in enumerate(rgb_patch_tokens)
            ]
        seg_tokens_rgb = [F.normalize(t, dim=-1) for t in seg_tokens_rgb]
        det_rgb = F.normalize(
            self.image_adapter["det_proj"](rgb_patch_tokens[-3]), dim=-1
        ).mean(1)

        if not use_t:
            # modality-dropped step: pure RGB output
            if return_align:
                return seg_tokens_rgb, det_rgb, total_load_balance_loss, total_etf_loss, None
            return seg_tokens_rgb, det_rgb, total_load_balance_loss, total_etf_loss

        # ---- Thermal readout (shared ln_post keeps features text-aligned) ----
        t_tok = self.image_encoder.ln_post(t_final)   # (B, L, C) batch-first
        t_patch = t_tok[:, 1:, :]
        shared_idx = self.use_paa and self.seg_proj_sharing_strategy == "shared"
        seg_tokens_t_raw = [
            self.thermal_adapter["seg_proj"][(k // 3) if shared_idx else k](t_patch)
            for k in range(len(seg_tokens_rgb))
        ]
        seg_tokens_t = [F.normalize(s, dim=-1) for s in seg_tokens_t_raw]
        det_t = F.normalize(self.thermal_adapter["det_proj"](t_patch), dim=-1).mean(1)

        # ---- Uncertainty-gated convex fusion ----
        # one learned gate per projection level; PAA "shared" expands each
        # level into 3 radius variants sharing the same projection (k // 3)
        g = torch.sigmoid(self.seg_gate_logits).to(seg_tokens_rgb[0].dtype)
        seg_tokens_fused = []
        for k, (s_r, s_t) in enumerate(zip(seg_tokens_rgb, seg_tokens_t)):
            g_k = g[(k // 3) if shared_idx else k]
            seg_tokens_fused.append(
                F.normalize(g_k * s_r + (1.0 - g_k) * s_t, dim=-1)
            )
        gd = torch.sigmoid(self.det_gate(torch.cat([det_rgb, det_t], dim=-1)))
        det_token = F.normalize(gd * det_rgb + (1.0 - gd) * det_t, dim=-1)

        align_pair = (
            F.normalize(rgb_patch_tokens[-3].mean(dim=1), dim=-1),
            F.normalize(t_patch.mean(dim=1), dim=-1),
        )

        if return_align:
            return (
                seg_tokens_fused,
                det_token,
                total_load_balance_loss,
                total_etf_loss,
                align_pair,
            )
        return seg_tokens_fused, det_token, total_load_balance_loss, total_etf_loss

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