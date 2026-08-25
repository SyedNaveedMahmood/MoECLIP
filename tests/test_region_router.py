"""Tests for zero-initialized region context in the RGB LoRA MoE router."""

from __future__ import annotations

import unittest

import torch

from model.config import MixLoraConfig
from model.moe_adapter import BaseIndependentMoE


def _config(num_experts: int = 4, top_k: int = 2) -> MixLoraConfig:
    return MixLoraConfig.from_config(
        {
            "bias": "none",
            "peft_type": "MIXLORA",
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.1,
            "target_modules": ["c_fc", "c_proj"],
            "routing_strategy": "mixlora",
            "num_experts": num_experts,
            "num_lora_experts": num_experts,
            "top_k": top_k,
            "router_aux_loss_coef": 0.001,
            "act_fn": "silu",
            "base_model_name_or_path": "CLIP_VIT",
            "task_type": "VISION",
        }
    )


class RegionRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def _router(self, num_context_experts=None) -> BaseIndependentMoE:
        return BaseIndependentMoE(
            d_model=16,
            config=_config(),
            use_fofs=False,
            router_context_dim=8,
            num_context_experts=num_context_experts,
        )

    def test_zero_initialized_context_preserves_base_router_logits(self) -> None:
        router = self._router()
        hidden = torch.randn(5, 2, 16)
        context = torch.randn(5, 2, 8)

        without_context = router.compute_router_logits(hidden)
        with_context = router.compute_router_logits(hidden, router_context=context)

        torch.testing.assert_close(with_context, without_context)
        self.assertEqual(router.context_gate.weight.abs().sum().item(), 0.0)

    def test_random_base_router_is_explicit_and_context_stays_zero(self) -> None:
        router = BaseIndependentMoE(
            d_model=16,
            config=_config(),
            use_fofs=False,
            router_context_dim=8,
            base_router_init="normal",
        )
        self.assertGreater(float(router.gate.weight.abs().sum()), 0.0)
        self.assertEqual(float(router.context_gate.weight.abs().sum()), 0.0)

    def test_configurable_context_subset_masks_only_logit_residual(self) -> None:
        router = self._router(num_context_experts=2)
        hidden = torch.randn(3, 1, 16)
        context = torch.ones(3, 1, 8)
        with torch.no_grad():
            router.context_gate.weight.fill_(1.0)

        residual = router.compute_router_logits(
            hidden, router_context=context
        ) - router.compute_router_logits(hidden)

        self.assertTrue((residual[:, :2] != 0.0).all())
        self.assertEqual(residual[:, 2:].abs().sum().item(), 0.0)
        self.assertIn("context_expert_mask", router.state_dict())

    def test_top_k_forward_keeps_expert_computation_in_rgb_width(self) -> None:
        router = self._router().train()
        hidden = torch.randn(4, 2, 16, requires_grad=True)
        context = torch.randn(4, 2, 8, requires_grad=True)
        with torch.no_grad():
            router.context_gate.weight.normal_(std=0.05)
            for expert_index, expert in enumerate(router.experts):
                expert.lora_B.weight.normal_(mean=0.01 * (expert_index + 1), std=0.02)

        output, balance, all_expert_outputs, selected = router(
            hidden, router_context=context
        )

        self.assertEqual(output.shape, hidden.shape)
        self.assertEqual(all_expert_outputs.shape, (8, 4, 16))
        self.assertEqual(selected.shape, (8, 2))
        self.assertLess(int(selected.max()), 4)
        self.assertTrue(torch.isfinite(balance))
        (output.square().mean() + balance).backward()
        self.assertGreater(float(context.grad.abs().sum()), 0.0)
        self.assertGreater(float(router.context_gate.weight.grad.abs().sum()), 0.0)
        active_expert_gradients = [
            expert.lora_B.weight.grad is not None
            and float(expert.lora_B.weight.grad.abs().sum()) > 0.0
            for expert in router.experts
        ]
        self.assertTrue(any(active_expert_gradients))

    def test_context_shape_and_disabled_context_fail_loudly(self) -> None:
        hidden = torch.randn(2, 1, 16)
        router = self._router()
        with self.assertRaisesRegex(ValueError, "router context must have shape"):
            router.compute_router_logits(
                hidden, router_context=torch.randn(2, 1, 7)
            )

        disabled = BaseIndependentMoE(
            d_model=16, config=_config(), use_fofs=False
        )
        with self.assertRaisesRegex(ValueError, "router_context_dim is disabled"):
            disabled.compute_router_logits(
                hidden, router_context=torch.randn(2, 1, 8)
            )


if __name__ == "__main__":
    unittest.main()
