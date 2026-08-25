"""End-to-end fixture tests for region-conditioned RGB-only MoE adaptation."""

from __future__ import annotations

import io
import unittest

import torch
from torch import nn

from model.moe_adapter import MoECLIP


class _IdentityResidualBlock(nn.Module):
    def forward(self, x, attn_mask=None):
        return x, None, None


class _VisualTransformer(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            _IdentityResidualBlock() for _ in range(depth)
        )


class _FakeVisual(nn.Module):
    def __init__(self, width: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=14, stride=14, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(width))
        self.positional_embedding = nn.Parameter(torch.randn(5, width) * 0.01)
        self.patch_dropout = nn.Identity()
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = _VisualTransformer(depth)
        self.ln_post = nn.LayerNorm(width)


class _FakeCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _FakeVisual()


def _model(**kwargs) -> MoECLIP:
    defaults = {
        "clip_model": _FakeCLIP(),
        "levels": [1, 2, 3, 4],
        "moe_layers": [0, 1, 2, 3],
        "moe_r": 2,
        "moe_lora_alpha": 4,
        "use_fofs": False,
        "thermal_width": 16,
        "region_context_dim": 16,
        "region_attention_heads": 4,
    }
    defaults.update(kwargs)
    return MoECLIP(**defaults)


def _inputs():
    image = torch.randn(1, 3, 28, 28)
    thermal = torch.rand(1, 1, 28, 28)
    region_map = torch.zeros(1, 28, 28, dtype=torch.long)
    region_map[:, :14, 14:] = 1
    region_map[:, 14:, :14] = 2
    region_map[:, 14:, 14:] = 3
    return image, thermal, region_map


def _activate_conditioning(model: MoECLIP) -> None:
    with torch.no_grad():
        for adapter in model.image_adapter["moe_adapters"]:
            adapter.context_gate.weight.normal_(std=0.05)
            for expert_index, expert in enumerate(adapter.experts):
                expert.lora_B.weight.normal_(
                    mean=0.01 * (expert_index + 1), std=0.02
                )


class RegionMoECLIPTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)

    def test_original_rgb_path_still_produces_standard_outputs(self) -> None:
        model = _model().eval()
        image, _, _ = _inputs()

        with torch.no_grad():
            segmentation, detection, balance, etf = model(image)

        self.assertEqual(len(segmentation), 12)
        for feature in segmentation:
            self.assertEqual(feature.shape, (1, 4, 768))
        self.assertEqual(detection.shape, (1, 768))
        self.assertEqual(balance.item(), 0.0)
        self.assertEqual(etf.item(), 0.0)

    def test_region_thermal_forward_remains_rgb_patch_derived(self) -> None:
        model = _model(use_thermal=True, use_region_routing=True).eval()
        _activate_conditioning(model)
        image, thermal, region_map = _inputs()

        with torch.no_grad():
            first = model(image, thermal=thermal, region_map=region_map)
            second = model(
                image,
                thermal=torch.rand_like(thermal),
                region_map=region_map,
            )

        self.assertEqual(len(first[0]), 12)
        self.assertEqual(first[0][0].shape, (1, 4, 768))
        self.assertEqual(first[1].shape, (1, 768))
        self.assertNotIn("thermal_adapter", dict(model.named_children()))
        self.assertFalse(hasattr(model, "seg_gate_logits"))
        self.assertFalse(torch.equal(first[0][0], second[0][0]))
        for adapter in model.image_adapter["moe_adapters"]:
            for expert in adapter.experts:
                self.assertEqual(expert.lora_A.in_features, 32)
                self.assertEqual(expert.lora_B.out_features, 32)

    def test_patch_conditioned_thermal_variant_needs_no_region_map(self) -> None:
        model = _model(use_thermal=True, use_region_routing=False).eval()
        image, thermal, _ = _inputs()

        with torch.no_grad():
            segmentation, detection, _, _ = model(image, thermal=thermal)

        self.assertEqual(len(segmentation), 12)
        self.assertEqual(segmentation[0].shape, (1, 4, 768))
        self.assertEqual(detection.shape, (1, 768))

    def test_rgb_region_variant_and_missing_thermal_fallback_work(self) -> None:
        rgb_region_model = _model(use_region_routing=True).eval()
        multimodal_model = _model(
            use_thermal=True, use_region_routing=True
        ).eval()
        image, _, region_map = _inputs()

        with torch.no_grad():
            rgb_region_outputs = rgb_region_model(image, region_map=region_map)
            missing_thermal_outputs = multimodal_model(
                image, region_map=region_map
            )

        self.assertEqual(len(rgb_region_outputs[0]), 12)
        self.assertEqual(len(missing_thermal_outputs[0]), 12)
        self.assertFalse(multimodal_model.last_thermal_available.any())

    def test_gradients_reach_conditioner_router_experts_and_projection(self) -> None:
        model = _model(use_thermal=True, use_region_routing=True).train()
        _activate_conditioning(model)
        image, thermal, region_map = _inputs()

        segmentation, detection, balance, etf = model(
            image, thermal=thermal, region_map=region_map
        )
        loss = sum(
            (feature * torch.randn_like(feature)).mean()
            for feature in segmentation
        )
        loss = loss + (detection * torch.randn_like(detection)).mean()
        loss = loss + balance + etf
        loss.backward()

        self.assertTrue(torch.isfinite(balance))
        self.assertTrue(torch.isfinite(etf))
        gradients = {
            "thermal": model.thermal_branch.patch_embed.weight.grad,
            "attention": model.image_adapter["region_contexts"][0]
            .thermal_attention.query.weight.grad,
            "context": model.image_adapter["region_contexts"][0]
            .context_mlp[-1].weight.grad,
            "router": model.image_adapter["moe_adapters"][0]
            .context_gate.weight.grad,
            "projection": model.image_adapter["seg_proj"][0].fc[0].weight.grad,
        }
        for name, gradient in gradients.items():
            with self.subTest(component=name):
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0.0)
        expert_gradients = [
            expert.lora_B.weight.grad
            for adapter in model.image_adapter["moe_adapters"]
            for expert in adapter.experts
            if expert.lora_B.weight.grad is not None
        ]
        self.assertTrue(expert_gradients)
        self.assertTrue(any(float(gradient.abs().sum()) > 0 for gradient in expert_gradients))

    def test_modality_dropout_uses_conditioner_training_state_only(self) -> None:
        model = _model(
            use_thermal=True,
            use_region_routing=True,
            modality_dropout=1.0,
        ).train()
        image, thermal, region_map = _inputs()

        model(image, thermal=thermal, region_map=region_map)
        self.assertFalse(model.last_thermal_available.any())
        self.assertFalse(model.clipmodel.training)
        self.assertFalse(model.image_encoder.patch_dropout.training)
        self.assertTrue(model.image_adapter["region_contexts"].training)

        model.eval()
        model(image, thermal=thermal, region_map=region_map)
        self.assertTrue(model.last_thermal_available.all())

    def test_full_state_round_trip_is_deterministic(self) -> None:
        model = _model(use_thermal=True, use_region_routing=True).eval()
        _activate_conditioning(model)
        image, thermal, region_map = _inputs()
        with torch.no_grad():
            expected = model(image, thermal=thermal, region_map=region_map)

        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = _model(use_thermal=True, use_region_routing=True).eval()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        with torch.no_grad():
            actual = restored(image, thermal=thermal, region_map=region_map)

        for expected_feature, actual_feature in zip(expected[0], actual[0]):
            torch.testing.assert_close(
                actual_feature, expected_feature, rtol=0.0, atol=0.0
            )
        torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[2], expected[2], rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual[3], expected[3], rtol=0.0, atol=0.0)

    def test_region_routing_rejects_missing_map_and_active_patch_dropout(self) -> None:
        model = _model(use_region_routing=True).eval()
        image, _, region_map = _inputs()
        with self.assertRaisesRegex(ValueError, "region_map is required"):
            model(image)

        model.image_encoder.patch_dropout.train()
        with self.assertRaisesRegex(RuntimeError, "PatchDropout"):
            model(image, region_map=region_map)


if __name__ == "__main__":
    unittest.main()
