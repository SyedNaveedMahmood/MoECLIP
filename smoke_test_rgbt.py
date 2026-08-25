"""Smoke test for MoE-TwinCLIP (RGB-T) forward/backward + backward compat."""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from model.moe_adapter import MoECLIP, etf_loss, BaseIndependentMoE
from model.clip import create_model
from utils import setup_seed

setup_seed(111)
device = "cuda:0"

print("== loading CLIP ViT-L-14-336 ==")
clip_model = create_model(
    model_name="ViT-L-14-336",
    img_size=518,
    device=device,
    pretrained="openai",
    require_pretrained=True,
).to(device)
clip_model.eval()

model = MoECLIP(
    clip_model=clip_model,
    use_paa=True,
    seg_proj_sharing_strategy="shared",
    image_adapt_weight=0.1,
    moe_num_experts=4,
    moe_top_k=2,
    use_thermal=True,
    thermal_depth=4,
    num_shared_experts=1,
    modality_dropout=0.0,
).to(device)
# NOTE: repo convention — model stays in eval() during training (see train.py),
# otherwise PatchDropout(0.2) breaks PAA's square reshape even in original code.
model.eval()

# ---- 1. expert partition check ----
moe0 = model.image_adapter["moe_adapters"][0]
print("expert groups:", moe0.expert_groups)
assert moe0.expert_groups["rgb"] == [0, 1, 2], "rgb group wrong"
assert moe0.expert_groups["thermal"] == [3, 2], "thermal group wrong"
assert hasattr(moe0, "cond_proj"), "cond_proj missing"
print("[OK] expert partition + cond_proj")

x = torch.randn(1, 3, 518, 518, device=device)
t = torch.randn(1, 3, 518, 518, device=device)

# ---- 2. RGB-only path (backward compat) ----
with torch.no_grad():
    seg_r, det_r, aux_r, sp_r = model(x)
print(f"[OK] rgb-only fwd: {len(seg_r)} seg maps, det {tuple(det_r.shape)}, aux={float(aux_r):.4f}")

# ---- 3. dual-stream forward ----
seg_f, det_f, aux, sp, align_pair = model(x, thermal=t, return_align=True)
print(f"[OK] rgb-t fwd: {len(seg_f)} fused maps, det {tuple(det_f.shape)}, aux={float(aux):.4f}, etf={float(sp):.6f}")
assert align_pair is not None
g0 = torch.sigmoid(model.seg_gate_logits).detach().cpu()
print(f"seg gates (init ~0.12): {[round(float(v),3) for v in g0]}")

# ---- 4. backward through dual stream ----
loss = sum(m.sum() for m in seg_f) + det_f.sum() + aux + sp
f_rgb, f_t = align_pair
loss = loss + (1.0 - F.cosine_similarity(f_rgb, f_t, dim=-1)).mean()
loss.backward()

grad_checks = {
    "thermal_branch": any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.thermal_branch.parameters()),
    "thermal_adapter": any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.thermal_adapter.parameters()),
    "seg_gate_logits": model.seg_gate_logits.grad is not None and float(model.seg_gate_logits.grad.abs().sum()) > 0,
    "det_gate": any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.det_gate.parameters()),
    "cond_proj": model.image_adapter["moe_adapters"][0].cond_proj.weight.grad is not None,
}
for k, v in grad_checks.items():
    print(f"[{'OK' if v else 'FAIL'}] grad reaches {k}")
assert all(grad_checks.values()), "gradient check failed"

# NOTE: with the zero-initialized router, eval-mode Top-k breaks value ties by
# index, so expert selection is arbitrary until the router trains; coverage of
# the thermal-private expert is asserted in the training-branch check below.

peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"peak CUDA memory (batch 1 dual-stream): {peak:.2f} GiB")

# ---- 6. MoE training-branch path (ETF / load-balance + routed grads) ----
# toggle ONLY the MoE adapters to train mode (patch_dropout stays eval -> safe)
for m in model.image_adapter["moe_adapters"]:
    m.train()
model.zero_grad(set_to_none=True)
seg_f2, det_f2, aux2, sp2, align2 = model(x, thermal=t, return_align=True)
print(f"[OK] moe-train fwd: lb={float(aux2):.6f}, etf={float(sp2):.6f} "
      f"(etf grad is 0 at init since all lora_B start at zero; "
      f"routed losses break that deadlock after step 1)")
assert float(aux2) > 0 and float(sp2) > 0

(seg_f2[0].sum() + det_f2.sum()).backward()
ok_all = []
for li, m in enumerate(model.image_adapter["moe_adapters"]):
    g_t = m.experts[3].lora_B.weight.grad
    g_s = m.experts[2].lora_B.weight.grad
    ok = (g_t is not None and float(g_t.abs().sum()) > 0
          and g_s is not None and float(g_s.abs().sum()) > 0)
    ok_all.append(ok)
    print(f"  layer {li}: thermal-private+shared routed grads "
          f"{'OK' if ok else 'MISSING'}")
assert all(ok_all), "some MoE layer's thermal/shared experts got no gradient"

print("\nALL SMOKE TESTS PASSED")
