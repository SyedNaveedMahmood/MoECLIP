# MoE-TwinCLIP: Extending MoECLIP to RGB-Thermal Zero-Shot Anomaly Detection

## 1. The Idea

MoECLIP adapts a frozen CLIP ViT-L/14-336 with a **Mixture-of-LoRA-Experts** (K=4,
Top-2) whose experts specialize on *tasks* (ETF decorrelation + load balancing).
Our extension promotes **modality itself to a routing signal**: instead of adding
a second vision-language model or naively stacking channels, we split each MoE
layer's experts into modality groups and let every routing decision see *both*
streams:

- **Expert partitioning** per MoE layer: `{0,1}` RGB-private, `{2}` shared,
  `{3}` thermal-private. Both streams invoke the *same* LoRA expert weights
  (true cross-modal weight sharing); masking restricts which experts each
  stream may select.
- **Cross-modal conditioned routing** (`cond_proj`): the router gate input is
  `cond_proj([own-stream token ; other-stream token])`, so fusion happens
  implicitly through routing weights rather than naive feature averaging.
- **Lightweight thermal branch** (`model/thermal_branch.py`): a 4-block,
  width-1024 transformer (~30M trainable params) encoding registered thermal
  images into CLIP-sized patch tokens. It advances one block right before each
  CLIP MoE layer ("interleaved stepping"), so every MoE-updated thermal state
  feeds all subsequent thermal computation.
- **Uncertainty-gated convex fusion** at readout: per-level learnable gates mix
  RGB/thermal similarity maps (init ~12% thermal), plus an input-dependent det
  gate `sigmoid(MLP([det_rgb ; det_t]))`. Fused features stay inside CLIP's
  text-aligned embedding space, preserving zero-shot transfer via adapted text
  embeddings — unchanged from MoECLIP.
- **Modality dropout** (p=0.3 during training): the thermal stream is randomly
  dropped and gates forced to RGB-only, giving graceful degradation to RGB-only
  inference.
- **Cross-modal alignment loss**: cosine alignment of pooled paired features,
  weighted toward normal samples.

Rationale: any fusion placed *before* the text comparison preserves zero-shot
generalization; late-fusing two independent models does not. Channel-stacking
thermal into CLIP's stem breaks pretraining statistics (ablation target), and
thermal-to-RGB translation introduces phantom-anomaly hallucinations.

## 2. Data Strategy

No public benchmark provides *registered* RGB+thermal pairs with pixel-level
defect masks (verified against FLIR ADAS / KAIST / LLVIP / VT-series /
InfraredSolarModules — each misses either pairing, registration, or defect GT).
We therefore generate **pseudo-thermal counterparts**
(`tools/make_pseudo_thermal.py`):

```
grayscale -> invert (warm/bright emits more) -> gamma 0.8 -> Gaussian blur sigma 2 -> +/-2% noise
```

Thermal files mirror the RGB relative paths under `data/<name>_T/`, so the
paired loader needs no metadata changes. The loader is source-agnostic: real
paired data (e.g., a self-collected FLIR set) drops in later without model-code
changes. Known limitation: pseudo-thermal cannot reproduce real emissivity
failures (glass transparency, metal reflectivity).

## 3. Implementation Map

| File | Change |
|---|---|
| `model/thermal_branch.py` | NEW - ThermalTransformer (embed/step interleave API) |
| `model/moe_adapter.py` | Expert groups, cond_proj, dual-stream `_forward_rgb_thermal`, `_fuse_readouts`, fusion gates |
| `dataset/__init__.py` | `PairedBaseDataset` (5-channel joint geometric augmentation), `get_dataset(use_thermal=)` |
| `dataset/constants.py` | `THERMAL_PATH` |
| `tools/make_pseudo_thermal.py` | NEW - pseudo-thermal generator |
| `train.py` | `--use_thermal --thermal_depth --num_shared_experts --modality_dropout --align_loss_lambda`, alignment loss, checkpoint/resume extensions |
| `test.py` | unchanged (RGB-only path identical by design) |

Backward compatibility is structural: `model(image)` returns the exact original
4-tuple through the untouched `_forward_rgb` path.

## 4. Validation (`smoke_test_rgbt.py`, all passing)

1. Expert partition `{rgb:[0,1,2], thermal:[3,2]}` + `cond_proj` present
2. RGB-only forward returns the original signature/output structure
3. Dual-stream forward produces 12 fused seg maps + fused detection token
4. Gradients reach every new module (branch, projections, gates, cond_proj)
5. All 4 MoE layers' thermal-private + shared experts receive routed gradients
6. Peak memory 7.9 GiB (batch 1 dual-stream fwd/bwd)
7. Full `train.py` epoch on MVTec pseudo-thermal completed (863 steps,
   ~0.66 s/step, avg loss 12.57); checkpoints saved under `ckpt/rgbt_smoke/`

### Engineering notes
- **Repo convention**: the model stays in `eval()` during training — global
  `.train()` activates PatchDropout(0.2), which breaks PAA's square reshape
  (pre-existing behavior). We toggle only the MoE adapters to train mode so the
  ETF/load-balance losses activate.
- **Upstream bug fixed**: load-balance coefficient was read from
  `router_aux_loss_coef` while the config stores `router_aux_loss_coef_`,
  silently disabling it; corrected.
- **ETF cold-start**: with all `lora_B` zero-initialized every expert output is
  exactly zero and the ETF gradient vanishes identically; routed task losses
  wake the RGB experts on step 1, after which ETF decorrelation engages for all
  groups.

## 5. Usage

```bash
# 1. pseudo-thermal generation (once per dataset)
python tools/make_pseudo_thermal.py --src data/MVTec --dst data/MVTec_T

# 2. RGB-T training
python train.py --dataset MVTec --use_thermal \
    --num_shared_experts 1 --modality_dropout 0.3 --align_loss_lambda 0.1 \
    --save_path ckpt/rgbt --epoch 20

# 3. sanity suite
python smoke_test_rgbt.py
```

## 6. Status & Next Steps

Validated: architecture trains, fuses, back-propagates, and checkpoints
end-to-end on MVTec + pseudo-thermal. Not yet run: full 20-epoch RGB-T training
+ ZSAD evaluation sweep, ablations (expert-split ratios, w/o conditioning,
w/o alignment, w/o modality dropout, +/- shared experts), real-thermal pilot.

