# Segment-Guided RGB-Thermal MoECLIP on MulSen-AD

Status: dataset audit/loader and the end-to-end RGB-only-expert,
region-conditioned MoECLIP model path are complete in offline smoke tests;
MulSen-AD training/evaluation CLI integration is not started.

Last updated: 2026-08-25

## Research question

Can RGB-region context and thermal evidence improve MoECLIP expert selection while
retaining RGB CLIP patch tokens, patch-level LoRA adaptation, and the original
normal/anomalous text-scoring pathway?

## FACTS

### Repository baseline

- Local `main` is synchronized with `origin/main` at commit `0e4e726`.
- The released-code reproduction and its paper/code differences are recorded in
  `RESULTS_SUMMARY.md`. In particular, the released RGB training path keeps the
  model in evaluation mode, so ETF, router balance loss, and LoRA dropout were not
  all active as described by the paper. Published reproduction numbers remain
  released-code baseline evidence and will not be relabeled as corrected runs.
- The exploratory MoE-TwinCLIP implementation is not the basis of the new model.
  Its loader, thermal fusion, modality dropout, router grouping, checkpointing,
  and evaluation paths require replacement. Original RGB checkpoints/results are
  retained.

### Official dataset and audit

- Official project: <https://github.com/MapleBoat/MulSen-AD>
- Official dataset host: <https://huggingface.co/datasets/orgjy314159/MulSen_AD>
- Audited archive: `MulSen_AD.zip`, 8,973,642,033 bytes.
- SHA-256:
  `455b5ea1732a8847c47433d9787eaa7d8f9b865b7967b0081b723baff4a8baca`.
- Extracted structure contains 15 categories, 1,391 paired normal training
  samples, and 644 paired test samples: 150 `good` plus 494 files in anomaly
  folders. All 2,035 RGB/IR pairs match by category, split, anomaly type, and
  filename.
- The release has 87 label CSVs. Labels are modality-specific columns named
  `RGB`, `infrared`, and `pointcloud`.
- Across the 494 anomaly-folder samples:
  - 282 are visible in both RGB and IR;
  - 78 are RGB-only;
  - 106 are IR-only;
  - 25 are point-cloud-only;
  - 3 have zero in all three CSV label columns.
- The three all-zero anomaly-folder records are
  `toothbrush/foreign_body/0`, `zipper/hole/1`, and `zipper/scratch/3`.
- Therefore, the primary RGB+IR image label is `RGB OR infrared`, not the anomaly
  folder name and not the official three-modality OR label.
- Modality-positive labels reconcile with 360 RGB masks and 388 IR masks.
  `nut/RGB/GT/color/8.png` is one orphan mask with no image or CSV row; it is
  warned about and ignored.
- RGB and IR masks are distinct, modality-specific annotations. They must never
  be silently unioned or assumed identical.

### Actual image storage

- All RGB images decode to `uint8`, 1280x960. The `.png`-named files contain 411
  BMP payloads and 1,624 PNG payloads; decoded modes are RGB or RGBA. Sampled
  RGBA alpha channels are opaque.
- All IR images decode to `uint8`, 640x480 and PIL mode RGB. The `.png`-named
  files contain 1,236 BMP payloads and 799 PNG payloads. This is not distributed
  raw 16-bit thermography.
- Full IR pixel audit:
  - decoded range: 0--255;
  - 2,005/2,035 images have exactly equal RGB channels;
  - 30 have sparse channel discrepancies;
  - discrepant pixels are 148,389/625,152,000 = 0.000237365;
  - maximum per-pixel channel spread is 20.
- Loader policy: validate the grayscale-like encoding, average decoded RGB
  channels symmetrically, divide by 255, and never perform per-image min/max
  normalization. Dataset mean/std, if used, must be estimated only from the
  final seen-category normal training images.

### Spatial correspondence

- The paper describes separate sequential capture: the IR camera is placed at
  random horizontal angles and the RGB pose/height is manually adjusted using
  the IR image, scale, robot, and software grid. It does not provide calibrated
  intrinsics, extrinsics, or a registration transform.
- A deterministic 16-pair audit covered all 15 categories, normal/anomalous
  examples, and RGB/IR label disagreements. RGB was resized to the native IR
  grid only for diagnosis.
- Best bounded edge-shift statistics were mean absolute `dx=17.06` and
  `dy=15.25` IR pixels; 12/16 pairs exceeded one 37x37 CLIP-grid patch in at
  least one dimension. Contact sheets also show sample-dependent scale and
  rotation differences.
- Conclusion: naive same-index RGB/IR patch correspondence is not scientifically
  justified. Common resizing is a tensor-shape operation, not registration.

## HYPOTHESES

### Core hypothesis

A patch should not be routed only from its RGB appearance. Its useful expert may
depend on its RGB structural region and on thermal evidence associated with that
region. Thermal evidence should influence expert selection without moving the
adapted representation out of CLIP visual space.

### V1 model boundary

- MoECLIP LoRA experts operate exclusively on RGB CLIP tokens.
- The lightweight thermal encoder produces conditioning features only. There is
  no separate thermal expert output and no late RGB/IR anomaly-map averaging.
- SLIC is computed on transformed RGB only, after shared geometry and before
  CLIP normalization. Ground-truth masks never enter SLIC or routing.
- Every RGB patch retains its own expert computation and CLIP-text comparison.
- Segment-aware PAA is deferred until the region-conditioned router works and is
  an independent ablation.

### Registration-tolerant thermal context

Directly selecting thermal token `i` for RGB patch `i` is rejected by the audit.
The serious region-guided model will instead use a lightweight correspondence
module:

1. RGB SLIC region `j` pools RGB CLIP features into `z_R[j]`.
2. `z_R[j]` queries the thermal 37x37 feature grid with cross-attention.
3. An optional broad normalized-coordinate bias acts as a soft prior, never a
   hard same-index constraint.
4. The attended thermal summary is `z_T[j]`.
5. A context MLP receives projected
   `[z_R, z_T, z_R - z_T, z_R * z_T]`.
6. The context produces a zero-initialized region logit residual `delta[j]`.
7. RGB patch `i` in region `R(i)` routes with
   `logits[i] = base_rgb_router(f_R[i]) + delta[R(i)]`.
8. Top-2 of four learned experts transforms only `f_R[i]`.

Zero initialization makes the new conditioning path start from the RGB router
rather than perturbing the released pathway arbitrarily. Experts are not assigned
hard-coded modality semantics in v1.

### Experimental variants

- **A -- RGB MoECLIP baseline:** original RGB pathway, with released-code behavior
  reported separately from any corrected training-mode run.
- **B -- RGB+IR patch-conditioned routing:** each RGB patch queries the thermal
  grid without SLIC; no same-index assumption.
- **C -- RGB-only region routing:** SLIC RGB region context conditions the router.
- **D -- RGB+IR region-conditioned routing:** proposed region-to-thermal attention
  and region router residual.
- **E -- D plus segment-aware PAA:** implemented only after D works.

## Proposed category-held-out ZSAD protocol

This split was selected from sample counts, modality visibility, material, and
shape metadata before any model result was available.

### Final seen categories `D_s` (10)

`button_cell, capsule, cube, flat_pad, light, plastic_cylinder, screen, screw,
spring_pad, zipper`

- 885 official normal training images.
- 331 anomaly-folder samples: 307 RGB-or-IR-visible, 22 point-cloud-only, and 2
  all-zero-label samples.

### Locked final unseen categories `D_u` (5)

`cotton, nut, piggy, solar_panel, toothbrush`

- Their 506 official normal training images are not used.
- Final official test has 50 `good` and 163 anomaly-folder samples.
- Primary RGB+IR test subset: 50 normal plus 159 RGB-or-IR-visible anomalies.
- The remaining three point-cloud-only and one all-zero-label samples are
  excluded from the primary metric, not relabeled as normal or anomalous.

### Development split inside `D_s`

- Development train (8): `button_cell, capsule, cube, flat_pad, light, screen,
  spring_pad, zipper`.
  - 705 normal train images.
  - 251 RGB-or-IR-visible labeled anomalies are eligible for auxiliary
    supervision.
  - 19 point-cloud-only and 2 all-zero-label samples are excluded from
    supervised anomaly training.
- Category-held-out development validation (2): `plastic_cylinder, screw`.
  - No image from either category enters development training or normalization.
  - Validation uses 20 normal and 56 RGB-or-IR-visible abnormal test samples.
- Select flags/hyperparameters and a fixed epoch budget only on this development
  split. Then refit on all ten `D_s` categories for that fixed budget and evaluate
  once on locked `D_u`.
- No `D_u` image, mask, modality label, or normalization statistic enters model
  fitting or selection. Class names may be supplied to the CLIP prompt at final
  inference, consistent with the CLIP-based zero-shot scoring pathway.
- One primary split is a prototype, not evidence of split robustness. A later
  multi-fold category protocol and multiple seeds are required for stronger
  claims.

## Loss and evaluation policy

- Preserve segmentation, classification, ETF, and router-balance objectives,
  but first correct and smoke-test their actual activation under module train/eval
  state.
- Optional cross-modal alignment is off by default in v1. If added, it must be
  limited or weighted using seen-category normal supervision; it must not force
  modality-exclusive anomalies to match.
- Modality dropout is training-only and must have an explicit activation test.
- Primary image label: `label_rgbt = label_rgb OR label_thermal`.
- `label_any` is retained for auditing only while point cloud is absent.
- RGB-space pixel metrics use RGB masks on RGB-visible anomalies plus normal
  samples. IR masks are preserved, but an RGB patch map is not evaluated against
  an IR mask until a defensible correspondence/transport method exists.
- Pixel maps remain RGB patch-derived. The wording is: the extension "preserves
  the CLIP-based zero-shot scoring pathway," not that it guarantees zero-shot
  generalization.

## Implementation status

- `tools/inspect_mulsen_alignment.py`: read-only pairing, encoding, label/mask,
  and edge-overlay audit.
- `dataset/mulsen_ad.py`: standalone strict RGB+IR loader with separate labels,
  separate masks, synchronized optional geometry, and RGB-only SLIC.
- `tests/test_mulsen_ad.py`: synthetic loader tests.
- `model/thermal_branch.py::ThermalEncoder`: 1.12M-parameter, four-block local
  thermal encoder. A 518x518 one-channel tensor produces four patch-only
  37x37 taps with width 256. It has no CLS token, no unregistered positional
  tensor, and is not yet connected to MoECLIP.
- `tests/test_thermal_encoder.py`: patch-count, input-contract, gradient,
  parameter-budget, and state-dict round-trip tests.
- `model/region_context.py`: modal-label reduction from RGB SLIC pixels to the
  CLIP grid, per-image region pooling, region-to-full-thermal-grid attention,
  a broad optional coordinate prior, and region-context broadcast back to
  individual RGB patches. Identity regions support variant B without assuming
  same-index RGB/IR fusion.
- `tests/test_region_context.py`: deterministic patch assignment/pooling,
  full-grid attention, soft-prior, missing-modality, padding, and gradient tests.
- `model/moe_adapter.py::BaseIndependentMoE`: optional region-context logit
  residual with a zero-initialized head. Its default exposes all experts to
  learned context; a checkpointed fixed subset supports the context-expert-count
  ablation. Expert inputs remain RGB hidden states only.
- `tests/test_region_router.py`: initialization equivalence, context masking,
  Top-k shape, RGB-width expert output, gradient, and failure-contract tests.
- `model/moe_adapter.py::MoECLIP`: variants A--D now share the RGB CLIP
  readout. Thermal taps and RGB region context add router-logit residuals at
  four MoE layers; only RGB tokens enter the LoRA experts. Frozen CLIP remains
  in evaluation mode when adapters train, preventing PatchDropout from breaking
  the square patch layout.
- The exploratory thermal-expert stream, thermal readout projections, final
  fusion gates, and unregistered lazy thermal positional state were removed.
- `tests/test_region_moeclip.py`: A/B/C/D forward contracts, 12 standard-PAA
  maps, RGB-only expert widths, dropout mode, gradients, PatchDropout guard, and
  deterministic full-state round-trip tests on a small CLIP-shaped fixture.
- `dataset/mulsen_protocol.py`: executable, locked development/final category
  partitions. Training composes official normal-train samples with only
  RGB-or-IR-visible seen-category anomalies; evaluation retains good and visible
  anomalies while filtering unavailable records without relabeling.
- `dataset/constants.py`: MulSen-AD prompt names/domain added and the old
  user-specific absolute data root replaced by a repository-relative path.
- `tests/test_mulsen_protocol.py`: category disjointness/completeness, selection
  rules, geometry-mode, and prompt-coverage tests.
- The existing RGB `get_dataset`, training, evaluation, model, checkpoints, and
  command-line interface are unchanged at this stage.

## Reproducible smoke commands

```powershell
conda activate moeclip
$Py = (Get-Command python).Source

& $Py -m unittest `
  tests.test_mulsen_ad `
  tests.test_thermal_encoder `
  tests.test_region_context `
  tests.test_region_router `
  tests.test_region_moeclip `
  tests.test_mulsen_protocol `
  tools.test_inspect_mulsen_alignment -v

& $Py tools\inspect_mulsen_alignment.py `
  --data-root "data\MulSenAD_official\MulSen_AD" `
  --output-dir "data\mulsen_alignment_audit" `
  --sample-count 16 `
  --seed 111 `
  --max-shift 32
```

No training or evaluation command is defined yet.

## RESULTS

### Observed

- Dataset/archive/pairing/encoding/mask audit: passed with the single documented
  orphan-mask warning.
- Ten synthetic inspector/loader tests: passed.
- Real loader smoke: 1,391 train and 644 test records; one sample from every
  category decoded; one 518x518 SLIC sample produced 63 contiguous regions when
  64 were requested (SLIC's segment count is approximate).
- Full 518x518 test-loader integrity pass: all 644 pairs decoded; all 360
  RGB-positive and 388 IR-positive masks remained nonempty after nearest-neighbor
  resizing; negative-modality masks remained zero.
- Thermal encoder smoke on the real `capsule/train/good/0.png` IR payload:
  input `[1,1,518,518]`, four `[1,1369,256]` taps, finite nonzero gradients at
  the patch stem, early/late spatial blocks, and output normalization/projection.
  The default encoder has 1,117,952 trainable parameters. State-dict round-trip
  produces bit-identical deterministic evaluation output in the unit fixture.
- Full-grid region-context smoke on that sample: 63 RGB SLIC regions remained
  after modal-label patch reduction; `[1,63,256]` region contexts were broadcast
  to `[1,1369,256]` RGB patch contexts. Every valid region attended over all
  1,369 thermal tokens, attention sums were numerically one, and finite nonzero
  gradients reached RGB inputs, the thermal stem, and the context MLP.
- Isolated MoE routing smoke: a zero context head exactly preserved base router
  logits; after a nonzero head perturbation, Top-2 routing remained patch-level,
  context gradients were nonzero, and experts produced only 16-wide RGB test
  features rather than consuming the 8-wide conditioning representation.
- End-to-end four-layer fixture: A/B/C/D each produced patch-derived outputs;
  standard PAA retained 12 maps, thermal perturbation changed D only through
  router conditioning, and gradients reached the thermal stem, full-grid
  attention, context MLP, router head, active RGB LoRA experts, and RGB
  segmentation projection. Training-only modality dropout activated at
  probability one and disabled in evaluation. A state-dict reload was
  bit-identical in deterministic evaluation.
- Real record-index smoke (no image decode): development training is 705 normal
  plus 251 visible anomalies, development validation is 20 normal plus 56
  visible anomalies, final refit is 885 plus 307, and locked unseen evaluation
  is 50 plus 159. These reproduce the predeclared audit counts.
- No model has been trained or evaluated for this extension.

### Not results

- The proposed router, end-to-end multimodal model, split performance, ablations,
  CUDA memory use, and interview claims are hypotheses/plans until experiments
  are actually run.

## Known limitations and next gates

- Sequential RGB/IR acquisition is not calibrated. Cross-attention may still
  learn spurious correspondences; attention maps and routing distributions must
  be inspected.
- IR localization cannot be scored by comparing an unregistered IR mask directly
  with an RGB patch map.
- Thermal mean/std has not been estimated. It must use only final `D_s` normal
  training images and be saved with the experiment config.
- The primary category split has not been tested and must remain locked before
  model results are observed.
- Next implementation gate: MulSen-AD protocol/CLI/checkpoint integration,
  followed by a real ViT-L/14 batch-one forward/backward and CUDA-memory smoke.
  No segment-aware PAA until this region-conditioned path passes that real-model
  gate.
