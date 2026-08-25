# Segment-Guided RGB-Thermal MoECLIP on MulSen-AD

Status: the audited MulSen-AD loader, locked protocol, RGB-only-expert RGB-T
model, thermal statistics, real ViT-L/14 smoke tests, and A/B/D seed-111
development runs are complete. The final held-out categories remain sealed and
unevaluated.

Last updated: 2026-08-25

## Research question

Can RGB-region context and thermal evidence improve MoECLIP expert selection while
retaining RGB CLIP patch tokens, patch-level LoRA adaptation, and the original
normal/anomalous text-scoring pathway?

## FACTS

### Repository baseline

- The audited v1 implementation and A/B/D development evidence are preserved
  through commit `39d9414`; the documentation-corrected save point is identified
  by the annotated tag `mulsen-v1-dev-audit`.
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
- The maximum *per-image* non-equal-channel fraction is 0.091858724 in
  `capsule/Infrared/train/5.png`; its maximum channel spread is only 4. The
  loader therefore uses independent audited caps of 10% discrepant pixels and
  channel spread 20. The earlier 5% cap was too strict and was corrected when
  the user-run statistics pass reached this file.
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
- Segment-aware PAA was implemented only after the region-conditioned router
  passed its real-model gate and remains an independent, parameter-free
  ablation. Standard PAA remains the default.

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
- **E -- D plus segment-aware PAA:** same D model with same-SLIC-neighbor PAA
  enabled at scales 1, 3, and 5.

## Locked category-held-out ZSAD protocol

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
- Development metrics are macro-averaged across `plastic_cylinder` and `screw`.
  The predeclared checkpoint-selection score is the mean of (a) macro image
  AUROC using MoECLIP's industrial 0.5 detection + 0.5 max-patch rule and (b)
  macro RGB-pixel AUROC. Exact ties select the earliest epoch.
- Detection-only image AUROC/AP and RGB-pixel AP are reported alongside the
  selection metrics. IR-only anomalies enter image metrics but are excluded
  from RGB-pixel metrics; they are never assigned fabricated RGB masks.
- The final unseen stage accepts one fixed checkpoint only. It cannot scan final
  epochs or select a checkpoint using `D_u`.

## Implementation status

- `tools/inspect_mulsen_alignment.py`: read-only pairing, encoding, label/mask,
  and edge-overlay audit.
- `dataset/mulsen_ad.py`: standalone strict RGB+IR loader with separate labels,
  separate masks, synchronized optional geometry, and RGB-only SLIC.
- `tests/test_mulsen_ad.py`: synthetic loader tests.
- `model/thermal_branch.py::ThermalEncoder`: 1.12M-parameter, four-block local
  thermal encoder. A 518x518 one-channel tensor produces four patch-only
  37x37 taps with width 256. It has no CLS token, no unregistered positional
  tensor, and supplies conditioning taps only to the region-context modules.
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
  maps, RGB-only expert widths, dropout mode, gradients, PatchDropout guard,
  deterministic full-state round-trip, and exact segment-PAA boundary tests on
  a small CLIP-shaped fixture.
- Optional segment-aware PAA applies the same spatial scales `{1,3,5}` but
  includes a neighbor only when its SLIC-derived patch region equals the center
  patch's region. Padding and other regions are excluded from the denominator.
  It is valid only for C/D, adds no parameters, and leaves standard PAA intact.
- `dataset/mulsen_protocol.py`: executable, locked development/final category
  partitions. Training composes official normal-train samples with only
  RGB-or-IR-visible seen-category anomalies; evaluation retains good and visible
  anomalies while filtering unavailable records without relabeling.
- `dataset/constants.py`: MulSen-AD prompt names/domain added and the old
  user-specific absolute data root replaced by a repository-relative path.
- `tests/test_mulsen_protocol.py`: category disjointness/completeness, selection
  rules, geometry-mode, and prompt-coverage tests.
- `tools/compute_mulsen_thermal_stats.py` and `dataset/mulsen_stats.py`: a
  user-run streaming normal-IR statistic command plus strict stage/category
  metadata validation. This prevents a development run from loading statistics
  contaminated by validation or unseen categories.
- `train_mulsen.py`: separate A/B/C/D training CLI. It requires stage-matched
  IR statistics for B/D, uses the locked dataset composition, keeps frozen CLIP
  in eval while adapters train, enables AMP on CUDA, and keeps cross-modal
  alignment disabled in v1.
- The MulSen CLI explicitly enables AMP-safe adapter norm matching with floor
  1.0 and initial loss scale 1,024; both are recorded in experiment config and
  overridable. `MoECLIP` defaults to the released norm expression, so existing
  `train.py`/`test.py` behavior is not silently changed.
- `mulsen_checkpoint.py`: strict component, optimizer, scheduler, scaler,
  experiment-config, global RNG, and DataLoader-generator checkpointing with
  atomic replacement. Training recreates workers each epoch so resumed shuffle
  order and joint-augmentation RNG derive from the restored generator state.
- `evaluate_mulsen.py`: reconstructs architecture only from checkpoint config,
  evaluates the locked category subset, keeps image/RGB-pixel label semantics
  separate, records checkpoint hashes and per-sample score provenance, and
  forbids multi-checkpoint selection on final unseen categories.
- `tests/test_evaluate_mulsen.py`: constant-score stability, IR-only exclusion
  from RGB-pixel metrics, inclusion in image metrics, and selection-score tests.
- New MulSen runs default to a small random base router to avoid deterministic
  zero-logit Top-k ties; `router_init=zero` remains the default in `MoECLIP`
  itself for released RGB-path compatibility. The context residual remains
  zero-initialized in both modes.
- The exploratory paired pseudo-thermal loader/generator and its obsolete
  `train.py` flags were removed. The normal released RGB CLI remains available;
  the new research path is deliberately isolated in `train_mulsen.py`.
- The existing RGB `get_dataset`, training, evaluation, model, checkpoints, and
  command-line interface are unchanged at this stage.

## Reproducible smoke commands

```powershell
$Conda = "C:\Users\user7\miniconda3\Scripts\conda.exe"

& $Conda run --no-capture-output -n moeclip python -m unittest `
  tests.test_mulsen_ad `
  tests.test_thermal_encoder `
  tests.test_region_context `
  tests.test_region_router `
  tests.test_region_moeclip `
  tests.test_mulsen_protocol `
  tests.test_mulsen_stats `
  tests.test_mulsen_checkpoint `
  tests.test_train_mulsen `
  tools.test_inspect_mulsen_alignment -v

& $Conda run --no-capture-output -n moeclip python `
  tools\inspect_mulsen_alignment.py `
  --data-root "data\MulSenAD_official\MulSen_AD" `
  --output-dir "data\mulsen_alignment_audit" `
  --sample-count 16 `
  --seed 111 `
  --max-shift 32

# User-run preprocessing; choose the stage being trained.
& $Conda run --no-capture-output -n moeclip python `
  tools\compute_mulsen_thermal_stats.py `
  --data-root "data\MulSenAD_official\MulSen_AD" `
  --protocol-stage development `
  --output "data\MulSenAD_official\thermal_stats_development.json"

& $Conda run --no-capture-output -n moeclip python `
  tools\smoke_mulsen_real_model.py `
  --data-root "data\MulSenAD_official\MulSen_AD" `
  --thermal-stats "data\MulSenAD_official\thermal_stats_development.json" `
  --protocol-stage development `
  --variant D `
  --sample-index 0 `
  --img-size 518 `
  --seed 111 `
  --amp-init-scale 1024 `
  --use-segment-paa
```

## User-run development commands

These are the exact commands used for the seed-111 A/B/D development runs.
They are retained for reproducibility; existing output directories must not be
overwritten.

```powershell
Set-Location "C:\Users\user7\Desktop\moeclip"
$Conda = "C:\Users\user7\miniconda3\Scripts\conda.exe"
$DataRoot = "data\MulSenAD_official\MulSen_AD"
$ThermalStats = "data\MulSenAD_official\thermal_stats_development.json"
$Common = @(
  "--dataset", "MulSenAD",
  "--data_root", $DataRoot,
  "--protocol_stage", "development",
  "--model_name", "ViT-L-14-336",
  "--img_size", "518",
  "--moe_r", "8",
  "--moe_lora_alpha", "16",
  "--moe_num_experts", "4",
  "--moe_top_k", "2",
  "--moe_layers", "5,11,17,23",
  "--router_init", "normal",
  "--image_adapt_weight", "0.1",
  "--seg_proj_sharing_strategy", "shared",
  "--slic_segments", "64",
  "--slic_compactness", "10.0",
  "--thermal_depth", "4",
  "--thermal_width", "256",
  "--region_context_dim", "256",
  "--region_attention_heads", "4",
  "--region_coordinate_bias", "1.0",
  "--region_coordinate_sigma", "0.75",
  "--num_context_experts", "4",
  "--modality_dropout", "0.2",
  "--align_loss_lambda", "0.0",
  "--adapter_norm_floor", "1.0",
  "--epochs", "20",
  "--batch_size", "1",
  "--workers", "4",
  "--lr", "5e-5",
  "--weight_decay", "0.0",
  "--lr_milestones", "12,16",
  "--lr_gamma", "0.1",
  "--balance_loss_lambda", "0.01",
  "--etf_loss_lambda", "0.01",
  "--amp_init_scale", "1024",
  "--seed", "111"
)

# A: RGB-only MoECLIP baseline.
& $Conda run --no-capture-output -n moeclip python train_mulsen.py @Common `
  --variant A `
  --output_dir "ckpt\mulsen_dev_A_seed111"

# B: thermal patch-conditioned routing, without SLIC.
& $Conda run --no-capture-output -n moeclip python train_mulsen.py @Common `
  --variant B `
  --thermal_stats $ThermalStats `
  --output_dir "ckpt\mulsen_dev_B_seed111"

# D: proposed RGB+IR segment-guided routing with standard PAA.
& $Conda run --no-capture-output -n moeclip python train_mulsen.py @Common `
  --variant D `
  --thermal_stats $ThermalStats `
  --output_dir "ckpt\mulsen_dev_D_seed111"

# Leakage-safe development validation and checkpoint selection (example A).
& $Conda run --no-capture-output -n moeclip python evaluate_mulsen.py `
  --checkpoint_dir "ckpt\mulsen_dev_A_seed111" `
  --data_root $DataRoot `
  --output "ckpt\mulsen_dev_A_seed111\development_validation.json" `
  --batch_size 1 `
  --workers 4

# Read-only selected-checkpoint routing audit (example A-18).
& $Conda run --no-capture-output -n moeclip python `
  tools\inspect_mulsen_routing.py `
  --checkpoint "ckpt\mulsen_dev_A_seed111\mulsen_epoch_018.pth" `
  --data_root $DataRoot `
  --output "ckpt\mulsen_dev_A_seed111\routing_audit_selected.json" `
  --batch_size 1 `
  --workers 4
```

These commands generated the local A/B/D checkpoints and development reports
described below. No final-stage model has been fit or evaluated.

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
- Offline training-loss wiring is finite and differentiable; fixture checkpoint
  round-trip restores image/MoE/region modules, thermal encoder, text adapter,
  optimizer, scheduler, config, and deterministic output. This is code evidence,
  not evidence that a MulSen model has trained successfully.
- User-run development thermal statistics completed over exactly 705 normal IR
  images (216,576,000 native pixels): mean 0.614252555096359 and population
  standard deviation 0.33349515475237973 after symmetric RGB-channel averaging
  and division by 255. The JSON SHA-256 is
  `8159917b95b1592053189b9552cba0c0871848730d6d1fa2193dc868589b7d0d`.
- Real ViT-L/14-336 variant-D CUDA smoke on a 518x518 `button_cell` normal pair
  passed with 12 `[1,1369,768]` patch-derived segmentation maps, one
  `[1,768]` image feature, 10,019,584 trainable parameters, 6,723.07 MiB peak
  allocated CUDA memory, and 7,052 MiB peak reserved memory. The first backward
  gave a finite nonzero gradient to the zero-initialized context-router head;
  after one in-memory AdamW update, finite nonzero gradients reached the thermal
  encoder, full-grid thermal attention, context MLP, router head, all 16 RGB
  LoRA expert instances, segmentation projection, and text adapter. The smoke
  wrote no checkpoint and is not a training result.
- The first real AMP smoke exposed non-finite backward gradients despite finite
  losses. Root causes were singular norm matching at zero-initialized LoRA
  outputs and half-precision ETF normalization; both are now hardened. AMP
  initial scale 65,536 still overflowed this batch, while an explicit scale
  1,024 was finite and is now the configurable, recorded MulSen default.
- Real variant-E (D plus segment-aware PAA) CUDA smoke passed on the same sample
  with 12 `[1,1369,768]` maps and finite gradients through every required
  component after the router-head stabilization update. It used 6,629.50 MiB
  peak allocated and 6,968 MiB peak reserved CUDA memory. These are feasibility
  measurements, not evidence that segment-aware PAA improves anomaly detection.
- Real A and B CUDA smokes also passed under the same sample, seed, checkpoint,
  and AMP settings. A required no thermal statistics, retained 12 maps, and used
  4,674,560 trainable parameters with 6,574.55 MiB peak allocation. B retained
  12 maps and activated thermal/context gradients after the stabilization step,
  with 10,019,584 trainable parameters and 6,980.26 MiB peak allocation. The
  generalized diagnostic therefore covers the prioritized A/B/D paths.
- Seed-111 development training completed for A, B, and D with the locked eight
  training categories and without any images, labels, or masks from the two
  held-out development categories entering training. All runs completed 20
  epochs without NaN/Inf, AMP, CUDA-memory, SLIC, or dataloader failures. Final
  training losses were A 2.894581, B 2.619047, and D 2.600920. The lower B/D
  training losses did **not** imply better held-out transfer.
- Leakage-safe validation scanned all 20 checkpoints on only
  `plastic_cylinder` and `screw`, using the predeclared selection score and
  earliest-epoch tie-break. This is one seed, not an estimate of variance:

  | Variant | Selected epoch | Selection | Image AUROC | Detection-only AUROC | RGB pixel AUROC | RGB pixel AP |
  |---|---:|---:|---:|---:|---:|---:|
  | A | 18 | 0.859809 | 0.731677 | 0.430839 | 0.987941 | 0.204185 |
  | B | 8 | 0.779219 | 0.570000 | 0.235548 | 0.988437 | 0.194250 |
  | D | 13 | 0.724571 | 0.458129 | 0.259871 | 0.991014 | 0.197952 |

  The result-file SHA-256 values are A
  `f1ecd3028332f38d3e53893ea3fcb19e86d2f00351174ab5eea319ed9ff38173`,
  B `20e4f6a03f602871cca8faebc14638e1b8a43103a3dd9addbd3543e36dab0750`,
  and D `943b710e7ec40dabfad55305c2e0cbae9e29657c97f31932d2f84bd06f424fdf`.
- The negative B/D result is concentrated at image level, not pixel ranking.
  Raw patch-maximum AUROC on `plastic_cylinder`/`screw` was A 0.756/0.774,
  B 0.712/0.642, and D 0.796/0.374. D therefore improved the patch-maximum
  ranking on `plastic_cylinder` but inverted it on `screw`. The development
  `plastic_cylinder` set contains 10 normal, 17 RGB-visible anomalous, and 8
  IR-only anomalous images. D's mean raw patch maximum was -266.10 for normal,
  -170.37 for RGB-visible anomalous, and -267.34 for IR-only anomalous images:
  thermal-conditioned routing did not make IR-only defects score anomalously in
  the retained RGB/text scoring space. `screw` has no IR-only anomalies in this
  split, so modality visibility alone does not explain its inversion.
- Read-only hooks audited the selected A-18, B-8, and D-13 routers over all 76
  held-out images. Soft routing did not collapse: effective expert counts stayed
  near four, and Top-2 selection used multiple experts at every layer. Context
  influence was concentrated at transformer block 17. For B, mean absolute
  context logits were 1.748 times base RGB logits and changed the patch Top-1
  expert for 47.5% of tokens; for D the corresponding values were 0.527 and
  37.6%. D's context/base ratio at block 23 was only 0.0027. CLS context logits
  were exactly zero at every layer by the v1 design. Thus the observed failure
  is not explained by inactive losses or total expert collapse; it is consistent
  with patch/image score calibration and uneven layer-wise context influence.
- The routing audit added two deterministic tests. The complete offline suite is
  now 50 repository tests plus 3 alignment-inspector tests, all passing.

### Not results

- Final unseen-category performance, multi-seed stability, segment-aware PAA
  effects, and generalization claims remain unknown. The table above is a single
  development seed and must not be described as variance or final ZSAD evidence.

## Known limitations and next gates

- Sequential RGB/IR acquisition is not calibrated. Cross-attention may still
  learn spurious correspondences; attention maps and routing distributions must
  be inspected.
- IR localization cannot be scored by comparing an unregistered IR mask directly
  with an RGB patch map.
- Development thermal mean/std is now fixed from only the eight development
  training categories. Final-refit statistics remain intentionally uncomputed;
  they must be recomputed from normal training images in final `D_s` only.
- The final unseen-category side of the primary split has not been tested and
  must remain locked. Development results have now been observed, but the five
  final unseen categories remain untouched.
- V1 supplies zero router context to CLS. Thermal/region information can affect
  the image feature only indirectly through patch-to-CLS attention in later
  blocks. This is a plausible reason that B/D retain pixel localization while
  degrading image-level calibration.
- **Hypothesis for v1.1, not a result:** condition the RGB CLS router with a
  pooled global summary of the same region contexts, and add an explicit,
  configurable per-layer context-residual scale. This preserves RGB CLIP-space
  expert outputs and the text-scoring pathway while making image-level thermal
  evidence direct and bounding layer-17 context dominance. It must be introduced
  as a new development ablation, not silently substituted for D.
- Next gate: add image-score subgroup diagnostics (RGB-visible versus IR-only),
  implement the single v1.1 ablation above, and compare it once against A/D on
  the same development categories. Do not run E or touch final unseen categories
  unless region-conditioned routing first clears this development gate.
