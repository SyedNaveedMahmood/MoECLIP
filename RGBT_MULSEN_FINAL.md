# Region-Guided RGB-T MoECLIP

## Research question

Can thermal and local structural evidence improve MoECLIP expert selection while
preserving RGB CLIP patch features and the original normal/anomalous text-scoring
pathway?

## Dataset and audit

MulSen-AD supplies paired RGB and infrared files, modality-specific labels and
masks, and 15 object categories. The distributed IR payloads decode as uint8 RGB
containers and are converted symmetrically to one channel; acquisition bit depth
was not assumed. A 2,035-pair audit found no missing pairs but showed that the
sequential RGB/IR captures are not reliably pixel-registered. D-v1.1 therefore
uses full-grid region-to-thermal attention rather than same-index patch fusion.

The pre-protocol sensor audit sampled pixels from all categories, including
categories later assigned to the final unseen set. It used no GT masks and no
model scores. No final-unseen sample, label, mask, score, or statistic enters
model fitting, checkpoint selection, development evaluation, or normalization.

## Frozen architecture

D-v1.1 leaves the MoECLIP LoRA experts exclusively in RGB CLIP space. RGB SLIC
regions pool RGB patch features, query the full thermal token grid, and combine
local multimodal region context with a count-weighted global context. That
context contributes a bounded residual to each patch router, with an independent
`sigmoid(a_l)` scale per MoE layer initialized near 0.2. CLS router context is
zero. Thermal reliability gating, thermal auxiliary classification,
segment-aware PAA, cross-modal alignment, and CLS conditioning are disabled.

RGB segmentation supervision is applied to good samples and RGB-visible
anomalies only; IR-only anomalies remain positive for image classification and
are excluded from RGB pixel metrics. The image score remains the predeclared
equal mixture of detection and category-normalized RGB patch maximum.

## Category-held-out protocol

- Development train: `button_cell`, `capsule`, `cube`, `flat_pad`, `light`,
  `screen`, `spring_pad`, `zipper`.
- Development validation: `plastic_cylinder`, `screw`.
- Final seen `D_s`: the ten categories above.
- Locked final unseen `D_u`: `cotton`, `nut`, `piggy`, `solar_panel`,
  `toothbrush`.
- Seed: 111.

## Development progression

Historical A/B/D-v1 results are preserved in `RGBT_MULSEN_PLAN.md`; they
motivated D-v1.1 but are not overwritten. The controlled corrected comparison is:

| Model | Epoch | Selection | Combined AUROC/AP | Detection AUROC/AP | RGB pixel AUROC/AP |
|---|---:|---:|---:|---:|---:|
| A-corrected | 9 | 0.785295 | 0.580645 / 0.819238 | 0.408387 / 0.730865 | 0.989946 / 0.212985 |
| D-v1.1 | 3 | 0.875575 | 0.768387 / 0.912996 | 0.573806 / 0.806293 | 0.982763 / 0.152964 |

D-v1.1 improved development image metrics while reducing RGB pixel metrics.
This is one seed; no statistical significance or variance claim is made. The
architecture was frozen before evaluating the five final unseen MulSen-AD
categories. Development-selected final budgets are A = 9 epochs and D-v1.1 = 3
epochs.

## Routing and modality observations

The learned layer scales at blocks 5/11/17/23 are 0.199965, 0.200566, 0.200504,
and 0.200118. Effective expert counts remain 3.994–4.000. Patch context/base
absolute-logit ratios are 0.078, 0.420, 0.597, and 0.067; context changes Top-1
selection for 6.1%, 28.7%, 19.5%, and 2.1% of patches. CLS context influence is
zero. Mean normalized thermal-attention entropy is 0.960, 0.941, 0.903, and
0.950, so attention is broad even when its qualitative aggregate is
object-centered.

IR-only combined scores remain below good-score means for both A-corrected and
D-v1.1 on development. D-v1.1 narrows that gap but does not demonstrate reliable
IR-only detection. The routing evidence shows an active, non-collapsed mechanism;
it does not by itself prove that thermal evidence caused the image-metric gain.

## Final results

Not yet run. This section will be populated only after the frozen source tag,
fixed final thermal statistics, fixed 9/3-epoch refits, and one evaluation of
each model on locked `D_u`.

## Limitations

- RGB/IR captures are unregistered and cross-attention can learn spurious
  correspondence.
- RGB patch outputs cannot support spatial evaluation against unregistered IR
  masks.
- Development evidence is one seed and one category split.
- The retained RGB/text image score has no direct thermal anomaly score.
- The early all-category sensor audit prevents the stronger claim that future
  `D_u` pixels were never viewed; the defensible claim is that no `D_u`
  performance feedback or fitting occurred before freeze.

## Future work

Thermal reliability gating, thermal auxiliary supervision, segment-aware PAA,
learned registration, improved multimodal image scoring, multiple seeds, and
multiple category folds remain future work and are outside the frozen experiment.

## Reproduction

Compact configs, hashes, metrics, routing diagnostics, figures, and commands are
under [`results/mulsen/development/`](results/mulsen/development/README.md).
The detailed research log is [`RGBT_MULSEN_PLAN.md`](RGBT_MULSEN_PLAN.md).
