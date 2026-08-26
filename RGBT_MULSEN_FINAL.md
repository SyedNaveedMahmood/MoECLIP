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

## Development routing and modality observations

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

The freeze commit and annotated tag were pushed before final refitting or unseen
evaluation. The final epoch budgets were fixed from development (A = 9,
D-v1.1 = 3). Both models were then refit from scratch on all ten `D_s`
categories and evaluated exactly once on the five locked `D_u` categories.

| Model | Combined AUROC/AP | Detection AUROC/AP | RGB pixel AUROC/AP |
|---|---:|---:|---:|
| A-final | 0.531840 / 0.782527 | 0.672881 / 0.852295 | 0.846033 / 0.064449 |
| D-v1.1-final | 0.385808 / 0.722930 | 0.501320 / 0.771992 | 0.920235 / 0.028958 |

Observed: D-v1.1 reduces combined image AUROC/AP by 0.146032/0.059597 and
detection-only AUROC/AP by 0.171561/0.080303. It raises macro RGB pixel AUROC by
0.074202 but lowers RGB pixel AP by 0.035491. It therefore does not outperform
the corrected RGB baseline overall. This is one seed and one category split;
these differences are not estimates of variance or statistical significance.

### Per-category final metrics

| Category | A combined | D combined | A detection | D detection | A RGB pixel | D RGB pixel |
|---|---:|---:|---:|---:|---:|---:|
| cotton | 0.5769 / 0.8447 | 0.6564 / 0.8966 | 0.7000 / 0.8878 | 0.4077 / 0.7574 | 0.9211 / 0.2073 | 0.9153 / 0.0291 |
| nut | 0.6172 / 0.8644 | 0.1069 / 0.5675 | 0.7034 / 0.9012 | 0.3414 / 0.7528 | 0.4445 / 0.0011 | 0.9303 / 0.0139 |
| piggy | 0.4833 / 0.7223 | 0.3833 / 0.7605 | 0.6833 / 0.8717 | 0.5200 / 0.7826 | 0.9344 / 0.0377 | 0.8456 / 0.0174 |
| solar_panel | 0.6590 / 0.9079 | 0.3051 / 0.7167 | 0.8231 / 0.9320 | 0.9103 / 0.9769 | 0.9480 / 0.0505 | 0.9320 / 0.0586 |
| toothbrush | 0.3227 / 0.5733 | 0.4773 / 0.6733 | 0.4545 / 0.6688 | 0.3273 / 0.5903 | 0.9823 / 0.0257 | 0.9779 / 0.0259 |

Each cell is AUROC/AP. D-v1.1 improves combined ranking on `cotton` and
`toothbrush`, and detection ranking on `solar_panel`, but the large regressions
on `nut` and `solar_panel` dominate macro image performance. Its macro pixel
AUROC gain is driven mainly by `nut`; the very low pixel AP shows that this does
not translate into precise anomaly localization.

### Modality subgroup diagnostics

The following post-hoc diagnostics compare each anomaly subgroup against the 50
good images using the already-recorded scores. They were not used for tuning or
model selection.

| Subgroup | A combined AUROC/AP | D combined AUROC/AP | A detection AUROC/AP | D detection AUROC/AP |
|---|---:|---:|---:|---:|
| RGB-only (23) | 0.6878 / 0.5361 | 0.3035 / 0.2332 | 0.6009 / 0.4141 | 0.2365 / 0.2205 |
| IR-only (61) | 0.3843 / 0.4991 | 0.5220 / 0.6682 | 0.5184 / 0.6839 | 0.6479 / 0.6685 |
| RGB+IR (75) | 0.5341 / 0.6589 | 0.4035 / 0.5754 | 0.6027 / 0.7107 | 0.4808 / 0.5583 |

Observed: D-v1.1 improves IR-only separation from good images but substantially
harms RGB-only and RGB+IR separation. This supports the narrow statement that
thermal-conditioned routing can expose some IR-only signal; it does not support
a claim of better overall RGB-T ZSAD.

### Final routing diagnostics

| Block | alpha | Context/base | Top-1 changed | Effective experts | Attention entropy |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.200058 | 0.0656 | 0.0417 | 3.9998 | 0.9446 |
| 11 | 0.201592 | 2.6975 | 0.2648 | 3.9996 | 0.9469 |
| 17 | 0.200895 | 1.2813 | 0.2974 | 3.9964 | 0.8472 |
| 23 | 0.199956 | 0.0480 | 0.0308 | 3.9987 | 0.8335 |

The learned alphas remain close to their 0.2 initialization. Context is weak at
the first and last MoE layers but dominates the mean absolute base logits in the
middle layers, changing about 26-30% of patch Top-1 choices there. Mean expert
probabilities stay close to uniform, so the router is active without global
expert collapse. Thermal attention remains broad: its effective support is
about 940, 951, 565, and 547 of 1,369 thermal tokens. These facts establish that
the mechanism operated; they do not establish that its routing decisions were
beneficial.

### Final provenance

- Final thermal statistics: 885 normal `D_s` training images, 271,872,000
  pixels, mean 0.6446795692, population standard deviation 0.3207445050,
  SHA-256 `40d71cf4fdf0601052118c208498e400d18290c5a1885c514be9949ab221ee46`.
- A-final epoch-9 checkpoint SHA-256:
  `5155948632a3dd1d0f2843cedc95fef695c2ce3a16c1453b25b911e0dc944df1`.
- D-v1.1-final epoch-3 checkpoint SHA-256:
  `3c941903018a60d3a530988c122595997db8799f46ef0a6dc78ae7a8be09b192`.
- Source evaluation JSON SHA-256: A
  `421be1456f61b494009d09fc939d1090de83775e5101535828697698bccc992c`,
  D-v1.1
  `919ef401a358534139f892786a7694553aacc651dc39b2a5ce85a86a1102d0b0`.
- Final verification passed 77 repository tests plus 3 alignment-inspector tests
  (80/80 total). A bounded one-sample D-v1.1 CUDA smoke also passed with 12
  `[1,1369,768]` maps, finite losses/gradients, and 6,747.31 MiB peak allocation.

No post-`D_u` model tuning or evaluation rerun occurred.

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
Frozen final configs, one-shot evaluations, summaries, routing diagnostics, and
commands are under [`results/mulsen/final/`](results/mulsen/final/README.md).
The detailed research log is [`RGBT_MULSEN_PLAN.md`](RGBT_MULSEN_PLAN.md).
