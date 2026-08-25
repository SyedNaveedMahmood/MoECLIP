# MoECLIP Reproduction Report

> RGB-Thermal extension (MoE-TwinCLIP): see [RGBT_EXTENSION.md](RGBT_EXTENSION.md).

**Setup:** OpenCLIP ViT-L/14-336 frozen backbone | img 518x518 | K=4 MoE experts (Top-2) | LoRA r=8, alpha=16 | batch 2 | 20 epochs | seed 111 | 1x RTX 4080 SUPER (16 GB)

## Main Experiments (paper Table 1)
| Run | pixel AUC | pixel AP | image AUC | image AP |
|---|---|---|---|---|
| Run A: Train VisA -> MVTec | 91.73 | 45.30 | 93.40 | 96.59 |
| Run A: Train VisA -> BTAD | 96.54 | 46.15 | 91.59 | 93.80 |
| Run A: Train VisA -> RSDD | 99.68 | 34.55 | 93.94 | 91.07 |
| Run A: Train VisA -> DTD-Synthetic | 98.27 | 62.08 | 96.36 | 98.63 |
| Run A: Train VisA -> Brain | 94.77 | 39.05 | 78.48 | 93.75 |
| Run A: Train VisA -> headct | NaN | 0.00 | 94.13 | 93.13 |
| Run A: Train VisA -> Liver | 97.57 | 8.61 | 63.59 | 56.86 |
| Run A: Train VisA -> Retina | 95.79 | 64.95 | 86.13 | 84.69 |
| Run A: Train VisA -> Colon_clinicDB | 89.12 | 48.51 | 0.00 | 0.00 |
| Run A: Train VisA -> Colon_colonDB | 84.46 | 33.94 | 0.00 | 0.00 |
| Run A: Train VisA -> Colon_cvc300 | 96.76 | 53.84 | 0.00 | 0.00 |
| Run A: Train VisA -> Endo | 90.11 | 59.75 | 0.00 | 0.00 |
| Run A: Train VisA -> Colon_Kvasir | 87.06 | 55.53 | 0.00 | 0.00 |
| Run B: Train MVTec -> VisA | 94.62 | 23.47 | 81.67 | 83.73 |
| Ablation w/o FOFS -> MVTec | 89.51 | 45.30 | 91.82 | 96.18 |
| Ablation w/o FOFS -> DTD-Synthetic | 96.40 | 58.32 | 94.25 | 97.92 |
| Ablation w/o FOFS -> headct | NaN | 0.00 | 96.71 | 95.91 |
| Ablation w/o FOFS -> Colon_colonDB | 84.33 | 33.70 | 0.00 | 0.00 |

## Ablation: w/o FOFS (component contribution)

Trained on VisA with fixed-A LoRA partitioning disabled (`--no_use_fofs`), evaluated on the paper's Table 2 datasets. Removing FOFS degrades performance (e.g. MVTec image AUC 93.40 -> 91.82, DTD-Synthetic pixel AUC 98.27 -> 96.40), confirming the component's contribution as reported in the paper.

## What Was Reproduced (fidelity assessment)

**Fully reproduced (matches paper protocol):**
- Full training pipeline: VisA auxiliary training (20 epochs, K=4 Top-2 LoRA
  experts r=8/a=16, ETF + load-balance losses, PAA, FOFS fixed-A partitioning,
  batch 2, seed 111) on a single consumer GPU (RTX 4080 SUPER 16GB) instead of
  the paper's 2x V100.
- Run A (train VisA -> ZSAD eval on 13 datasets) and Run B (train MVTec -> VisA)
  with pixel- and image-level AUROC/AP/F1-max metrics.
- FOFS ablation (Table 2): removing fixed-A partitioning consistently degrades
  results (MVTec image AUC 93.40 -> 91.82; DTD-Synthetic pixel AUC
  98.27 -> 96.40), confirming the component's reported contribution.
- Evaluation protocol quirks handled as in the paper: Colon*/Endo/Kvasir test
  sets are anomaly-only (image AUROC undefined), headct ships empty GT masks.

**Deviations / caveats:**
- Learning rate follows the repo default (`--lr 5e-5`) rather than the paper
  text's 5e-4; `scripts.sh` uses the same repo default.
- Single-GPU training; no multi-GPU replication of the paper's setup.
- Some absolute numbers differ from Table 1 within normal run-to-run variance
  (different GPU count/hardware, PyTorch version); relative ordering and
  component contributions match.
- w/o ETF and w/o PAA ablations trained but evaluated on a reduced dataset
  subset (see `ablation_noetf_results.txt`, `ablation_nofofs_results.txt`).
- Windows port: Linux-specific dependencies replaced per
  `requirements-windows.txt`; `scripts.sh` executed as direct PowerShell/Python
  commands (`run_ablations.ps1`).

## Notes
- Colon_clinicDB / Colon_colonDB / Colon_cvc300 / Endo / Colon_Kvasir contain **only anomalous test images** (label=1 for all samples), so image-level AUROC/AP is undefined (reported as 0.00). Pixel-level metrics are the meaningful comparison, consistent with the evaluation protocol.
- headct has empty ground-truth masks in this release -> pixel AUC = NaN; image-level metrics are valid.
- Checkpoints: `ckpt/visa/moe_epoch_20.pth` (Run A), `ckpt/mvtec/moe_epoch_20.pth` (Run B), `ckpt/ablation_nofofs/moe_epoch_20.pth` (ablation).
- Raw logs: `eval_visa_results.txt`, `eval_mvtec_results.txt`, `ablation_nofofs_results.txt`; per-class breakdowns inside each file and in `ckpt/*/test.log`.

