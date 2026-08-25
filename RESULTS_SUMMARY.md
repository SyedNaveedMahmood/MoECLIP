# MoECLIP Released-Code Reproduction Report

> This report documents a reproduction of the authors' released MoECLIP implementation and compares it with the paper. It should not be interpreted as an exact paper-spec reproduction: several paper/code differences are documented below.
>
> RGB-Thermal extension work is documented separately in [RGBT_EXTENSION.md](RGBT_EXTENSION.md).

**Setup:** OpenCLIP ViT-L/14-336 frozen backbone | image 518x518 | K=4 MoE experts (Top-2) | LoRA r=8, alpha=16 | batch 2 | 20 epochs | seed 111 | 1x RTX 4080 SUPER (16 GB)

## Main Experiments

The main reproduction follows the released-code cross-dataset protocol: Run A trains on VisA and evaluates the remaining datasets; Run B trains on MVTec-AD and evaluates VisA.

| Run | pixel AUC | pixel AP | image AUC | image AP |
|---|---:|---:|---:|---:|
| Run A: Train VisA -> MVTec | 91.73 | 45.30 | 93.40 | 96.59 |
| Run A: Train VisA -> BTAD | 96.54 | 46.15 | 91.59 | 93.80 |
| Run A: Train VisA -> RSDD | 99.68 | 34.55 | 93.94 | 91.07 |
| Run A: Train VisA -> DTD-Synthetic | 98.27 | 62.08 | 96.36 | 98.63 |
| Run A: Train VisA -> Brain MRI | 94.77 | 39.05 | 78.48 | 93.75 |
| Run A: Train VisA -> Head CT | NaN | 0.00 | 94.13 | 93.13 |
| Run A: Train VisA -> Liver CT | 97.57 | 8.61 | 63.59 | 56.86 |
| Run A: Train VisA -> Retina OCT | 95.79 | 64.95 | 86.13 | 84.69 |
| Run A: Train VisA -> ClinicDB | 89.12 | 48.51 | 0.00 | 0.00 |
| Run A: Train VisA -> ColonDB | 84.46 | 33.94 | 0.00 | 0.00 |
| Run A: Train VisA -> CVC-300 | 96.76 | 53.84 | 0.00 | 0.00 |
| Run A: Train VisA -> Endo | 90.11 | 59.75 | 0.00 | 0.00 |
| Run A: Train VisA -> Kvasir | 87.06 | 55.53 | 0.00 | 0.00 |
| Run B: Train MVTec -> VisA | 94.62 | 23.47 | 81.67 | 83.73 |

### Aggregate comparison with the paper

Using the same valid-dataset averaging convention as the paper (Head CT excluded from pixel metrics; anomaly-only colonoscopy datasets excluded from image metrics):

| Metric | Paper | Reproduction | Gap |
|---|---:|---:|---:|
| Image AUROC | 89.60 | 86.59 | -3.01 |
| Image AP | 90.60 | 88.03 | -2.57 |
| Pixel AUROC | 94.30 | 93.58 | -0.72 |
| Pixel AP | 47.50 | 44.29 | -3.21 |

Pixel-level AUROC reproduced most closely overall. MVTec-AD is also close to the reported result (91.73 vs. 92.5 pixel AUROC; 93.40 vs. 93.9 image AUROC). Larger differences remain in several medical-domain results, especially Brain MRI and Liver CT. Because this reproduction currently uses one seed, these gaps are reported directly rather than attributed to run-to-run variance.

## FOFS Ablation

A model was trained on VisA with fixed-A FOFS partitioning disabled (`--no_use_fofs`) and evaluated on the four Table-2 datasets available in the run.

| Dataset | Full model | w/o FOFS | Observation |
|---|---|---|---|
| MVTec image AUROC | 93.40 | 91.82 | -1.58 |
| MVTec pixel AUROC | 91.73 | 89.51 | -2.22 |
| DTD-Synthetic image AUROC | 96.36 | 94.25 | -2.11 |
| DTD-Synthetic pixel AUROC | 98.27 | 96.40 | -1.87 |
| Head CT image AUROC | 94.13 | 96.71 | +2.58 |
| ColonDB pixel AUROC | 84.46 | 84.33 | -0.13 |

The ablation supports a useful FOFS contribution on MVTec-AD and DTD-Synthetic, but the effect is not uniform across all evaluated datasets. Head CT improves in this run without FOFS, while ColonDB changes only marginally. A multi-seed rerun would be needed before making a stronger statistical claim.

## Fidelity Assessment

### Reproduced from the released implementation

- OpenCLIP ViT-L/14-336 backbone with 518x518 inputs.
- Four LoRA experts with Top-2 routing, LoRA rank 8 / alpha 16, PAA, and the released text/image adaptation pipeline.
- 20-epoch VisA auxiliary training followed by cross-dataset ZSAD evaluation on the remaining datasets.
- Reverse MVTec-AD -> VisA run for the VisA comparison.
- Pixel- and image-level AUROC/AP evaluation, including the dataset-specific evaluation limitations below.
- Training and evaluation completed on a single RTX 4080 SUPER 16 GB rather than the paper's 2x V100 setup.

### Paper/code differences and implementation caveats

1. **Learning rate:** the paper states an initial learning rate of `5e-4`, while the released `train.py` and `scripts.sh` use the repository default `5e-5`. The reported reproduction follows the released-code value.

2. **Single-GPU execution:** the paper reports 2x NVIDIA V100 16 GB GPUs; this reproduction uses one RTX 4080 SUPER 16 GB.

3. **Training-mode / auxiliary-loss behavior:** the released RGB training script keeps the model in `eval()` mode. In the released MoE implementation, load-balancing and collection of all expert outputs are executed only in the MoE training branch, while ETF loss depends on those expert outputs. Therefore the reported runs should be interpreted as a reproduction of the released execution path, not as proof that every paper-described auxiliary objective was active exactly as specified. This behavior is being treated as an implementation discrepancy to validate separately rather than silently corrected in the reported baseline numbers.

4. **Load-balance configuration name:** the upstream MoE code reads `router_aux_loss_coef`, whereas the configuration object stores `router_aux_loss_coef_`. The fork now corrects the attribute name. The baseline numbers above are retained as the original released-code reproduction and are not relabeled as corrected-paper runs.

5. **Ablation coverage:** the completed and committed quantitative ablation is the w/o-FOFS run. `ablation_noetf_results.txt` is currently empty, so no quantitative no-ETF reproduction claim is made here.

6. **Platform:** the reproduction was performed on Windows; Linux-specific dependencies/commands were adapted for the local environment.

## Dataset Evaluation Notes

- ClinicDB, ColonDB, CVC-300, Endo, and Kvasir contain only anomalous test images in this release. Image-level AUROC is therefore undefined; the logs display `0.00`, while pixel-level metrics are the meaningful comparison.
- Head CT contains no usable pixel-level masks in the provided release, so pixel AUROC is `NaN`; image-level metrics remain valid.

## Reproduction Artifacts

- Main checkpoints: `ckpt/visa/moe_epoch_20.pth` and `ckpt/mvtec/moe_epoch_20.pth`.
- FOFS-ablation checkpoint: `ckpt/ablation_nofofs/moe_epoch_20.pth`.
- Raw evaluation logs: `eval_visa_results.txt`, `eval_mvtec_results.txt`, and `ablation_nofofs_results.txt`.
- Per-class results are preserved in the raw logs and checkpoint test logs.
