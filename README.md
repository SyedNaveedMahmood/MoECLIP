# MoECLIP Reproduction and RGB-T Extension

This repository contains an independent reproduction of **MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection** (CVPR 2026) together with an experimental RGB-thermal extension on MulSen-AD.

The repository started from the authors' released implementation. The original RGB MoECLIP training and evaluation path is retained, while the RGB-T experiments use separate MulSen-specific training and evaluation code.

- Paper: https://arxiv.org/abs/2603.03101
- CVPR paper: https://openaccess.thecvf.com/content/CVPR2026/html/Park_MoECLIP_Patch-Specialized_Experts_for_Zero-shot_Anomaly_Detection_CVPR_2026_paper.html

## What is included

- Released-code reproduction of MoECLIP on the cross-dataset zero-shot anomaly-detection protocol.
- FOFS ablation and raw evaluation logs.
- MulSen-AD RGB-T data loading, integrity checks, training, evaluation, and routing diagnostics.
- A region-conditioned RGB MoE extension in which thermal information conditions expert routing while anomaly prediction remains anchored to the RGB CLIP pathway.
- Machine-readable development and final evaluation artifacts under `results/mulsen/`.

## Released-code reproduction

The reproduction uses OpenCLIP ViT-L/14-336 with 518x518 inputs, four LoRA experts, Top-2 routing, LoRA rank 8, and seed 111. Training and evaluation were run on a single RTX 4080 SUPER 16 GB.

Aggregate results across the valid benchmark datasets were:

| Metric | Paper | Reproduction |
|---|---:|---:|
| Image AUROC | 89.60 | 86.59 |
| Image AP | 90.60 | 88.03 |
| Pixel AUROC | 94.30 | 93.58 |
| Pixel AP | 47.50 | 44.29 |

Pixel AUROC reproduced most closely. The reported numbers should be interpreted as a reproduction of the released execution path rather than a claim of exact paper-specification fidelity, because the released code contains several implementation/configuration differences from the paper.

Two discrepancies are especially important when interpreting this reproduction. The paper's experimental setup specifies Adam with an initial learning rate of `5e-4`, whereas the released `train.py` defaults to `5e-5`. In addition, the released training script leaves the adapted model in evaluation mode during optimization; in the released MoE implementation, training-state-dependent auxiliary routing and ETF computations are therefore not exercised in the same way implied by the paper description. For this reason, the results above intentionally document the released execution path and should not be read as an exact reconstruction of the written paper specification.

Raw reproduction logs are retained in the repository, including `eval_visa_results.txt`, `eval_mvtec_results.txt`, and `ablation_nofofs_results.txt`.

## RGB-T extension on MulSen-AD

The RGB-T extension investigates whether thermal information can improve routing in a patch-specialized RGB MoE without replacing the pretrained CLIP representation with a separate thermal prediction branch.

The final D-v1.1 design uses RGB region context and thermal context to condition routing at selected MoE blocks. The final experiment was frozen before evaluating the held-out MulSen-AD categories.

Final one-shot held-out results were:

| Model | Combined AUROC / AP | Detection AUROC / AP | RGB pixel AUROC / AP |
|---|---:|---:|---:|
| RGB baseline | 0.531840 / 0.782527 | 0.672881 / 0.852295 | 0.846033 / 0.064449 |
| RGB-T D-v1.1 | 0.385808 / 0.722930 | 0.501320 / 0.771992 | 0.920235 / 0.028958 |

The RGB-T model did **not** improve the primary final image-level metrics. It increased macro RGB pixel AUROC and improved separation for some IR-only anomalies, but reduced pixel AP and overall image-level performance. This result is kept as a negative result rather than tuned after observing the held-out set.

Detailed machine-readable metrics, per-category results, routing diagnostics, portable configs, and artifact hashes are available in `results/mulsen/`.

## Installation

```bash
conda create -n moeclip python=3.10.18 -y
conda activate moeclip
pip install -r requirements.txt
```

For the upstream MoECLIP workflow, configure the dataset base path in `dataset/constants.py` and place the OpenCLIP ViT-L/14-336px weights under `model/`.

## Running the original MoECLIP path

```bash
python train.py
python test.py
```

The original convenience script is also retained:

```bash
bash scripts.sh
```

## Running the MulSen path

MulSen-AD data is not redistributed. The extension code uses:

```bash
python train_mulsen.py --help
python evaluate_mulsen.py --help
```

Portable experiment configs and recorded outputs are under `results/mulsen/`.

## Repository structure

- `model/` — MoECLIP model components and RGB-T routing extensions
- `dataset/` — dataset loaders and metadata
- `train.py`, `test.py` — original/reproduction training and evaluation path
- `train_mulsen.py`, `evaluate_mulsen.py` — MulSen-AD RGB-T experiments
- `results/mulsen/` — compact development/final results and provenance
- `assets/` — figures from the original MoECLIP repository

## Acknowledgements

This work builds on the official MoECLIP implementation by Park et al. The original authors retain credit for the MoECLIP method and released code. The reproduction and RGB-T experiments in this fork are independent follow-up work.

## Citation

```bibtex
@InProceedings{Park_2026_CVPR,
    author    = {Park, Jun Yeong and Seo, JunYoung and Kang, Minji and Park, Yu Rang},
    title     = {MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {35534-35544}
}
```
