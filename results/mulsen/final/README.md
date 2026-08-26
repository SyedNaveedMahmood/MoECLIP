# Frozen final unseen evaluation

These are the seed-111 results from the architecture and protocol frozen at
`mulsen-rgbt-extension-frozen-v1`. Final epoch budgets were selected using only
development categories: A-corrected = 9 epochs and D-v1.1 = 3 epochs. Both
models were refit from scratch on the ten seen categories and evaluated exactly
once on `cotton`, `nut`, `piggy`, `solar_panel`, and `toothbrush`. No model,
prompt, normalization, threshold, score, or checkpoint choice was changed after
the unseen results were opened.

| Model | Combined AUROC/AP | Detection AUROC/AP | RGB pixel AUROC/AP |
|---|---:|---:|---:|
| A-final | 0.531840 / 0.782527 | 0.672881 / 0.852295 | 0.846033 / 0.064449 |
| D-v1.1-final | 0.385808 / 0.722930 | 0.501320 / 0.771992 | 0.920235 / 0.028958 |

D-v1.1 does not outperform A-final overall. It raises macro RGB pixel AUROC but
lowers pixel AP and both image-level metric pairs. This is a single seed and a
single category split; no variance or statistical-significance claim is made.

## Files

- `A_final_summary.json`, `D_v1_1_final_summary.json`: compact metrics,
  per-category results, subgroup distributions, training losses, checkpoint
  metadata, and source hashes.
- `A_final_evaluation.json`, `D_v1_1_final_evaluation.json`: path-sanitized
  one-shot evaluation reports retaining per-sample score arrays.
- `D_v1_1_final_routing.json`: all-layer expert-routing and thermal-attention
  diagnostics for the fixed D-v1.1 checkpoint.
- `configs/`: exact final hyperparameters with machine paths replaced by
  documented portable placeholders.
- `thermal_stats_final.json`: byte-identical statistics from 885 normal training
  images in the ten seen categories only.
- `artifact_manifest.json`: SHA-256 inventory for tracked result JSONs, local
  source reports, and untracked selected checkpoints.
- `commands.ps1`: PowerShell reconstruction of the frozen final workflow.

Checkpoint binaries remain under ignored local `ckpt/` directories because A is
56.7 MB and D-v1.1 is 130.5 MB. Their filenames, sizes, SHA-256 values, seed,
epoch, and exact configs are recorded in the summaries.
