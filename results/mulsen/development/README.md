# Frozen development gate

These are the seed-111 results used to freeze D-v1.1 before final unseen
evaluation. Both runs used the same eight training categories, two development
categories, optimizer, scheduler, 20-epoch ceiling, corrected RGB segmentation
supervision, and predeclared selector. This is one seed and is not a variance or
statistical-significance estimate.

| Model | Epoch | Selection | Combined AUROC/AP | Detection AUROC/AP | RGB pixel AUROC/AP |
|---|---:|---:|---:|---:|---:|
| A-corrected | 9 | 0.785295 | 0.580645 / 0.819238 | 0.408387 / 0.730865 | 0.989946 / 0.212985 |
| D-v1.1 | 3 | 0.875575 | 0.768387 / 0.912996 | 0.573806 / 0.806293 | 0.982763 / 0.152964 |

D-v1.1 improved development image ranking and reduced RGB pixel AUROC/AP. The
architecture is frozen regardless of that mixed outcome. Final refit budgets are
therefore A-corrected = 9 epochs and D-v1.1 = 3 epochs.

## Files

- `A_corrected_summary.json`, `D_v1_1_summary.json`: compact metrics, subgroup
  distributions, training losses, source hashes, and selected-checkpoint hashes.
- `D_v1_1_routing.json`: all-layer routing and thermal-attention diagnostics.
- `configs/`: exact hyperparameters with only machine-specific paths replaced by
  documented portable placeholders; raw config hashes remain in each summary.
- `thermal_stats_development.json`: byte-identical development thermal statistics.
- `qualitative_manifest.json`, `figures/`: one lexicographically selected sample
  from each modality subgroup. GT masks are evaluation-only overlays.
- `commands.ps1`: PowerShell reconstruction of the completed development workflow.

Checkpoint binaries and full evaluation prediction arrays remain under ignored
local `ckpt/` directories. Their filenames, sizes, and SHA-256 values are recorded
in the summaries.
