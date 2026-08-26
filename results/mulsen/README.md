# MulSen-AD RGB-T results

This directory contains compact, Git-tracked provenance for the category-held-out
MulSen-AD experiments. Dataset files and checkpoint binaries are intentionally
excluded.

- [`development/`](development/README.md) records the seed-111 A-corrected versus
  D-v1.1 development comparison that fixed the architecture and final epoch
  budgets.
- [`final/`](final/README.md) records the fixed 9/3-epoch refits and one-time
  unseen evaluation, including portable score arrays, configs, checkpoint and
  result hashes, routing diagnostics, thermal-stat provenance, and commands.

The protocol and interpretation are documented in
[`RGBT_MULSEN_PLAN.md`](../../RGBT_MULSEN_PLAN.md) and
[`RGBT_MULSEN_FINAL.md`](../../RGBT_MULSEN_FINAL.md).
