# MulSen-AD RGB-T results

This directory contains compact, Git-tracked provenance for the category-held-out
MulSen-AD experiments. Dataset files and checkpoint binaries are intentionally
excluded.

- [`development/`](development/README.md) records the seed-111 A-corrected versus
  D-v1.1 development comparison that fixed the architecture and final epoch
  budgets.
- `final/` will be created only after the frozen final refits and one-time unseen
  evaluation are complete.

The protocol and interpretation are documented in
[`RGBT_MULSEN_PLAN.md`](../../RGBT_MULSEN_PLAN.md) and
[`RGBT_MULSEN_FINAL.md`](../../RGBT_MULSEN_FINAL.md).
