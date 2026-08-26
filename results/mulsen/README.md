# MulSen-AD RGB-T experiment artifacts

This directory contains compact, Git-tracked artifacts for the category-held-out MulSen-AD experiments. Dataset files and checkpoint binaries are not redistributed.

- `development/` contains the development comparison used to select the fixed epoch budgets and freeze the RGB-T design.
- `final/` contains the fixed refits and one-shot held-out evaluation, including metrics, score arrays, portable configs, routing diagnostics, thermal statistics, and artifact hashes.

The top-level repository README summarizes the experimental design and main results. Machine-readable JSON/config artifacts in these directories are the primary record for the reported numbers.
