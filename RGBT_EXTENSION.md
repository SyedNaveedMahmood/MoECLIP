# Superseded exploratory RGB-T extension

The earlier MoE-TwinCLIP prototype has been removed from the working model. It
sent thermal tokens through LoRA experts, created a second thermal readout, and
fused final RGB/thermal scores. Those choices conflict with the current v1
research boundary and with the empirical MulSen-AD registration audit.

The active, scientifically reviewed design and observed smoke-test evidence are
maintained in [`RGBT_MULSEN_PLAN.md`](RGBT_MULSEN_PLAN.md). Git history retains
the exploratory implementation for provenance; it must not be cited as a result
of the segment-guided MulSen-AD project.
