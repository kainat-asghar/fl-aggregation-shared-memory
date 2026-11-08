# Ablation: fixed 1 MB inner tile (pre-fix behaviour)

Full sweep recorded with the historical hardcoded `BLOCK = 262_144` (1 MB of
float32) used by every worker regardless of worker count. Kept as the "before"
arm of the tile-size ablation in the paper.

Scheme A is unaffected by the fix (a single worker is budgeted the same 1 MB
tile either way), so A times are directly comparable between this run and the
main results; the difference is confined to the multi-worker schemes.

Same machine, same seed, same protocol as the main run. 294/294 configurations
passed the allclose check; max abs diff 5.59e-9.
