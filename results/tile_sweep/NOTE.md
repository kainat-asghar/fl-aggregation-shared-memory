# Tile-size sweep (Fig. 7 / Sec. V-D)

Scheme C (slice) and Scheme A measured at inner tile sizes of
16, 32, 64, 128, 256, 512, 1024 and 2048 KB, crossed with W = 1, 2, 4, 8,
on MobileNetV2 (N=50, 668 MB) and ResNet-18 (N=25, 1066 MB).
10 repetitions each, same machine and protocol as the main sweep.

Each `t<KB>` directory is one value of `FL_AGG_TILE_KB`, which overrides the
worker-count-aware rule in `aggregators.py`. Note that Scheme A is affected by
the tile too, so compare absolute times rather than speedups across tiles: at a
16 KB tile the sequential baseline itself is ~2x slower, which would otherwise
flatter the parallel schemes.

Measured optimum (both models agree): W=1 -> 512 KB, W=2 -> 128 KB,
W=4 -> 64 KB, W=8 -> 64 KB. These are interior optima, bracketed on both
sides, which is why 16 and 32 KB were added after the first pass.

Regenerate the figure with:
    python paper/make_tile_fig.py results/tile_sweep
