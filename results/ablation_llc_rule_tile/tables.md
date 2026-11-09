# Result tables

## Aggregation time at W = 4 (median ms, speedup vs A)

| Model | N | Data (MB) | A | B | C slice | C clients | C e2e | Numba | C slice speedup | B/C |
|---|---|---|---|---|---|---|---|---|---|---|
| LeNet (62K) | 10 | 2 | 0.23 | 3.19 | 0.59 | 0.69 | 0.70 | 0.38 | 0.39x | 5.4x |
| LeNet (62K) | 50 | 12 | 0.98 | 10.51 | 0.90 | 1.15 | 1.84 | 0.66 | 1.09x | 11.7x |
| LeNet (62K) | 100 | 24 | 2.05 | 20.07 | 1.64 | 1.90 | 4.47 | 1.17 | 1.25x | 12.3x |
| LeNet (62K) | 250 | 59 | 5.40 | 48.70 | 3.53 | 4.09 | 10.94 | 2.60 | 1.53x | 13.8x |
| LeNet (62K) | 500 | 118 | 9.76 | 90.77 | 6.09 | 7.03 | 21.07 | 5.15 | 1.60x | 14.9x |
| LeNet (62K) | 1000 | 236 | 19.09 | 176.44 | 11.64 | 13.46 | 40.92 | 9.89 | 1.64x | 15.2x |
| MobileNetV2 (3.5M) | 10 | 134 | 14.30 | 168.65 | 11.31 | 23.51 | 21.19 | 6.64 | 1.26x | 14.9x |
| MobileNetV2 (3.5M) | 50 | 668 | 58.86 | 546.90 | 44.43 | 56.35 | 93.48 | 26.50 | 1.32x | 12.3x |
| MobileNetV2 (3.5M) | 100 | 1337 | 112.86 | 1009.04 | 83.85 | 98.21 | 180.19 | 51.23 | 1.35x | 12.0x |
| MobileNetV2 (3.5M) | 250 | 3342 | 281.48 | 2454.17 | 205.21 | 229.75 | 435.64 | 127.73 | 1.37x | 12.0x |
| ResNet-18 (11.2M) | 10 | 426 | 43.95 | 536.61 | 34.57 | 73.94 | 65.52 | 20.49 | 1.27x | 15.5x |
| ResNet-18 (11.2M) | 25 | 1066 | 97.64 | 1026.10 | 75.16 | 114.90 | 153.88 | 43.92 | 1.30x | 13.7x |
| ResNet-18 (11.2M) | 50 | 2131 | 187.54 | 1829.02 | 144.62 | 183.98 | 298.07 | 83.72 | 1.30x | 12.6x |
| ResNet-18 (11.2M) | 75 | 3197 | 274.40 | 2719.53 | 207.47 | 258.49 | 439.21 | 121.55 | 1.32x | 13.1x |

## Scaling with W at the largest N (speedup / efficiency / busy cores / peak MB)

| Model | N | Scheme | W | median ms | speedup | efficiency | busy cores | CPU mean % | peak MB |
|---|---|---|---|---|---|---|---|---|---|
| LeNet (62K) | 1000 | A: sequential | 1 | 19.09 | 1.00 | 1.00 | 1.28 | 16 | 376 |
| LeNet (62K) | 1000 | B: multiprocessing + pickling | 1 | 197.06 | 0.10 | 0.10 | 1.45 | 13 | 654 |
| LeNet (62K) | 1000 | B: multiprocessing + pickling | 2 | 177.97 | 0.11 | 0.05 | 1.75 | 13 | 765 |
| LeNet (62K) | 1000 | B: multiprocessing + pickling | 4 | 176.44 | 0.11 | 0.03 | 1.75 | 16 | 759 |
| LeNet (62K) | 1000 | B: multiprocessing + pickling | 8 | 171.28 | 0.11 | 0.01 | 2.11 | 19 | 814 |
| LeNet (62K) | 1000 | C: shared memory (clients) | 1 | 19.40 | 0.98 | 0.98 | 0.97 | 8 | 397 |
| LeNet (62K) | 1000 | C: shared memory (clients) | 2 | 13.55 | 1.41 | 0.70 | 1.87 | 13 | 418 |
| LeNet (62K) | 1000 | C: shared memory (clients) | 4 | 13.46 | 1.42 | 0.35 | 3.98 | 27 | 463 |
| LeNet (62K) | 1000 | C: shared memory (clients) | 8 | 16.22 | 1.18 | 0.15 | 6.98 | 51 | 545 |
| LeNet (62K) | 1000 | C: slice, incl. write to shm | 1 | 48.88 | 0.39 | 0.39 | 1.77 | 12 | 397 |
| LeNet (62K) | 1000 | C: slice, incl. write to shm | 2 | 44.51 | 0.43 | 0.21 | 1.36 | 9 | 418 |
| LeNet (62K) | 1000 | C: slice, incl. write to shm | 4 | 40.92 | 0.47 | 0.12 | 1.81 | 14 | 463 |
| LeNet (62K) | 1000 | C: slice, incl. write to shm | 8 | 41.18 | 0.46 | 0.06 | 2.78 | 18 | 545 |
| LeNet (62K) | 1000 | C: shared memory (slice) | 1 | 18.76 | 1.02 | 1.02 | 0.98 | 6 | 397 |
| LeNet (62K) | 1000 | C: shared memory (slice) | 2 | 13.76 | 1.39 | 0.69 | 1.95 | 15 | 418 |
| LeNet (62K) | 1000 | C: shared memory (slice) | 4 | 11.64 | 1.64 | 0.41 | 3.93 | 25 | 463 |
| LeNet (62K) | 1000 | C: shared memory (slice) | 8 | 11.72 | 1.63 | 0.20 | 7.01 | 54 | 545 |
| LeNet (62K) | 1000 | Numba threads [6] | 1 | 12.24 | 1.56 | 1.56 | 0.99 | 8 | 376 |
| LeNet (62K) | 1000 | Numba threads [6] | 2 | 9.73 | 1.96 | 0.98 | 2.02 | 15 | 376 |
| LeNet (62K) | 1000 | Numba threads [6] | 4 | 9.89 | 1.93 | 0.48 | 4.00 | 26 | 376 |
| LeNet (62K) | 1000 | Numba threads [6] | 8 | 10.28 | 1.86 | 0.23 | 7.68 | 54 | 376 |
| MobileNetV2 (3.5M) | 250 | A: sequential | 1 | 281.48 | 1.00 | 1.00 | 0.99 | 7 | 3967 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 1 | 2739.56 | 0.10 | 0.10 | 1.51 | 13 | 4235 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 2 | 2439.67 | 0.12 | 0.06 | 2.02 | 18 | 4396 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 4 | 2454.17 | 0.12 | 0.03 | 2.08 | 17 | 4611 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 8 | 2607.13 | 0.11 | 0.01 | 2.28 | 26 | 5041 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 1 | 295.29 | 0.95 | 0.95 | 1.01 | 7 | 4007 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 2 | 240.22 | 1.17 | 0.59 | 1.97 | 14 | 4048 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 4 | 229.75 | 1.23 | 0.31 | 3.91 | 27 | 4129 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 8 | 243.30 | 1.16 | 0.14 | 7.55 | 50 | 4290 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 1 | 509.22 | 0.55 | 0.55 | 1.75 | 12 | 4007 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 2 | 456.09 | 0.62 | 0.31 | 1.47 | 10 | 4048 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 4 | 435.64 | 0.65 | 0.16 | 2.40 | 18 | 4129 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 8 | 442.25 | 0.64 | 0.08 | 4.25 | 28 | 4291 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 1 | 279.13 | 1.01 | 1.01 | 0.99 | 7 | 4007 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 2 | 226.52 | 1.24 | 0.62 | 1.99 | 14 | 4048 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 4 | 205.21 | 1.37 | 0.34 | 3.89 | 32 | 4129 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 8 | 213.36 | 1.32 | 0.17 | 7.72 | 51 | 4290 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 1 | 154.21 | 1.82 | 1.82 | 0.98 | 7 | 3967 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 2 | 124.25 | 2.27 | 1.13 | 2.02 | 14 | 3967 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 4 | 127.73 | 2.20 | 0.55 | 3.96 | 26 | 3967 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 8 | 136.88 | 2.06 | 0.26 | 7.85 | 52 | 3967 |
| ResNet-18 (11.2M) | 75 | A: sequential | 1 | 274.40 | 1.00 | 1.00 | 0.99 | 7 | 4903 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 1 | 2858.45 | 0.10 | 0.10 | 1.46 | 13 | 5186 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 2 | 2626.97 | 0.10 | 0.05 | 2.10 | 17 | 5385 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 4 | 2719.53 | 0.10 | 0.03 | 2.14 | 19 | 5796 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 8 | 3249.79 | 0.08 | 0.01 | 2.48 | 26 | 6782 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 1 | 283.96 | 0.97 | 0.97 | 1.01 | 7 | 4973 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 2 | 264.87 | 1.04 | 0.52 | 1.95 | 14 | 5043 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 4 | 258.49 | 1.06 | 0.27 | 3.70 | 25 | 5182 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 8 | 302.26 | 0.91 | 0.11 | 6.87 | 45 | 5464 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 1 | 501.01 | 0.55 | 0.55 | 1.62 | 17 | 4973 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 2 | 459.21 | 0.60 | 0.30 | 1.52 | 13 | 5043 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 4 | 439.21 | 0.62 | 0.16 | 2.39 | 16 | 5183 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 8 | 446.28 | 0.61 | 0.08 | 4.21 | 28 | 5464 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 1 | 272.11 | 1.01 | 1.01 | 0.99 | 9 | 4973 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 2 | 228.64 | 1.20 | 0.60 | 1.99 | 15 | 5043 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 4 | 207.47 | 1.32 | 0.33 | 3.93 | 26 | 5182 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 8 | 215.28 | 1.27 | 0.16 | 7.77 | 50 | 5464 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 1 | 150.34 | 1.82 | 1.82 | 0.99 | 7 | 4903 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 2 | 118.69 | 2.31 | 1.16 | 2.01 | 14 | 4903 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 4 | 121.55 | 2.26 | 0.56 | 3.99 | 26 | 4903 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 8 | 129.17 | 2.12 | 0.27 | 7.95 | 52 | 4903 |

## Correctness

All 294 scheme/configuration pairs passed allclose(atol=1e-6) against Scheme A; largest max-abs difference = 5.59e-09.

## Worker-pool start-up (spawn, excluded from round times)

| W | start-up (ms) |
|---|---|
| 1 | 176.9 |
| 2 | 182.3 |
| 4 | 205.4 |
| 8 | 288.8 |

## Federated training (correctness track)

| Dataset | Scheme | final acc (%) | max abs acc diff vs A (pp) | max param diff vs seq | median agg ms |
|---|---|---|---|---|---|
| cifar10 | A | 53.58 | 0.00 | 0 | 0.485 |
| cifar10 | B | 53.26 | 0.32 | 8.94e-08 | 5.806 |
| cifar10 | C | 53.58 | 0.00 | 0 | 1.378 |
| femnist | A | 80.20 | 0.00 | 0 | 0.541 |
| femnist | B | 80.00 | 0.26 | 1.19e-07 | 5.961 |
| femnist | C | 80.20 | 0.00 | 0 | 1.481 |
