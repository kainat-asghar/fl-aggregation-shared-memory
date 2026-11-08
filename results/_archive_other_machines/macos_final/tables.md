# Result tables

## Aggregation time at W = 8 (median ms, speedup vs A)

| Model | N | Data (MB) | A | B | C slice | C clients | C e2e | Numba | C slice speedup | B/C |
|---|---|---|---|---|---|---|---|---|---|---|
| LeNet (62K) | 10 | 2 | 0.11 | 3.06 | 0.31 | 0.42 | 0.36 | 0.10 | 0.34x | 9.8x |
| LeNet (62K) | 50 | 12 | 0.49 | 9.10 | 0.49 | 0.51 | 0.64 | 0.15 | 0.98x | 18.4x |
| LeNet (62K) | 100 | 24 | 1.03 | 17.95 | 0.64 | 0.72 | 1.03 | 0.37 | 1.60x | 27.8x |
| LeNet (62K) | 250 | 59 | 2.44 | 39.99 | 1.26 | 1.26 | 2.20 | 0.98 | 1.93x | 31.7x |
| LeNet (62K) | 500 | 118 | 5.03 | 79.75 | 2.27 | 2.15 | 4.15 | 1.97 | 2.22x | 35.1x |
| LeNet (62K) | 1000 | 236 | 10.27 | 153.80 | 4.92 | 4.69 | 7.91 | 4.00 | 2.09x | 31.3x |
| MobileNetV2 (3.5M) | 10 | 134 | 5.69 | 163.85 | 3.63 | 9.26 | 6.62 | 2.00 | 1.57x | 45.1x |
| MobileNetV2 (3.5M) | 50 | 668 | 29.18 | 825.27 | 18.24 | 47.46 | 38.14 | 10.38 | 1.60x | 45.2x |
| MobileNetV2 (3.5M) | 100 | 1337 | 74.04 | 1453.05 | 38.41 | 68.47 | 61.86 | 27.15 | 1.93x | 37.8x |
| MobileNetV2 (3.5M) | 250 | 3342 | 191.57 | 5188.29 | 115.52 | 253.14 | 363.32 | 66.92 | 1.66x | 44.9x |
| ResNet-18 (11.2M) | 10 | 426 | 23.21 | 825.18 | 11.85 | 71.74 | 27.45 | 9.04 | 1.96x | 69.6x |
| ResNet-18 (11.2M) | 25 | 1066 | 57.74 | 1366.75 | 29.34 | 130.91 | 60.55 | 15.59 | 1.97x | 46.6x |
| ResNet-18 (11.2M) | 50 | 2131 | 134.02 | 3061.88 | 71.17 | 374.28 | 208.97 | 51.10 | 1.88x | 43.0x |
| ResNet-18 (11.2M) | 75 | 3197 | 302.16 | 4510.43 | 103.69 | 724.43 | 251.53 | 89.24 | 2.91x | 43.5x |

## Scaling with W at the largest N (speedup / efficiency / busy cores / peak MB)

| Model | N | Scheme | W | median ms | speedup | efficiency | busy cores | CPU mean % | peak MB |
|---|---|---|---|---|---|---|---|---|---|
| LeNet (62K) | 1000 | A: sequential | 1 | 10.27 | 1.00 | 1.00 | 1.00 | 15 | 1712 |
| LeNet (62K) | 1000 | B: multiprocessing + pickling | 1 | 176.54 | 0.06 | 0.06 | 1.01 | 24 | 3910 |
| LeNet (62K) | 1000 | B: multiprocessing + pickling | 2 | 165.43 | 0.06 | 0.03 | 1.05 | 25 | 3475 |
| LeNet (62K) | 1000 | B: multiprocessing + pickling | 4 | 179.12 | 0.06 | 0.01 | 0.95 | 23 | 4423 |
| LeNet (62K) | 1000 | B: multiprocessing + pickling | 8 | 153.80 | 0.07 | 0.01 | 1.53 | 18 | 3915 |
| LeNet (62K) | 1000 | C: shared memory (clients) | 1 | 10.44 | 0.98 | 0.98 | 1.00 | 12 | 2732 |
| LeNet (62K) | 1000 | C: shared memory (clients) | 2 | 6.54 | 1.57 | 0.79 | 1.88 | 20 | 2247 |
| LeNet (62K) | 1000 | C: shared memory (clients) | 4 | 4.53 | 2.27 | 0.57 | 3.66 | 47 | 1498 |
| LeNet (62K) | 1000 | C: shared memory (clients) | 8 | 4.69 | 2.19 | 0.27 | 6.07 | 63 | 3916 |
| LeNet (62K) | 1000 | C: slice, incl. write to shm | 1 | 14.26 | 0.72 | 0.72 | 1.00 | 13 | 2163 |
| LeNet (62K) | 1000 | C: slice, incl. write to shm | 2 | 9.88 | 1.04 | 0.52 | 1.62 | 18 | 2602 |
| LeNet (62K) | 1000 | C: slice, incl. write to shm | 4 | 8.71 | 1.18 | 0.29 | 2.61 | 27 | 2391 |
| LeNet (62K) | 1000 | C: slice, incl. write to shm | 8 | 7.91 | 1.30 | 0.16 | 3.80 | 39 | 3916 |
| LeNet (62K) | 1000 | C: shared memory (slice) | 1 | 10.45 | 0.98 | 0.98 | 1.00 | 11 | 2732 |
| LeNet (62K) | 1000 | C: shared memory (slice) | 2 | 6.54 | 1.57 | 0.79 | 1.98 | 22 | 2612 |
| LeNet (62K) | 1000 | C: shared memory (slice) | 4 | 4.77 | 2.15 | 0.54 | 3.76 | 43 | 1696 |
| LeNet (62K) | 1000 | C: shared memory (slice) | 8 | 4.92 | 2.09 | 0.26 | 6.14 | 63 | 3912 |
| LeNet (62K) | 1000 | Numba threads [6] | 1 | 10.20 | 1.01 | 1.01 | 1.02 | 13 | 1339 |
| LeNet (62K) | 1000 | Numba threads [6] | 2 | 7.27 | 1.41 | 0.71 | 1.96 | 21 | 938 |
| LeNet (62K) | 1000 | Numba threads [6] | 4 | 6.06 | 1.69 | 0.42 | 3.84 | 41 | 587 |
| LeNet (62K) | 1000 | Numba threads [6] | 8 | 4.00 | 2.57 | 0.32 | 6.92 | 74 | 1480 |
| MobileNetV2 (3.5M) | 250 | A: sequential | 1 | 191.57 | 1.00 | 1.00 | 0.94 | 16 | 4827 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 1 | 4006.32 | 0.05 | 0.05 | 0.89 | 40 | 6088 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 2 | 3794.31 | 0.05 | 0.03 | 0.98 | 33 | 7340 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 4 | 4187.96 | 0.05 | 0.01 | 0.93 | 36 | 7947 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 8 | 5188.29 | 0.04 | 0.01 | 0.83 | 33 | 5928 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 1 | 370.92 | 0.52 | 0.52 | 0.88 | 29 | 7842 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 2 | 348.00 | 0.55 | 0.28 | 1.75 | 19 | 9077 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 4 | 264.14 | 0.72 | 0.18 | 3.30 | 71 | 7857 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 8 | 253.14 | 0.76 | 0.10 | 6.15 | 69 | 9612 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 1 | 463.33 | 0.41 | 0.41 | 0.79 | 10 | 7675 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 2 | 376.21 | 0.51 | 0.26 | 1.45 | 16 | 7697 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 4 | 300.61 | 0.64 | 0.16 | 2.58 | 29 | 7917 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 8 | 363.32 | 0.53 | 0.07 | 3.93 | 40 | 12685 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 1 | 346.78 | 0.55 | 0.55 | 0.92 | 18 | 8244 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 2 | 210.46 | 0.91 | 0.46 | 1.95 | 22 | 7343 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 4 | 139.65 | 1.37 | 0.34 | 3.38 | 84 | 5233 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 8 | 115.52 | 1.66 | 0.21 | 5.46 | 85 | 6464 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 1 | 216.77 | 0.88 | 0.88 | 0.98 | 12 | 4295 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 2 | 144.56 | 1.32 | 0.66 | 1.94 | 22 | 3984 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 4 | 90.92 | 2.11 | 0.53 | 3.96 | 45 | 3579 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 8 | 66.92 | 2.86 | 0.36 | 7.58 | 79 | 4258 |
| ResNet-18 (11.2M) | 75 | A: sequential | 1 | 302.16 | 1.00 | 1.00 | 0.89 | 12 | 3790 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 1 | 3614.92 | 0.08 | 0.08 | 1.09 | 28 | 7339 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 2 | 3881.19 | 0.08 | 0.04 | 0.86 | 36 | 7301 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 4 | 4002.33 | 0.07 | 0.02 | 1.10 | 25 | 8139 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 8 | 4510.43 | 0.07 | 0.01 | 0.86 | 33 | 6879 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 1 | 562.34 | 0.54 | 0.54 | 0.81 | 25 | 8985 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 2 | 463.58 | 0.65 | 0.33 | 0.80 | 11 | 7911 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 4 | 461.18 | 0.66 | 0.16 | 2.72 | 28 | 10436 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 8 | 724.43 | 0.42 | 0.05 | 4.01 | 57 | 9032 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 1 | 596.91 | 0.51 | 0.51 | 0.50 | 17 | 7507 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 2 | 482.99 | 0.63 | 0.31 | 1.41 | 18 | 7546 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 4 | 316.85 | 0.95 | 0.24 | 2.57 | 36 | 7584 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 8 | 251.53 | 1.20 | 0.15 | 3.96 | 41 | 12213 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 1 | 297.38 | 1.02 | 1.02 | 0.93 | 19 | 8857 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 2 | 203.27 | 1.49 | 0.74 | 1.99 | 26 | 6105 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 4 | 161.68 | 1.87 | 0.47 | 3.76 | 39 | 7667 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 8 | 103.69 | 2.91 | 0.36 | 7.03 | 79 | 5511 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 1 | 227.80 | 1.33 | 1.33 | 0.91 | 22 | 5028 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 2 | 128.96 | 2.34 | 1.17 | 1.91 | 20 | 4039 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 4 | 107.21 | 2.82 | 0.70 | 3.83 | 40 | 4588 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 8 | 89.24 | 3.39 | 0.42 | 7.16 | 91 | 4089 |

## Correctness

All 294 scheme/configuration pairs passed allclose(atol=1e-6) against Scheme A; largest max-abs difference = 5.59e-09.

## Worker-pool start-up (spawn, excluded from round times)

| W | start-up (ms) |
|---|---|
| 1 | 70.4 |
| 2 | 63.6 |
| 4 | 68.2 |
| 8 | 98.0 |

## Federated training (correctness track)

| Dataset | Scheme | final acc (%) | max abs acc diff vs A (pp) | max param diff vs seq | median agg ms |
|---|---|---|---|---|---|
| cifar10 | A | 53.41 | 0.00 | 0 | 0.220 |
| cifar10 | B | 53.39 | 0.27 | 5.96e-08 | 3.948 |
| cifar10 | C | 53.41 | 0.00 | 0 | 0.495 |
| femnist | A | 80.50 | 0.00 | 0 | 0.234 |
| femnist | B | 80.27 | 0.51 | 1.19e-07 | 4.137 |
| femnist | C | 80.50 | 0.00 | 0 | 0.520 |
