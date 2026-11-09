# Result tables

## Aggregation time at W = 4 (median ms, speedup vs A)

| Model | N | Data (MB) | A | B | C slice | C clients | C e2e | Numba | C slice speedup | B/C |
|---|---|---|---|---|---|---|---|---|---|---|
| LeNet (62K) | 10 | 2 | 0.20 | 2.97 | 0.49 | 0.57 | 0.61 | 0.29 | 0.41x | 6.0x |
| LeNet (62K) | 50 | 12 | 0.96 | 10.83 | 0.92 | 1.05 | 1.93 | 0.72 | 1.04x | 11.8x |
| LeNet (62K) | 100 | 24 | 2.08 | 21.58 | 1.75 | 1.96 | 4.48 | 1.31 | 1.19x | 12.3x |
| LeNet (62K) | 250 | 59 | 4.74 | 47.60 | 3.42 | 3.48 | 10.86 | 2.77 | 1.39x | 13.9x |
| LeNet (62K) | 500 | 118 | 9.67 | 92.45 | 6.20 | 6.41 | 21.20 | 5.51 | 1.56x | 14.9x |
| LeNet (62K) | 1000 | 236 | 20.24 | 194.88 | 13.32 | 13.50 | 43.53 | 11.05 | 1.52x | 14.6x |
| MobileNetV2 (3.5M) | 10 | 134 | 15.56 | 187.09 | 10.46 | 24.34 | 21.69 | 7.56 | 1.49x | 17.9x |
| MobileNetV2 (3.5M) | 50 | 668 | 56.89 | 550.62 | 33.17 | 44.41 | 82.06 | 27.73 | 1.72x | 16.6x |
| MobileNetV2 (3.5M) | 100 | 1337 | 111.32 | 1003.60 | 63.51 | 72.90 | 161.08 | 51.47 | 1.75x | 15.8x |
| MobileNetV2 (3.5M) | 250 | 3342 | 270.58 | 2416.73 | 155.36 | 161.02 | 420.92 | 131.06 | 1.74x | 15.6x |
| ResNet-18 (11.2M) | 10 | 426 | 42.83 | 534.31 | 28.13 | 68.47 | 58.82 | 20.38 | 1.52x | 19.0x |
| ResNet-18 (11.2M) | 25 | 1066 | 94.34 | 1023.55 | 54.80 | 96.59 | 132.94 | 44.10 | 1.72x | 18.7x |
| ResNet-18 (11.2M) | 50 | 2131 | 185.94 | 1889.16 | 107.57 | 153.09 | 272.87 | 87.54 | 1.73x | 17.6x |
| ResNet-18 (11.2M) | 75 | 3197 | 258.24 | 2875.75 | 153.31 | 186.24 | 388.74 | 122.12 | 1.68x | 18.8x |

## Scaling with W at the largest N (speedup / efficiency / busy cores / peak MB)

| Model | N | Scheme | W | median ms | speedup | efficiency | busy cores | CPU mean % | peak MB |
|---|---|---|---|---|---|---|---|---|---|
| LeNet (62K) | 1000 | A: sequential | 1 | 20.24 | 1.00 | 1.00 | 1.28 | 12 | 376 |
| LeNet (62K) | 1000 | B: multiprocessing + pickling | 1 | 217.89 | 0.09 | 0.09 | 1.42 | 12 | 653 |
| LeNet (62K) | 1000 | B: multiprocessing + pickling | 2 | 199.32 | 0.10 | 0.05 | 1.72 | 16 | 719 |
| LeNet (62K) | 1000 | B: multiprocessing + pickling | 4 | 194.88 | 0.10 | 0.03 | 1.82 | 17 | 759 |
| LeNet (62K) | 1000 | B: multiprocessing + pickling | 8 | 190.48 | 0.11 | 0.01 | 2.18 | 52 | 814 |
| LeNet (62K) | 1000 | C: shared memory (clients) | 1 | 21.07 | 0.96 | 0.96 | 1.05 | 9 | 397 |
| LeNet (62K) | 1000 | C: shared memory (clients) | 2 | 15.66 | 1.29 | 0.65 | 2.00 | 21 | 417 |
| LeNet (62K) | 1000 | C: shared memory (clients) | 4 | 13.50 | 1.50 | 0.38 | 3.87 | 28 | 461 |
| LeNet (62K) | 1000 | C: shared memory (clients) | 8 | 14.01 | 1.44 | 0.18 | 7.01 | 84 | 542 |
| LeNet (62K) | 1000 | C: slice, incl. write to shm | 1 | 52.86 | 0.38 | 0.38 | 1.72 | 28 | 397 |
| LeNet (62K) | 1000 | C: slice, incl. write to shm | 2 | 48.00 | 0.42 | 0.21 | 1.32 | 32 | 417 |
| LeNet (62K) | 1000 | C: slice, incl. write to shm | 4 | 43.53 | 0.47 | 0.12 | 1.87 | 31 | 461 |
| LeNet (62K) | 1000 | C: slice, incl. write to shm | 8 | 43.38 | 0.47 | 0.06 | 2.99 | 33 | 542 |
| LeNet (62K) | 1000 | C: shared memory (slice) | 1 | 21.29 | 0.95 | 0.95 | 0.98 | 8 | 397 |
| LeNet (62K) | 1000 | C: shared memory (slice) | 2 | 16.16 | 1.25 | 0.63 | 1.96 | 14 | 417 |
| LeNet (62K) | 1000 | C: shared memory (slice) | 4 | 13.32 | 1.52 | 0.38 | 3.89 | 29 | 461 |
| LeNet (62K) | 1000 | C: shared memory (slice) | 8 | 12.49 | 1.62 | 0.20 | 7.50 | 82 | 542 |
| LeNet (62K) | 1000 | Numba threads [6] | 1 | 13.59 | 1.49 | 1.49 | 1.03 | 12 | 376 |
| LeNet (62K) | 1000 | Numba threads [6] | 2 | 11.76 | 1.72 | 0.86 | 1.96 | 18 | 376 |
| LeNet (62K) | 1000 | Numba threads [6] | 4 | 11.05 | 1.83 | 0.46 | 3.93 | 46 | 376 |
| LeNet (62K) | 1000 | Numba threads [6] | 8 | 11.55 | 1.75 | 0.22 | 7.73 | 82 | 376 |
| MobileNetV2 (3.5M) | 250 | A: sequential | 1 | 270.58 | 1.00 | 1.00 | 1.00 | 8 | 3966 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 1 | 2742.35 | 0.10 | 0.10 | 1.49 | 13 | 4234 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 2 | 2395.55 | 0.11 | 0.06 | 1.98 | 16 | 4395 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 4 | 2416.73 | 0.11 | 0.03 | 2.09 | 17 | 4610 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 8 | 2602.25 | 0.10 | 0.01 | 2.28 | 22 | 5040 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 1 | 265.34 | 1.02 | 1.02 | 1.00 | 7 | 4006 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 2 | 191.05 | 1.42 | 0.71 | 2.00 | 13 | 4047 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 4 | 161.02 | 1.68 | 0.42 | 3.86 | 26 | 4127 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 8 | 192.67 | 1.40 | 0.18 | 6.98 | 56 | 4290 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 1 | 531.25 | 0.51 | 0.51 | 1.71 | 12 | 4007 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 2 | 451.49 | 0.60 | 0.30 | 1.44 | 12 | 4047 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 4 | 420.92 | 0.64 | 0.16 | 2.14 | 14 | 4128 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 8 | 429.98 | 0.63 | 0.08 | 3.41 | 24 | 4290 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 1 | 268.15 | 1.01 | 1.01 | 0.98 | 7 | 4006 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 2 | 183.79 | 1.47 | 0.74 | 1.98 | 15 | 4047 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 4 | 155.36 | 1.74 | 0.43 | 4.00 | 26 | 4127 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 8 | 157.16 | 1.72 | 0.21 | 7.83 | 52 | 4290 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 1 | 163.87 | 1.65 | 1.65 | 1.00 | 7 | 3966 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 2 | 126.46 | 2.14 | 1.07 | 2.00 | 14 | 3966 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 4 | 131.06 | 2.06 | 0.52 | 3.97 | 29 | 3966 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 8 | 138.24 | 1.96 | 0.24 | 7.97 | 55 | 3966 |
| ResNet-18 (11.2M) | 75 | A: sequential | 1 | 258.24 | 1.00 | 1.00 | 1.01 | 7 | 4898 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 1 | 2916.82 | 0.09 | 0.09 | 1.48 | 12 | 5181 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 2 | 2615.43 | 0.10 | 0.05 | 2.11 | 16 | 5336 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 4 | 2875.75 | 0.09 | 0.02 | 2.21 | 18 | 5732 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 8 | 3234.76 | 0.08 | 0.01 | 2.56 | 26 | 6777 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 1 | 262.13 | 0.98 | 0.98 | 1.02 | 7 | 4967 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 2 | 196.88 | 1.31 | 0.66 | 1.95 | 14 | 5037 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 4 | 186.24 | 1.39 | 0.35 | 3.55 | 25 | 5176 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 8 | 247.17 | 1.04 | 0.13 | 6.43 | 45 | 5458 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 1 | 491.44 | 0.53 | 0.53 | 1.44 | 16 | 4967 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 2 | 423.43 | 0.61 | 0.30 | 1.45 | 10 | 5037 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 4 | 388.74 | 0.66 | 0.17 | 2.14 | 14 | 5176 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 8 | 390.13 | 0.66 | 0.08 | 3.79 | 25 | 5458 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 1 | 261.30 | 0.99 | 0.99 | 0.99 | 8 | 4967 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 2 | 191.48 | 1.35 | 0.67 | 1.98 | 14 | 5037 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 4 | 153.31 | 1.68 | 0.42 | 3.98 | 26 | 5176 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 8 | 156.37 | 1.65 | 0.21 | 7.75 | 50 | 5458 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 1 | 148.87 | 1.74 | 1.74 | 1.00 | 7 | 4898 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 2 | 118.52 | 2.18 | 1.09 | 2.01 | 13 | 4898 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 4 | 122.12 | 2.12 | 0.53 | 3.95 | 25 | 4898 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 8 | 129.97 | 1.99 | 0.25 | 8.00 | 53 | 4898 |

## Correctness

All 294 scheme/configuration pairs passed allclose(atol=1e-6) against Scheme A; largest max-abs difference = 5.59e-09.

## Worker-pool start-up (spawn, excluded from round times)

| W | start-up (ms) |
|---|---|
| 1 | 232.0 |
| 2 | 216.2 |
| 4 | 263.8 |
| 8 | 331.1 |

## Federated training (correctness track)

| Dataset | Scheme | final acc (%) | max abs acc diff vs A (pp) | max param diff vs seq | median agg ms |
|---|---|---|---|---|---|
| cifar10 | A | 53.58 | 0.00 | 0 | 0.485 |
| cifar10 | B | 53.26 | 0.32 | 8.94e-08 | 5.806 |
| cifar10 | C | 53.58 | 0.00 | 0 | 1.378 |
| femnist | A | 80.20 | 0.00 | 0 | 0.541 |
| femnist | B | 80.00 | 0.26 | 1.19e-07 | 5.961 |
| femnist | C | 80.20 | 0.00 | 0 | 1.481 |
