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
| MobileNetV2 (3.5M) | 10 | 134 | 6.33 | 164.01 | 3.54 | 11.52 | 6.57 | 1.96 | 1.79x | 46.3x |
| MobileNetV2 (3.5M) | 50 | 668 | 29.36 | 876.87 | 17.78 | 44.92 | 42.51 | 11.38 | 1.65x | 49.3x |
| MobileNetV2 (3.5M) | 100 | 1337 | 58.42 | 1715.64 | 30.13 | 75.72 | 75.64 | 52.90 | 1.94x | 56.9x |
| MobileNetV2 (3.5M) | 250 | 3342 | 255.31 | 5135.05 | 107.11 | 316.77 | 393.28 | 86.72 | 2.38x | 47.9x |
| ResNet-18 (11.2M) | 10 | 426 | 22.86 | 587.43 | 10.62 | 69.33 | 20.18 | 6.73 | 2.15x | 55.3x |
| ResNet-18 (11.2M) | 25 | 1066 | 52.13 | 1245.45 | 23.67 | 139.92 | 110.46 | 19.03 | 2.20x | 52.6x |
| ResNet-18 (11.2M) | 50 | 2131 | 157.69 | 4692.12 | 85.85 | 327.97 | 238.61 | 54.97 | 1.84x | 54.7x |
| ResNet-18 (11.2M) | 75 | 3197 | 351.64 | 6270.64 | 124.70 | 1192.30 | 225.89 | 78.57 | 2.82x | 50.3x |

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
| MobileNetV2 (3.5M) | 250 | A: sequential | 1 | 255.31 | 1.00 | 1.00 | 1.00 | 15 | 4362 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 1 | 3863.65 | 0.07 | 0.07 | 0.92 | 36 | 5664 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 2 | 4377.45 | 0.06 | 0.03 | 1.02 | 35 | 7066 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 4 | 5556.86 | 0.05 | 0.01 | 0.77 | 32 | 7065 |
| MobileNetV2 (3.5M) | 250 | B: multiprocessing + pickling | 8 | 5135.05 | 0.05 | 0.01 | 0.92 | 31 | 6898 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 1 | 299.44 | 0.85 | 0.85 | 0.91 | 20 | 7446 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 2 | 605.41 | 0.42 | 0.21 | 1.73 | 25 | 10635 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 4 | 566.38 | 0.45 | 0.11 | 3.09 | 42 | 11526 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (clients) | 8 | 316.77 | 0.81 | 0.10 | 4.87 | 86 | 8947 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 1 | 572.04 | 0.45 | 0.45 | 0.82 | 60 | 7221 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 2 | 263.30 | 0.97 | 0.48 | 1.49 | 40 | 7261 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 4 | 230.41 | 1.11 | 0.28 | 2.57 | 38 | 7428 |
| MobileNetV2 (3.5M) | 250 | C: slice, incl. write to shm | 8 | 393.28 | 0.65 | 0.08 | 3.83 | 58 | 11531 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 1 | 386.32 | 0.66 | 0.66 | 0.92 | 28 | 7406 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 2 | 218.12 | 1.17 | 0.58 | 2.00 | 41 | 8921 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 4 | 122.63 | 2.08 | 0.52 | 3.64 | 53 | 9242 |
| MobileNetV2 (3.5M) | 250 | C: shared memory (slice) | 8 | 107.11 | 2.38 | 0.30 | 6.03 | 95 | 5799 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 1 | 227.07 | 1.12 | 1.12 | 0.94 | 22 | 4465 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 2 | 240.22 | 1.06 | 0.53 | 1.87 | 36 | 4522 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 4 | 92.94 | 2.75 | 0.69 | 2.95 | 62 | 4663 |
| MobileNetV2 (3.5M) | 250 | Numba threads [6] | 8 | 86.72 | 2.94 | 0.37 | 6.69 | 98 | 3819 |
| ResNet-18 (11.2M) | 75 | A: sequential | 1 | 351.64 | 1.00 | 1.00 | 0.99 | 17 | 4282 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 1 | 3893.94 | 0.09 | 0.09 | 0.92 | 28 | 5603 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 2 | 4020.43 | 0.09 | 0.04 | 0.89 | 24 | 6831 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 4 | 4888.56 | 0.07 | 0.02 | 1.06 | 26 | 7124 |
| ResNet-18 (11.2M) | 75 | B: multiprocessing + pickling | 8 | 6270.64 | 0.06 | 0.01 | 0.97 | 30 | 6508 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 1 | 862.20 | 0.41 | 0.41 | 1.00 | 12 | 5638 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 2 | 698.52 | 0.50 | 0.25 | 0.49 | 7 | 7302 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 4 | 638.47 | 0.55 | 0.14 | 2.64 | 28 | 10300 |
| ResNet-18 (11.2M) | 75 | C: shared memory (clients) | 8 | 1192.30 | 0.29 | 0.04 | 2.49 | 27 | 12676 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 1 | 545.66 | 0.64 | 0.64 | 0.66 | 20 | 6156 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 2 | 595.62 | 0.59 | 0.29 | 1.26 | 24 | 7291 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 4 | 310.64 | 1.13 | 0.28 | 2.53 | 27 | 7412 |
| ResNet-18 (11.2M) | 75 | C: slice, incl. write to shm | 8 | 225.89 | 1.56 | 0.20 | 3.99 | 50 | 10202 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 1 | 319.30 | 1.10 | 1.10 | 1.00 | 13 | 5510 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 2 | 221.48 | 1.59 | 0.79 | 1.38 | 18 | 5492 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 4 | 137.34 | 2.56 | 0.64 | 3.73 | 39 | 7519 |
| ResNet-18 (11.2M) | 75 | C: shared memory (slice) | 8 | 124.70 | 2.82 | 0.35 | 7.05 | 74 | 9154 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 1 | 401.83 | 0.88 | 0.88 | 1.00 | 12 | 3938 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 2 | 212.20 | 1.66 | 0.83 | 2.00 | 22 | 3872 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 4 | 128.52 | 2.74 | 0.68 | 3.72 | 39 | 4757 |
| ResNet-18 (11.2M) | 75 | Numba threads [6] | 8 | 78.57 | 4.48 | 0.56 | 7.14 | 79 | 4909 |

## Correctness

All 294 scheme/configuration pairs passed allclose(atol=1e-6) against Scheme A; largest max-abs difference = 5.59e-09.

## Worker-pool start-up (spawn, excluded from round times)

| W | start-up (ms) |
|---|---|
| 1 | 70.3 |
| 2 | 62.1 |
| 4 | 68.9 |
| 8 | 98.6 |
