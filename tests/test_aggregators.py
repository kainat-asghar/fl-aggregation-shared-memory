"""
Unit tests for aggregation kernels, partitioning edge cases, and scheme equivalence.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import aggregators as ag


def naive_ref(U, w):
    """Reference implementation of weighted federated averaging."""
    return (w[:, None] * U).sum(axis=0)


# (a) Kernels vs naive reference
@pytest.mark.parametrize("kernel", ["blocked", "axpy", "gemv"])
@pytest.mark.parametrize("N,P", [(1, 7), (10, 1000), (33, 4097)])
def test_kernels_match_naive(kernel, N, P):
    rng = np.random.default_rng(42)
    U = rng.standard_normal((N, P)).astype(np.float32)
    w = rng.random(N).astype(np.float32)
    w /= w.sum()

    out = np.zeros(P, dtype=np.float32)
    tmp = np.zeros(P, dtype=np.float32)
    ag.accumulate(U, w, out, tmp, kernel=kernel)

    np.testing.assert_allclose(out, naive_ref(U, w), rtol=1e-4, atol=1e-5)


# (b) Boundary and partitioning edge cases (W > N, P % ALIGN != 0, W * ALIGN > P)
@pytest.mark.parametrize("N,P,W", [
    (4, 100, 8),     # W > N
    (10, 1001, 3),   # P % ALIGN != 0
    (5, 20, 8),      # W * ALIGN > P
    (1, 15, 2),      # small edge case
    (16, 256, 4),    # perfectly divisible
])
def test_bounds_partition(N, P, W):
    # Test slice_bounds covering P
    slice_covered = 0
    prev_end = 0
    for k in range(W):
        a, b = ag.slice_bounds(P, W, k)
        assert 0 <= a <= b <= P
        assert a == prev_end or a == P
        slice_covered += (b - a)
        prev_end = b
    assert slice_covered == P

    # Test client_bounds covering N
    client_covered = 0
    prev_end = 0
    for k in range(W):
        lo, hi = ag.client_bounds(N, W, k)
        assert 0 <= lo <= hi <= N
        assert lo == prev_end
        client_covered += (hi - lo)
        prev_end = hi
    assert client_covered == N


# (c) Slice vs client partition equivalence
@pytest.mark.parametrize("N,P,W", [
    (10, 500, 2),
    (20, 1024, 4),
    (8, 256, 2),
])
def test_slice_vs_client_equivalence(N, P, W):
    rng = np.random.default_rng(123)
    U = rng.standard_normal((N, P)).astype(np.float32)
    w = rng.random(N).astype(np.float32)
    w /= w.sum()

    # Sequential reference
    seq = ag.SequentialAggregator()
    ref = seq.aggregate(U, w).copy()

    # Worker pool for multi-process schemes
    pool = ag.WorkerPool(W=W)
    try:
        agg_slice = ag.SharedMemoryAggregator(pool, partition="slice")
        out_slice = agg_slice.aggregate(U, w).copy()
        agg_slice.release()

        agg_clients = ag.SharedMemoryAggregator(pool, partition="clients")
        out_clients = agg_clients.aggregate(U, w).copy()
        agg_clients.release()

        # Both slice and client partition must match sequential reference
        np.testing.assert_allclose(out_slice, ref, atol=1e-5)
        np.testing.assert_allclose(out_clients, ref, atol=1e-5)
        # And therefore match each other
        np.testing.assert_allclose(out_slice, out_clients, atol=1e-5)
    finally:
        pool.close()
