"""
Correctness track: real federated training (FedAvg) with every aggregation scheme.

For each dataset, the same federation (same data split, same initial model, same
client seeds, same straggler arrival order) is trained once per scheme, so the
only difference between runs is the aggregator. We record the per-round test
accuracy of each run, and in every round we also compare the scheme's global
model with a sequential reference on the same updates (max abs difference).

    python fl_train.py --quick                     # ~2 minute smoke test
    python fl_train.py                             # CIFAR-10 + FEMNIST, 20 clients, 20 rounds
    python fl_train.py --datasets cifar10 --rounds 30 --clients 10

Data: CIFAR-10 is downloaded with torchvision and split non-IID with a
Dirichlet(alpha=0.5) label distribution. FEMNIST (LEAF) is loaded from the
Hugging Face mirror "flwrlabs/femnist" (needs `pip install datasets`); each
client is a group of real writers, so the split is naturally non-IID.
"""
import aggregators as ag   # first: pins BLAS threads before numpy/torch load

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import psutil

# torch is imported in main() only: with "spawn" (Windows) every aggregation
# worker re-imports this file, and the workers must not load PyTorch.
torch = nn = F = None


def _import_torch():
    global torch, nn, F
    import torch as _torch
    torch, nn, F = _torch, _torch.nn, _torch.nn.functional


# ----------------------------------------------------------------------- model
def build_lenet(in_ch, n_classes, img):
    """LeNet-5 (62,006 parameters for CIFAR-10, ~66K for FEMNIST)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, 6, 5, padding=2 if img == 28 else 0), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, n_classes))


def get_vec(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy().copy()


def set_vec(model, vec):
    """Copy vec INTO the parameters (vector_to_parameters would make them views of
    vec, and local training would then overwrite the global model in place)."""
    src = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32))
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(src[i:i + n].view_as(p))
            i += n


# ------------------------------------------------------------------------ data
def dirichlet_split(y, n_clients, alpha, rng, min_size=20):
    classes = np.unique(y)
    while True:
        idx = [[] for _ in range(n_clients)]
        for k in classes:
            ids = rng.permutation(np.where(y == k)[0])
            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for c, part in enumerate(np.split(ids, cuts)):
                idx[c].extend(part.tolist())
        if min(len(i) for i in idx) >= min_size:
            return [np.array(sorted(i)) for i in idx]


def _cifar10_arrays(root):
    """CIFAR-10 as uint8 (N, 32, 32, 3) + labels. Uses the Hugging Face mirror
    (fast CDN); falls back to torchvision's download from the Toronto server."""
    cache = os.path.join(root, "cifar10.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    try:
        from datasets import load_dataset
        print("  loading CIFAR-10 (uoft-cs/cifar10) - first run downloads ~145 MB ...")
        ds = load_dataset("uoft-cs/cifar10", cache_dir=os.path.join(root, "hf"))

        def conv(split):
            s = ds[split]
            X = np.stack([np.asarray(im.convert("RGB"), np.uint8) for im in s["img"]])
            return X, np.asarray(s["label"], np.int64)
        Xtr, ytr = conv("train")
        Xte, yte = conv("test")
    except Exception as e:
        print("  Hugging Face download failed (%s); using torchvision instead" % e)
        import torchvision
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        Xtr, ytr = tr.data, np.array(tr.targets, np.int64)
        Xte, yte = te.data, np.array(te.targets, np.int64)
    np.savez(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


def load_cifar10(root, n_clients, alpha, rng, quick):
    Xtr_u8, ytr, Xte_u8, yte = _cifar10_arrays(root)
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32)
    std = np.array([0.2470, 0.2435, 0.2616], np.float32)

    def prep(a):
        return ((a.astype(np.float32) / 255 - mean) / std).transpose(0, 3, 1, 2).copy()
    Xtr, Xte = prep(Xtr_u8), prep(Xte_u8)
    if quick:
        Xtr, ytr, Xte, yte = Xtr[:5000], ytr[:5000], Xte[:2000], yte[:2000]
    parts = dirichlet_split(ytr, n_clients, alpha, rng)
    clients = [(torch.from_numpy(Xtr[i]), torch.from_numpy(ytr[i])) for i in parts]
    meta = dict(in_ch=3, n_classes=10, img=32, split="dirichlet alpha=%g" % alpha)
    return clients, (torch.from_numpy(Xte), torch.from_numpy(yte)), meta


def load_femnist(root, n_clients, writers_per_client, rng, seed):
    cache = os.path.join(root, "femnist_%dx%d_seed%d.npz" % (n_clients, writers_per_client, seed))
    if not os.path.exists(cache):
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("FEMNIST needs the Hugging Face 'datasets' package: pip install datasets")
        print("  loading FEMNIST (flwrlabs/femnist) - first run downloads ~200 MB ...")
        ds = load_dataset("flwrlabs/femnist", split="train", cache_dir=os.path.join(root, "hf"))
        writers = np.array(sorted(ds.unique("writer_id")))
        chosen = rng.choice(writers, n_clients * writers_per_client, replace=False)
        group = {wid: i // writers_per_client for i, wid in enumerate(chosen)}
        sub = ds.filter(lambda ws: [x in group for x in ws], input_columns="writer_id",
                        batched=True, batch_size=20000)
        X = np.stack([np.asarray(im.convert("L"), np.uint8) for im in sub["image"]])
        y = np.asarray(sub["character"], np.int64)
        g = np.asarray([group[x] for x in sub["writer_id"]], np.int64)
        wid = np.asarray(sub["writer_id"])
        # LEAF-style per-writer 90/10 train/test split
        is_test = np.zeros(len(y), bool)
        for u in np.unique(wid):
            ids = rng.permutation(np.where(wid == u)[0])
            is_test[ids[:max(1, len(ids) // 10)]] = True
        np.savez_compressed(cache, X=X, y=y, g=g, is_test=is_test)
    d = np.load(cache)
    # HF FEMNIST is dark ink on a white background; invert to MNIST style (bright
    # strokes on 0) and standardize, otherwise LeNet stalls at majority-class accuracy.
    # Normalization statistics are computed from the training split (~is_test: mean ≈ 0.0366, std ≈ 0.1600).
    X = 1.0 - d["X"].astype(np.float32) / 255.0
    norm_cache = os.path.join(root, "femnist_norm.json")
    if os.path.exists(norm_cache):
        import json
        with open(norm_cache, "r") as f:
            stats = json.load(f)
        mean, std = stats["mean"], stats["std"]
    else:
        import json
        train_mask = ~d["is_test"]
        mean, std = float(X[train_mask].mean()), float(X[train_mask].std())
        try:
            with open(norm_cache, "w") as f:
                json.dump({"mean": mean, "std": std}, f)
        except Exception:
            pass
    X = ((X - mean) / std)[:, None]
    y, g, is_test = d["y"], d["g"], d["is_test"]
    clients = []
    for c in range(n_clients):
        m = (g == c) & ~is_test
        clients.append((torch.from_numpy(X[m]), torch.from_numpy(y[m])))
    meta = dict(in_ch=1, n_classes=62, img=28,
                split="%d writers per client" % writers_per_client)
    return clients, (torch.from_numpy(X[is_test]), torch.from_numpy(y[is_test])), meta


# -------------------------------------------------------------------- training
def local_train(model, vec, X, y, epochs, bs, lr, seed):
    set_vec(model, vec)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(X[b]), y[b]).backward()
            opt.step()
    return get_vec(model)


def evaluate(model, vec, test):
    set_vec(model, vec)
    model.eval()
    with torch.no_grad():
        return _evaluate(model, test)


def _evaluate(model, test):
    X, y = test
    correct, loss = 0, 0.0
    for i in range(0, len(y), 2000):
        out = model(X[i:i + 2000])
        loss += F.cross_entropy(out, y[i:i + 2000], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + 2000]).sum().item()
    return correct / len(y), loss / len(y)


def arrival_order(n_samples, rnd, seed):
    """FedScale-style stragglers on a virtual clock: arrival = compute (data size x
    device speed) + network delay. Updates are aggregated in arrival order; the
    delay is recorded but never slept on and is excluded from aggregation time."""
    rng = np.random.default_rng([seed, rnd])
    speed = rng.lognormal(0.0, 0.5, size=n_samples.size)
    t = n_samples * 2e-3 * speed + rng.lognormal(np.log(0.5), 0.6, size=n_samples.size)
    return np.argsort(t, kind="stable"), t


def run_federation(scheme, agg, clients, test, meta, args, writer, dataset):
    torch.manual_seed(args.seed)                       # identical init for every scheme
    model = build_lenet(meta["in_ch"], meta["n_classes"], meta["img"])
    vec = get_vec(model)
    P, K = vec.size, len(clients)
    n_samples = np.array([len(c[1]) for c in clients])
    U = np.empty((K, P), np.float32)
    ref, tmp = np.empty(P, np.float32), np.empty(P, np.float32)
    acc0, loss0 = evaluate(model, vec, test)
    writer.writerow(dict(dataset=dataset, scheme=scheme, round=0, test_acc="%.4f" % acc0,
                         test_loss="%.4f" % loss0, agg_time_ms="", maxdiff_vs_seq="",
                         sim_round_s="", train_s=""))
    accs = []
    for r in range(1, args.rounds + 1):
        order, arrive = arrival_order(n_samples, r, args.seed)
        t0 = time.perf_counter()
        for i, c in enumerate(order):                  # row i = i-th update to arrive
            X, y = clients[c]
            U[i] = local_train(model, vec, X, y, args.local_epochs, args.batch_size,
                               args.lr, seed=args.seed * 100003 + r * 1009 + int(c))
        train_s = time.perf_counter() - t0
        w = (n_samples[order] / n_samples.sum()).astype(np.float32)

        t0 = time.perf_counter()
        g = agg.aggregate(U, w)
        agg_s = time.perf_counter() - t0
        new_vec = np.array(g, copy=True)

        if scheme == "A":
            diff = 0.0                                     # g already IS the sequential reference
        else:
            ref[...] = 0
            ag.accumulate(U, w, ref, tmp)                  # sequential reference, same updates
            diff = float(np.max(np.abs(new_vec - ref)))
            if not np.allclose(ref, new_vec, atol=1e-6):
                print("  !! %s round %d differs from the sequential reference (%.3g)" % (scheme, r, diff))
        vec = new_vec
        acc, loss = evaluate(model, vec, test)
        accs.append(acc)
        writer.writerow(dict(dataset=dataset, scheme=scheme, round=r, test_acc="%.4f" % acc,
                             test_loss="%.4f" % loss, agg_time_ms="%.4f" % (agg_s * 1e3),
                             maxdiff_vs_seq="%.3g" % diff,
                             sim_round_s="%.3f" % (arrive.max() + agg_s),
                             train_s="%.2f" % train_s))
        print("  [%s] %-8s round %2d  acc %.4f  loss %.4f  agg %.3f ms  maxdiff %.2g"
              % (dataset, scheme, r, acc, loss, agg_s * 1e3, diff))
    return accs, P


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["cifar10", "femnist"],
                   choices=["cifar10", "femnist"])
    p.add_argument("--schemes", nargs="+", default=["A", "B", "C"],
                   choices=["A", "B", "C", "C_clients", "numba"])
    p.add_argument("--clients", type=int, default=20)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (CIFAR-10)")
    p.add_argument("--writers-per-client", type=int, default=10, help="FEMNIST")
    p.add_argument("--workers", type=int, default=4, help="W for schemes B and C")
    p.add_argument("--torch-threads", type=int, default=psutil.cpu_count(logical=False) or 4)
    p.add_argument("--start-method", choices=["spawn", "fork", "forkserver"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.clients, a.rounds, a.writers_per_client = min(a.clients, 5), min(a.rounds, 3), 3
    return a


def main():
    args = parse_args()
    _import_torch()
    torch.set_num_threads(args.torch_threads)
    os.makedirs(args.data, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    osname = "windows" if os.name == "nt" else sys.platform.replace("darwin", "macos")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.out, "fl_%s_%s.csv" % (osname, stamp))
    f = open(path, "w", newline="")
    wr = csv.DictWriter(f, ["dataset", "scheme", "round", "test_acc", "test_loss",
                            "agg_time_ms", "maxdiff_vs_seq", "sim_round_s", "train_s"])
    wr.writeheader()

    pool = ag.WorkerPool(args.workers, args.start_method)
    print("worker pool W=%d (%s) started in %.3fs" % (args.workers, pool.start_method, pool.startup_s))
    aggs = {"A": ag.SequentialAggregator(),
            "B": ag.PickleAggregator(pool),
            "C": ag.SharedMemoryAggregator(pool, "slice"),
            "C_clients": ag.SharedMemoryAggregator(pool, "clients")}
    if "numba" in args.schemes:
        aggs["numba"] = ag.NumbaAggregator(args.workers)
        aggs["numba"].aggregate(np.ones((2, 64), np.float32), np.ones(2, np.float32))  # JIT
    summary = dict(os=osname, start_method=pool.start_method, workers=args.workers,
                   args=vars(args), datasets={})
    try:
        for ds in args.datasets:
            rng = np.random.default_rng(args.seed)
            print("\n=== %s ===" % ds)
            if ds == "cifar10":
                clients, test, meta = load_cifar10(args.data, args.clients, args.alpha, rng, args.quick)
            else:
                clients, test, meta = load_femnist(args.data, args.clients,
                                                   args.writers_per_client, rng, args.seed)
            sizes = [len(c[1]) for c in clients]
            print("  %d clients, %d-%d samples each (%s), %d test samples"
                  % (len(clients), min(sizes), max(sizes), meta["split"], len(test[1])))
            res = {}
            for s in args.schemes:
                accs, P = run_federation(s, aggs[s], clients, test, meta, args, wr, ds)
                res[s] = accs
                f.flush()
            base = np.array(res.get("A", next(iter(res.values()))))
            summary["datasets"][ds] = dict(
                params=int(P), clients=len(clients), client_sizes=sizes, **meta,
                final_acc={s: v[-1] for s, v in res.items()},
                max_abs_acc_diff_vs_A={s: float(np.max(np.abs(np.array(v) - base)))
                                       for s, v in res.items()})
            print("  final accuracy:", {s: round(v[-1], 4) for s, v in res.items()})
            print("  max |acc - acc_A| over rounds:",
                  summary["datasets"][ds]["max_abs_acc_diff_vs_A"])
    finally:
        f.close()
        for a in aggs.values():
            a.close()
        pool.close()
    with open(path.replace(".csv", "_summary.json"), "w") as jf:
        json.dump(summary, jf, indent=2)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
