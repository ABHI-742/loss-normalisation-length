"""E6: does the full length law hold for realistic, heavy-tailed lengths?

E1 used only two lengths (2 and 8). Here document lengths run from 1 to 32 with a
log-normal shape whose spread matches Alpaca (coefficient of variation about 1).
The theory predicts the whole learned length distribution, not only one number:

    learned chance of length L  is proportional to  p(L) x w(L),
    w(L) = E[ 1 / (L + total length of the other documents in the averaging group) ]

and the cost in held-out loss per document, KL(p || q) = log E[w] - E[log w].

Content tokens are uniform and independent of length, so the model can only learn
the end-of-text chance at each position (the hazard h_k). The learned length
distribution follows from the hazards: q(L) = h_L x product over k < L of (1 - h_k).
Hazards are read by feeding random content (no end-of-text) and averaged over the
last 20% of training.

Methods (32 documents per update): micro1, micro4, global, fixed.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import TinyLM, lr_at

N_CONTENT = 8
BOS, EOS, PAD = N_CONTENT, N_CONTENT + 1, N_CONTENT + 2
VOCAB = N_CONTENT + 3
MAX_L = 32
LS = np.arange(1, MAX_L + 1)
_raw = np.exp(-(np.log(LS) - 1.5) ** 2 / (2 * 1.2 ** 2)) / LS
P = _raw / _raw.sum()
MU = float((P * LS).sum())
GLOBAL_B = 32
METHODS = ("micro1", "micro4", "global", "fixed")
GROUP = {"micro1": 1, "micro4": 4, "global": GLOBAL_B}


def theory_q(group):
    """Exact learned length distribution for averaging over groups of `group` documents."""
    if group is None:
        return P.copy()
    # distribution of the total length of the other group - 1 documents, by convolution
    pmf = np.zeros(MAX_L + 1)
    pmf[LS] = P
    others = np.array([1.0])
    for _ in range(group - 1):
        others = np.convolve(others, pmf)
    totals = np.arange(len(others))
    w = np.array([(others / (L + totals)).sum() for L in LS])
    q = P * w
    return q / q.sum()


def kl(p, q):
    return float((p * np.log(p / q)).sum())


def sample_batch(n, gen):
    idx = torch.multinomial(torch.tensor(P, dtype=torch.float), n, replacement=True, generator=gen)
    lengths = torch.tensor(LS, dtype=torch.long)[idx]
    seq = torch.full((n, MAX_L + 1), PAD, dtype=torch.long)
    seq[:, 0] = BOS
    content = torch.randint(0, N_CONTENT, (n, MAX_L), generator=gen)
    pos = torch.arange(MAX_L + 1).unsqueeze(0)
    body = (pos >= 1) & (pos < lengths.unsqueeze(1))
    seq[:, 1:][body[:, 1:]] = content[body[:, 1:]]
    seq[torch.arange(n), lengths] = EOS
    return seq[:, :-1], seq[:, 1:], lengths


def doc_losses(model, x, y):
    logits = model(x)
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=PAD, reduction="none")
    return ce.view(y.shape).sum(dim=1)


def doc_weights(method, lengths):
    n = lengths.shape[0]
    if method == "fixed":
        return torch.full((n,), 1.0 / (n * MU))
    g = GROUP[method]
    per = lengths.view(-1, g).sum(dim=1).repeat_interleave(g).float()
    return 1.0 / (per * (n // g))


@torch.no_grad()
def hazards(model, gen, n=128):
    """End-of-text chance at prediction k = 1..MAX_L, averaged over random content."""
    x = torch.randint(0, N_CONTENT, (n, MAX_L), generator=gen)
    x[:, 0] = BOS
    return F.softmax(model(x), dim=-1)[:, :, EOS].mean(dim=0).double().numpy()


def length_dist(h):
    h = h.copy()
    h[-1] = 1.0  # no document is longer than MAX_L
    survive = np.concatenate([[1.0], np.cumprod(1 - h[:-1])])
    return survive * h


def run(method, seed, steps, lr=3e-3):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(6000 + seed)
    eval_gen = torch.Generator().manual_seed(7000 + seed)
    model = TinyLM(vocab=VOCAB, max_len=MAX_L)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    h_sum, count = np.zeros(MAX_L), 0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        x, y, lengths = sample_batch(GLOBAL_B, gen)
        opt.zero_grad()
        (doc_losses(model, x, y) * doc_weights(method, lengths)).sum().backward()
        opt.step()
        if step >= int(0.8 * steps) and step % 5 == 0:
            h_sum += hazards(model, eval_gen)
            count += 1
    return length_dist(h_sum / count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    res = pathlib.Path(__file__).resolve().parent.parent / "results"
    out = res / ("e6_smoke.jsonl" if args.smoke else f"e6_many_lengths_w{args.worker}.jsonl")
    done = set()
    for f in res.glob("e6_many_lengths_w*.jsonl") if not args.smoke else []:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            done.add((r["method"], r["seed"]))
    print(f"length law: mean {MU:.2f}, cv {np.sqrt((P * LS ** 2).sum() - MU ** 2) / MU:.2f}")
    all_jobs = [(m, s) for s in range(args.seeds) for m in METHODS]
    for i, (method, seed) in enumerate(all_jobs):
        if i % args.workers != args.worker or (method, seed) in done:
            continue
        steps = 200 if args.smoke else args.steps
        t0 = time.time()
        q = run(method, seed, steps)
        qt = theory_q(GROUP.get(method))
        row = {"method": method, "seed": seed, "steps": steps,
               "learned_mean": float((q * LS).sum()), "theory_mean": float((qt * LS).sum()), "true_mean": MU,
               "kl_learned": kl(P, q), "kl_theory": kl(P, qt),
               "tv_to_theory": float(0.5 * np.abs(q - qt).sum()), "tv_to_truth": float(0.5 * np.abs(q - P).sum()),
               "q": q.tolist(), "q_theory": qt.tolist()}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{method:6s} seed={seed} learned mean={row['learned_mean']:.3f} theory={row['theory_mean']:.3f} "
              f"(true {MU:.3f}) KL learned={row['kl_learned']:.4f} theory={row['kl_theory']:.4f} "
              f"TV to theory={row['tv_to_theory']:.4f} to truth={row['tv_to_truth']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if args.smoke:
            break


if __name__ == "__main__":
    main()
