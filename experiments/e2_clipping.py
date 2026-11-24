"""E2: does clipping undo the fixed divisor, and does a length-aware threshold fix it?

Step 1 (calibration): train with the fixed divisor and no clipping, and measure
  - the typical gradient size late in training (median of |g|), and
  - the gradient-growth power alpha, from regressing log|A| on log N.
Step 2 (sweep): thresholds c = multiple x median, for
  fixed        - fixed divisor, constant threshold
  global       - global token count, constant threshold
  fixed_aware  - fixed divisor, threshold c * (N / (B mu))^alpha
For each run, the theory's prediction is computed from that run's own clip factors:
  w(L) = sum over batches of (s / Z) * n_L  /  sum of n_L,   q(short) = p w(2) / (p w(2) + (1-p) w(8))
(theorem T9, which assumes the clip factor is independent of the end-of-text gradients given the lengths).
"""
import argparse
import json
import math
import pathlib
import statistics

import torch

from common import LENGTHS, MU, P_SHORT, TinyLM, doc_losses, eos_prob_at_2, lr_at, sample_batch

B = 32
BMU = B * MU


def grad_norm(params):
    return math.sqrt(sum(float(p.grad.pow(2).sum()) for p in params))


def run(method, c, alpha, seed, steps, log_norms=False):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(2000 + seed)
    model = TinyLM()
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    tail, count = 0.0, 0
    num = {L: 0.0 for L in LENGTHS}
    den = {L: 0.0 for L in LENGTHS}
    norms = []
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, 3e-3)
        x, y, lengths = sample_batch(B, gen)
        n_tokens = lengths.sum().item()
        divisor = n_tokens if method == "global" else BMU
        opt.zero_grad()
        (doc_losses(model, x, y) / divisor).sum().backward()
        norm = grad_norm(params)
        threshold = c
        if method == "fixed_aware" and math.isfinite(c):
            threshold = c * (n_tokens / BMU) ** alpha
        s = min(1.0, threshold / norm) if math.isfinite(threshold) else 1.0
        if s < 1.0:
            for p in params:
                p.grad.mul_(s)
        opt.step()
        if step >= steps // 2:
            for L in LENGTHS:
                n_l = int((lengths == L).sum())
                num[L] += s / divisor * n_l
                den[L] += n_l
            if log_norms:
                norms.append((n_tokens, norm * divisor))
        if step >= int(0.8 * steps) and step % 5 == 0:
            tail += eos_prob_at_2(model)
            count += 1
    w = {L: num[L] / den[L] for L in LENGTHS}
    predicted = P_SHORT * w[2] / (P_SHORT * w[2] + (1 - P_SHORT) * w[8])
    return tail / count, predicted, norms


def calibrate(steps):
    _, _, norms = run("fixed", math.inf, 0.0, seed=99, steps=steps, log_norms=True)
    xs = [math.log(n) for n, _ in norms]
    ys = [math.log(a) for _, a in norms]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    alpha = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    late = norms[int(0.6 * len(norms)):]
    median_g = statistics.median(a / BMU for _, a in late)
    return alpha, median_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="../results/e2_clipping.jsonl")
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = pathlib.Path(__file__).parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    rows = [r for r in rows if r["steps"] == args.steps]
    cal = next((r for r in rows if r["kind"] == "calibration"), None)
    if cal is None:
        alpha, median_g = calibrate(args.steps)
        cal = {"steps": args.steps, "kind": "calibration", "alpha": alpha, "median_g": median_g}
        with out.open("a") as f:
            f.write(json.dumps(cal) + "\n")
    alpha, median_g = cal["alpha"], cal["median_g"]
    print(f"calibration: alpha={alpha:.3f} median late |g|={median_g:.4f}", flush=True)
    done = {(r["mult"], r["method"], r["seed"]) for r in rows if r["kind"] == "run"}
    for mult in (math.inf, 1.0, 0.5, 0.2, 0.05):
        c = mult * median_g
        key_mult = mult if math.isfinite(mult) else None
        for method in ("fixed", "global", "fixed_aware"):
            if method == "fixed_aware" and not math.isfinite(mult):
                continue
            for seed in range(args.seeds):
                if (key_mult, method, seed) in done:
                    continue
                q, pred, _ = run(method, c, alpha, seed, args.steps)
                row = {"steps": args.steps, "kind": "run", "mult": key_mult, "method": method,
                       "seed": seed, "q_short": q, "predicted": pred}
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"c={mult}x {method:11s} seed={seed} q_short={q:.4f} predicted={pred:.4f}", flush=True)


if __name__ == "__main__":
    main()
