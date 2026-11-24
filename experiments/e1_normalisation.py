"""E1: does a real transformer learn the length law the theory predicts?

Global batch of 32 documents per update. Methods:
  micro1   - average over each single document, then average 32 (batch-average law at B=1)
  micro4   - average over micro-batches of 4, then average 8   (law at B=4)
  global   - divide by the total token count of all 32          (law at B=32)
  fixed    - divide by 32 * average length (the fixed divisor)   (truth)
  regress  - fixed divisor plus a control-variate correction     (truth)
"""
import argparse
import json
import pathlib
import time

import torch

from common import MU, TinyLM, doc_losses, eos_prob_at_2, lr_at, sample_batch, theory_q_short

GLOBAL_B = 32
PREDICTED = {
    "micro1": theory_q_short(1),
    "micro4": theory_q_short(4),
    "global": theory_q_short(GLOBAL_B),
    "fixed": 0.5,
    "regress": 0.5,
}


def doc_weights(method, lengths):
    n = lengths.shape[0]
    if method == "fixed":
        return torch.full((n,), 1.0 / (n * MU))
    if method == "global":
        return torch.full((n,), 1.0 / lengths.sum().item())
    micro = 1 if method == "micro1" else 4
    per_micro = lengths.view(-1, micro).sum(dim=1).repeat_interleave(micro).float()
    return 1.0 / (per_micro * (n // micro))


def run(method, opt_name, seed, steps):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1000 + seed)
    model = TinyLM()
    params = list(model.parameters())
    base_lr = 3e-3 if opt_name == "adam" else 0.5
    opt = torch.optim.Adam(params, lr=base_lr) if opt_name == "adam" else torch.optim.SGD(params, lr=base_lr)
    bmu = GLOBAL_B * MU
    ema_an = [torch.zeros_like(p) for p in params]
    ema_nn = 0.0
    tail, count = 0.0, 0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, base_lr)
        x, y, lengths = sample_batch(GLOBAL_B, gen)
        losses = doc_losses(model, x, y)
        opt.zero_grad()
        if method == "regress":
            losses.sum().backward()
            d = (lengths.sum().item() - bmu) / bmu
            v_ready = ema_nn > 0
            for p, e in zip(params, ema_an):
                a = p.grad.clone()
                v = e / ema_nn if v_ready else torch.zeros_like(a)
                p.grad = a / bmu - v * d
                e.mul_(0.99).add_(0.01 * (a / bmu) * d)
            ema_nn = 0.99 * ema_nn + 0.01 * d * d
        else:
            (losses * doc_weights(method, lengths)).sum().backward()
        opt.step()
        if step >= int(0.8 * steps) and step % 5 == 0:
            tail += eos_prob_at_2(model)
            count += 1
    return tail / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="../results/e1_normalisation.jsonl")
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = pathlib.Path(__file__).parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r["steps"] == args.steps:
                done.add((r["opt"], r["method"], r["seed"]))
    for opt_name in ("adam", "sgd"):
        for method in PREDICTED:
            for seed in range(args.seeds):
                if (opt_name, method, seed) in done:
                    continue
                t0 = time.time()
                q = run(method, opt_name, seed, args.steps)
                row = {"steps": args.steps, "opt": opt_name, "method": method, "seed": seed,
                       "q_short": q, "predicted": PREDICTED[method]}
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{opt_name:4s} {method:7s} seed={seed} q_short={q:.4f} "
                      f"predicted={PREDICTED[method]:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
