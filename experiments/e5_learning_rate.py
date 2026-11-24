"""E5: is the small offset of the fixed divisor training noise, and are the gaps
between methods independent of the learning rate?

E1 found every method about +0.0025 above its predicted value, the fixed divisor
included. Two theory claims are tested here, on the E1 task (lengths 2 and 8):

  1. The gap between two methods (for example global minus fixed) is set by the
     loss each method minimises, so it should not change with the learning rate.
  2. The common offset should come from step-size noise, so it should shrink
     towards zero as the learning rate falls (with steps scaled up so that every
     run takes the same total learning-rate distance).

Learning rates are E1's base rate times 1, 1/3 and 1/10, with 4000, 12000 and
40000 steps. Results go to one file per worker, so workers can run in parallel.
"""
import argparse
import json
import pathlib
import time

import torch

from common import MU, TinyLM, doc_losses, eos_prob_at_2, lr_at, sample_batch, theory_q_short

GLOBAL_B = 32
METHODS = ("fixed", "global", "micro4")
PREDICTED = {"fixed": 0.5, "global": theory_q_short(GLOBAL_B), "micro4": theory_q_short(4)}
BASE = {"adam": 3e-3, "sgd": 0.5}
SCALES = ((1.0, 4000), (1 / 3, 12000), (0.1, 40000))


def doc_weights(method, lengths):
    n = lengths.shape[0]
    if method == "fixed":
        return torch.full((n,), 1.0 / (n * MU))
    if method == "global":
        return torch.full((n,), 1.0 / lengths.sum().item())
    per_micro = lengths.view(-1, 4).sum(dim=1).repeat_interleave(4).float()
    return 1.0 / (per_micro * (n // 4))


def run(method, opt_name, lr, steps, seed):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(5000 + seed)
    model = TinyLM()
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr) if opt_name == "adam" else torch.optim.SGD(params, lr=lr)
    tail, count = 0.0, 0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        x, y, lengths = sample_batch(GLOBAL_B, gen)
        opt.zero_grad()
        (doc_losses(model, x, y) * doc_weights(method, lengths)).sum().backward()
        opt.step()
        if step >= int(0.8 * steps) and step % max(1, steps // 800) == 0:
            tail += eos_prob_at_2(model)
            count += 1
    return tail / count, eos_prob_at_2(model)


def jobs(seeds, seed_start=0, methods=METHODS, scales=SCALES):
    for opt_name in ("adam", "sgd"):
        for scale, steps in scales:
            for method in methods:
                for seed in range(seed_start, seed_start + seeds):
                    yield opt_name, scale, steps, method, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extension", action="store_true",
                    help="post-hoc: seeds 5-14, fixed divisor only, scales 1 and 1/10")
    args = ap.parse_args()
    torch.set_num_threads(4)
    res = pathlib.Path(__file__).resolve().parent.parent / "results"
    stem = "e5_extension" if args.extension else "e5_learning_rate"
    out = res / ("e5_smoke.jsonl" if args.smoke else f"{stem}_w{args.worker}.jsonl")
    done = set()
    for f in res.glob(f"{stem}_w*.jsonl") if not args.smoke else []:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            done.add((r["opt"], r["scale"], r["method"], r["seed"]))
    if args.extension:
        all_jobs = list(jobs(10, seed_start=5, methods=("fixed",), scales=(SCALES[0], SCALES[2])))
    else:
        all_jobs = list(jobs(args.seeds))
    # longest jobs first, dealt round-robin so workers finish together
    all_jobs.sort(key=lambda j: -j[2])
    for i, (opt_name, scale, steps, method, seed) in enumerate(all_jobs):
        if i % args.workers != args.worker or (opt_name, round(scale, 4), method, seed) in done:
            continue
        if args.smoke:
            steps = 200
        t0 = time.time()
        lr = BASE[opt_name] * scale
        q_tail, q_end = run(method, opt_name, lr, steps, seed)
        row = {"opt": opt_name, "scale": round(scale, 4), "lr": lr, "steps": steps, "method": method,
               "seed": seed, "q_short": q_tail, "q_end": q_end, "predicted": PREDICTED[method]}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{opt_name} lr={lr:.2g} steps={steps} {method:6s} seed={seed} q={q_tail:.4f} "
              f"end={q_end:.4f} predicted={PREDICTED[method]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if args.smoke:
            break


if __name__ == "__main__":
    main()
