"""Analyse E9, exactly as pre-registered in results/e9_preregistration.txt.

Primary: for m in (micro1, micro4), D_m = 100 x (len(fixed) - len(m)) / len(m) per seed;
one-sided paired t-test that mean D_m > 0; the two p-values are Holm-corrected (pass if < 0.05).
Fewer than 3 complete seed pairs for a comparison: INCOMPLETE. Secondary S1-S5 are descriptive.
"""
import argparse
import json
import math
import pathlib

from scipy import stats

from analyse_e7 import load

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = tuple(range(300, 304))
METHODS = ("micro1", "micro4")
MIN_PAIRS = 3


def summary(d, level):
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return mean, sd, se, mean - t * se, mean + t * se


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def e4_shares(path):
    """E4's shifts as a share of their prompt-blind limits, for comparison (S5)."""
    if not pathlib.Path(path).exists():
        return {}
    R = load(path)
    out = {}
    for m in METHODS:
        seeds = [s for s in (0, 1) if (m, s) in R and ("fixed", s) in R]
        if not seeds:
            continue
        shift = sum(100 * (R[(m, s)]["gen_mean_len"] - R[("fixed", s)]["gen_mean_len"]) / R[("fixed", s)]["gen_mean_len"]
                    for s in seeds) / len(seeds)
        lim = 100 * (R[(m, seeds[0])]["theory_upper_limit"] / R[(m, seeds[0])]["train_mean_len"] - 1)
        out[m] = (shift, lim, shift / lim)
    return out


def analyse(R, e4_path=None):
    out = []
    p = out.append
    p("E9: pre-registered scale test, TinyLlama 1.1B with LoRA on Dolly (results/e9_preregistration.txt)")
    info = next(iter(R.values()), {})
    p(f"model {info.get('model')}, dataset {info.get('dataset')}, LoRA rank {info.get('lora_r')}, "
      f"training answers {info.get('n_train')}, mean length {info.get('train_mean_len', float('nan')):.1f}")

    pairs, d, pv = {}, {}, {}
    for m in METHODS:
        pairs[m] = [s for s in SEEDS if (m, s) in R and ("fixed", s) in R]
        d[m] = [100 * (R[("fixed", s)]["gen_mean_len"] - R[(m, s)]["gen_mean_len"]) / R[(m, s)]["gen_mean_len"]
                for s in pairs[m]]
        if len(d[m]) >= 2:
            mean, sd, se, _, _ = summary(d[m], 0.90)
            t = mean / se if se > 0 else math.inf
            pv[m] = (t, 1 - stats.t.cdf(t, len(d[m]) - 1))
    ms = [m for m in METHODS if m in pv]
    adj = dict(zip(ms, holm([pv[m][1] for m in ms])))

    p("\nPer seed, D = 100 x (fixed - method) / method (positive = method shorter):")
    for m in METHODS:
        p(f"  {m:7s} " + "  ".join(f"{s}: {x:+.2f}" for s, x in zip(pairs[m], d[m])))

    p("\nPRIMARY (one-sided paired t-tests, Holm-corrected over H1 and H2)")
    for h, m in (("H1", "micro1"), ("H2", "micro4")):
        n = len(pairs[m])
        if m not in pv:
            p(f"  {h} {m}: fewer than 2 pairs, no test")
            continue
        t, raw = pv[m]
        if n < MIN_PAIRS:
            verdict = "INCOMPLETE"
        else:
            verdict = "PASS" if adj[m] < 0.05 else "FAIL"
        mean = sum(d[m]) / n
        p(f"  {h} {m} shorter than fixed: mean D {mean:+.2f}%, t = {t:.2f}, df {n - 1}, p = {raw:.4f}, "
          f"Holm p = {adj[m]:.4f}, {n} seed pairs -> {verdict}")

    p("\nSECONDARY (descriptive)")
    p("  S1 shift relative to the constant divisor, and share of the prompt-blind limit:")
    for m in METHODS:
        if len(pairs[m]) < 2:
            continue
        rel = [100 * (R[(m, s)]["gen_mean_len"] - R[("fixed", s)]["gen_mean_len"]) / R[("fixed", s)]["gen_mean_len"]
               for s in pairs[m]]
        mean, _, _, lo, hi = summary(rel, 0.90)
        r0 = R[(m, pairs[m][0])]
        lim = 100 * (r0["theory_upper_limit"] / r0["train_mean_len"] - 1) if r0.get("theory_upper_limit") else float("nan")
        p(f"    {m:7s} {mean:+.2f}% (90% interval {lo:+.2f} to {hi:+.2f}); prompt-blind limit {lim:+.1f}%; "
          f"share {mean / lim:.2f}")
    p("  S2 held-out loss, method minus fixed (95% interval):")
    for m in METHODS:
        if len(pairs[m]) < 2:
            continue
        dl = [R[(m, s)]["eval_loss"] - R[("fixed", s)]["eval_loss"] for s in pairs[m]]
        mean, _, _, lo, hi = summary(dl, 0.95)
        p(f"    {m:7s} {mean:+.5f} ({lo:+.5f} to {hi:+.5f})")
    fr = [r["frac_norm_over_1"] for r in R.values() if r.get("frac_norm_over_1") is not None]
    al = [r["growth_power"] for r in R.values() if r.get("growth_power") is not None]
    if fr:
        p(f"  S3 share of updates with adapter-gradient norm above 1: {min(fr):.3f} to {max(fr):.3f}")
    if al:
        p(f"     growth exponent alpha: {min(al):.3f} to {max(al):.3f} (mean {sum(al) / len(al):.3f})")
    p("  S4 per-prompt paired length difference (fixed minus method), and medians:")
    for m in METHODS:
        for s in pairs[m]:
            f, g = R[("fixed", s)], R[(m, s)]
            if f.get("gen_lengths") and g.get("gen_lengths") and len(f["gen_lengths"]) == len(g["gen_lengths"]):
                dd = [a - b for a, b in zip(f["gen_lengths"], g["gen_lengths"])]
                mm = sum(dd) / len(dd)
                se = math.sqrt(sum((x - mm) ** 2 for x in dd) / (len(dd) - 1) / len(dd))
                p(f"    {m:7s} seed {s}: {mm:+.2f} tokens (se {se:.2f}); median fixed {f['gen_median_len']:.1f}, "
                  f"{m} {g['gen_median_len']:.1f}")
    if e4_path:
        sh = e4_shares(e4_path)
        if sh:
            p("  S5 E4 (SmolLM2-135M, Alpaca) for comparison: " + "; ".join(
                f"{m} {v[0]:+.1f}% of limit {v[1]:+.1f}% (share {v[2]:.2f})" for m, v in sh.items()))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "results" / "e9_results.jsonl"))
    ap.add_argument("--e4", default=str(ROOT / "results" / "e4_results.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "e9_tests.txt"))
    a = ap.parse_args()
    text = analyse(load(a.input), a.e4)
    pathlib.Path(a.out).write_text(text)
    print(text)
