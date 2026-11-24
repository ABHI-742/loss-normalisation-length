"""Analyse E7, exactly as pre-registered in results/e7_preregistration.txt.

Primary: two one-sided t-tests (TOST, alpha 0.05 each) on the paired seed differences
  D_loss = loss(fixed) - loss(global), margin +/-0.001
  D_len  = 100 x (len(fixed) - len(global)) / len(global), margin +/-5 (percent)
"On par" needs both. Fewer than 6 complete seed pairs: INCOMPLETE.
Secondary S1-S4 are descriptive.
"""
import argparse
import json
import math
import pathlib

from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = tuple(range(100, 108))
MARGIN = {"loss": 0.001, "len": 5.0}
MIN_PAIRS = 6


def load(path):
    rows = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    return {(r["method"], r["seed"]): r for r in rows}


def pairs(R, seeds):
    return [s for s in seeds if ("fixed", s) in R and ("global", s) in R]


def diffs(R, seeds):
    d_loss = [R[("fixed", s)]["eval_loss"] - R[("global", s)]["eval_loss"] for s in seeds]
    d_len = [100 * (R[("fixed", s)]["gen_mean_len"] - R[("global", s)]["gen_mean_len"]) / R[("global", s)]["gen_mean_len"]
             for s in seeds]
    return d_loss, d_len


def tost(d, margin):
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    t = stats.t.ppf(0.95, n - 1)
    lo, hi = mean - t * se, mean + t * se
    if se == 0:
        p = 0.0 if -margin < mean < margin else 1.0
    else:
        p_low = 1 - stats.t.cdf((mean + margin) / se, n - 1)  # H0: mean <= -margin
        p_high = stats.t.cdf((mean - margin) / se, n - 1)     # H0: mean >= +margin
        p = max(p_low, p_high)
    return {"n": n, "mean": mean, "sd": sd, "lo": lo, "hi": hi, "p": p, "pass": -margin < lo and hi < margin}


def analyse(R, R4=None):
    out = []
    p = out.append
    have = pairs(R, SEEDS)
    p("E7: pre-registered equivalence test, fixed divisor against global (results/e7_preregistration.txt)")
    p(f"complete seed pairs: {len(have)} of {len(SEEDS)} {have}")
    complete = len(have) >= MIN_PAIRS
    if not complete:
        p(f"INCOMPLETE: fewer than {MIN_PAIRS} pairs. Intervals below are reported, but the primary tests are not decided.")
    if len(have) < 2:
        return "\n".join(out) + "\n"
    d_loss, d_len = diffs(R, have)

    p("\nPer seed (fixed minus global):")
    p(f"  {'seed':>4s} {'loss':>9s} {'length %':>9s}")
    for s, a, b in zip(have, d_loss, d_len):
        p(f"  {s:4d} {a:+9.5f} {b:+9.2f}")

    p("\nPRIMARY (90% interval must lie inside the margin)")
    res = {}
    for key, d, unit, fmt in (("loss", d_loss, "", "+.5f"), ("len", d_len, "%", "+.2f")):
        r = tost(d, MARGIN[key])
        res[key] = r
        verdict = ("PASS" if r["pass"] else "FAIL") if complete else "not decided (incomplete)"
        p(f"  H{1 if key == 'loss' else 2} {key:4s}: mean {r['mean']:{fmt}}{unit}, 90% interval "
          f"[{r['lo']:{fmt}}, {r['hi']:{fmt}}]{unit}, margin +/-{MARGIN[key]}{unit}, TOST p = {r['p']:.4f} -> {verdict}")
    if complete:
        both = res["loss"]["pass"] and res["len"]["pass"]
        if both:
            p("  CLAIM: the fixed divisor performs on par with the standard method "
              "(equivalent within +/-0.001 loss and +/-5% length).")
        elif res["loss"]["pass"] or res["len"]["pass"]:
            ok = "held-out loss" if res["loss"]["pass"] else "answer length"
            no = "answer length" if res["loss"]["pass"] else "held-out loss"
            p(f"  CLAIM: equivalent in {ok} only; equivalence in {no} not shown.")
        else:
            p("  CLAIM: equivalence not shown for either endpoint; drop 'on par' and report the intervals.")

    p("\nSECONDARY (descriptive)")
    p(f"  S1 direction: mean length difference {res['len']['mean']:+.2f}% "
      f"({'global shorter, as the theory predicts' if res['len']['mean'] > 0 else 'global not shorter'})")
    p("  S2 per-prompt paired length difference (fixed minus global), and medians:")
    for s in have:
        f, g = R[("fixed", s)], R[("global", s)]
        if f.get("gen_lengths") and g.get("gen_lengths") and len(f["gen_lengths"]) == len(g["gen_lengths"]):
            dd = [a - b for a, b in zip(f["gen_lengths"], g["gen_lengths"])]
            m = sum(dd) / len(dd)
            se = math.sqrt(sum((x - m) ** 2 for x in dd) / (len(dd) - 1) / len(dd))
            p(f"    seed {s}: {m:+.2f} tokens (se {se:.2f}, {len(dd)} prompts); "
              f"median fixed {f['gen_median_len']:.1f}, global {g['gen_median_len']:.1f}")
        else:
            p(f"    seed {s}: per-answer lengths missing")
    cvs = [(R[("fixed", s)].get("norm_cv"), R[("global", s)].get("norm_cv")) for s in have]
    cvs = [c for c in cvs if None not in c]
    if cvs:
        mf = sum(c[0] for c in cvs) / len(cvs)
        mg = sum(c[1] for c in cvs) / len(cvs)
        higher = sum(c[0] > c[1] for c in cvs)
        p(f"  S3 step-to-step spread of gradient size (coefficient of variation): fixed {mf:.3f}, global {mg:.3f}; "
          f"fixed higher on {higher} of {len(cvs)} seeds")
    if R4:
        both = {**R, **{k: v for k, v in R4.items() if k[0] in ("fixed", "global")}}
        s4 = pairs(both, (0, 1) + SEEDS)
        a, b = diffs(both, s4)
        ra, rb = tost(a, MARGIN["loss"]), tost(b, MARGIN["len"])
        p(f"  S4 with E4 seeds 0 and 1 added ({len(s4)} pairs, exploratory): "
          f"loss [{ra['lo']:+.5f}, {ra['hi']:+.5f}] {'inside' if ra['pass'] else 'not inside'} margin; "
          f"length [{rb['lo']:+.2f}, {rb['hi']:+.2f}]% {'inside' if rb['pass'] else 'not inside'} margin")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "results" / "e7_results.jsonl"))
    ap.add_argument("--e4", default=str(ROOT / "results" / "e4_results.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "e7_tests.txt"))
    a = ap.parse_args()
    R4 = load(a.e4) if pathlib.Path(a.e4).exists() else None
    text = analyse(load(a.input), R4)
    pathlib.Path(a.out).write_text(text)
    print(text)
