"""Analyse E8, exactly as pre-registered in results/e8_preregistration.txt.

Primary: one-sided paired t-test that D_len = 100 x (len(fixed) - len(global)) / len(global) > 0
over seeds 200-207 (global shorter at 4 answers per update). Fewer than 6 pairs: INCOMPLETE.
Secondary S1-S5 are descriptive.
"""
import argparse
import math
import pathlib

from scipy import stats

from analyse_e7 import diffs, load, pairs

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = tuple(range(200, 208))
MIN_PAIRS = 6
THEORY_LIMIT = -23.8   # percent, prompt-blind, B = 4
E4_MICRO4 = -6.3       # percent, E4 micro4 against fixed


def summary(d, level):
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return mean, sd, se, mean - t * se, mean + t * se


def analyse(R, R7=None):
    out = []
    p = out.append
    have = pairs(R, SEEDS)
    p("E8: pre-registered small-batch test, 4 answers per update (results/e8_preregistration.txt)")
    p(f"complete seed pairs: {len(have)} of {len(SEEDS)} {have}")
    groups = {R[k].get("group") for k in R}
    p(f"answers per update recorded in the results: {sorted(g for g in groups if g is not None)}")
    complete = len(have) >= MIN_PAIRS
    if not complete:
        p(f"INCOMPLETE: fewer than {MIN_PAIRS} pairs. Intervals are reported, but H1 is not decided.")
    if len(have) < 2:
        return "\n".join(out) + "\n"
    d_loss, d_len = diffs(R, have)

    p("\nPer seed (fixed minus global):")
    p(f"  {'seed':>4s} {'length %':>9s} {'loss':>9s}")
    for s, a, b in zip(have, d_len, d_loss):
        p(f"  {s:4d} {a:+9.2f} {b:+9.5f}")

    mean, sd, se, lo90, hi90 = summary(d_len, 0.90)
    t = mean / se if se > 0 else math.inf
    pval = 1 - stats.t.cdf(t, len(d_len) - 1)
    p("\nPRIMARY")
    verdict = ("PASS" if pval < 0.05 else "FAIL") if complete else "not decided (incomplete)"
    p(f"  H1 (global shorter): mean D_len {mean:+.2f}%, one-sided paired t = {t:.2f}, df {len(d_len) - 1}, "
      f"p = {pval:.4f} -> {verdict}")
    # S1 is stated relative to fixed (the unbiased reference), as in the pre-registration
    rel = [100 * (R[("global", s)]["gen_mean_len"] - R[("fixed", s)]["gen_mean_len"]) / R[("fixed", s)]["gen_mean_len"]
           for s in have]
    shift, _, _, slo, shi = summary(rel, 0.90)
    if complete:
        if pval < 0.05:
            p(f"  CLAIM: at 4 answers per update the standard method shortens answers by {mean:.1f}% "
              f"(90% interval {lo90:.1f} to {hi90:.1f}%) relative to the constant divisor.")
        else:
            p(f"  CLAIM: no detectable drift at 4 answers per update (90% interval of D_len "
              f"{lo90:+.1f} to {hi90:+.1f}%); the constant divisor's advantage remains theoretical here.")

    p("\nSECONDARY (descriptive)")
    p(f"  S1 global against fixed: {shift:+.2f}% (90% interval {slo:+.2f} to {shi:+.2f}%); "
      f"share of the theory limit ({THEORY_LIMIT}%): {shift / THEORY_LIMIT:.2f}; E4 micro4 was {E4_MICRO4}%")
    lm, _, _, llo, lhi = summary(d_loss, 0.95)
    p(f"  S2 held-out loss, fixed minus global: {lm:+.5f} (95% interval {llo:+.5f} to {lhi:+.5f})")
    cvs = [(R[("fixed", s)].get("norm_cv"), R[("global", s)].get("norm_cv")) for s in have]
    cvs = [c for c in cvs if None not in c]
    if cvs:
        mf = sum(c[0] for c in cvs) / len(cvs)
        mg = sum(c[1] for c in cvs) / len(cvs)
        p(f"  S3 spread of gradient size: fixed {mf:.3f}, global {mg:.3f} ({100 * (mf / mg - 1):+.0f}%); "
          f"fixed higher on {sum(c[0] > c[1] for c in cvs)} of {len(cvs)} seeds (E7: +4%)")
    p("  S4 per-prompt paired length difference (fixed minus global), and medians:")
    for s in have:
        f, g = R[("fixed", s)], R[("global", s)]
        if f.get("gen_lengths") and g.get("gen_lengths") and len(f["gen_lengths"]) == len(g["gen_lengths"]):
            dd = [a - b for a, b in zip(f["gen_lengths"], g["gen_lengths"])]
            m = sum(dd) / len(dd)
            sem = math.sqrt(sum((x - m) ** 2 for x in dd) / (len(dd) - 1) / len(dd))
            p(f"    seed {s}: {m:+.2f} tokens (se {sem:.2f}); median fixed {f['gen_median_len']:.1f}, "
              f"global {g['gen_median_len']:.1f}")
        else:
            p(f"    seed {s}: per-answer lengths missing")
    if R7:
        p("  S5 batch size 4 (E8) against 32 (E7), mean generated length (Welch, unpaired):")
        for m in ("fixed", "global"):
            a = [R[(m, s)]["gen_mean_len"] for s in have]
            b = [r["gen_mean_len"] for k, r in R7.items() if k[0] == m]
            if len(b) >= 2:
                w = stats.ttest_ind(a, b, equal_var=False)
                p(f"    {m:6s}: B=4 {sum(a) / len(a):.2f}, B=32 {sum(b) / len(b):.2f}, "
                  f"difference {sum(a) / len(a) - sum(b) / len(b):+.2f} tokens, Welch p = {w.pvalue:.3f}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "results" / "e8_results.jsonl"))
    ap.add_argument("--e7", default=str(ROOT / "results" / "e7_results.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "e8_tests.txt"))
    a = ap.parse_args()
    R7 = load(a.e7) if pathlib.Path(a.e7).exists() else None
    text = analyse(load(a.input), R7)
    pathlib.Path(a.out).write_text(text)
    print(text)
