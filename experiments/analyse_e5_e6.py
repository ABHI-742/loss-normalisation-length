"""Tests for E5 and E6, exactly as pre-registered in results/e5_e6_preregistration.txt."""
import json
import pathlib

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load(pattern):
    return [json.loads(l) for f in sorted(RES.glob(pattern)) for l in f.read_text().splitlines()]


def ci(d):
    d = np.asarray(d, dtype=float)
    if len(d) < 2:
        return d.mean(), float("nan")
    return d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


def fmt(d):
    m, h = ci(d)
    return f"{m:+.4f} [{m - h:+.4f}, {m + h:+.4f}] (n={len(d)})"


lines = []

# ---------------- E5 ----------------
e5 = load("e5_learning_rate_w*.jsonl")
lines.append("E5: learning rate sweep")
if e5:
    def val(opt, scale, method, key="q_short"):
        return {r["seed"]: r[key] for r in e5 if r["opt"] == opt and r["scale"] == scale and r["method"] == method}

    for opt in ("adam", "sgd"):
        lines.append(f"  {opt}")
        for scale in (1.0, 0.3333, 0.1):
            f = val(opt, scale, "fixed")
            if not f:
                continue
            row = f"    scale {scale:<6} fixed - 0.5: {fmt([v - 0.5 for v in f.values()])}"
            for m, th in (("global", 0.0096), ("micro4", 0.0891)):
                o = val(opt, scale, m)
                seeds = sorted(set(f) & set(o))
                if seeds:
                    row += f" | {m} - fixed: {fmt([o[s] - f[s] for s in seeds])} (theory {th:+.4f})"
            lines.append(row)
        f1, f10 = val(opt, 1.0, "fixed"), val(opt, 0.1, "fixed")
        for m in ("global", "micro4"):
            o1, o10 = val(opt, 1.0, m), val(opt, 0.1, m)
            seeds = sorted(set(f1) & set(f10) & set(o1) & set(o10))
            if len(seeds) >= 2:
                d = [(o1[s] - f1[s]) - (o10[s] - f10[s]) for s in seeds]
                m_, h = ci(d)
                verdict = "PASS" if (m_ - h > -0.002 and m_ + h < 0.002) else (
                    "FAIL" if (m_ - h > 0.002 or m_ + h < -0.002) else "INCONCLUSIVE")
                lines.append(f"    P1 {m} gap, scale 1 minus scale 1/10: {fmt(d)} -> {verdict}")
        seeds = sorted(set(f1) & set(f10))
        if len(seeds) >= 2:
            d = [(f1[s] - 0.5) - (f10[s] - 0.5) for s in seeds]
            t, p = stats.ttest_1samp(d, 0.0, alternative="greater")
            m10, h10 = ci([f10[s] - 0.5 for s in seeds])
            ok = p < 0.05 and abs(m10) + h10 <= 0.001
            lines.append(f"    P2 offset shrink (scale 1 minus 1/10): {fmt(d)} one-sided p={p:.2g}; "
                         f"offset at 1/10 {m10:+.4f}+/-{h10:.4f} -> {'PASS' if ok else 'FAIL'}")
        # end-of-training value (no tail averaging), for reference
        fe = val(opt, 0.1, "fixed", "q_end")
        if fe:
            lines.append(f"    fixed at scale 1/10, final step only, minus 0.5: {fmt([v - 0.5 for v in fe.values()])}")

# ---------------- E5 post-hoc extension ----------------
ext = load("e5_extension_w*.jsonl")
if ext:
    lines += ["", "E5 POST-HOC EXTENSION (exploratory, not pre-registered; pre-registered P2 stays FAIL)",
              "  fixed divisor, seeds 0-4 from the main run plus seeds 5-14"]
    both = [r for r in e5 if r["method"] == "fixed" and r["scale"] in (1.0, 0.1)] + ext
    for opt in ("adam", "sgd"):
        v1 = {r["seed"]: r["q_short"] - 0.5 for r in both if r["opt"] == opt and r["scale"] == 1.0}
        v10 = {r["seed"]: r["q_short"] - 0.5 for r in both if r["opt"] == opt and r["scale"] == 0.1}
        seeds = sorted(set(v1) & set(v10))
        if len(seeds) < 2:
            continue
        d = [v1[s] - v10[s] for s in seeds]
        t, p = stats.ttest_1samp(d, 0.0, alternative="greater")
        m10, h10 = ci([v10[s] for s in seeds])
        ok = p < 0.05 and abs(m10) + h10 <= 0.001
        lines.append(f"  {opt}: offset at scale 1 {fmt([v1[s] for s in seeds])}; at 1/10 {fmt([v10[s] for s in seeds])}")
        lines.append(f"    shrink (1 minus 1/10) {fmt(d)} one-sided p={p:.2g} -> P2 rule {'met' if ok else 'not met'}")
        fe = [r["q_end"] - 0.5 for r in both if r["opt"] == opt and r["scale"] == 0.1 and r["seed"] in seeds]
        lines.append(f"    final step only at 1/10: {fmt(fe)}")

# ---------------- E6 ----------------
e6 = load("e6_many_lengths_w*.jsonl")
lines += ["", "E6: many lengths (1-32, coefficient of variation 0.97)"]
if e6:
    by = {(r["method"], r["seed"]): r for r in e6}
    fixed = {s: r for (m, s), r in by.items() if m == "fixed"}
    if fixed:
        lines.append(f"  fixed learned mean - true mean: {fmt([r['learned_mean'] - r['true_mean'] for r in fixed.values()])}")
    for m in ("micro1", "micro4", "global"):
        runs = {s: r for (mm, s), r in by.items() if mm == m}
        if not runs:
            continue
        th_gap = next(iter(runs.values()))["theory_mean"] - next(iter(runs.values()))["true_mean"]
        seeds = sorted(set(runs) & set(fixed))
        gaps = [runs[s]["learned_mean"] - fixed[s]["learned_mean"] for s in seeds]
        mean_, h = ci(gaps)
        if m == "global":
            ok3 = (mean_ - h <= th_gap <= mean_ + h) and (mean_ + h < 0)
        else:
            ok3 = (mean_ - h >= th_gap * 1.1) and (mean_ + h <= th_gap * 0.9)
        lines.append(f"  {m}: learned mean {np.mean([r['learned_mean'] for r in runs.values()]):.3f} "
                     f"(theory {next(iter(runs.values()))['theory_mean']:.3f})")
        lines.append(f"    P3 gap to fixed {fmt(gaps)} vs theory {th_gap:+.3f} -> {'PASS' if ok3 else 'FAIL'}")
        tvt = [r["tv_to_theory"] for r in runs.values()]
        tvp = [r["tv_to_truth"] for r in runs.values()]
        line = f"    total variation: to theory {np.mean(tvt):.4f}, to truth {np.mean(tvp):.4f}"
        if m != "global":
            line += f" -> P4 {'PASS' if all(a < b for a, b in zip(tvt, tvp)) else 'FAIL'}"
        lines.append(line)
        kl_ex = [runs[s]["kl_learned"] - fixed[s]["kl_learned"] for s in seeds]
        kl_th = next(iter(runs.values()))["kl_theory"]
        line = f"    KL excess over fixed {fmt(kl_ex)} vs theory {kl_th:.4f}"
        if m != "global":
            km, kh = ci(kl_ex)
            line += f" -> P5 {'PASS' if abs(km - kl_th) <= 0.2 * kl_th else 'FAIL'}"
        lines.append(line)
    # per-length comparison for the most biased method, averaged over seeds
    m1 = [r for (m, s), r in by.items() if m == "micro1"]
    if m1:
        q = np.mean([r["q"] for r in m1], axis=0)
        qt = np.array(m1[0]["q_theory"])
        lines.append("  micro1 learned vs theory chance of each length (first 12):")
        lines.append("    " + "  ".join(f"L{L + 1}:{q[L]:.3f}/{qt[L]:.3f}" for L in range(12)))

text = "\n".join(lines)
print(text)
(RES / "e5_e6_tests.txt").write_text(text + "\n")
