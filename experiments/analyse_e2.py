"""Paired tests for E2 (clipping). Every method uses the same seeds, so we compare
methods within each seed. Two-sided paired t-tests, Holm correction across the
family of planned comparisons, plus a check of measured against predicted values.
"""
import json
import pathlib

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "results" / "e2_clipping.jsonl").read_text().splitlines()]
runs = [r for r in rows if r["kind"] == "run"]


def complete_seeds():
    """Seeds that finished every (threshold, method) cell, so all pairs line up."""
    cells = {(r["mult"], r["method"]) for r in runs}
    return sorted(s for s in {r["seed"] for r in runs}
                  if all(any(r["seed"] == s and (r["mult"], r["method"]) == c for r in runs) for c in cells))


SEEDS = complete_seeds()


def q(mult, method, key="q_short"):
    by_seed = {r["seed"]: r[key] for r in runs if r["mult"] == mult and r["method"] == method}
    return np.array([by_seed[s] for s in SEEDS])


# (label, a, b, expected direction of a - b)
comparisons = [("no clipping: global - fixed", q(None, "global"), q(None, "fixed"), "+")]
for m in (1.0, 0.5, 0.2, 0.05):
    comparisons += [
        (f"c={m}x: fixed clipped - fixed unclipped", q(m, "fixed"), q(None, "fixed"), "+"),
        (f"c={m}x: global - fixed", q(m, "global"), q(m, "fixed"), "0"),
        (f"c={m}x: fixed - fixed_aware", q(m, "fixed"), q(m, "fixed_aware"), "+"),
    ]

results = []
for label, a, b, expect in comparisons:
    d = a - b
    t, p = stats.ttest_rel(a, b)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    results.append((label, expect, d, d.mean(), half, t, p))

# Holm correction over the planned comparisons
order = np.argsort([r[6] for r in results])
holm = np.empty(len(results))
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (len(results) - rank) * results[i][6]))
    holm[i] = running

lines = [f"Paired comparisons over {len(SEEDS)} seeds {SEEDS} "
         "(difference in learned chance of a short document, per seed)", ""]
for (label, expect, d, mean, half, t, p), ph in zip(results, holm):
    per_seed = ", ".join(f"{x:+.4f}" for x in d)
    lines.append(f"{label:42s} expect {expect}  mean {mean:+.4f}  95% CI [{mean - half:+.4f}, {mean + half:+.4f}]"
                 f"  t={t:+.2f}  p={p:.4f}  Holm p={ph:.4f}  seeds: {per_seed}")

lines += ["", "Measured minus predicted (T9, from each run's own clip factors)",
          "Runs that share a seed are correlated, so the test uses one average per seed."]
by_seed = {s: np.mean([r["q_short"] - r["predicted"] for r in runs if r["seed"] == s]) for s in SEEDS}
vals = np.array(list(by_seed.values()))
t, p = stats.ttest_1samp(vals, 0.0)
half = stats.t.ppf(0.975, len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals))
lines.append(f"seeds={len(vals)} mean {vals.mean():+.4f} 95% CI [{vals.mean() - half:+.4f}, {vals.mean() + half:+.4f}]"
             f" t={t:+.2f} p={p:.2g}")
lines.append("mean by seed: " + ", ".join(f"seed {s}: {v:+.4f}" for s, v in by_seed.items()))
for method in ("fixed", "global", "fixed_aware"):
    per = np.array([np.mean([r["q_short"] - r["predicted"] for r in runs
                             if r["seed"] == s and r["method"] == method]) for s in SEEDS])
    t, p = stats.ttest_1samp(per, 0.0)
    lines.append(f"  {method:11s} mean {per.mean():+.4f} t={t:+.2f} p={p:.2g}")

text = "\n".join(lines)
print(text)
(ROOT / "results" / "e2_paired_tests.txt").write_text(text + "\n")
