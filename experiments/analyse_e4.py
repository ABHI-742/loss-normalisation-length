"""Analyse E4 (SmolLM2-135M fine-tuned on Alpaca, 2 seeds). Not pre-registered: exploratory.

Every method is compared with the fixed divisor (fixed) trained on the same seed.
Per-answer lengths were not saved, so the sampling error of a 500-answer mean is
approximated from the spread of the training answers (sd 55.1 tokens). Methods share
prompts and the sampling seed, so treating them as independent overstates the error.
"""
import json, math, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in (ROOT / "results" / "e4_results.jsonl").read_text().splitlines() if l.strip()]
R = {(r["method"], r["seed"]): r for r in rows}
SEEDS = sorted({r["seed"] for r in rows})
SD, N = 55.1, 500
se_diff = math.sqrt(2) * SD / math.sqrt(N) / math.sqrt(len(SEEDS))  # difference of two means, averaged over seeds
train_mean = rows[0]["train_mean_len"]
out = []
p = out.append

p("E4: SmolLM2-135M on Alpaca, 2 epochs, lr 1e-4, 16k training answers, 500 held-out generations per run")
p(f"seeds {SEEDS}; training answer mean {train_mean:.1f}; held-out reference mean {rows[0]['ref_mean_len']:.1f}")
p("EXPLORATORY: not pre-registered. Two seeds only.")
p(f"approximate sampling error of a difference averaged over seeds: +/-{se_diff:.1f} tokens (1 se)\n")

p("1. Generated answer length, each method minus fixed on the same seed")
p(f"{'method':12s} {'seed0':>8s} {'seed1':>8s} {'mean':>8s} {'z':>6s} {'%':>7s} {'theory %':>9s} {'share':>6s}")
for m, base in [("micro1", "fixed"), ("micro4", "fixed"), ("global", "fixed"), ("global_clip", "fixed_clip")]:
    d = [R[(m, s)]["gen_mean_len"] - R[(base, s)]["gen_mean_len"] for s in SEEDS]
    ref = sum(R[(base, s)]["gen_mean_len"] for s in SEEDS) / len(SEEDS)
    mean = sum(d) / len(d)
    lim = R[(m, SEEDS[0])]["theory_upper_limit"]
    th = (lim / train_mean - 1) * 100 if lim else None
    share = f"{(mean / ref * 100) / th:6.2f}" if th else "   -"
    label = m if base == "fixed" else "gclip-fclip"
    p(f"{label:12s} {d[0]:+8.2f} {d[1]:+8.2f} {mean:+8.2f} {mean / se_diff:+6.1f} "
      f"{mean / ref * 100:+6.1f}% {(f'{th:+8.1f}%' if th else '        -')} {share}")
p("('theory %' assumes the prompt says nothing about length; 'share' = observed shift / theory limit)\n")

p("2. Median generated length")
for m in ("micro1", "micro4", "global", "fixed", "global_clip", "fixed_clip"):
    p(f"  {m:12s} " + "  ".join(f"{R[(m, s)]['gen_median_len']:5.1f}" for s in SEEDS))

p("\n3. Held-out token-mean loss (the objective the fixed divisor estimates without bias)")
for m in ("micro1", "micro4", "global", "fixed", "global_clip", "fixed_clip"):
    v = [R[(m, s)]["eval_loss"] for s in SEEDS]
    base = "fixed_clip" if m.endswith("clip") else "fixed"
    d = [R[(m, s)]["eval_loss"] - R[(base, s)]["eval_loss"] for s in SEEDS]
    p(f"  {m:12s} " + "  ".join(f"{x:.4f}" for x in v) + f"   minus {base}: " + "  ".join(f"{x:+.4f}" for x in d))

p("\n4. Heavy clipping (Hugging Face default threshold 1.0)")
fr = [r["frac_norm_over_1"] for r in rows]
p(f"  fraction of steps with gradient norm above 1: min {min(fr):.3f}, max {max(fr):.3f}")
for s in SEEDS:
    g, f = R[("global_clip", s)], R[("fixed_clip", s)]
    p(f"  seed {s}: fixed_clip - global_clip: length {f['gen_mean_len'] - g['gen_mean_len']:+.2f}, "
      f"loss {f['eval_loss'] - g['eval_loss']:+.5f}")
for s in SEEDS:
    p(f"  seed {s}: clipping cost to fixed (fixed_clip - fixed loss): {R[('fixed_clip', s)]['eval_loss'] - R[('fixed', s)]['eval_loss']:+.4f}")

p("\n5. Gradient-norm growth power (alpha), for the length-aware clipping threshold")
al = [r["growth_power"] for r in rows]
for m in ("micro1", "micro4", "global", "fixed", "global_clip", "fixed_clip"):
    p(f"  {m:12s} " + "  ".join(f"{R[(m, s)]['growth_power']:.3f}" for s in SEEDS))
p(f"  all runs: mean {sum(al) / len(al):.3f}, range {min(al):.3f}-{max(al):.3f} (synthetic E2: 0.449)")

text = "\n".join(out) + "\n"
(ROOT / "results" / "e4_tests.txt").write_text(text)
print(text)
