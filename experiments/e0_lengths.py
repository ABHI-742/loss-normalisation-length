"""E0: how big is the predicted bias for real fine-tuning datasets?

For each dataset, count the loss-bearing tokens of every example (the response
plus one end-of-text token, with the prompt masked, as in standard fine-tuning),
then evaluate the exact formulas for the batch average at micro-batch size B:
  learned mean length  = 1 / (B E[1/S_B])                     (T2)
  bounds               = mu / (1 + var / (B mu l_min or l_max)) (T3)
  extra test loss      = log E[w] - E[log w],  w(L) = E[1/(L + S_{B-1})]   (T6)
Expectations over S are estimated by Monte Carlo with a fixed seed.

Runs anywhere with internet access: pip install datasets transformers
"""
import json
import math
import pathlib

import numpy as np

DATASETS = {
    "alpaca": ("tatsu-lab/alpaca", None, "train", lambda r: [r["output"]]),
    "dolly": ("databricks/databricks-dolly-15k", None, "train", lambda r: [r["response"]]),
    "gsm8k": ("openai/gsm8k", "main", "train", lambda r: [r["answer"]]),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft",
                  lambda r: [m["content"] for m in r["messages"] if m["role"] == "assistant"]),
}
MICRO_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
MAX_EXAMPLES = 20000
MC_SAMPLES = 20000


def response_lengths(tokenizer, name):
    from datasets import load_dataset

    repo, config, split, get = DATASETS[name]
    ds = load_dataset(repo, config, split=split)
    if len(ds) > MAX_EXAMPLES:
        ds = ds.shuffle(seed=0).select(range(MAX_EXAMPLES))
    lengths = []
    for row in ds:
        for text in get(row):
            lengths.append(len(tokenizer(text)["input_ids"]) + 1)  # + end-of-text
    return np.array(lengths, dtype=np.int64)


def predictions(lengths, rng):
    mu = lengths.mean()
    var = lengths.var()
    l_min, l_max = lengths.min(), lengths.max()
    out = {"n": int(len(lengths)), "mean": float(mu), "sd": float(math.sqrt(var)),
           "cv": float(math.sqrt(var) / mu), "min": int(l_min), "max": int(l_max), "by_B": {}}
    uniq, counts = np.unique(lengths, return_counts=True)
    p = counts / counts.sum()
    for B in MICRO_BATCH_SIZES:
        # S_{B-1}: sum of B-1 random lengths, shared Monte Carlo draws
        others = rng.choice(lengths, size=(MC_SAMPLES, B - 1)).sum(axis=1) if B > 1 else np.zeros(MC_SAMPLES)
        w = np.array([np.mean(1.0 / (L + others)) for L in uniq])
        ew = float(np.sum(p * w))
        learned_mean = float(np.sum(p * w * uniq) / ew)
        kl = float(math.log(ew) - np.sum(p * np.log(w)))
        out["by_B"][B] = {
            "learned_mean": learned_mean,
            "shift_pct": 100 * (learned_mean - mu) / mu,
            "lower": float(mu / (1 + var / (B * mu * l_min))),
            "upper": float(mu / (1 + var / (B * mu * l_max))),
            "extra_loss_per_doc": kl,
            "extra_loss_per_token": kl / mu,
        }
    return out


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    rng = np.random.default_rng(0)
    results = {}
    for name in DATASETS:
        try:
            lengths = response_lengths(tokenizer, name)
        except Exception as exc:  # report and continue with the others
            print(f"{name}: failed to load ({exc})")
            continue
        results[name] = predictions(lengths, rng)
        r = results[name]
        print(f"\n{name}: n={r['n']} mean={r['mean']:.1f} sd={r['sd']:.1f} cv={r['cv']:.2f} "
              f"range={r['min']}-{r['max']}")
        for B, v in r["by_B"].items():
            print(f"  B={B:2d}: learned mean={v['learned_mean']:.1f} ({v['shift_pct']:+.1f}%) "
                  f"bounds=[{v['lower']:.1f}, {v['upper']:.1f}] "
                  f"extra loss/token={v['extra_loss_per_token']:.2e} nats")
    out = pathlib.Path("e0_lengths.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"\nsaved {out.resolve()}")


if __name__ == "__main__":
    main()
