"""E4: does real fine-tuning shorten answers the way the theory predicts?

Fine-tunes SmolLM2-135M on Alpaca (prompt masked, response + end-of-text are the
loss-bearing tokens), with 32 examples per update, under six methods:
  micro1       - average each example's loss over its own tokens, then average 32
  micro4       - average over micro-batches of 4, then average 8
  global       - divide by the total response tokens of all 32 (the Hugging Face fix)
  fixed        - divide by 32 x average response length (the fixed divisor)
  global_clip  - global, with gradient clipping at 1.0 (the Hugging Face default)
  fixed_clip   - fixed, with gradient clipping at 1.0
Every method runs the same forward passes in chunks of 8; only the per-example
weights differ, which reproduces each method's gradient exactly.

Measured after training: held-out per-token loss, and the length of sampled
answers (temperature 1, no top-k/top-p, so samples follow the model's own
distribution) on held-out prompts. The script also logs gradient size against
token count, giving the gradient-growth power alpha and how often clipping at 1.0
is active.

Theory: E0-style predictions use the marginal length distribution, so they are
upper limits; when the prompt already pins down the answer length, the learned
shift is smaller (the reweighting acts within each prompt's own length spread).

E7 (pre-registered equivalence test, results/e7_preregistration.txt) reuses this
script: --tag e7 --methods global fixed --n-gen 1000 --seeds 100 ... 107.
E8 (pre-registered small-batch test, results/e8_preregistration.txt):
--tag e8 --methods global fixed --group 4 --lr 3.5e-5 --n-gen 1000 --seeds 200 ... 207.
E9 (pre-registered scale test, results/e9_preregistration.txt): TinyLlama 1.1B with LoRA on Dolly:
--tag e9 --model TinyLlama/TinyLlama_v1.1 --dataset dolly --lora-r 16 --grad-ckpt --lr 2e-4
--epochs 1 --n-train 12000 --methods micro1 micro4 fixed --n-gen 1000 --seeds 300 301 302 303.
"""
import json
import math
import os
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

MODEL = "HuggingFaceTB/SmolLM2-135M"
MAX_TOTAL = 512
GROUP = 32
CHUNK = 8
METHODS = ("micro1", "micro4", "global", "fixed", "global_clip", "fixed_clip")
WORK = pathlib.Path("/kaggle/working") if os.path.isdir("/kaggle/working") else pathlib.Path(".")
INPUT = pathlib.Path("/kaggle/input")  # earlier results uploaded as a dataset are found here


def format_prompt(r):
    if r["input"]:
        return f"### Instruction:\n{r['instruction']}\n\n### Input:\n{r['input']}\n\n### Response:\n"
    return f"### Instruction:\n{r['instruction']}\n\n### Response:\n"


DATASETS = {
    # name: (hub id, field for the optional input, field for the response)
    "alpaca": ("tatsu-lab/alpaca", "input", "output"),
    "dolly": ("databricks/databricks-dolly-15k", "context", "response"),
}


def response_ids(tok, text):
    """Token ids of a response, without a start-of-text token (Llama tokenizers add one)."""
    ids = tok(text)["input_ids"]
    if tok.bos_token_id is not None and ids and ids[0] == tok.bos_token_id:
        ids = ids[1:]
    return ids


def build_data(tok, n_train, n_eval, seed=0, dataset="alpaca"):
    from datasets import load_dataset

    hub_id, input_field, output_field = DATASETS[dataset]
    ds = load_dataset(hub_id, split="train").shuffle(seed=seed)
    examples = []
    for row in ds:
        r = {"instruction": row["instruction"], "input": row[input_field], "output": row[output_field]}
        if not r["output"].strip():
            continue
        p = tok(format_prompt(r))["input_ids"]
        y = response_ids(tok, r["output"]) + [tok.eos_token_id]
        if len(p) + len(y) > MAX_TOTAL:
            continue
        examples.append((p, y))
        if len(examples) >= n_train + n_eval:
            break
    return examples[:n_train], examples[n_train:n_train + n_eval]


def collate(batch, pad_id):
    T = max(len(p) + len(y) for p, y in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), T), dtype=torch.long)
    lab = torch.full((len(batch), T), -100, dtype=torch.long)
    for i, (p, y) in enumerate(batch):
        s = p + y
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        lab[i, len(p):len(s)] = torch.tensor(y)
    return ids, att, lab


def doc_losses(model, ids, att, lab):
    """Summed cross-entropy over each example's response tokens."""
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tgt = lab[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                         ignore_index=-100, reduction="none").view(tgt.shape)
    return ce.sum(dim=1)


def example_weights(method, lengths, mu):
    L = torch.tensor(lengths, dtype=torch.float64)
    n = len(L)
    if method.startswith("fixed"):
        return torch.full((n,), 1.0 / (n * mu), dtype=torch.float64)
    if method.startswith("global"):
        return torch.full((n,), 1.0 / float(L.sum()), dtype=torch.float64)
    m = 1 if method == "micro1" else 4
    per = L.view(-1, m).sum(dim=1).repeat_interleave(m)
    return 1.0 / (per * (n // m))


def theory_mean(lengths, B, rng, samples=20000):
    """Exact learned mean length for the batch average at batch size B (marginal, upper limit)."""
    L = np.asarray(lengths)
    uniq, cnt = np.unique(L, return_counts=True)
    p = cnt / cnt.sum()
    others = rng.choice(L, size=(samples, B - 1)).sum(axis=1) if B > 1 else np.zeros(samples)
    w = np.array([np.mean(1.0 / (u + others)) for u in uniq])
    return float(np.sum(p * w * uniq) / np.sum(p * w))


def growth_power(grads):
    pts = [(math.log(n), math.log(a)) for n, a in grads if a > 0 and math.isfinite(a)]
    if len(pts) < 10:
        return None
    xs, ys = zip(*pts)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else None


INIT_SCALE = 2.0 ** 16  # PyTorch's default starting loss scale for fp16; E9 uses a lower one
TRAIN_STATS = {}  # filled by train(): updates, skipped (non-finite) updates, first finite update


def train(model, method, seed, train_ex, mu, epochs, lr, device, pad_id, use_amp):
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=INIT_SCALE, enabled=use_amp)
    updates, skipped, first_finite = 0, 0, None
    n_groups = len(train_ex) // GROUP
    total = epochs * n_groups
    warm = max(1, total // 30)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    model.train()
    grads, norms, over_one, steps = [], [], 0, 0
    for _ in range(epochs):
        order = list(range(len(train_ex)))
        rng.shuffle(order)
        for g in range(n_groups):
            group = [train_ex[i] for i in order[g * GROUP:(g + 1) * GROUP]]
            lengths = [len(y) for _, y in group]
            w = example_weights(method, lengths, mu).float()
            opt.zero_grad(set_to_none=True)
            for c in range(0, GROUP, CHUNK):
                ids, att, lab = (t.to(device) for t in collate(group[c:c + CHUNK], pad_id))
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    dl = doc_losses(model, ids, att, lab)
                scaler.scale((dl * w[c:c + CHUNK].to(device)).sum()).backward()
            scaler.unscale_(opt)
            norm = float(torch.nn.utils.clip_grad_norm_(params, float("inf")))
            updates += 1
            if not math.isfinite(norm):
                skipped += 1  # the gradient scaler skips this update and halves its scale
            elif first_finite is None:
                first_finite = updates
            if math.isfinite(norm):
                # gradient size of the unnormalised summed gradient, for the growth power
                scale = 1.0 / float(w.mean())
                grads.append((sum(lengths), norm * scale))
                norms.append(norm)
                over_one += norm > 1.0
                steps += 1
            if method.endswith("_clip"):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
    # spread of the actual (normalised) gradient size from step to step
    norm_cv = float(np.std(norms) / np.mean(norms)) if norms else None
    TRAIN_STATS.clear()
    TRAIN_STATS.update(updates=updates, skipped_updates=skipped, first_finite_update=first_finite)
    return grads, over_one / max(1, steps), norm_cv


@torch.no_grad()
def eval_loss(model, ex, device, pad_id, use_amp):
    model.eval()
    tot, ntok = 0.0, 0
    for c in range(0, len(ex), 16):
        batch = ex[c:c + 16]
        ids, att, lab = (t.to(device) for t in collate(batch, pad_id))
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            tot += float(doc_losses(model, ids, att, lab).sum())
        ntok += sum(len(y) for _, y in batch)
    return tot / ntok


@torch.no_grad()
def sample_lengths(model, ex, device, eos_id, seed, max_new, use_amp, batch_size=32):
    model.eval()
    torch.manual_seed(seed)
    lengths, censored = [], 0
    for c in range(0, len(ex), batch_size):
        prompts = [p for p, _ in ex[c:c + batch_size]]
        T = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), T), eos_id, dtype=torch.long)
        att = torch.zeros((len(prompts), T), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, T - len(p):] = torch.tensor(p)
            att[i, T - len(p):] = 1
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), do_sample=True,
                                 temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=max_new,
                                 pad_token_id=eos_id, eos_token_id=eos_id)
        for row in out[:, T:].tolist():
            if eos_id in row:
                lengths.append(row.index(eos_id) + 1)
            else:
                lengths.append(len(row))
                censored += 1
    return lengths, censored


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model(model_name, device, lora_r=0, grad_ckpt=False):
    """Full fine-tuning (lora_r=0, as in E4/E7/E8) or LoRA adapters on all projection matrices."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.enable_input_require_grads()
    if lora_r:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                         target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    return model.to(device)


def main(smoke=False, seeds=(0, 1), epochs=2, lr=1e-4, methods=METHODS, n_gen=None, tag="e4", group=32,
         model_name=MODEL, dataset="alpaca", n_train=None, n_eval=None, lora_r=0, grad_ckpt=False,
         init_scale=2.0 ** 16, train_fp32=False):
    from transformers import AutoTokenizer

    global GROUP, INIT_SCALE
    GROUP = group  # answers per update; train() and theory_mean() read it
    INIT_SCALE = init_scale

    device = "cuda"
    use_amp = True
    tok = AutoTokenizer.from_pretrained(model_name)
    eos = tok.eos_token_id
    if smoke:
        n_train, n_eval, n_gen, epochs = 256, 64, 32, 1
    else:
        n_train, n_eval = n_train or 16000, n_eval or 1000
        n_gen = 500 if n_gen is None else min(n_gen, n_eval)
    train_ex, eval_ex = build_data(tok, n_train, n_eval, dataset=dataset)
    train_len = [len(y) for _, y in train_ex]
    ref_len = [len(y) for _, y in eval_ex[:n_gen]]
    mu = float(np.mean(train_len))
    rng = np.random.default_rng(0)
    pred = {"micro1": theory_mean(train_len, 1, rng), "micro4": theory_mean(train_len, 4, rng),
            "global": theory_mean(train_len, GROUP, rng), "fixed": mu}
    print(f"train examples={len(train_ex)} mean response length={mu:.1f} sd={np.std(train_len):.1f}; "
          f"held-out reference mean={np.mean(ref_len):.1f}")
    print("theory upper limits for learned mean length:", {k: round(v, 1) for k, v in pred.items()})
    # Each process writes only its own file (so two GPUs never write the same file);
    # merge() combines them into <tag>_results.jsonl at the end.
    if smoke:
        out = WORK / f"{tag}_smoke.jsonl"
        out.unlink(missing_ok=True)
        done = set()
    else:
        out = WORK / (f"{tag}_results_seeds_" + "_".join(map(str, seeds)) + ".jsonl")
        done = set(load_rows(tag))
        if done:
            print(f"resuming: {len(done)} finished runs found: {sorted(done)}")
    for seed in seeds:
        for method in methods:
            if (method, seed) in done:
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            model = load_model(model_name, device, lora_r, grad_ckpt)
            # train_fp32: full-precision training (TinyLlama's fp16 backward overflows); evaluation and
            # generation below stay in fp16, whose forward pass is finite
            grads, over_one, norm_cv = train(model, method, seed, train_ex, mu, epochs, lr, device, eos,
                                             use_amp and not train_fp32)
            loss = eval_loss(model, eval_ex, device, eos, use_amp)
            if grad_ckpt:
                model.config.use_cache = True  # fast generation; checkpointing only matters in training
            gen, cens = sample_lengths(model, eval_ex[:n_gen], device, eos, seed, MAX_TOTAL, use_amp)
            row = {"method": method, "seed": seed, "epochs": epochs, "lr": lr, "group": GROUP,
                   "model": model_name, "dataset": dataset, "lora_r": lora_r, "n_train": len(train_ex),
                   "train_mean_len": mu, "ref_mean_len": float(np.mean(ref_len)),
                   "gen_mean_len": float(np.mean(gen)), "gen_median_len": float(np.median(gen)),
                   "censored": cens, "n_gen": len(gen), "eval_loss": loss,
                   "theory_upper_limit": pred.get(method.replace("_clip", "")) if not method.endswith("_clip") else None,
                   "growth_power": growth_power(grads), "frac_norm_over_1": over_one,
                   "norm_cv": norm_cv, **TRAIN_STATS, "init_scale": INIT_SCALE, "train_fp32": train_fp32,
                   "minutes": (time.time() - t0) / 60, "gen_lengths": gen}
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            # Also print the row: the log of a committed version is kept even if the
            # output file is lost, and each row can be copied back from it.
            print("RESULT " + json.dumps(row), flush=True)
            print(f"{method:11s} seed={seed} generated mean={row['gen_mean_len']:.1f} "
                  f"(reference {row['ref_mean_len']:.1f}, censored {cens}) eval loss={loss:.4f} "
                  f"alpha={row['growth_power']} |g|>1 on {100 * over_one:.0f}% of steps; "
                  f"skipped {row.get('skipped_updates')} of {row.get('updates')} updates "
                  f"({row['minutes']:.1f} min)", flush=True)
            del model
            torch.cuda.empty_cache()
    print(f"saved {out}", flush=True)


def load_rows(tag="e4"):
    """Finished runs from every results file here, plus earlier results uploaded as a
    Kaggle dataset (Add Input), keyed by (method, seed)."""
    files = sorted(WORK.glob(f"{tag}_results*.jsonl"))
    if INPUT.is_dir():
        files += sorted(INPUT.rglob(f"{tag}_results*.jsonl"))
    rows = {}
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["method"], r["seed"])] = r
    return rows


def merge(tag="e4", seeds=(0, 1), methods=METHODS):
    """Combine all per-seed files into <tag>_results.jsonl and print it, so the committed
    log holds every result even if the output file is lost."""
    rows = load_rows(tag)
    order = sorted(rows, key=lambda k: (k[1], METHODS.index(k[0])))
    name = f"{tag}_results.jsonl"
    out = WORK / name
    out.write_text("".join(json.dumps(rows[k]) + "\n" for k in order))
    missing = [(m, s) for s in sorted({k[1] for k in rows} | set(seeds)) for m in methods if (m, s) not in rows]
    print(f"merged {len(rows)} runs into {out}; missing: {missing or 'none'}")
    print(f"ALL RESULTS (if the output file is lost, copy the lines between the markers into {name})")
    print(f"=== BEGIN {name} ===")
    print(out.read_text(), end="")
    print(f"=== END {name} ===", flush=True)
    return rows


def run_parallel(seeds, n_gpus=None, child_cmd=None, methods=METHODS, n_gen=None, tag="e4", group=32, lr=1e-4,
                 extra=()):
    """Run the seeds side by side, one process per GPU, streaming each process's output
    with a [gpuN] prefix, then merge. child_cmd(seeds) overrides the command (for tests)."""
    import subprocess
    import sys
    import threading

    if n_gpus is None:
        n_gpus = max(1, torch.cuda.device_count())
    seed_sets = [g for g in (list(seeds)[i::n_gpus] for i in range(n_gpus)) if g]
    procs, threads = [], []
    for gpu, gpu_seeds in enumerate(seed_sets):
        if child_cmd:
            cmd = child_cmd(gpu_seeds)
        else:
            cmd = [sys.executable, "-u", os.path.abspath(__file__), "--seeds", *map(str, gpu_seeds),
                   "--methods", *methods, "--tag", tag, "--group", str(group), "--lr", repr(lr)]
            if n_gen is not None:
                cmd += ["--n-gen", str(n_gen)]
            cmd += list(extra)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

        def pump(p=p, gpu=gpu):
            for line in p.stdout:
                print(f"[gpu{gpu}] {line}", end="", flush=True)

        t = threading.Thread(target=pump)
        t.start()
        procs.append(p)
        threads.append(t)
        print(f"started seeds {gpu_seeds} on GPU {gpu}", flush=True)
    codes = [p.wait() for p in procs]
    for t in threads:
        t.join()
    rows = merge(tag, seeds, methods)
    if any(codes):
        print(f"WARNING: exit codes {codes}; rerun the same cell (finished runs are skipped)", flush=True)
    return rows, codes


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run of every method to check the setup")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--parallel", action="store_true", help="one process per GPU, then merge")
    ap.add_argument("--merge", action="store_true", help="only combine the result files")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--n-gen", type=int, default=None, help="held-out prompts to sample (default 500, max 1000)")
    ap.add_argument("--tag", default="e4", help="prefix of the result files (e4, e7, e8)")
    ap.add_argument("--group", type=int, default=32, help="answers per update (default 32)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--model", default=MODEL, help="Hugging Face model id (default SmolLM2-135M)")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="alpaca")
    ap.add_argument("--n-train", type=int, default=None, help="training examples (default 16000)")
    ap.add_argument("--n-eval", type=int, default=None, help="held-out examples (default 1000)")
    ap.add_argument("--lora-r", type=int, default=0, help="LoRA rank; 0 means full fine-tuning")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (saves GPU memory)")
    ap.add_argument("--init-scale", type=float, default=2.0 ** 16,
                    help="starting fp16 loss scale (PyTorch default 65536; lower avoids skipped first updates)")
    ap.add_argument("--train-fp32", action="store_true",
                    help="train in full precision (evaluation and generation stay fp16)")
    a = ap.parse_args()
    methods = tuple(a.methods)
    # settings beyond E4's, passed unchanged to each GPU process
    extra = ["--epochs", str(a.epochs), "--model", a.model, "--dataset", a.dataset, "--lora-r", str(a.lora_r),
             "--init-scale", repr(a.init_scale)]
    if a.n_train:
        extra += ["--n-train", str(a.n_train)]
    if a.n_eval:
        extra += ["--n-eval", str(a.n_eval)]
    if a.grad_ckpt:
        extra.append("--grad-ckpt")
    if a.train_fp32:
        extra.append("--train-fp32")
    if a.merge:
        merge(a.tag, a.seeds, methods)
    elif a.parallel:
        run_parallel(a.seeds, methods=methods, n_gen=a.n_gen, tag=a.tag, group=a.group, lr=a.lr, extra=extra)
    else:
        main(smoke=a.smoke, seeds=tuple(a.seeds), methods=methods, n_gen=a.n_gen, tag=a.tag,
             group=a.group, lr=a.lr, epochs=a.epochs, model_name=a.model, dataset=a.dataset,
             n_train=a.n_train, n_eval=a.n_eval, lora_r=a.lora_r, grad_ckpt=a.grad_ckpt,
             init_scale=a.init_scale, train_fp32=a.train_fp32)
