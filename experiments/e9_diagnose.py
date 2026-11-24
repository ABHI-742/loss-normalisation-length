"""E9 diagnostic: which training set-up gives finite gradients for TinyLlama + LoRA on a T4?

The E9 smoke test skipped every update even at loss scale 1024 (forward loss finite, backward not).
This runs ONE backward pass on 8 Dolly answers under four set-ups and reports, for each, whether the
loss and the LoRA gradients are finite, at loss scale 1 and 1024, plus time and peak memory.
Run in a Kaggle editor session after the install and %%writefile cells:  !python e9_diagnose.py
"""
import math
import time

import torch

import e4_finetune as e4

MODEL = "TinyLlama/TinyLlama_v1.1"


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    train_ex, _ = e4.build_data(tok, 64, 0, dataset="dolly")
    mu = sum(len(y) for _, y in train_ex) / len(train_ex)
    batch = train_ex[:8]
    print(f"8 answers, {sum(len(p) + len(y) for p, y in batch)} tokens in total; torch {torch.__version__}")

    def trial(name, attn="sdpa", ckpt=True, amp=True):
        for scale in (1.0, 1024.0):
            torch.manual_seed(0)
            torch.cuda.reset_peak_memory_stats()
            model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation=attn)
            if ckpt:
                model.gradient_checkpointing_enable()
                model.config.use_cache = False
                model.enable_input_require_grads()
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=e4.LORA_TARGETS, task_type="CAUSAL_LM")
            model = get_peft_model(model, cfg).to("cuda")
            model.train()
            ids, att, lab = (t.cuda() for t in e4.collate(batch, tok.eos_token_id))
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                dl = e4.doc_losses(model, ids, att, lab)
            loss = dl.sum() / (len(batch) * mu)
            (loss * scale).backward()
            torch.cuda.synchronize()
            with_grad = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
            bad = [n for n, g in with_grad if not torch.isfinite(g).all()]
            lv = float(loss)
            print(f"{name:34s} scale {scale:6.0f}: loss {lv:.4f} ({'finite' if math.isfinite(lv) else 'NON-FINITE'}), "
                  f"{len(with_grad)} params with grads, {len(bad)} non-finite"
                  f"{' e.g. ' + bad[0] if bad else ''}; {time.time() - t0:.1f}s, "
                  f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            del model, dl, loss
            torch.cuda.empty_cache()

    trial("A: current (sdpa, checkpointing, fp16)")
    trial("B: eager attention", attn="eager")
    trial("C: no checkpointing", ckpt=False)
    trial("D: full precision (no fp16)", amp=False)


if __name__ == "__main__":
    main()
