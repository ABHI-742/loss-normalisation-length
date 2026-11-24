"""Shared pieces for the synthetic experiments.

Each document is BOS, then L-1 random content tokens, then EOS, so it has exactly
L loss-bearing predictions. Content tokens are uniform and independent of length,
so the only thing the model can learn about length is the end-of-text hazard by
position. With lengths 2 and 8 at probability one half, the true chance of ending
at the second prediction is 0.5, and the theory gives exact predictions for each
loss-normalisation method.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CONTENT = 8
BOS, EOS, PAD = N_CONTENT, N_CONTENT + 1, N_CONTENT + 2
VOCAB = N_CONTENT + 3
LENGTHS = (2, 8)
P_SHORT = 0.5
MU = P_SHORT * LENGTHS[0] + (1 - P_SHORT) * LENGTHS[1]
MAX_LEN = max(LENGTHS)


class TinyLM(nn.Module):
    def __init__(self, d=32, heads=2, layers=2, vocab=VOCAB, max_len=MAX_LEN):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len + 1, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, vocab)
        mask = torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        t = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(t, device=x.device))
        h = self.body(h, mask=self.mask[:t, :t], is_causal=True)
        return self.out(h)


def sample_batch(n, gen):
    """Returns inputs, targets, per-document lengths (loss-bearing tokens)."""
    long_mask = torch.rand(n, generator=gen) >= P_SHORT
    lengths = torch.where(long_mask, LENGTHS[1], LENGTHS[0])
    seq = torch.full((n, MAX_LEN + 1), PAD, dtype=torch.long)
    seq[:, 0] = BOS
    content = torch.randint(0, N_CONTENT, (n, MAX_LEN), generator=gen)
    for i, L in enumerate(lengths.tolist()):
        seq[i, 1:L] = content[i, : L - 1]
        seq[i, L] = EOS
    return seq[:, :-1], seq[:, 1:], lengths


def doc_losses(model, x, y):
    """Summed cross-entropy per document (padding excluded)."""
    logits = model(x)
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=PAD, reduction="none")
    return ce.view(y.shape).sum(dim=1)


@torch.no_grad()
def eos_prob_at_2(model):
    """Learned chance of ending at the second prediction, averaged over first tokens."""
    x = torch.stack([torch.tensor([BOS, c]) for c in range(N_CONTENT)])
    probs = F.softmax(model(x)[:, 1, :], dim=-1)[:, EOS]
    return probs.mean().item()


def theory_q_short(B):
    """Exact learned chance of a short document for the batch average at batch size B."""
    w = {}
    for L in LENGTHS:
        tot = 0.0
        for k in range(B):
            pr = math.comb(B - 1, k) * P_SHORT ** (B - 1 - k) * (1 - P_SHORT) ** k
            tot += pr / (L + LENGTHS[0] * (B - 1 - k) + LENGTHS[1] * k)
        w[L] = tot
    return P_SHORT * w[LENGTHS[0]] / (P_SHORT * w[LENGTHS[0]] + (1 - P_SHORT) * w[LENGTHS[1]])


def lr_at(step, steps, base):
    """Constant for the first half, then cosine decay to 1% of base."""
    half = steps // 2
    if step < half:
        return base
    frac = (step - half) / (steps - half)
    return base * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * frac)))
