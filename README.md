# Per-batch loss normalisation and learned length: code and results

This repository holds the experiment code, raw results and pre-registrations for the paper
*How per-batch loss normalisation shapes what a language model learns about length*.

Dividing the summed token loss by the number of loss-bearing tokens in each batch weights every
document by w(L) = E[1/(L + S)], where S is the total length of the other documents in its batch. The trained
model's length distribution then becomes q(L) ∝ p(L)·w(L). The experiments test this and the related
results on clipping and on a constant divisor.

## Contents

| Path | What it is |
|---|---|
| `experiments/common.py` | Shared pieces for the synthetic experiments: the tiny transformer, data, exact theory values |
| `experiments/e1_normalisation.py` | E1: length law with two lengths (2 and 8), five normalisers, Adam and SGD |
| `experiments/e2_clipping.py` | E2: clipping, heavy-clipping equivalence, length-aware threshold |
| `experiments/e5_learning_rate.py` | E5 (pre-registered): learning-rate sweep, plus the exploratory 15-seed extension |
| `experiments/e6_many_lengths.py` | E6 (pre-registered): lengths 1–32, full learned distribution and held-out cost |
| `experiments/e0_lengths.py`, `e0_lengths_kaggle.ipynb` | E0: prompt-blind limits for Alpaca, Dolly, GSM8K and UltraChat |
| `experiments/e4_finetune.py` | Fine-tuning SmolLM2-135M on Alpaca. Used for E4, and with flags for E7 and E8 (below) |
| `experiments/e4_finetune_kaggle.ipynb`, `e7_equivalence_kaggle.ipynb`, `e8_small_batch_kaggle.ipynb` | Kaggle notebooks (2 × T4) that ran E4, E7 and E8 |
| `experiments/analyse_*.py` | Analysis scripts; each writes `results/*_tests.txt` |
| `results/*.jsonl`, `results/e0_lengths.json` | Raw per-run results |
| `results/*_tests.txt` | Analysis outputs quoted in the paper |
| `results/*_preregistration.txt` | Pre-registrations, written before the runs of E5/E6, E7 and E8 |
| `results/logs/` | Console logs of the synthetic runs |

## Reproducing

**Synthetic experiments (CPU).** Python 3.11 with PyTorch, NumPy and SciPy. The runs used PyTorch 2.14.1 (CPU), NumPy 2.4.6 and SciPy 1.17.1.

```
python experiments/e1_normalisation.py
python experiments/e2_clipping.py
python experiments/e5_learning_rate.py
python experiments/e6_many_lengths.py
python experiments/analyse_e2.py
python experiments/analyse_e5_e6.py
```

Each script has `--help`. E5 and E6 can be split across workers (`--worker`, `--workers`).

**Fine-tuning experiments (GPU).** Run the notebooks on Kaggle with two T4 GPUs and internet access. They install `datasets` and write the script with `%%writefile`. The equivalent command lines are:

```
# E4: six normalisers, 32 answers per update, seeds 0 and 1
python experiments/e4_finetune.py --parallel --seeds 0 1

# E7 (pre-registered): constant divisor vs global, 32 answers per update
python experiments/e4_finetune.py --parallel --tag e7 --methods global fixed --n-gen 1000 --seeds 100 101 102 103 104 105 106 107

# E8 (pre-registered): constant divisor vs global, 4 answers per update
python experiments/e4_finetune.py --parallel --tag e8 --methods global fixed --group 4 --lr 3.5e-5 --n-gen 1000 --seeds 200 201 202 203 204 205 206 207
```

Then put the merged `*_results.jsonl` files in `results/` and run `analyse_e4.py`, `analyse_e7.py` and `analyse_e8.py`.

In the code, `fixed` is the constant divisor A/(B·μ) and `global` is division by the batch token count.

## Pre-registration

The files `results/e5_e6_preregistration.txt`, `results/e7_preregistration.txt` and `results/e8_preregistration.txt` were written before any run of the experiments they describe. Later additions are marked as post hoc inside the files. E1, E2 and E4 were not pre-registered.
