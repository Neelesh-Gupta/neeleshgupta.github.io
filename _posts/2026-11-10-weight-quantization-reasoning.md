---
layout: post
title: "If Weight Quantization Degrades Reasoning, Why Is It the Industry Standard for LLM Scaling?"
date: 2026-11-10 09:00:00+0530
description: Why 4-bit weights hurt reasoning, why the industry quantizes anyway, and how PTQ, QAT and mixed-precision QAT fit together, with NumPy code you can run in Google Colab.
tags: quantization llm qat ptq inference
categories: machine-learning
related_posts: false
toc:
  sidebar: left
---

If you've ever pulled an open large language model (LLM) onto your laptop, you've probably seen a name like gemma3:4b-it-q4_K_M and hit enter without a second thought. The "4b" means 4 billion parameters, "it" means instruction-tuned, and q4_K_M is llama.cpp's medium 4-bit format. Ollama lists the same Gemma 3 4B model at 8.6 GB in 16-bit and 3.4 GB as q4_K_M: about 60% smaller, and it fits on machines the full version won't.

So what was in those missing 5 GB? Most of the weights went from 16-bit numbers to roughly 4-bit ones, with far fewer values to choose from. That has to cost something. In a quick chat you may never notice, but research on quantized reasoning models finds that below 4 bits, the hardest problems are where accuracy goes first.

Yet quantized weights are everywhere in production, from 8-bit formats in datacenters to 4-bit on laptops and phones. Part of the reason is hardware. A 70B model needs 140 GB for its 16-bit weights, and NVIDIA's H100, the datacenter GPU much of the industry runs on, has 80 GB. You can't even load that model on one card without shrinking it first.

So lower precision costs reasoning, and the industry runs on it anyway. Why? Below, we'll see what quantization does to a weight, how that damage reaches reasoning, and why the industry takes the deal regardless, from post-training quantization (PTQ) to quantization-aware training (QAT), plus FP8 (8-bit floating point) serving, KV-cache quantization and on-device models.

You only need a rough idea of what a transformer layer is and how training works. There's runnable code along the way, and the notes for running it in Google Colab come right before the first snippet.

## The paradox and the bet

Whatever lets a model keep an argument straight lives in the fine differences between its weights, and a 4-bit weight can take one of only 16 values. But generating text is limited by how fast you can read weights out of memory, so the useful question is: what's the best model I can fit in the memory I've got? Asked that way, a big model at 4 bits usually beats a smaller one at 16 bits ([Dettmers & Zettlemoyer, 2022](https://arxiv.org/abs/2212.09720)). Quantization is a bet on scale. Most models make it with PTQ; labs that can't afford to lose reasoning make it with QAT, more and more often in mixed precision.

## What weight quantization actually is

Models are usually trained in BF16 (bfloat16, short for "brain floating point") or FP16 (standard 16-bit floating point). BF16 spends 1 bit on the sign, 8 on the exponent and 7 on the mantissa, and that exponent lets it represent tiny and huge values with roughly the same *relative* precision. Integer formats don't have one, so INT8 and INT4 (8-bit and 4-bit integers) are evenly spaced grids of 256 and 16 values. FP8 sits in between; its E4M3 variant uses 4 exponent bits and 3 mantissa bits.

![Bit layouts of FP32, BF16, FP16, FP8 (E4M3), INT8 and INT4, colour-coded by sign, exponent, mantissa and integer bits, with bytes per weight and the weight memory of a 70B model in each format]({{ '/assets/blog/weight-quantization-reasoning/number-formats.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Quantizing a tensor means picking a grid and snapping every weight to its nearest point. Symmetric absmax (absolute-maximum) quantization, the simplest scheme, sets the step size $$\Delta$$ from the largest weight on the grid:

$$\Delta = \frac{\max_i |w_i|}{2^{b-1}-1}, \qquad q_i = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w_i}{\Delta}\right),\, -(2^{b-1}-1),\, 2^{b-1}-1\right), \qquad \hat w_i = \Delta\, q_i$$

Here $$b$$ is the bit width, $$w_i$$ the original weight, $$q_i$$ the stored integer code (−7 to 7 for INT4), and $$\hat w_i$$ the value the model computes with. Asymmetric schemes add a zero-point $$z$$, $$\hat w_i = \Delta(q_i - z)$$, and nothing below changes. Grid points sit $$\Delta$$ apart, so no weight moves more than half a step:

$$|w_i - \hat w_i| \le \frac{\Delta}{2}$$

Hold on to that inequality, because it's the whole cost model. Precision lost scales with $$\Delta$$, $$\Delta$$ is set by the largest value on the grid, and going from 8 to 4 bits makes $$\Delta$$ about 18 times bigger.

![Sixty weights on a number line snapping onto a symmetric quantization grid as the bit width drops from 8 to 4 to 2 bits; the yellow rounding errors lengthen and at 2 bits most weights collapse onto 0]({{ '/assets/blog/weight-quantization-reasoning/quant-grid-snap.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

One more choice: how many weights share a $$\Delta$$. Per-tensor uses one scale for the whole matrix and per-channel one per output row. Group-wise, the default for 4-bit LLM weights, gives each block of $$g$$ weights its own scale (usually $$g = 128$$) for 0.125 extra bits per weight. Dettmers & Zettlemoyer found small block sizes were one of only two changes that reliably improved the bits-versus-accuracy trade-off.

### Running the code in Google Colab

Every snippet in this post is ready to paste into a free Google Colab notebook. Here's all the setup you need:

- Open a new notebook at colab.research.google.com and sign in with any Google account. The free tier is enough.
- Keep the default CPU (central processing unit) runtime. The code is plain NumPy, so the free tier's T4 GPU wouldn't speed anything up; under Runtime → Change runtime type, CPU is the right choice. If a GPU runtime is already selected, the code still runs, just on the CPU.
- There's nothing else to configure: no files to upload, no downloads and no API (application programming interface) keys. The only libraries are NumPy and matplotlib, which Colab preinstalls, and cell 1 installs them anyway if they're missing, so the snippets also work in any fresh Python 3.10+ notebook.
- Paste each snippet into its own cell and run them in order, because later cells reuse functions and variables from earlier ones. If the runtime disconnects, start again from cell 1.
- The whole run takes about a minute and peaks at roughly 300 MB of memory, a small fraction of what the free tier gives you.

Cell 1 sets up that environment.

```python
# Tested with numpy 2.0.2 to 2.4.6 and matplotlib 3.8.4 to 3.11.2: identical results.
# Google Colab already ships both, so this line normally installs nothing and needs no runtime restart.
%pip install -q "numpy>=2.0" "matplotlib>=3.8"
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

SEED = 0
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({"figure.figsize": (8, 3.6), "figure.dpi": 120,
                     "axes.spines.top": False, "axes.spines.right": False})
print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__} | seed={SEED}")
```
```text
numpy 2.3.3 | matplotlib 3.10.8 | seed=0
```

The `%pip` line only installs NumPy and matplotlib if they're missing or older than the minimum versions, so on Colab it's normally a no-op and never asks you to restart. After that come the imports, a fixed random seed (0) so every run produces the same numbers, `rng`, the random generator the later cells draw from, and a few print and plot settings.

Running it prints a single line with your library versions and the seed. Don't worry if your versions differ from mine: I ran the whole post on NumPy 2.0.2 through 2.4.6 and matplotlib 3.8.4 through 3.11.2, including a fresh environment where cell 1 had to install both, and every cell printed exactly the same numbers.

Cell 2 turns the equations into code and tests the $$\Delta/2$$ bound.

```python
def quantize(w: np.ndarray, bits: int, group_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric absmax quantization. Returns integer codes and one scale (Δ) per group."""
    qmax = 2 ** (bits - 1) - 1                      # INT8 -> 127, INT4 -> 7
    flat = w.reshape(-1, group_size) if group_size else w.reshape(1, -1)
    scale = np.abs(flat).max(axis=1, keepdims=True) / qmax   # Δ: the grid's step size
    codes = np.clip(np.round(flat / scale), -qmax, qmax)
    return codes, scale

def fake_quant(w: np.ndarray, bits: int, group_size: int | None = None) -> np.ndarray:
    """Quantize then dequantize: the float weights a low-bit model effectively uses."""
    codes, scale = quantize(w, bits, group_size)
    return (codes * scale).reshape(w.shape)

W = rng.normal(0.0, 0.02, size=(512, 2048))         # a typical-looking LLM weight matrix
for bits in (8, 4):
    codes, delta = quantize(W, bits)
    err = np.abs(W - fake_quant(W, bits))
    print(f"INT{bits}: levels={len(np.unique(codes)):3d}  Δ={delta.item():.5f}  "
          f"max|err|={err.max():.5f} <= Δ/2={delta.item() / 2:.5f}: {err.max() <= delta.item() / 2 + 1e-12}  "
          f"MSE={np.mean(err ** 2):.2e}")

fig, ax = plt.subplots()
ax.hist(W.ravel(), bins=200, density=True, alpha=0.5, label="original (BF16-like)")
vals, counts = np.unique(fake_quant(W, 4), return_counts=True)
ax.vlines(vals, 0, counts / counts.max() * ax.get_ylim()[1], colors="crimson", label="INT4 grid (bar height = count)")
ax.set(xlabel="weight value", ylabel="density", title="16-bit weights collapse onto 15 INT4 levels")
ax.legend(); plt.show()
```
```text
INT8: levels=235  Δ=0.00079  max|err|=0.00039 <= Δ/2=0.00039: True  MSE=5.16e-08
INT4: levels= 15  Δ=0.01428  max|err|=0.00714 <= Δ/2=0.00714: True  MSE=1.70e-05
```
![Output of cell 2]({{ '/assets/blog/weight-quantization-reasoning/cell-02-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Two small functions do the work. `quantize` sets each group's step $$\Delta$$ (`scale` in the code) to the group's largest absolute value divided by the largest code (127 for INT8, 7 for INT4), then rounds every weight to an integer code $$q$$. `fake_quant` multiplies the codes back by the step to get $$\hat w$$. The rest of the cell quantizes a 512 × 2048 matrix drawn from $$\mathcal N(0, 0.02^2)$$, a typical scale for LLM weights, to INT8 and INT4, prints the levels used, step size, largest error and MSE (mean squared error) for each, and plots the original weights against the INT4 values.

Both printed lines end in `True`. The worst error really is exactly $$\Delta/2$$. INT4's step (0.01428) is 18 times INT8's (0.00079), and since squared error grows with $$\Delta^2$$, INT4's MSE (1.70e-05) is about 330 times INT8's (5.16e-08). In the plot, a smooth blue bell curve collapses onto a handful of red spikes.

## Why quantization degrades reasoning

Why should an error that small matter? A dot product sums thousands of those $$\Delta/2$$ errors, and the sum acts like noise on every layer's output. At INT8 it's negligible. At INT4 it isn't, and it reaches reasoning in three ways.

### Outliers stretch the grid

Real LLM numbers aren't well-behaved Gaussians. [Dettmers et al. (2022)](https://arxiv.org/abs/2208.07339) found that by roughly 6.7B parameters, a handful of hidden dimensions carry "emergent outlier features" in every layer: values far larger than the rest that dominate attention and the model's predictions. Since $$\Delta$$ is set by the largest value on the grid, an outlier $$k$$ times bigger than its neighbours makes $$\Delta$$ roughly $$k$$ times bigger, and once $$\Delta/2$$ exceeds a typical weight, nearly everything rounds to zero. Strictly, those features live in the *activations*. W4A16 (4-bit weights, 16-bit activations) lets them pass straight through, while W8A8 (8-bit weights and activations) has to quantize them. Weights have milder outliers of their own, and the arithmetic is identical.

![One weight grows into an outlier from 0.05 to 1.0; the INT4 grid step widens from 0.007 to 0.14 and the share of ordinary weights rounded to 0 climbs from 22% to 100%]({{ '/assets/blog/weight-quantization-reasoning/outlier-stretch.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Cell 3 plants an artificial outlier and measures the damage.

```python
W_out = W.copy()
W_out[:, 7] *= 20                                    # one input channel 20x larger, like an LLM outlier feature
normal = np.ones_like(W, dtype=bool); normal[:, 7] = False

def report(name: str, w_hat: np.ndarray, delta: float) -> tuple[float, float]:
    mse = np.mean((W_out[normal] - w_hat[normal]) ** 2)        # error on the *ordinary* weights
    zeroed = np.mean(w_hat[normal] == 0) * 100
    print(f"{name:<32} Δ={delta:.4f}  ordinary weights -> 0: {zeroed:5.1f}%  MSE={mse:.2e}")
    return mse, zeroed

_, d_clean = quantize(W, 4)
_, d_out = quantize(W_out, 4)
print(f"outlier stretched per-tensor Δ by {d_out.item() / d_clean.item():.1f}x\n")
cases = {
    "clean, per-tensor INT4": (fake_quant(W, 4), d_clean.item()),
    "outlier, per-tensor INT4": (fake_quant(W_out, 4), d_out.item()),
    "outlier, group-wise INT4 (g=128)": (fake_quant(W_out, 4, 128), np.median(quantize(W_out, 4, 128)[1])),
}
results = {k: report(k, w_hat, d) for k, (w_hat, d) in cases.items()}   # g=128 row shows the median Δ

fig, (a1, a2) = plt.subplots(1, 2)
labels, colors = ["clean\nper-tensor", "outlier\nper-tensor", "outlier\ngroup-wise g=128"], ["grey", "crimson", "seagreen"]
a1.bar(labels, [m for m, _ in results.values()], color=colors)
a1.set(yscale="log", ylabel="MSE on ordinary weights (log)")
a2.bar(labels, [z for _, z in results.values()], color=colors)
a2.set(ylabel="% ordinary weights rounded to 0")
for a in (a1, a2): a.tick_params(axis="x", labelsize=8)
fig.suptitle("One outlier channel vs. INT4 scaling granularity"); fig.tight_layout(); plt.show()
```
```text
outlier stretched per-tensor Δ by 15.0x

clean, per-tensor INT4           Δ=0.0143  ordinary weights -> 0:  27.9%  MSE=1.70e-05
outlier, per-tensor INT4         Δ=0.2148  ordinary weights -> 0: 100.0%  MSE=4.01e-04
outlier, group-wise INT4 (g=128) Δ=0.0081  ordinary weights -> 0:  19.0%  MSE=1.57e-05
```
![Output of cell 3]({{ '/assets/blog/weight-quantization-reasoning/cell-03-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

The setup is deliberately crude: take the matrix from cell 2 and multiply one input column (column 7) by 20 to stand in for an outlier channel. A `report` helper then looks only at the other, ordinary weights and prints the step size, the share rounded to exactly zero and the MSE for three setups: clean per-tensor INT4, per-tensor INT4 with the outlier, and group-wise INT4 ($$g = 128$$) with the outlier. Two bar charts put them side by side.

The first line says the outlier stretched the per-tensor step 15.0 times. Even the clean per-tensor case zeroes 27.9% of weights, higher than most people would guess. With the outlier, the step jumps from 0.0143 to 0.2148, `100.0%` of ordinary weights become zero, and the MSE of 4.01e-04 equals the weights' own variance ($$0.02^2$$). They've been wiped out. Group-wise INT4 keeps the damage to one group per row and lands at an MSE of 1.57e-05, slightly better than clean per-tensor. In both charts the red middle bar towers over the others, which is why 4-bit LLM weights are quantized group-wise.

### Errors compound with depth

Each quantized layer adds its noise to an input that already carries the noise of every layer before it. Cell 4 measures how fast that builds up.

```python
DEPTH, DIM = 24, 512
layers = [rng.normal(0.0, np.sqrt(2.0 / DIM), size=(DIM, DIM)) for _ in range(DEPTH)]  # He init keeps norms stable
x0 = rng.normal(size=(64, DIM))

def run_stack(ws: list[np.ndarray], x: np.ndarray) -> list[np.ndarray]:
    """Linear + ReLU blocks; returns the hidden state after every layer."""
    states = []
    for w in ws:
        x = np.maximum(x @ w, 0.0)
        states.append(x)
    return states

ref = run_stack(layers, x0)
curves = {}
for bits in (8, 4):
    q_states = run_stack([fake_quant(w, bits, 128) for w in layers], x0)
    curves[bits] = [np.linalg.norm(q - r) / np.linalg.norm(r) for q, r in zip(q_states, ref)]
    print(f"INT{bits} (g=128) relative error: layer 1 = {curves[bits][0]:.3f}, "
          f"layer 12 = {curves[bits][11]:.3f}, layer 24 = {curves[bits][-1]:.3f}")

fig, ax = plt.subplots()
for bits, c in zip((8, 4), ("steelblue", "crimson")):
    ax.plot(range(1, DEPTH + 1), curves[bits], marker="o", ms=3, color=c, label=f"INT{bits}")
ax.set(yscale="log", xlabel="layer", ylabel="relative error vs. full precision (log)",
       title="Rounding noise compounds with depth")
ax.legend(); plt.show()
```
```text
INT8 (g=128) relative error: layer 1 = 0.006, layer 12 = 0.023, layer 24 = 0.032
INT4 (g=128) relative error: layer 1 = 0.115, layer 12 = 0.393, layer 24 = 0.477
```
![Output of cell 4]({{ '/assets/blog/weight-quantization-reasoning/cell-04-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Our stand-in for a model is a stack of 24 random 512 × 512 layers. They use He initialization (weights scaled by the square root of 2 over the layer width, which keeps activations a stable size). `run_stack` pushes 64 random inputs through them. Each layer is a matrix multiply followed by ReLU (rectified linear unit, which zeroes negative values). The cell runs the stack with full-precision, group-wise INT8 and group-wise INT4 weights, prints the relative error at layers 1, 12 and 24, and plots it for every layer.

INT8 barely moves: 0.006, 0.023, 0.032. INT4 goes 0.115, 0.393, 0.477, so by the last layer the hidden state is off by nearly half its own size, and its red line sits far above INT8's. Residual connections and normalization damp this a lot in real transformers (it's why 4-bit models work at all), but the direction is the same.

### Reasoning compounds per-step errors

A chain of thought adds another kind of depth: steps. If a chain has $$n$$ independent steps, each right with probability $$p$$, the whole chain is right with probability

$$P(\text{chain correct}) = p^{\,n}$$

Cell 5 puts numbers on it.

```python
steps = np.arange(1, 41)
for p in (0.99, 0.97, 0.95):
    print(f"per-step accuracy {p:.2f}: " + "  ".join(f"n={n:2d} -> {p ** n:5.1%}" for n in (5, 10, 20, 40)))

fig, ax = plt.subplots()
for p, c in zip((0.99, 0.97, 0.95), ("steelblue", "orange", "crimson")):
    ax.plot(steps, p ** steps, color=c, label=f"p = {p}")
ax.axvline(20, ls="--", color="grey", lw=1)
ax.set(xlabel="number of reasoning steps n", ylabel="P(whole chain correct) = pⁿ",
       title="Small per-step losses become large chain failures")
ax.legend(); plt.show()
```
```text
per-step accuracy 0.99: n= 5 -> 95.1%  n=10 -> 90.4%  n=20 -> 81.8%  n=40 -> 66.9%
per-step accuracy 0.97: n= 5 -> 85.9%  n=10 -> 73.7%  n=20 -> 54.4%  n=40 -> 29.6%
per-step accuracy 0.95: n= 5 -> 77.4%  n=10 -> 59.9%  n=20 -> 35.8%  n=40 -> 12.9%
```
![Output of cell 5]({{ '/assets/blog/weight-quantization-reasoning/cell-05-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

This one is pure arithmetic: the formula above for per-step accuracies of 99%, 97% and 95% at 5, 10, 20 and 40 steps, plus a plot of all three curves up to 40 steps.

At 99% per step, a 20-step chain is right 81.8% of the time. At 97%, only 54.4%. Two points lost per step become 27 points over 20 steps, and the curves keep pulling apart as chains get longer. Real steps aren't independent, but this is how one quantized model can look fine on short benchmarks and worse at long maths and code.

![Two 20-step reasoning chains: at 99% per-step accuracy the whole chain is right 82% of the time, at 97% only 54%]({{ '/assets/blog/weight-quantization-reasoning/chain-compounding.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

### What the evidence says

[Liu et al. (2025)](https://arxiv.org/abs/2504.04823) ran the first systematic study of quantized reasoning models, from 1.5B to 70B parameters, on AIME (the American Invitational Mathematics Examination), MATH-500 (500 competition maths problems), GPQA (Graduate-Level Google-Proof Q&A, a hard science benchmark) and LiveCodeBench (a coding benchmark). W8A8 and W4A16 came out close to lossless. Lower bit widths brought significant accuracy risks, with model size, origin and task difficulty deciding how much. And post-training quantization hurts more the more data a model was trained on ([Kumar et al., 2024](https://arxiv.org/abs/2411.04330); [Ouyang et al., 2024](https://arxiv.org/abs/2411.17691)).

I call the result of naive post-hoc rounding "quantization shock" (my label, not a standard term): the network gets noisy weights after training ends and never gets a chance to adjust.

## Why the industry takes the bet anyway

### The bet on scale

So why does anyone quantize? Every new token means reading every weight from GPU memory once, with little arithmetic per weight, so at batch size 1 memory bandwidth is the speed limit:

$$\text{weight memory} = N_\text{params} \times \frac{b}{8}\ \text{bytes}, \qquad \text{tokens/s} \lesssim \frac{\text{memory bandwidth}}{\text{weight memory}}$$

Cell 6 runs those formulas for real hardware.

```python
BANDWIDTH_TB_S, GPU_GB = 3.35, 80             # H100 SXM: HBM3 bandwidth and capacity
formats = {"FP16": 16, "FP8": 8, "INT4 (g=128)": 4 + 16 / 128}   # group-wise adds one FP16 scale per 128 weights

print(f"{'model':>5} {'format':>13} {'weights GB':>11} {'fits 80 GB?':>12} {'max tok/s @ batch 1':>20}")
for params_b in (7, 13, 70):
    for fmt, bits in formats.items():
        gb = params_b * 1e9 * bits / 8 / 1e9
        tok_s = BANDWIDTH_TB_S * 1e12 / (gb * 1e9)   # each new token reads every weight once
        print(f"{params_b:>4}B {fmt:>13} {gb:>11.1f} {'yes' if gb < GPU_GB else 'NO':>12} {tok_s:>20.0f}")
print("\n(weights only: the KV cache and activations need room on top of this)")
```
```text
model        format  weights GB  fits 80 GB?  max tok/s @ batch 1
   7B          FP16        14.0          yes                  239
   7B           FP8         7.0          yes                  479
   7B  INT4 (g=128)         3.6          yes                  928
  13B          FP16        26.0          yes                  129
  13B           FP8        13.0          yes                  258
  13B  INT4 (g=128)         6.7          yes                  500
  70B          FP16       140.0           NO                   24
  70B           FP8        70.0          yes                   48
  70B  INT4 (g=128)        36.1          yes                   93

(weights only: the KV cache and activations need room on top of this)
```

No plot this time, just a table. The cell takes an NVIDIA H100 SXM (the server-module version of the card, with 80 GB of memory and 3.35 TB/s, terabytes per second, of bandwidth) and works out weight memory, whether it fits, and the speed ceiling in tokens per second for 7B, 13B and 70B models in FP16, FP8 and group-wise INT4. INT4 is counted as 4.125 bits per weight so the scales are included.

Look at the 70B rows. In FP16 the model needs 140.0 GB and doesn't fit; in INT4 it needs 36.1 GB, leaves room for the KV cache, and its ceiling rises from 24 to 93 tokens/s. So on one card, your realistic options are a 13B model at full precision (26.0 GB) or a 70B model at 4 bits.

![Three 80 GB GPUs: a 13B FP16 model uses 26 GB, a 70B INT4 model uses 36 GB, and a 70B FP16 model at 140 GB overflows the card]({{ '/assets/blog/weight-quantization-reasoning/scale-bet.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Which should you pick? Across more than 35,000 experiments on models from 19M to 176B parameters, holding total model bits fixed, Dettmers & Zettlemoyer found 4-bit precision "almost universally optimal" for zero-shot accuracy (accuracy on tasks the model wasn't given examples for). The bigger, coarser model usually wins, because the rounding noise costs less than the extra parameters buy you.

### Post-training quantization: the default

The cheapest way to make the bet is PTQ: quantize a finished 16-bit model, usually with a small calibration set and no retraining. It takes minutes to hours, which is why most quantized models you can download were made this way, and it's much smarter than rounding to nearest:

- GPTQ ([Frantar et al., 2022](https://arxiv.org/abs/2210.17323)), a PTQ method built for generative pre-trained transformers, quantizes one column at a time. It adjusts the remaining weights to cancel the error, using approximate second-order information.
- AWQ (Activation-aware Weight Quantization, [Lin et al., 2023](https://arxiv.org/abs/2306.00978)) scales up the few weight channels that see large activations, so the weights that matter most lose the least.
- SmoothQuant ([Xiao et al., 2022](https://arxiv.org/abs/2211.10438)) makes W8A8 practical by shifting the difficulty from activations into weights.
- Rotation methods (QuaRot, [Ashkboos et al., 2024](https://arxiv.org/abs/2404.00456); SpinQuant, [Liu et al., 2024](https://arxiv.org/abs/2405.16406)) spread outliers evenly across channels without changing the output.

SmoothQuant and the rotation methods are what handle activation outliers; QAT of the weights alone leaves them in place. PTQ's real limitation is that the model never sees its own rounding error. At 8 bits or W4A16 that rarely matters. Below that, especially on long reasoning, it does.

![Two pipelines: PTQ calibrates and rounds a finished BF16 model to INT4, while QAT trains with fake-quantized weights, straight-through gradients and FP32 master weights before exporting INT4]({{ '/assets/blog/weight-quantization-reasoning/ptq-vs-qat.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

### Quantization-aware training: the frontier

QAT puts the rounding inside the training loop ([Jacob et al., 2017](https://arxiv.org/abs/1712.05877)). The model keeps full-precision *master weights*, and each step does three things:

1. It runs the forward pass on fake-quantized weights (`fake_quant` from cell 2), so the loss sees what the deployed model will see.
2. It passes the gradient straight through the rounding. `round()` has zero gradient almost everywhere, so the straight-through estimator, or STE ([Bengio et al., 2013](https://arxiv.org/abs/1308.3432)), treats it as the identity, $$\partial \hat w / \partial w \approx 1$$, and applies the gradient for $$\hat w$$ to $$w$$.
3. It updates the master weights, which drift in full precision until they cross a rounding boundary and, over training, settle into values that survive the grid.

![The forward pass uses the INT3 staircase while the backward pass uses the straight-through estimator, a gradient of 1 everywhere, instead of the true gradient of 0]({{ '/assets/blog/weight-quantization-reasoning/qat-ste.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

![A full-precision master weight climbs from 0.10 to 0.385 under straight-through gradient steps while its deployed quantized copy jumps from 0 to 0.25 to 0.5 and the loss drops to zero]({{ '/assets/blog/weight-quantization-reasoning/qat-master-weights.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Does it help? Cell 7 compares PTQ and QAT head to head.

```python
data_rng = np.random.default_rng(1)
X = data_rng.normal(size=(2048, 8))                  # toy regression task from a random "teacher" network
teacher = [data_rng.normal(0, 1.2 / np.sqrt(m), (m, n)) for m, n in ((8, 32), (32, 1))]
y = np.tanh(X @ teacher[0]) @ teacher[1] + 0.01 * data_rng.normal(size=(2048, 1))
X_tr, y_tr, X_te, y_te = X[:1536], y[:1536], X[1536:], y[1536:]
QAT_BITS = 4

def forward(ws: list[np.ndarray], bs: list[np.ndarray], x: np.ndarray) -> list[np.ndarray]:
    acts = [x]
    for i, (w, b) in enumerate(zip(ws, bs)):
        z = acts[-1] @ w + b
        acts.append(np.tanh(z) if i < len(ws) - 1 else z)
    return acts

def train(bits: int | None, steps: int = 800, lr: float = 3e-3) -> tuple[list, list]:
    init = np.random.default_rng(2)
    sizes = [8, 64, 64, 1]
    ws = [init.normal(0, 1 / np.sqrt(m), (m, n)) for m, n in zip(sizes, sizes[1:])]   # FP32 master weights
    bs = [np.zeros(n) for n in sizes[1:]]
    params = ws + bs
    m1, m2 = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
    for t in range(1, steps + 1):
        wq = [fake_quant(w, bits) if bits else w for w in ws]     # forward pass sees the INT grid
        acts = forward(wq, bs, X_tr)
        g = 2 * (acts[-1] - y_tr) / len(X_tr)
        grads_w, grads_b = [], []
        for i in reversed(range(len(wq))):
            if i < len(wq) - 1:
                g = g * (1 - acts[i + 1] ** 2)
            grads_w.insert(0, acts[i].T @ g); grads_b.insert(0, g.sum(0))
            g = g @ wq[i].T
        # STE: treat round() as identity, so dL/dW_quant is applied straight to the master weights
        for p, gr, a, b_ in zip(params, grads_w + grads_b, m1, m2):
            a[:] = 0.9 * a + 0.1 * gr; b_[:] = 0.999 * b_ + 0.001 * gr ** 2
            p -= lr * (a / (1 - 0.9 ** t)) / (np.sqrt(b_ / (1 - 0.999 ** t)) + 1e-8)
    return ws, bs

def test_mse(ws: list, bs: list, bits: int | None) -> float:
    wq = [fake_quant(w, bits) if bits else w for w in ws]
    return float(np.mean((forward(wq, bs, X_te)[-1] - y_te) ** 2))

fp_ws, fp_bs = train(bits=None)
qat_ws, qat_bs = train(bits=QAT_BITS)
scores = {"FP32 baseline": test_mse(fp_ws, fp_bs, None),
          f"PTQ INT{QAT_BITS} (round after training)": test_mse(fp_ws, fp_bs, QAT_BITS),
          f"QAT INT{QAT_BITS} (STE, fake-quant in loop)": test_mse(qat_ws, qat_bs, QAT_BITS)}
for k, v in scores.items():
    print(f"{k:<38} test MSE = {v:.4f}   ({v / scores['FP32 baseline']:.1f}x baseline)")

fig, ax = plt.subplots()
ax.bar(["FP32\nbaseline", f"PTQ\nINT{QAT_BITS}", f"QAT\nINT{QAT_BITS}"], list(scores.values()),
       color=["grey", "crimson", "seagreen"])
ax.set(ylabel="test MSE", title="Training with the grid in the loop recovers most of the loss")
plt.show()
```
```text
FP32 baseline                          test MSE = 0.0031   (1.0x baseline)
PTQ INT4 (round after training)        test MSE = 0.0294   (9.5x baseline)
QAT INT4 (STE, fake-quant in loop)     test MSE = 0.0052   (1.7x baseline)
```
![Output of cell 7]({{ '/assets/blog/weight-quantization-reasoning/cell-07-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

This is the longest snippet and the slowest, at 10 to 20 seconds. It builds a toy regression task (targets from a small random "teacher" network, with 1,536 training and 512 test examples) and a student MLP (multi-layer perceptron) with layers of 8, 64, 64 and 1 units and tanh (hyperbolic tangent) activations. The student is trained twice from the same starting weights, for 800 steps of the Adam optimizer: once in full precision, and once with QAT, using fake-quantized INT4 weights and the STE. Then the cell tests the float model, that model rounded to INT4 (PTQ), and the QAT model.

Three numbers come out. The FP32 (32-bit floating point) baseline scores a test MSE of 0.0031, PTQ at INT4 rises to 0.0294 (9.5 times the baseline), and QAT at INT4 reaches 0.0052, just 1.7 times. In the chart, the red PTQ bar towers over the other two. Same bits, same architecture: the only difference is that one model saw the grid while it learned.

Real releases follow suit. Google's [Gemma 3 QAT checkpoints](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/) shrink the 27B model from 54 GB in BF16 to 14.1 GB in int4. Meta's [quantized Llama 3.2](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/) shipped QAT with LoRA (low-rank adaptation) adaptors as its high-accuracy option. Moonshot's [Kimi K2 Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) applied INT4 weight-only QAT during post-training because long reasoning traces made ordinary quantization lose too much. My rule of thumb: PTQ when the model is finished, QAT when the low-bit model *is* the product.

### Mixed-precision QAT: spending bits where they matter

Not every layer is equally sensitive. The token embedding and the LM head (the language-model head, which turns the final hidden state into word probabilities) sit right on the path from input tokens to output, so their errors reach everything downstream. The middle blocks hold most of the parameters but only add updates to the residual stream, and much of their capacity is redundant. So mixed-precision schemes keep the embeddings and head at BF16, FP8 or INT8 and push the middle blocks to INT4, sometimes INT3 (3-bit integers). Below that, the training changes character: [ParetoQ (Liu et al., 2025)](https://arxiv.org/abs/2502.02631) found that at 3 bits and above, quantized models stay close to their pre-trained distributions, while at 2 bits and below the representations change drastically.

The recipe has three steps. Measure each layer's sensitivity by quantizing it alone, hand out bits under a fixed average budget, then run QAT with that bit map. Cell 8 does the first two.

```python
N_LAYERS, D, ALPHA = 12, 512, 0.2
stack = [rng.normal(0.0, np.sqrt(2.0 / D), size=(D, D)) for _ in range(N_LAYERS)]
x_in = rng.normal(size=(64, D))

def run_residual(ws: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    """Transformer-shaped toy: input projection -> residual blocks -> output head."""
    h = x @ ws[0]                                   # "embedding": on the direct path
    for w in ws[1:-1]:
        h = h + ALPHA * np.maximum(h @ w, 0.0)      # middle blocks write small updates to a residual stream
    return h @ ws[-1]                               # "LM head": on the direct path

ref_out = run_residual(stack, x_in)

def stack_error(bits_per_layer: list[int | None]) -> float:
    ws = [w if b is None else fake_quant(w, b, 128) for w, b in zip(stack, bits_per_layer)]
    out = run_residual(ws, x_in)
    return float(np.linalg.norm(out - ref_out) / np.linalg.norm(ref_out))

# 1) sensitivity: push ONE layer to INT3, keep the rest in full precision
sens = [stack_error([3 if j == i else None for j in range(N_LAYERS)]) for i in range(N_LAYERS)]
# 2) same 4.0-bit average budget: 2 layers @ 8, 2 @ 4, 8 @ 3 = 48 bits = 12 x 4
order = np.argsort(sens)[::-1]
alloc = [3] * N_LAYERS
for rank, i in enumerate(order[:4]):
    alloc[i] = 8 if rank < 2 else 4
print("sensitivity (INT3, one layer):", " ".join(f"{s:.3f}" for s in sens))
print("mixed bits per layer:         ", alloc, f"avg = {np.mean(alloc):.2f}")
print(f"uniform INT4 output error = {stack_error([4] * N_LAYERS):.3f}")
print(f"mixed    (avg 4 bits)     = {stack_error(alloc):.3f}")

fig, ax = plt.subplots()
bar_colors = {8: "seagreen", 4: "orange", 3: "crimson"}
ax.bar(range(N_LAYERS), sens, color=[bar_colors[b] for b in alloc])
for b, c in bar_colors.items():
    ax.bar(0, 0, color=c, label=f"assigned INT{b}")
ax.set_xticks(range(N_LAYERS), ["emb"] + [str(i) for i in range(1, N_LAYERS - 1)] + ["head"])
ax.set(yscale="log", xlabel="layer", ylabel="output error, only this layer INT3 (log)",
       title="Sensitivity scan decides where the bits go")
ax.legend(); plt.show()
```
```text
sensitivity (INT3, one layer): 0.183 0.035 0.036 0.035 0.037 0.038 0.039 0.042 0.046 0.047 0.050 0.287
mixed bits per layer:          [8, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 8] avg = 4.00
uniform INT4 output error = 0.157
mixed    (avg 4 bits)     = 0.116
```
![Output of cell 8]({{ '/assets/blog/weight-quantization-reasoning/cell-08-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

The toy here is shaped like a transformer: an input projection ("emb"), ten middle blocks that each add a 0.2-scaled ReLU update to a running hidden state, and an output projection ("head"). `stack_error` quantizes chosen layers group-wise and returns the final output's relative error. The cell pushes one layer at a time to INT3 to measure sensitivity, then gives INT8 to the two most sensitive layers, INT4 to the next two and INT3 to the rest, an average of exactly 4 bits ($$2 \times 8 + 2 \times 4 + 8 \times 3 = 12 \times 4$$). It prints the results and plots each layer's sensitivity, coloured by its bits.

The sensitivities tell the story. "emb" scores 0.183 and "head" 0.287, against 0.035 to 0.050 for the middle blocks, so the scan finds the input and output projections without being told. The allocation comes out as [8, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 8], averaging 4.00. Uniform INT4 gives an output error of 0.157; the mixed allocation gives 0.116, about 26% less for the same memory. In this toy the sensitivity comes purely from where a layer sits; real models have to be measured one by one, but the method is the same.

![A 32-block transformer coloured by bit width: embedding and LM head at BF16 or FP8 on the residual stream's direct path, blocks 1-2 and 31-32 at INT8, and the middle blocks at INT4 or INT3]({{ '/assets/blog/weight-quantization-reasoning/mixed-precision-map.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

## Other methods in brief

### High-precision cloud serving

In datacenters, FP8 ([Micikevicius et al., 2022](https://arxiv.org/abs/2209.05433)) is a common alternative to integer weights. It keeps an exponent, so it follows the dynamic range of weights and activations better than INT8. It also runs natively on NVIDIA's Hopper generation (the H100's) and typically serves close to BF16 quality. Blackwell, the next generation, adds 4-bit floating-point formats with a shared scale per small block: MXFP4 (microscaling 4-bit floating point, [Rouhani et al., 2023](https://arxiv.org/abs/2310.10537)) and NVFP4, NVIDIA's own 4-bit floating-point format. OpenAI's [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) ships its MoE (mixture-of-experts) weights in MXFP4, which is how it runs on one 80 GB GPU, and NVFP4 now works even for pretraining ([NVIDIA, 2025](https://arxiv.org/abs/2509.25149)).

### KV-cache quantization

The model stores the attention keys and values for every token it has seen. On long contexts, that cache can outgrow the weights. Quantizing it to FP8 or lower frees memory for longer contexts and more users without touching the weights, and KIVI ([Liu et al., 2024](https://arxiv.org/abs/2402.02750)) gets all the way down to 2 bits by quantizing keys per channel and values per token, with 2.6× less peak memory and up to 4× larger batches.

### Edge and on-device models

On laptops and phones memory is the hard limit, so 4-bit is the default. GGUF, the model file format used by llama.cpp, holds formats like the q4_K_M from the top of this post, mostly made with PTQ. QAT checkpoints are arriving too: Gemma 3's QAT models ship as Q4_0 GGUFs (llama.cpp's basic 4-bit format), and Meta's quantized Llama 3.2 targets phones with a reported 56% smaller model and 2 to 4× faster inference.

![Three deployment tiers from most to fewest bits: cloud APIs with FP8 and FP4, self-hosted INT4 GPTQ or AWQ, and on-device GGUF Q4 and QAT checkpoints, with KV-cache quantization at every tier]({{ '/assets/blog/weight-quantization-reasoning/deployment-spectrum.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

## Try it yourself, and what to take away

This last experiment runs on your own computer rather than in Colab. With Ollama installed and around 16 GB of RAM (random-access memory), give one multi-step maths problem to the same model in three forms:

```bash
ollama run gemma3:4b-it-fp16     # 16-bit reference (8.6 GB)
ollama run gemma3:4b-it-q4_K_M   # 4-bit, post-training quantized (3.4 GB)
ollama run gemma3:4b-it-qat      # 4-bit, quantization-aware trained (4.0 GB)
```

If you keep four things from this post, make it these:

- Each weight moves by at most $$\Delta/2$$, and $$\Delta$$ is set by the largest value on the grid, so one outlier can erase a per-tensor INT4 matrix. Group-wise scales contain it.
- Per-step losses compound ($$0.99^{20} \approx 82\%$$, $$0.97^{20} \approx 54\%$$), so reasoning shows the damage first.
- Generation is bandwidth-bound, so at fixed memory a bigger 4-bit model usually beats a smaller 16-bit one.
- Most models ship with PTQ. Labs that need 4-bit models to keep reasoning use QAT, increasingly in mixed precision.

So yes, quantization degrades reasoning. It's still the standard because, done well, the parameters it buys are worth more than the precision it costs.

## Further reading

- Dettmers & Zettlemoyer (2022), [The case for 4-bit precision: k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09720). The bet on scale.
- Dettmers et al. (2022), [LLM.int8()](https://arxiv.org/abs/2208.07339). Emergent outlier features.
- Liu et al. (2025), [Quantization Hurts Reasoning?](https://arxiv.org/abs/2504.04823). Quantized reasoning models, tested systematically.
- Kumar et al. (2024), [Scaling Laws for Precision](https://arxiv.org/abs/2411.04330). Why PTQ gets harder with more training data.
- Jacob et al. (2017), [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877). The fake-quantization recipe QAT still uses.
- Google (2025), [Gemma 3 QAT models](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/). A production QAT release.
