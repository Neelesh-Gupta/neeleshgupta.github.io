---
layout: post
title: "Decoders Cost More Than Encoders. So Why Is Almost Every LLM a Decoder?"
date: 2026-11-20 10:00:00+0530
description: Why a decoder pays far more per generated token than an encoder pays per input token, and why the industry builds decoder-only models anyway, through roofline math, KV-cache arithmetic and runnable code.
tags: llm inference transformers kv-cache decoding
categories: machine-learning
related_posts: false
toc:
  sidebar: left
---

<style>
  /* This post only: use the window's width so code and figures fit without sideways scrolling. */
  @media (min-width: 576px) {
    .container[role="main"], .navbar .container, footer .container { max-width: min(calc(100vw - 32px), 1520px); }
    .container[role="main"] > .row > .col-sm-3 { flex: 0 0 18%; max-width: 18%; }
    .container[role="main"] > .row > .col-sm-9 { flex: 0 0 82%; max-width: 82%; }
  }
  .post-content img { max-width: min(100%, 960px) !important; }
  .post-content pre.highlight, .post-content pre.highlight code { white-space: pre-wrap; overflow-wrap: anywhere; }
  .post-content table { display: block; max-width: 100%; overflow-x: auto; }
  @media (max-width: 575px) {
    .post-content mjx-container[display] { font-size: 90% !important; }
    .post-content table { font-size: 75%; }
    .post-content table th, .post-content table td { padding: .3rem .35rem; }
  }
</style>

Give BERT a 512-token document and it reads the whole thing in one forward pass. That's all it does. A decoder-only LLM reads it the same way. Then it starts writing, and every token it outputs costs another full forward pass, hundreds or thousands of them in a row.

That makes decoders the expensive kind of transformer, and yet almost every frontier model of the last few years is one. I wanted to understand both halves of that properly, where the extra cost comes from and why the industry chose to pay it.

It helps to pin down what "costlier" means first. An encoder can't generate text at all, so you can't race the two on the same job, but you can compare cost per token. An encoder, like a decoder reading your prompt, pays for its input tokens in one parallel pass, whereas a decoder pays for each *generated* token with a forward pass of its own. That per-generated-token price is where the gap comes from.

The first half of the post ranks the four reasons that price is high, and the second ranks the five reasons the industry pays it anyway. Where I could, I've given each idea an intuition, a picture, a formula and some runnable code; a few only got three of the four.

## Part 1: Why decoders cost more than encoders

### One model, two modes: prefill and decode

A decoder-only model works in two modes, and only one of them is expensive.

Prefill is the reading phase. All $$L$$ prompt tokens go through each layer together, so every weight matrix multiplies an $$L \times d$$ activation matrix in one matrix-matrix product (a GEMM). That's an encoder's forward pass, give or take the causal mask. On the way through, prefill writes each prompt token's keys and values into the KV cache, and at the end it samples the first output token. The time all that takes is the time to first token, or TTFT.

Decode is everything that follows. Each step sends one new token per request through the entire network, reads back the cached keys and values of every earlier token, and samples one more. The weights now multiply a single $$1 \times d$$ vector per request, a matrix-vector product (GEMV). The gap between consecutive output tokens is the inter-token latency (ITL), and its average over a response is the time per output token (TPOT). Serving engines such as vLLM schedule the two phases differently, for the reasons in the rest of Part 1.

I picture a scribe who has to carry a whole reference shelf to the desk before writing anything. For prefill they make one trip and copy out a full page. For decode they haul the same shelf back and forth for every word, so the walking sets the pace and the writing hardly registers. The shelf is the weights, and the walking is memory traffic.

![One decoder-only model, two phases: prefill reads the prompt in one compute-bound GEMM pass and writes the KV cache, then the bandwidth-bound decode loop re-reads the weights and the KV cache from GPU HBM for every token until EOS.]({{ '/assets/blog/decoder-vs-encoder-cost/prefill-vs-decode.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

### 1. The GPU spends decode waiting on memory

During decode the GPU mostly waits. Its arithmetic units sit idle while data arrives from memory, and the roofline model (Williams, Waterman and Patterson, *CACM* 2009) shows why more cleanly than anything else I know.

Every kernel has an arithmetic intensity, the floating-point work it does for each byte it moves between HBM (the GPU's large main memory) and the on-chip SRAM where the arithmetic happens:

$$I = \frac{\text{FLOPs}}{\text{bytes moved}}$$

A GPU has two ceilings, peak compute in FLOP/s and HBM bandwidth in bytes/s, and a kernel runs at $$\min(\text{peak FLOP/s},\ I \times \text{bandwidth})$$. On log-log axes that gives the roofline, a slope that rises with $$I$$ while you're bandwidth-bound and flattens into a roof once you're compute-bound. The corner, peak FLOP/s divided by bandwidth, is called the ridge point. An H100 SXM moves 3.35 TB/s and does about 989 TFLOP/s of dense BF16 ([NVIDIA](https://www.nvidia.com/en-us/data-center/h100/) quotes 1,979 with sparsity, and dense is half that), so its ridge sits near 295 FLOPs per byte. Any kernel to the left of that leaves tensor cores waiting.

Take a single linear layer $$Y = XW$$ with $$W \in \mathbb{R}^{d \times d}$$ in BF16 and feed it $$T$$ token vectors stacked in $$X$$. It does $$2Td^2$$ FLOPs (one multiply and one add per weight per token) and moves $$2d^2$$ bytes of weights plus $$4Td$$ bytes of activations, so

$$I = \frac{2Td^2}{2d^2 + 4Td} = \frac{T}{1 + 2T/d}$$

The $$T$$ in the numerator counts the tokens that share one read of the weights. The $$2T/d$$ in the denominator is activation traffic, which only starts to matter as $$T$$ approaches $$d$$, and so when $$T \ll d$$ you get $$I \approx T$$. In words, arithmetic intensity is roughly the number of tokens you process per trip to memory.

Most of Part 1 follows from that approximation. An encoder pass or a prefill has $$T = B \times L$$ (batch size times sequence length), easily thousands of tokens. A decode step has $$L = 1$$, which makes $$T = B$$, nothing more than the batch size, and puts decode far down the slope while prefill sits up on the roof.

Every number in the post comes from one reference setup, defined in cell 1. It's an H100 plus a Llama-3-70B-like model with 80 layers, $$d = 8192$$, 64 query heads, 8 KV heads and $$d_\text{head} = 128$$, all in BF16. In the code, `H100_FLOPS / H100_BW` is the ridge point and `MODEL["d_model"]` is $$d$$.

```python
%pip install -q "torch==2.13.0" "numpy==2.3.3" "matplotlib==3.10.8"
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_printoptions(precision=4, sci_mode=False)
GiB, GB = 2**30, 1e9

# Reference hardware: one H100 SXM (NVIDIA datasheet; dense BF16, no sparsity)
H100_FLOPS = 989e12   # peak BF16 Tensor Core FLOP/s
H100_BW = 3.35e12     # HBM3 bandwidth, bytes/s
H100_VRAM = 80 * GB

# Reference model: a Llama-3-70B-like decoder with grouped-query attention, BF16
MODEL = dict(params=70e9, n_layers=80, d_model=8192, n_q_heads=64, n_kv_heads=8, d_head=128, bytes_per_el=2)
print(f"torch {torch.__version__} | numpy {np.__version__} | CPU threads={torch.get_num_threads()}")
print(f"H100 ridge point = {H100_FLOPS / H100_BW:.0f} FLOPs/byte")
```
```text
torch 2.13.0+cpu | numpy 2.3.3 | CPU threads=8
H100 ridge point = 295 FLOPs/byte
```

The next cell computes $$I(T)$$ for $$d = 8192$$ at a few token counts, using `linear_intensity` for the formula above and `attainable_flops` for the roofline. The row to look for is where $$I$$ crosses 295.

```python
def linear_intensity(T: int, d: int, bytes_per_el: int = 2) -> float:
    """FLOPs per byte of HBM traffic for Y = X @ W with X: (T, d) and W: (d, d)."""
    flops = 2 * T * d * d                         # one multiply + one add per weight per token
    traffic = bytes_per_el * (d * d + 2 * T * d)  # read W once, read X, write Y
    return flops / traffic

def attainable_flops(I: float) -> float:
    """Roofline: you get the lower of the compute roof and the bandwidth slope."""
    return min(H100_FLOPS, I * H100_BW)

d, ridge = MODEL["d_model"], H100_FLOPS / H100_BW
print(f"{'tokens/step T':>13} | {'I (FLOPs/B)':>11} | {'% of peak':>9} | regime")
for T in [1, 16, 64, 256, 1024, 4096]:
    I = linear_intensity(T, d)
    regime = "compute-bound" if I >= ridge else "bandwidth-bound"
    print(f"{T:>13} | {I:>11.1f} | {100 * attainable_flops(I) / H100_FLOPS:>8.1f}% | {regime}")

I_grid = np.logspace(0, 4, 400)
plt.figure(figsize=(7, 4))
plt.loglog(I_grid, [attainable_flops(i) / 1e12 for i in I_grid], color="black", lw=2, label="H100 roofline")
for T, label, color in [(1, "decode, B=1", "tab:red"), (16, "decode, B=16", "tab:orange"),
                        (4096, "prefill / encoder, T=4096", "tab:blue")]:
    I = linear_intensity(T, d)
    plt.scatter(I, attainable_flops(I) / 1e12, s=70, color=color, zorder=3, label=f"{label} (I={I:.0f})")
plt.axvline(ridge, ls="--", color="gray"); plt.text(ridge * 1.15, 5, f"ridge ~{ridge:.0f}", color="gray")
plt.xlabel("arithmetic intensity I (FLOPs / byte)"); plt.ylabel("attainable TFLOP/s")
plt.title(f"One d x d linear layer, d={d}, BF16"); plt.legend(loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()
```
```text
tokens/step T | I (FLOPs/B) | % of peak | regime
            1 |         1.0 |      0.3% | bandwidth-bound
           16 |        15.9 |      5.4% | bandwidth-bound
           64 |        63.0 |     21.3% | bandwidth-bound
          256 |       240.9 |     81.6% | bandwidth-bound
         1024 |       819.2 |    100.0% | compute-bound
         4096 |      2048.0 |    100.0% | compute-bound
```
![Output of cell 2]({{ '/assets/blog/decoder-vs-encoder-cost/cell-02-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

With one token per weight load ($$T = 1$$) the layer reaches $$I = 1.0$$ and uses 0.3% of the GPU. A decode batch of 16 reaches $$I = 15.9$$ and 5.4% of peak, which leaves roughly 95% of the tensor cores idle. $$I$$ only clears the ridge somewhere between $$T = 256$$ (81.6%) and $$T = 1024$$ (100%). A 4,096-token prefill gets to $$I = 2048.0$$ and sits on the roof, the blue dot in the plot, far from the red and orange decode dots on the slope.

Only two things move a dot to the right: sharing each trip through memory among more tokens, or making the trip carry fewer bytes. Speculative decoding and quantization, both in Part 2, do exactly that.

![A dot slides along the H100 roofline as tokens per weight read T grows from 1 to 4,096: it climbs the bandwidth slope (red, under 1% to ~80% of peak) and pins to the compute roof (blue) once intensity passes the ~295 FLOPs/byte ridge.]({{ '/assets/blog/decoder-vs-encoder-cost/roofline-slide.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

You can see the same effect on a laptop. The next cell pushes $$T$$ tokens through a 4096 × 4096 FP32 matrix on the CPU. At 64 MiB it won't fit in cache, so every product streams it from RAM, which stands in for HBM. `per_token_cost` reports the median time per token and the FLOP rate the CPU reached.

```python
d_cpu = 4096
W = torch.randn(d_cpu, d_cpu)  # fp32, 64 MiB: larger than CPU caches, so each matmul streams W from RAM

def per_token_cost(T: int, min_seconds: float = 0.5) -> tuple[float, float]:
    """Median microseconds per token and achieved GFLOP/s for X @ W with T tokens."""
    X = torch.randn(T, d_cpu)
    for _ in range(3):
        X @ W  # warm-up
    times: list[float] = []
    t_end = time.perf_counter() + min_seconds
    while time.perf_counter() < t_end or len(times) < 5:
        t0 = time.perf_counter(); X @ W; times.append(time.perf_counter() - t0)
    t = float(np.median(times))
    return 1e6 * t / T, 2 * T * d_cpu * d_cpu / t / 1e9

results = {T: per_token_cost(T) for T in [1, 8, 64, 512]}
base = results[1][0]
print(f"{'T':>4} | {'us / token':>10} | {'GFLOP/s':>8} | cost per token vs T=1")
for T, (us, gflops) in results.items():
    print(f"{T:>4} | {us:>10.1f} | {gflops:>8.1f} | {us / base:>6.3f}x")
```
```text
   T | us / token |  GFLOP/s | cost per token vs T=1
   1 |     1691.1 |     19.8 |  1.000x
   8 |      406.8 |     82.5 |  0.241x
  64 |      175.2 |    191.5 |  0.104x
 512 |      147.0 |    228.3 |  0.087x
```

Processing one token at a time costs 1691.1 µs per token at 19.8 GFLOP/s, and when 512 tokens share the read each one costs 147.0 µs, which is 0.087× the single-token cost (about 11× cheaper) at 228.3 GFLOP/s. Your absolute numbers will differ. The curve won't. On my CPU, paying for a full weight read to get one token's worth of math, which is what a GEMV does, costs an 11× premium per token.

### 2. The KV cache caps your batch

If decode is starved because $$T = B$$ is small, the natural fix is to batch more requests and push $$B$$ up toward the ridge. The KV cache blocks that in two separate ways.

Without a cache, producing token $$t$$ would mean recomputing keys and values for all $$t - 1$$ earlier tokens at every step. So a decoder keeps them. It stores each token's K and V vectors in every layer, appends one entry per step, and holds a separate cache for each request, since user A's cache is no use to user B. The total size is

$$
\begin{aligned}
\text{KV bytes} = {} & 2 \times n_\text{layers} \times n_\text{kv heads} \times d_\text{head} \\
& \times L_\text{ctx} \times B \times \text{bytes per element}
\end{aligned}
$$

where the 2 counts one K and one V. $$n_\text{kv heads}$$ is the number of key/value heads, which grouped-query attention (GQA, [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) makes smaller than the number of query heads. $$L_\text{ctx}$$ is how many tokens a request holds, and $$B$$ is how many requests run at once. All the factors are linear, so the cache grows like a staircase, one step per user per generated token, stacked on a fixed slab of weights until it reaches the top of GPU memory.

The first problem is plain capacity. `kv_bytes_per_token` evaluates the formula for our 70B under three attention schemes: ordinary multi-head attention (MHA, 64 KV heads), GQA (8 KV heads), and the multi-head latent attention (MLA) from [DeepSeek-V2](https://arxiv.org/abs/2405.04434). MLA caches one compressed 512-dimensional latent and one 64-dimensional rotary key per layer instead of per-head K and V. The MLA row is hypothetical; I kept the same 80 layers so that only the cache shape changes.

```python
def kv_bytes_per_token(n_layers: int, n_kv_heads: int, d_head: int, bytes_per_el: int = 2) -> int:
    """Bytes cached per token across every layer (the 2 is one K vector + one V vector)."""
    return 2 * n_layers * n_kv_heads * d_head * bytes_per_el

L, dh, b = MODEL["n_layers"], MODEL["d_head"], MODEL["bytes_per_el"]
kv_per_token = {
    "MHA (64 KV heads)": kv_bytes_per_token(L, MODEL["n_q_heads"], dh, b),
    "GQA (8 KV heads)": kv_bytes_per_token(L, MODEL["n_kv_heads"], dh, b),
    "MLA (512+64 latent)": L * (512 + 64) * b,  # DeepSeek-V2 style: one compressed latent per layer
}
gqa = kv_per_token["GQA (8 KV heads)"]
weights = MODEL["params"] * b
kv_budget = 8 * H100_VRAM - weights  # 8x H100 node: what is left after the weights (activations ignored)
print(f"weights: {weights / GB:.0f} GB ({weights / GiB:.0f} GiB) | KV budget on 8x H100: {kv_budget / GiB:.0f} GiB\n")
print(f"{'attention':<20} | {'KiB/token':>9} | {'GiB/32k req':>11} | {'max users @4k':>13} | {'max users @32k':>14}")
for name, per_tok in kv_per_token.items():
    print(f"{name:<20} | {per_tok / 1024:>9.0f} | {per_tok * 32768 / GiB:>11.2f} | "
          f"{int(kv_budget // (per_tok * 4096)):>13} | {int(kv_budget // (per_tok * 32768)):>14}")

print(f"\n32 users x 32k context (GQA): {32 * 32768 * gqa / GiB:.0f} GiB of KV vs {weights / GiB:.0f} GiB of weights")
block = 2 * 16 * MODEL["n_kv_heads"] * dh * b  # vLLM: one 16-token block in one layer
print(f"vLLM KV block (16 tokens, 1 layer): {block / 1024:.0f} KiB | a 17-token prompt needs "
      f"ceil(17/16) = {math.ceil(17 / 16)} blocks per layer")
```
```text
weights: 140 GB (130 GiB) | KV budget on 8x H100: 466 GiB

attention            | KiB/token | GiB/32k req | max users @4k | max users @32k
MHA (64 KV heads)    |      2560 |       80.00 |            46 |              5
GQA (8 KV heads)     |       320 |       10.00 |           372 |             46
MLA (512+64 latent)  |        90 |        2.81 |          1324 |            165

32 users x 32k context (GQA): 320 GiB of KV vs 130 GiB of weights
vLLM KV block (16 tokens, 1 layer): 64 KiB | a 17-token prompt needs ceil(17/16) = 2 blocks per layer
```

The weights take 140 GB (130 GiB), which leaves 466 GiB of the 8× H100 node for the cache. GQA costs 320 KiB per token, so a single 32k-token conversation holds 10.00 GiB and 32 of them need 320 GiB, more than twice the size of the weights. The node fits 46 users at 32k context, or 372 at 4k.

The linear layers wanted $$B$$ somewhere near 295 before leaving the bandwidth slope, so at long context memory runs out long before compute does. Plain MHA is eight times worse. It needs 2560 KiB per token and fits only 5 users at 32k, while MLA's latent brings the cost down to 90 KiB and lifts the cap to 165.

The last line of the output is about how serving engines store all this. vLLM splits the cache into fixed 16-token blocks, 64 KiB per layer for this model, and allocates them on demand, so a 17-token prompt takes $$\lceil 17/16 \rceil = 2$$ blocks per layer. Because the blocks don't have to be contiguous, the cache can be paged like virtual memory, and no request has to reserve room for its worst-case length in advance. That idea is PagedAttention ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)). Continuous batching ([Yu et al., OSDI '22](https://www.usenix.org/conference/osdi22/presentation/yu)) complements it. New requests are admitted and finished ones retired at every decode step instead of after a whole batch drains, which keeps $$B$$ close to the memory cap.

The second problem survives even unlimited memory. Attention doesn't batch. The linear layers share one set of weights across the whole batch, and that sharing is the only reason batching helps them. Attention reads each request's own cache, so there is nothing to share. For every cached token, each KV head means reading its K and V ($$4 d_\text{head}$$ bytes in BF16) to compute one dot product and one weighted sum ($$4 d_\text{head}$$ FLOPs per query head). With MHA that's about 1 FLOP per byte. GQA lets $$g = n_\text{q heads} / n_\text{kv heads}$$ query heads reuse each KV head, which raises that to

$$I_\text{attn} \approx g$$

For our 70B that's 8. It's better than 1, but still about 37× short of the ridge, and batch size doesn't change it. At long enough contexts these cache reads cost more bandwidth than the weight reads, as the next section shows.

![GPU memory on an 8x H100 node fills with paged 16-token KV blocks, one per user per step, on top of a fixed 140 GB weight slab; new users join mid-stream until the cache reaches the memory wall and a seventh user can't be admitted.]({{ '/assets/blog/decoder-vs-encoder-cost/kv-staircase.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

### 3. You can't parallelize time

Token $$y_t$$ is sampled conditioned on $$y_1, \dots, y_{t-1}$$, which means $$N$$ tokens need $$N$$ forward passes in strict order, and no amount of hardware can run them side by side. Each pass also has to stream at least the weights and the active KV cache out of HBM. Together those two facts put a floor under each user's latency:

$$\text{TPOT}_\text{min} \approx \frac{\text{weight bytes} + \text{active KV bytes}}{\text{aggregate HBM bandwidth}}$$

The denominator is one GPU's bandwidth times the number of GPUs the model is split across, and end-to-end latency is the TTFT plus all the inter-token gaps. Serving systems also track goodput. It counts throughput only over requests that met their latency target, because a batch that finishes quickly overall but misses its TPOT budget doesn't help someone waiting in a chat window.

`decode_step_seconds` in the next cell takes the slower of memory time and compute time (I approximate compute as 2 FLOPs per parameter per token and ignore attention FLOPs), then sweeps the batch size to find where compute takes over.

```python
def decode_step_seconds(B: int, ctx: int, n_gpus: int, weight_bytes: float, kv_per_tok: float) -> float:
    """One decode step for B users: limited by HBM traffic or by compute, whichever is slower."""
    traffic = weight_bytes + B * ctx * kv_per_tok  # every step reads all weights + every user's KV cache
    flops = 2 * MODEL["params"] * B                # ~2 FLOPs per parameter per token (attention FLOPs ignored)
    return max(traffic / (n_gpus * H100_BW), flops / (n_gpus * H100_FLOPS))

print(f"{'GPUs':>4} | {'ctx':>6} | {'B':>3} | {'TPOT_min ms':>11} | {'tok/s/user':>10} | KV share of bytes read")
for n_gpus, ctx, B in [(2, 1024, 1), (8, 1024, 1), (8, 32768, 1), (8, 32768, 16), (8, 32768, 40)]:
    t, kv = decode_step_seconds(B, ctx, n_gpus, weights, gqa), B * ctx * gqa
    print(f"{n_gpus:>4} | {ctx:>6} | {B:>3} | {1e3 * t:>11.2f} | {1 / t:>10.0f} | {100 * kv / (kv + weights):>5.1f}%")

batch, ctx = np.arange(1, 1025), 512
step_ms = np.array([1e3 * decode_step_seconds(int(B), ctx, 8, weights, gqa) for B in batch])
mem_ms = 1e3 * (weights + batch * ctx * gqa) / (8 * H100_BW)
b_sat = int(batch[np.argmax(step_ms > mem_ms + 1e-9)])  # first batch where compute, not bandwidth, sets the step time
print(f"\nB_sat (ctx={ctx}, 8x H100) ~ {b_sat} | step time: B=1 {step_ms[0]:.2f} ms, B=64 {step_ms[63]:.2f} ms, "
      f"B={b_sat} {step_ms[b_sat - 1]:.2f} ms, B=1024 {step_ms[-1]:.2f} ms")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
ax1.plot(batch, step_ms); ax1.set_xlabel("batch size B"); ax1.set_ylabel("decode step time (ms)")
ax2.plot(batch, batch / (step_ms / 1e3)); ax2.set_xlabel("batch size B"); ax2.set_ylabel("output tokens / s (all users)")
for ax in (ax1, ax2):
    ax.axvline(b_sat, ls="--", color="gray"); ax.text(b_sat * 1.05, ax.get_ylim()[0], f" B_sat~{b_sat}", color="gray", va="bottom")
fig.suptitle(f"70B decode on 8x H100, ctx={ctx}: extra users are nearly free until B_sat, then compute-bound"); fig.tight_layout(); plt.show()
```
```text
GPUs |    ctx |   B | TPOT_min ms | tok/s/user | KV share of bytes read
   2 |   1024 |   1 |       20.95 |         48 |   0.2%
   8 |   1024 |   1 |        5.24 |        191 |   0.2%
   8 |  32768 |   1 |        5.62 |        178 |   7.1%
   8 |  32768 |  16 |       11.63 |         86 |  55.1%
   8 |  32768 |  40 |       21.25 |         47 |  75.4%

B_sat (ctx=512, 8x H100) ~ 457 | step time: B=1 5.23 ms, B=64 5.62 ms, B=457 8.09 ms, B=1024 18.12 ms
```
![Output of cell 5]({{ '/assets/blog/decoder-vs-encoder-cost/cell-05-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

A 70B model doesn't fit on one 80 GB card. The smallest workable setup in the table is 2 GPUs with tensor parallelism, at 20.95 ms per token or 48 tokens/s for one user, and on 8 GPUs each card reads a quarter as many bytes, giving 5.24 ms (191 tokens/s) for four times the hardware. At 32k context one user's cache is 7.1% of the bytes read; with sixteen users it's 55.1% and the floor rises to 11.63 ms, and with forty it's 75.4% and 21.25 ms. By that point the cache uses more bandwidth than the weights do.

The sweep uses a short 512-token context, and I find it the most encouraging plot in the post. Going from $$B = 1$$ to $$B = 64$$ moves step time only from 5.23 ms to 5.62 ms, because the extra 63 users ride on a weight read the GPU was paying for anyway; below the knee, decoding one token or ten takes about the same time, and that is why every serving engine batches. The knee itself is at $$B_\text{sat} \approx 457$$. Past it, compute time has caught up with memory time, so step time grows in proportion to $$B$$ (18.12 ms at $$B = 1024$$) and total throughput stops rising, as the right-hand plot shows. It lands above the weights-only ridge of 295 because every added user also brings its own KV reads.

Any provider has to manage this trade-off. Batching up to the knee gives the most tokens per second per dollar, but each user's TPOT grows with $$B$$. To guarantee someone 50-100 tokens/s you have to under-batch and shard across extra GPUs, and pay for compute that sits idle. Encoders avoid the problem entirely, since their whole output exists as soon as their single parallel pass finishes.

### 4. Cross-attention, if you kept the encoder

The smallest cost applies only to the classic encoder-decoder design used by the original [Transformer](https://arxiv.org/abs/1706.03762), [T5](https://arxiv.org/abs/1910.10683) and BART. Their decoder blocks have three sublayers rather than two: masked self-attention, then cross-attention (queries from the decoder, keys and values from the encoder's output), then the FFN. Cross-attention brings its own $$W_Q$$, $$W_K$$, $$W_V$$ and $$W_O$$, four more $$d \times d$$ matrices. An encoder block has about $$12d^2$$ parameters ($$4d^2$$ in attention and $$8d^2$$ in an FFN with a $$4d$$ hidden layer), so this decoder block comes to about $$16d^2$$, roughly 33% more parameters and FLOPs.

It ranks last because decoder-only models dropped cross-attention altogether. A GPT- or Llama-style block is an encoder block with a causal mask and the same parameter count. For the models people deploy today, the whole cost gap comes from the inference mechanics in the three sections above.

![Parameter anatomy of three transformer blocks: an encoder block and a decoder-only block both hold about 12d² (4d² attention + 8d² FFN), while an encoder-decoder decoder block adds a 4d² cross-attention sublayer fed K and V from the encoder's output states, for about 16d² (+33%).]({{ '/assets/blog/decoder-vs-encoder-cost/block-anatomy.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

## Part 2: Why the industry prefers decoder-only models anyway

Part 1 adds up to a per-token tax made of low arithmetic intensity, a cache that limits the batch, and a latency floor that can't be parallelized away. The five reasons below, biggest first, explain why the industry pays it.

### 1. One model for every task

Next-token prediction turns almost every language task into the same shape, text in and text out. A classifier becomes a model that generates a label and an extractor becomes one that generates JSON, while translation, summarization, coding, multi-step reasoning and tool calls are all a matter of generating the right continuation. One model behind one serving stack covers all of it.

I think this economic argument is the one that actually settles the question. Paying more per generated token for a single generalist is cheaper than building and running hundreds of task-specific encoders, each with its own labelled data, fine-tuning run, alignment, evaluation suite and deployment. Post-training (instruction tuning, RLHF, RL on verifiable rewards) is done once and reaches every downstream task through the prompt.

The research supports the zero-shot half of this argument, with one condition. [Wang et al. (2022)](https://arxiv.org/abs/2204.05832) compared architectures and pretraining objectives at scale and found that straight after unsupervised pretraining, causal decoder-only models trained autoregressively had the best zero-shot generalization. After multitask finetuning, non-causal models trained with masked language modelling came out ahead. Industry mostly operates in the first setting, with one model asked to do things nobody fine-tuned it for, steered in context and improved by general-purpose post-training rather than curated multitask mixtures.

### 2. Every token is a training example

At frontier pretraining budgets you want each FLOP to teach the model as much as possible, and the two objectives differ a lot on that measure.

BERT's masked language modelling (MLM) hides 15% of the tokens and asks the model to fill them back in ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805)), so the other 85% supply context but no loss. Causal language modelling (CLM) puts a loss on every position at once. With teacher forcing the input is the true sequence, and the lower-triangular mask lets position $$i$$ see only tokens $$0..i$$ while it predicts token $$i+1$$. For a sequence of length $$L$$,

$$\frac{\text{CLM loss positions}}{\text{MLM loss positions}} = \frac{L}{0.15\,L} \approx 6.7$$

Geometrically, each row of that lower-triangular mask is a separate training example, a prefix of length $$i+1$$ labelled with the next token, so one forward pass trains on all $$L$$ prefixes in parallel. The next cell counts loss positions under both objectives and prints the triangle.

```python
seq_len, mask_rate = 512, 0.15
rng = np.random.default_rng(SEED)
mlm_targets = np.zeros(seq_len, dtype=bool)     # BERT-style: only the masked 15% of positions get a loss
mlm_targets[rng.choice(seq_len, size=round(mask_rate * seq_len), replace=False)] = True
clm_targets = np.ones(seq_len, dtype=bool)      # GPT-style: every position predicts its next token
print(f"loss positions per {seq_len}-token sequence: CLM={clm_targets.sum()} | MLM={mlm_targets.sum()} "
      f"| ratio={clm_targets.sum() / mlm_targets.sum():.1f}x (expected {1 / mask_rate:.1f}x)")

causal_mask = torch.ones(6, 6, dtype=torch.int64).tril()  # row i = query token i; 1 = may attend
print("\ncausal mask for 6 tokens (row i sees tokens 0..i and predicts token i+1):")
print(causal_mask)
```
```text
loss positions per 512-token sequence: CLM=512 | MLM=77 | ratio=6.6x (expected 6.7x)

causal mask for 6 tokens (row i sees tokens 0..i and predicts token i+1):
tensor([[1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1]])
```

CLM gets 512 loss positions and MLM 77. That's 6.6×, slightly under the 6.7× ideal because the mask rounds to a whole number of tokens. The printed mask shows a second advantage. Row 0 is a one-token prompt and row 5 a six-token prompt, so CLM trains on every prefix length from 1 to $$L$$, which matches inference, where prompts arrive at any length. MLM has a mismatch instead, since the `[MASK]` token it learns from never appears in real input. BERT softens that by replacing only 80% of the chosen tokens with `[MASK]`, 10% with random tokens, and leaving 10% unchanged.

In MLM's defence, each of its loss positions sees context on both sides, so the raw 6.6× overstates the gap somewhat. Even so, I don't know of an objective that beats causal LM per pretraining FLOP at scale.

### 3. A stack of identical blocks scales cleanly

Training across tens of thousands of GPUs rewards uniformity. A decoder-only model is about as uniform as an architecture can be, $$N$$ identical blocks from bottom to top.

That makes parallelism unremarkable, which is what you want from it. Tensor parallelism splits each weight matrix across the GPUs in a node in the same way for every layer. Pipeline parallelism cuts the stack into stages that balance without tuning, because every block is the same. Nothing hands off from an encoder to a decoder to create pipeline bubbles, and no decoder stage waits on the encoder through cross-attention. Context parallelism splits long sequences across GPUs, also uniformly.

An encoder-decoder model also forces you to decide up front how to divide the parameter budget between its two halves. Summarization, with long inputs and short outputs, wants a heavy encoder, while code generation and step-by-step reasoning, with short inputs and long outputs, want a heavy decoder, and no fixed 50/50 or 75/25 split suits both. A decoder-only model avoids the question entirely, since 100% of its parameters read the prompt during prefill and 100% of them generate during decode.

### 4. The cache survives the next turn

For chat and agents, this is the big one. It comes straight from the causal mask.

Under a causal mask the hidden state of token $$i$$ in layer $$\ell$$ depends only on tokens $$1..i$$, so its key and value vectors

$$
\begin{gathered}
k_i^{(\ell)} = W_K\, h_i^{(\ell)}, \qquad v_i^{(\ell)} = W_V\, h_i^{(\ell)}, \\
h_i^{(\ell)} = f_\ell(x_1, \dots, x_i)
\end{gathered}
$$

are fixed as soon as they're computed, and nothing appended after position $$i$$ can change them. Here $$W_K$$ and $$W_V$$ are the layer's key and value projections, $$h_i^{(\ell)}$$ is the input to layer $$\ell$$ at position $$i$$, and $$f_\ell$$ stands for everything below that layer. I'll call this causal invariance. It's why a KV cache gives correct results at all, and it pays off in two ways.

In a 20-turn conversation, or an agent loop that keeps appending tool output, each new turn only needs prefill for its new tokens. Everything earlier comes from the cache. Shared prefixes get the same treatment across requests: system prompts, few-shot examples and shared RAG documents are prefilled once and reused by every request that starts with them. vLLM hashes each full 16-token block together with the hash of the block before it, which turns finding a shared prefix into matching a chain of block hashes. This speeds up prefill (TTFT) and does nothing for decode.

A bidirectional encoder can't do either. Token 1 attends to token 1,000, so appending one token changes the attention weights, and therefore the hidden states, of every earlier token. Drawn as attention matrices, appending a token to the causal one adds a single row at the bottom, while every cell of the bidirectional square changes. The encoder has to re-encode the whole history, at $$O(L^2)$$ attention cost, on every turn.

To test this I built a tiny 2-layer transformer with random weights and ran it with and without the causal mask. In the code, `cache[i]` holds layer $$i$$'s $$(K, V)$$, and `tril(k.shape[1] - q.shape[1])` is the causal mask, offset so that one new query can see every cached key.

```python
class TinyLM(nn.Module):
    """2-layer pre-LN transformer with random weights. causal=True is a decoder, False an encoder."""

    def __init__(self, vocab: int = 100, d: int = 64, n_layers: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.emb, self.pos, self.h = nn.Embedding(vocab, d), nn.Embedding(512, d), n_heads
        self.layers = nn.ModuleList(nn.ModuleDict(dict(
            ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), o=nn.Linear(d, d), ln2=nn.LayerNorm(d),
            ffn=nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)))) for _ in range(n_layers))

    def forward(self, ids: torch.Tensor, causal: bool = True, cache: list | None = None) -> torch.Tensor:
        """ids: (T,) new tokens. cache: per-layer (K, V) of earlier tokens, extended in place."""
        start = cache[0][0].shape[1] if cache and cache[0] is not None else 0
        x = self.emb(ids) + self.pos(torch.arange(start, start + len(ids)))
        for i, layer in enumerate(self.layers):
            q, k, v = layer["qkv"](layer["ln1"](x)).view(len(ids), 3, self.h, -1).permute(1, 2, 0, 3)
            if cache is not None:
                if cache[i] is not None:  # decode: reuse earlier K, V instead of recomputing them
                    k, v = torch.cat([cache[i][0], k], 1), torch.cat([cache[i][1], v], 1)
                cache[i] = (k, v)
            mask = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool).tril(k.shape[1] - q.shape[1]) if causal else None
            x = x + layer["o"](F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(0, 1).reshape(len(ids), -1))
            x = x + layer["ffn"](layer["ln2"](x))
        return x

lm = TinyLM().eval()
ids, new = torch.randint(0, 100, (12,)), torch.randint(0, 100, (1,))
with torch.no_grad():
    for causal, name in [(True, "causal (decoder)"), (False, "bidirectional (encoder)")]:
        before, after = lm(ids, causal), lm(torch.cat([ids, new]), causal)
        print(f"{name:<24} max |change| in the 12 earlier states after appending 1 token: {(after[:12] - before).abs().max():.2e}")
    cache = [None, None]
    lm(ids, True, cache)                                  # prefill: computes and stores K, V for 12 tokens
    step, full = lm(new, True, cache), lm(torch.cat([ids, new]), True)
    print(f"\ndecode with KV cache == full recompute: {torch.allclose(step[0], full[-1], atol=1e-5)} "
          f"| layer-0 K cache shape (heads, tokens, d_head): {tuple(cache[0][0].shape)}")
```
```text
causal (decoder)         max |change| in the 12 earlier states after appending 1 token: 0.00e+00
bidirectional (encoder)  max |change| in the 12 earlier states after appending 1 token: 1.61e-01

decode with KV cache == full recompute: True | layer-0 K cache shape (heads, tokens, d_head): (4, 13, 16)
```

Appending one token changes the 12 earlier causal hidden states by exactly 0.00e+00, because those positions run the same arithmetic as before and there isn't even float noise. Without the mask, the same model moves them by up to 1.61e-01. The second line prefills 12 tokens, decodes the 13th using only the cached K and V, and matches a full recompute (`True`). The layer-0 key cache has shape (4, 13, 16), meaning 4 heads, 13 tokens and 16 dimensions per head.

![Appending one token to a 6-token sequence: the causal attention matrix only gains a new bottom row while its earlier cells stay unchanged, whereas every cell of the bidirectional matrix turns orange because all earlier states must be recomputed.]({{ '/assets/blog/decoder-vs-encoder-cost/causal-vs-bidirectional-append.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

### 5. Then the systems people went to work

The effect here is large, but it ranks last because it came after the decision. Once the industry had settled on decoder-only models for reasons 1-4, hardware and systems research worked through Part 1's bottlenecks one formula term at a time.

| Part 1 bottleneck | Fix | Term it changes |
|---|---|---|
| Weight bandwidth (1, 3) | Weight quantization: FP8, INT4 (W4A16) | weight bytes in $$I$$ and $$\text{TPOT}_\text{min}$$ |
| One token per weight read (1) | Speculative decoding, multi-token prediction | tokens per weight read ($$T$$ in $$I$$) |
| KV footprint (2) | GQA, MLA, KV-cache quantization | $$n_\text{kv heads} \times d_\text{head}$$ and bytes per element |
| KV fragmentation (2) | PagedAttention | usable share of the KV budget |
| Small batches (1, 2) | Continuous batching | $$B$$ |
| Long prompts stalling the batch (3) | Chunked prefill | everyone else's TTFT and ITL |
| Repeated prompts | Prefix caching | prefill tokens per request |
| Prefill and decode competing (1) | Prefill-decode disaggregation | each regime gets its own hardware |

Speculative decoding goes straight at the GEMV. A cheap drafter (a small model, a few extra prediction heads, or even an n-gram lookup) guesses the next $$k$$ tokens, and the large model scores all of them in one forward pass, a GEMM over $$k + 1$$ positions instead of $$k$$ GEMVs, accepting them left to right with a rejection rule that provably leaves the output distribution unchanged ([Leviathan et al., 2022](https://arxiv.org/abs/2211.17192); [Chen et al., 2023](https://arxiv.org/abs/2302.01318)). DeepSeek-V3 trains multi-token-prediction modules that can serve as that drafter ([DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437)).

Two of the fixes in the table are scheduling changes. Chunked prefill ([Agrawal et al., 2023](https://arxiv.org/abs/2308.16369)) splits long prompts into chunks that share each step with other requests' decodes. Disaggregation ([Zhong et al., 2024](https://arxiv.org/abs/2401.09670)) runs compute-bound prefill and bandwidth-bound decode on separately sized GPU pools, shipping the KV cache between them.

Cell 8 applies every fix the earlier formulas can express to a single scenario, our 70B on 8× H100 with 8k context and 32 users decoding. `scenario()` recomputes capacity and $$\text{TPOT}_\text{min}$$ with one term changed per row. Two rows go the other way and remove GQA or PagedAttention, to show what those now-standard fixes are worth. For the no-PagedAttention row I assume 30% of KV memory is usable, in line with the 20.4-38.2% utilization Kwon et al. measured in earlier serving systems. For speculative decoding I assume 3 tokens accepted per verify pass, and a verify pass that stays bandwidth-bound.

```python
def scenario(w_bytes_per_param: float = 2, kv_per_tok: float = gqa, kv_usable: float = 1.0,
             tokens_per_step: float = 1.0, ctx: int = 8192, B: int = 32, n_gpus: int = 8) -> tuple[float, int, float]:
    """(weights GB, max concurrent users at ctx, TPOT ms at B users) for the 70B model on 8x H100."""
    w = MODEL["params"] * w_bytes_per_param
    max_users = int((n_gpus * H100_VRAM - w) * kv_usable // (ctx * kv_per_tok))
    tpot = decode_step_seconds(B, ctx, n_gpus, w, kv_per_tok) / tokens_per_step  # verify pass assumed bandwidth-bound
    return w / GB, max_users, 1e3 * tpot

fixes = [  # (fix, formula term it changes, overrides)
    ("baseline: BF16 W, GQA, BF16 KV, paged", "-", {}),
    ("without GQA (plain MHA)", "n_kv_heads 8 -> 64", dict(kv_per_tok=kv_per_token["MHA (64 KV heads)"])),
    ("without PagedAttention (~70% waste)", "usable KV memory", dict(kv_usable=0.3)),
    ("FP8 weights", "weight bytes / 2", dict(w_bytes_per_param=1)),
    ("INT4 weights (W4A16)", "weight bytes / 4", dict(w_bytes_per_param=0.5)),
    ("FP8 KV cache", "KV bytes / 2", dict(kv_per_tok=gqa / 2)),
    ("MLA latent KV", "KV bytes per token", dict(kv_per_tok=kv_per_token["MLA (512+64 latent)"])),
    ("speculative decoding (3 tok/step)", "tokens per weight read", dict(tokens_per_step=3.0)),
    ("FP8 W + FP8 KV + spec. decoding", "all three", dict(w_bytes_per_param=1, kv_per_tok=gqa / 2, tokens_per_step=3.0)),
]
print(f"70B on 8x H100, 8k context, 32 users decoding\n"
      f"{'fix':<37} | {'term changed':<22} | {'weights GB':>10} | {'max users':>9} | {'TPOT ms':>7}")
for name, term, overrides in fixes:
    w_gb, users, tpot = scenario(**overrides)
    print(f"{name:<37} | {term:<22} | {w_gb:>10.0f} | {users:>9} | {tpot:>7.2f}")

prefix = 4096  # shared system prompt + few-shot examples
print(f"\nprefix caching: re-prefilling a {prefix}-token shared prefix costs ~"
      f"{1e3 * 2 * MODEL['params'] * prefix / (8 * H100_FLOPS):.0f} ms of TTFT per request; a cache hit skips it "
      f"(decode is unchanged)")
```
```text
70B on 8x H100, 8k context, 32 users decoding
fix                                   | term changed           | weights GB | max users | TPOT ms
baseline: BF16 W, GQA, BF16 KV, paged | -                      |        140 |       186 |    8.43
without GQA (plain MHA)               | n_kv_heads 8 -> 64     |        140 |        23 |   30.87
without PagedAttention (~70% waste)   | usable KV memory       |        140 |        55 |    8.43
FP8 weights                           | weight bytes / 2       |         70 |       212 |    5.82
INT4 weights (W4A16)                  | weight bytes / 4       |         35 |       225 |    4.51
FP8 KV cache                          | KV bytes / 2           |        140 |       372 |    6.83
MLA latent KV                         | KV bytes per token     |        140 |       662 |    6.13
speculative decoding (3 tok/step)     | tokens per weight read |        140 |       186 |    2.81
FP8 W + FP8 KV + spec. decoding       | all three              |         70 |       424 |    1.40

prefix caching: re-prefilling a 4096-token shared prefix costs ~72 ms of TTFT per request; a cache hit skips it (decode is unchanged)
```

Removing GQA multiplies KV bytes per token by 8. Capacity falls from 186 users to 23, and TPOT climbs from 8.43 to 30.87 ms as cache reads take over the traffic, so GQA accounts for much of why the baseline can be served at all. Removing PagedAttention leaves TPOT at 8.43 ms but cuts capacity from 186 to 55, so fragmentation costs capacity and has no effect on speed.

Each of the other fixes changes its own term. FP8 and INT4 weights shrink the weight term to 70 GB and 35 GB, cut TPOT to 5.82 and 4.51 ms, and free a little room for cache (212 and 225 users). FP8 KV halves the cache term, which doubles capacity to 372 users and brings TPOT to 6.83 ms, and MLA's latent goes further on capacity, to 662 users at 6.13 ms. Speculative decoding works differently. It's the only fix that changes tokens per weight read, and at 3 tokens per step TPOT drops to 2.81 ms with no change in memory. Combining FP8 weights, FP8 KV and speculative decoding gives 1.40 ms per token with room for 424 users, which is 6× faster than the baseline with 2.3× the capacity.

The final line covers prefill. Re-reading a 4,096-token shared prefix costs about 72 ms of TTFT per request, and a prefix-cache hit skips that. None of these fixes makes decoding as cheap per token as encoding. Together, though, they turned a 70B decoder that was too slow and memory-hungry to serve widely into something a single node can serve to hundreds of concurrent users at interactive speed.

![Each Part 1 decode bottleneck mapped to the Part 2 fix that attacks it, labelled with the formula term the fix changes: weight bytes, tokens per weight read, KV bytes per token, usable KV memory, batch size B, others' TTFT/ITL, prefill tokens, and separate hardware per regime.]({{ '/assets/blog/decoder-vs-encoder-cost/fix-to-bottleneck-map.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

## Where encoders still win

Encoders still have a clear place. When the job is turning text into a representation instead of generating more text, an encoder does it in one parallel pass with context from both sides, and that is what dense retrieval embeddings, cross-encoder rerankers and high-throughput classifiers need. Those workloads sit on the compute roof from cell 2 and never touch a KV cache. Encoder research hasn't stalled either. ModernBERT ([Warner et al., 2024](https://arxiv.org/abs/2412.13663)) updates the BERT recipe for long context and fast inference, and Encoder-Decoder Gemma ([Zhang et al., 2025](https://arxiv.org/abs/2504.06225)) adapts decoder-only checkpoints into encoder-decoder models for a better quality-efficiency trade-off.

A decoder costs more per generated token because each step re-reads the whole model and every user's cache to do one token's worth of math. The cache limits how many users can share that read, and the steps can't run in parallel. Almost every LLM is a decoder anyway because next-token prediction gives one interface for every task, the densest training signal, the simplest stack to scale and a cache that survives each new turn, and because systems work keeps shrinking the bill.

## Optional: check it on GPT-2 and BERT

Everything so far used calculators and a toy model. Cell 9 repeats the two central measurements on real pretrained weights, GPT-2 (124M parameters, decoder-only) and BERT-base (110M, encoder). It's optional because it installs `transformers` and downloads about 1 GB from the Hugging Face Hub the first time; after that it runs on a CPU in about 15 seconds.

The first half times GPT-2 prefilling a 256-token prompt and then 32 decode steps that reuse `past_key_values` (the KV cache), and checks that cached decoding picks the same greedy tokens as recomputing everything from scratch. The second half appends one word to a short sentence and measures how far each model's earlier hidden states move.

```python
%pip install -q "transformers==5.22.2"
# Optional: downloads GPT-2 (~550 MB) and BERT-base (~440 MB) from the Hugging Face Hub on first run.
import os
import warnings
os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1", HF_HUB_VERBOSITY="error")
warnings.filterwarnings("ignore")
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()

gpt_tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
prompt = gpt_tok("Decoders pay a memory tax on every token they generate. " * 40, return_tensors="pt").input_ids[:, :256]
with torch.no_grad():
    t0 = time.perf_counter(); out = gpt2(prompt, use_cache=True); prefill_s = time.perf_counter() - t0
    past, nxt, generated = out.past_key_values, out.logits[:, -1:].argmax(-1), []
    t0 = time.perf_counter()
    for _ in range(32):  # decode: one token per forward pass, reusing the KV cache
        generated.append(nxt); out = gpt2(nxt, past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    decode_s = time.perf_counter() - t0
    recompute = gpt2(torch.cat([prompt, *generated], 1)).logits[:, prompt.shape[1] - 1:-1].argmax(-1)
pre_ms, dec_ms = 1e3 * prefill_s / prompt.shape[1], 1e3 * decode_s / 32
print(f"GPT-2 prefill: {pre_ms:.2f} ms/token | decode: {dec_ms:.2f} ms/token ({dec_ms / pre_ms:.0f}x more per token)")
print(f"cached decode == no-cache recompute (32 greedy tokens): {torch.equal(recompute, torch.cat(generated, 1))}\n")

bert_tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert = AutoModel.from_pretrained("google-bert/bert-base-uncased").eval()
text, extra = "the kv cache makes decoding", " cheap"
with torch.no_grad():
    for name, tok, encoder, n_special_end in [("GPT-2 (causal)", gpt_tok, gpt2.transformer, 0),
                                              ("BERT (bidirectional)", bert_tok, bert, 1)]:
        a, b = tok(text, return_tensors="pt"), tok(text + extra, return_tensors="pt")
        n = a.input_ids.shape[1] - n_special_end  # compare only tokens shared by both inputs
        shift = (encoder(**b).last_hidden_state[0, :n] - encoder(**a).last_hidden_state[0, :n]).abs().max()
        print(f"{name:<21} max |change| in earlier hidden states after appending '{extra.strip()}': {shift:.2e}")
```
```text
GPT-2 prefill: 1.56 ms/token | decode: 22.57 ms/token (14x more per token)
cached decode == no-cache recompute (32 greedy tokens): True

GPT-2 (causal)        max |change| in earlier hidden states after appending 'cheap': 0.00e+00
BERT (bidirectional)  max |change| in earlier hidden states after appending 'cheap': 9.86e-01
```

On my laptop CPU, prefill costs 1.56 ms per token and decode 22.57 ms per token, so each generated token costs 14× more than each prompt token. It's the same GEMM-versus-GEMV gap as in cell 3, inside a real model this time. Treat the absolute timings as illustrative rather than as GPU benchmarks. Cached decoding matches the no-cache recompute (`True`), and appending "cheap" moves GPT-2's earlier hidden states by 0.00e+00 and BERT's by 9.86e-01, the same split cell 7 showed with random weights.

## Further reading

- [The vLLM anatomy post](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (Gordić, September 2025) covers the engine internals I skipped: the scheduler, KV-cache manager, prefix-cache hashing, chunked prefill and P/D connectors.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023), for paged KV blocks and the fragmentation numbers.
- [GQA](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023) and [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (DeepSeek-AI, 2024) are the two main ways to shrink KV bytes per token.
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022) is the original; [EAGLE-3](https://arxiv.org/abs/2503.01840) (Li et al., 2025) is a current draft-head variant.
- [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/abs/2204.05832) (Wang et al., 2022), the controlled comparison behind the first reason in Part 2.
- [SARATHI](https://arxiv.org/abs/2308.16369) (Agrawal et al., 2023) on chunked prefill, and [DistServe](https://arxiv.org/abs/2401.09670) (Zhong et al., 2024) on putting prefill and decode on separate hardware.
