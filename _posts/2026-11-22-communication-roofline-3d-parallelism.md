---
layout: post
title: "Why Adding More GPUs Can Make Training Slower: The Communication Roofline of 3D Parallelism"
date: 2026-11-22 17:00:00+0530
description: Why stretching tensor parallelism across two 8-GPU H100 nodes buys almost nothing while data or pipeline parallelism nearly doubles throughput, worked out with an alpha-beta cost model, ring collectives, a communication roofline, a pipeline-bubble simulator and a step-time model.
tags: distributed-training parallelism gpu llm training
categories: machine-learning
related_posts: false
toc:
  sidebar: left
---

<style>
  /* This post only: fill the whole window, and keep everything inside it so nothing slides sideways. */
  @media (min-width: 576px) {
    .container[role="main"], .navbar .container, footer .container { max-width: calc(100vw - 32px); }
    .container[role="main"] > .row > .col-sm-3 { flex: 0 0 16%; max-width: 16%; }
    .container[role="main"] > .row > .col-sm-9 { flex: 0 0 84%; max-width: 84%; }
  }
  .post-content img { max-width: 100% !important; height: auto; }
  /* display equations: the widest ones are split onto several lines; any leftover shrinks inside its own box */
  mjx-container[display="true"] { display: block; max-width: 100%; overflow-x: auto; overflow-y: hidden; }
  /* code and output lines wrap inside the column */
  article pre, article pre code { white-space: pre-wrap !important; overflow-wrap: anywhere; word-break: normal; overflow-x: hidden !important; }
  article p code, article li code, article td code { overflow-wrap: anywhere; }
  /* tables fit the column and wrap their text */
  .post-content table { width: 100%; table-layout: fixed; }
  .post-content th, .post-content td { overflow-wrap: anywhere; word-break: normal; }
  @media (max-width: 575px) {
    .post-content mjx-container[display] { font-size: 90% !important; }
    .post-content table { font-size: 82%; }
    .post-content table td, .post-content table th { padding: .4rem .3rem; }
  }
</style>

Adding GPUs is supposed to be the dependable way to train faster. Double the hardware, keep the batch the same, and the step time should roughly halve. On a multi-node cluster that only holds if each kind of parallelism lands on the right link. Put the wrong one across the boundary between servers and the second server adds almost nothing.

This post works out why, starting from the bytes. It begins with what it costs to send a message, moves on to the ring collectives that data, tensor and pipeline parallelism all rely on, places each of them on a communication roofline, and ends with a small model that predicts what happens when an 8-GPU job grows to 16. Along the way you'll see why tensor parallelism stays inside a server while pipeline and data parallelism can cross between servers, and how to reason about the idle time a pipeline leaves behind.

## The paradox: twice the GPUs, nowhere near twice the speed

Take a 70B-class transformer (hidden size $$h = 8192$$, $$L = 80$$ layers) training on one 8-GPU H100 node, with Megatron-style tensor parallelism (TP) across all eight GPUs because the weights don't fit on one. Each optimizer step sees 64 sequences of 4,096 tokens.

Then a second node shows up. The obvious move is TP = 16: same code, same batch, twice the hardware. In the cost model this post builds, the step goes from 19.0 s to 17.8 s. Twice the GPUs buy 7% more throughput, and model FLOPs utilization (MFU, the share of peak FLOP/s spent on the model's own math) falls from 67% to 36%.

Use the second node for a data-parallel copy instead, with TP = 8 inside each node, and the step drops to 9.5 s. Twice the throughput.

The difference is in what each layout asks of the link between the nodes. Inside a node each GPU talks to the others over NVLink at 450 GB/s per direction. Between nodes each GPU has its own [400 Gb/s NDR InfiniBand NIC](https://networking-docs.nvidia.com/connectx7hw/Introduction), which works out to 50 GB/s per direction, a 9x drop. (The "18x" you sometimes see compares NVLink's 900 GB/s two-way total with InfiniBand's one-way figure.) Data parallelism sends its gradients across that link once per step. TP = 16 sends activations across it in four all-reduces per layer per micro-batch, each one a ring of 16 GPUs with twice as many steps as before, and each GPU now has only half as much math to do while it waits.

The sections below build that picture up one piece at a time, and cell 10 at the end reproduces these numbers.

## Primer: three ways to split a model

Training on many GPUs comes down to two decisions: what each GPU holds, and what it has to tell the others. There are three basic answers. Big runs use all three at once, hence "3D parallelism".

In **data parallelism (DP)** every GPU holds the whole model and processes its own slice of the batch. After the backward pass the GPUs average their gradients with an *all-reduce*, then step the optimizer. That is one exchange per step, on a message the size of the gradients. PyTorch's DDP works this way.

**Fully sharded data parallelism (FSDP, or ZeRO-3)** is data parallelism where each GPU stores only $$1/d$$ of the weights, gradients and optimizer state, with $$d$$ the number of GPUs. Before a layer runs, the GPUs *all-gather* its weights. After its backward pass they *reduce-scatter* its gradients, so each GPU keeps only its own shard. That moves 1.5x DDP's bytes ([Rajbhandari et al., 2019](https://arxiv.org/abs/1910.02054)) in exchange for a large memory saving.

In **tensor parallelism (TP)** each weight matrix is split across $$t$$ GPUs, so every GPU does part of every matrix multiply. Megatron-LM splits the MLP's first matrix by columns and its second by rows. Each GPU can then run its share of the MLP without talking to anyone, and one all-reduce at the end sums the partial outputs. Attention is split by heads the same way ([Shoeybi et al., 2019](https://arxiv.org/abs/1909.08053)). The price is all-reduces on activations inside every layer, forward and backward.

In **pipeline parallelism (PP)** the layers are cut into $$p$$ consecutive stages, each on its own GPU or group of GPUs. Stage 1 runs the first $$L/p$$ layers and hands its activations to stage 2, and gradients flow back the other way. To keep every stage busy, the batch is cut into $$m$$ *micro-batches* that follow each other down the pipe. Communication is one point-to-point send per micro-batch at each stage boundary.

A 3D layout gives every GPU a coordinate in all three, with degrees $$t \times p \times d$$ = number of GPUs, and the question for this post is which of those coordinates should change when you step from one 8-GPU node to the next.

![Two 8-GPU nodes joined by NVLink inside each node and a 9x slower InfiniBand link between them, with three ways to use the second node: TP = 16 spanning all 16 GPUs, PP = 2 sending activations from Node 0 GPU 3 to Node 1 GPU 3, and DP = 2 all-reducing gradients between the two GPU 5s]({{ '/assets/blog/communication-roofline-3d-parallelism/cluster-grid.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Each box is one server with 8 GPUs. Inside a box the GPUs talk over NVLink, which is fast. The red line between the boxes is the InfiniBand network, 9x slower per GPU. The three coloured paths are three ways to use the second server, and the rest of the post works out how much traffic each one pushes across that red line.*

| Symbol | Meaning | In the code |
|---|---|---|
| $$h$$, $$L$$, $$s$$ | hidden size, number of layers, sequence length | `H`, `L_LAYERS`, `SEQ` |
| $$t$$, $$p$$, $$d$$ | tensor-, pipeline-, data-parallel degree | `t`, `p`, `d` |
| $$m$$, $$v$$ | micro-batches per step, model chunks per GPU (interleaved PP) | `M`, `V` |
| $$\Psi$$ | number of parameters | `PSI` |
| $$n_{\text{tok}}$$ | tokens each GPU processes per optimizer step | `tokens` |
| $$C$$ | peak FLOP/s of one GPU | `C_PEAK` |
| $$B_{\text{link}}$$, $$\alpha$$ | link bandwidth (bytes/s, one direction, per GPU), per-message latency | `link.bandwidth`, `link.latency` |

## The cost of talking: two ceilings

A message costs a fixed toll to get started (software stack, NIC, switches) plus a part that grows with its size. The standard cost model for collectives writes the time to send $$n$$ bytes as

$$T_{\text{comm}}(n) = \alpha + \frac{n}{B_{\text{link}}}$$

where $$\alpha$$ is the latency and $$B_{\text{link}}$$ the bandwidth ([Thakur et al., 2005](https://doi.org/10.1177/1094342005051521)). Compute is simpler still: a GPU with peak throughput $$C$$ needs $$T_{\text{comp}} = \text{FLOPs}/C$$.

The roofline model ([Williams et al., 2009](https://doi.org/10.1145/1498765.1498785)) asks whether arithmetic or memory traffic limits a kernel on one chip. It compares the kernel's FLOPs per byte with the hardware's FLOP/s per byte/s. The network gets the same treatment. If communication overlaps with compute, the network costs nothing as long as $$T_{\text{comp}} \ge T_{\text{comm}}$$, and for large messages, where $$\alpha$$ is negligible, that becomes

$$\frac{\text{FLOPs}}{\text{bytes sent}} \;\ge\; \frac{C}{B_{\text{link}}}$$

The left side belongs to the algorithm: how much math a parallelism scheme does per byte it sends. I'll call it the *compute-to-communication ratio*. The right side belongs to the hardware and acts as a ceiling the ratio has to clear. An H100 SXM does about 989 TFLOP/s of dense BF16 ([NVIDIA lists](https://www.nvidia.com/en-us/data-center/h100/) 1,979 with structured sparsity; dense is half), and its two links give very different ceilings. Cell 1 sets up the constants.

```python
%pip install -q "numpy==2.3.3" "matplotlib==3.10.8"
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

SEED = 0
rng = np.random.default_rng(SEED)
np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})

@dataclass(frozen=True)
class Link:
    """One interconnect: per-GPU one-way bandwidth and per-message latency."""

    name: str
    bandwidth: float  # bytes/s, unidirectional, per GPU
    latency: float  # seconds (alpha); illustrative order of magnitude

C_PEAK = 989e12  # H100 SXM5 dense BF16 FLOP/s
NVLINK = Link("NVLink 4 (intra-node)", 450e9, 3e-6)
IB = Link("NDR400 InfiniBand (inter-node)", 50e9, 8e-6)
NICS_PER_NODE = 8  # DGX-style H100 node: one NDR400 NIC per GPU
# A ring that spans nodes: NCCL runs one ring per NIC, so 1 hop in 8
# crosses nodes and the cross-node hops share all 8 NICs. Each GPU sends
# 1/8 of its bytes through its NIC and 7/8 over NVLink.
IB_RING = Link(
    "Ring across nodes (8 NICs)",
    min(
        NICS_PER_NODE * IB.bandwidth,
        NVLINK.bandwidth * NICS_PER_NODE / (NICS_PER_NODE - 1),
    ),
    IB.latency,  # every ring step waits for its slowest (cross-node) hop
)
BYTES_BF16 = 2

for link in (NVLINK, IB, IB_RING):
    print(
        f"{link.name:30s} {link.bandwidth / 1e9:3.0f} GB/s"
        f" | alpha {link.latency * 1e6:.0f} us"
        f" | C/B = {C_PEAK / link.bandwidth:6,.0f} FLOPs/byte"
    )
print(
    f"cliff per GPU: {NVLINK.bandwidth / IB.bandwidth:.0f}x"
    f" | a ring across nodes keeps"
    f" {IB_RING.bandwidth / NVLINK.bandwidth:.0%} of NVLink"
)
print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__}")
```
```text
NVLink 4 (intra-node)          450 GB/s | alpha 3 us | C/B =  2,198 FLOPs/byte
NDR400 InfiniBand (inter-node)  50 GB/s | alpha 8 us | C/B = 19,780 FLOPs/byte
Ring across nodes (8 NICs)     400 GB/s | alpha 8 us | C/B =  2,472 FLOPs/byte
cliff per GPU: 9x | a ring across nodes keeps 89% of NVLink
numpy 2.3.3 | matplotlib 3.10.8
```

On NVLink a GPU needs about 2,200 FLOPs per byte sent to stay compute-bound. On InfiniBand it needs 19,780, nine times as much math for every byte. The third line is a ring that spans both nodes, which the next section explains: it keeps 89% of NVLink's bandwidth, because only one hop in eight crosses the slow link. In the code `C_PEAK` is $$C$$, `link.bandwidth` is $$B_{\text{link}}$$ and `link.latency` is $$\alpha$$. I picked the latencies, 3 µs and 8 µs, as order-of-magnitude guesses. They matter for small messages in the next cell, and for the thousands of collectives per step in cell 10. NVIDIA's [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) racks push against the same gap by stretching one NVLink domain over 72 GPUs before anything touches InfiniBand.

The $$\alpha$$ term only matters for small messages. Divide a message's size by its transfer time to get the *effective* bandwidth it sees; it reaches half the link's peak at $$n_{1/2} = \alpha B_{\text{link}}$$. Cell 2 sweeps messages from 1 KB to 1 GB and marks one size that comes back later, a single TP activation.

```python
def transfer_time(n_bytes: np.ndarray, link: Link) -> np.ndarray:
    """Alpha-beta model: a fixed latency toll plus bytes / bandwidth."""
    return link.latency + n_bytes / link.bandwidth

def fmt_bytes(n: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.3g} {unit}"
    return f"{n:g} B"

# one b*s*h activation for b=1, s=4096, h=8192
TP_ACTIVATION = 1 * 4096 * 8192 * BYTES_BF16
sizes = np.logspace(3, 9, 300)  # 1 KB .. 1 GB

print(
    f"{'message':>9} | {'NVLink time':>11} {'eff. GB/s':>9}"
    f" | {'IB time':>9} {'eff. GB/s':>9}"
)
for n in (4e3, 64e3, 1e6, 16e6, TP_ACTIVATION, 1e9):
    row = []
    for link in (NVLINK, IB):
        t = transfer_time(np.array(n), link)
        row.append(f"{t * 1e6:9.1f}us {n / t / 1e9:9.1f}")
    print(f"{fmt_bytes(round(n, -3)):>9} | {row[0]:>11} | {row[1]}")
for link in (NVLINK, IB):
    # n_half: the message size at which half the peak bandwidth is reached
    n_half = fmt_bytes(link.latency * link.bandwidth)
    print(f"n_half({link.name.split()[0]}) = alpha * B = {n_half}")

fig, ax = plt.subplots(figsize=(8, 4.5))
for link, color in ((NVLINK, "tab:green"), (IB, "tab:red")):
    eff = sizes / transfer_time(sizes, link) / 1e9
    ax.loglog(sizes, eff, color=color, label=f"{link.name}")
    ax.axhline(link.bandwidth / 1e9, color=color, ls="--", lw=0.8)
    ax.axvline(link.latency * link.bandwidth, color=color, ls=":", lw=0.8)
ax.axvline(
    TP_ACTIVATION,
    color="black",
    lw=1.2,
    label="TP activation (b=1, s=4096, h=8192)",
)
ax.set_xlabel("message size (bytes)")
ax.set_ylabel("effective bandwidth (GB/s)")
ax.set_title("Alpha-beta model: latency-bound left, bandwidth-bound right")
ax.legend(loc="lower right", fontsize=8)
plt.show()
```
```text
  message | NVLink time eff. GB/s |   IB time eff. GB/s
     4 KB |       3.0us       1.3 |       8.1us       0.5
    64 KB |       3.1us      20.4 |       9.3us       6.9
     1 MB |       5.2us     191.5 |      28.0us      35.7
    16 MB |      38.6us     415.0 |     328.0us      48.8
  67.1 MB |     152.1us     441.1 |    1350.2us      49.7
     1 GB |    2225.2us     449.4 |   20008.0us      50.0
n_half(NVLink) = alpha * B = 1.35 MB
n_half(NDR400) = alpha * B = 400 KB
```
![Output of cell 2]({{ '/assets/blog/communication-roofline-3d-parallelism/cell-02-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The horizontal axis is message size and the vertical axis is the speed a message actually gets. Small messages on the left are slow on both links, because the fixed start-up cost dominates. Large messages on the right reach the dashed line, the link's full speed. The dotted lines mark where a message gets half that speed, and the black line is one tensor-parallel message, already well into the fast region.*

A 4 KB message gets 1.3 GB/s over NVLink. That is under 0.3% of the link, because the transfer time is nearly all latency. Half the peak arrives at $$n_{1/2}$$ = 1.35 MB on NVLink and 400 KB on InfiniBand, and by 16 MB both links are above 90%.

The black line is what TP sends in every layer: one micro-batch's activation, $$b \cdot s \cdot h$$ = 4,096 × 8,192 BF16 values = 67.1 MB. It moves at 441 GB/s over NVLink and 49.7 GB/s over InfiniBand. That is firmly bandwidth-bound, so from here on I drop $$\alpha$$ and count bytes. Small gradient buckets and inference decoding live in the latency regime, where NCCL often switches to tree algorithms. In-switch reduction (NVLS, SHARP) also changes the byte counts. Both are outside this post.

## What a collective costs: the ring

Parallel training leans on three collectives, each run by $$N$$ GPUs that each hold a tensor of $$M$$ bytes:

- A reduce-scatter sums the $$N$$ tensors element by element and leaves each GPU a different $$1/N$$ slice of the sum.
- An all-gather starts with each GPU holding a different $$1/N$$ slice and ends with every GPU holding all of them.
- An all-reduce leaves the full sum on every GPU. It is a reduce-scatter followed by an all-gather.

The ring algorithm sits the GPUs in a circle and cuts the tensor into $$N$$ chunks. In the reduce-scatter, each GPU sends one chunk to its right-hand neighbour at every step; the neighbour adds it to its own copy and passes the running sum on next step. After $$N-1$$ steps each chunk has gone nearly all the way round, picked up all $$N$$ contributions, and sits fully summed on one GPU. The all-gather runs the same loop again, copying finished chunks instead of adding. Every GPU sends exactly one chunk of $$M/N$$ bytes per step, so each sends

$$V_{\text{RS}} = V_{\text{AG}} = \frac{N-1}{N}M, \qquad V_{\text{AR}} = 2\,\frac{N-1}{N}M$$

The ring is bandwidth-optimal for all-reduce ([Patarasuk & Yuan, 2009](https://doi.org/10.1016/j.jpdc.2008.09.002)), and these are the same factors [nccl-tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) uses to turn a measured time into "bus bandwidth". Two consequences carry through the rest of the post. An all-reduce costs twice an all-gather or a reduce-scatter. And the bytes per GPU barely depend on $$N$$; what grows with $$N$$ is the number of steps, $$2(N-1)$$, and with it the latency term.

![Ring all-reduce on 8 GPUs: in each of 7 reduce-scatter steps every GPU sends one chunk clockwise and the receiver adds it, until each GPU owns one fully summed chunk; 7 all-gather steps then copy those chunks around until every GPU holds the full sum, for 2(N-1)/N times M bytes sent per GPU]({{ '/assets/blog/communication-roofline-3d-parallelism/ring-allreduce.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Eight GPUs sit in a circle, each with its own copy of a tensor cut into 8 pieces. In the first half, pieces hop one neighbour at a time and get added up along the way, until each GPU owns one fully summed piece. In the second half, those finished pieces go round again so every GPU ends up with the whole sum. No GPU ever talks to anyone but its neighbours.*

Cell 3 implements both halves in numpy. Each of the $$N$$ GPUs is a list of $$N$$ chunks, and the code moves them as described, counts the bytes each GPU sends, and checks the result against a direct sum.

```python
def ring_reduce_scatter(chunks: list[list[np.ndarray]]) -> np.ndarray:
    """In-place ring reduce-scatter; chunks[r][c] is rank r's chunk c.

    Returns the bytes each rank sent.
    """
    n = len(chunks)
    sent = np.zeros(n)
    for k in range(n - 1):
        # every rank sends one chunk to its right neighbour at the same time
        msgs = [
            (r, (r + 1) % n, (r - k) % n, chunks[r][(r - k) % n].copy())
            for r in range(n)
        ]
        for src, dst, c, payload in msgs:
            chunks[dst][c] = chunks[dst][c] + payload
            sent[src] += payload.nbytes
    return sent  # rank r now owns the fully reduced chunk (r + 1) % n

def ring_all_gather(chunks: list[list[np.ndarray]]) -> np.ndarray:
    """In-place ring all-gather; rank r starts owning chunk (r + 1) % n."""
    n = len(chunks)
    sent = np.zeros(n)
    for k in range(n - 1):
        msgs = []
        for r in range(n):
            c = (r + 1 - k) % n
            msgs.append((r, (r + 1) % n, c, chunks[r][c].copy()))
        for src, dst, c, payload in msgs:
            chunks[dst][c] = payload
            sent[src] += payload.nbytes
    return sent

ELEMS = 4096  # float32 per rank -> M = 16 KB, divisible by every N below
print(
    " N | RS sent/M  (N-1)/N | AG sent/M"
    " | AR sent/M  2(N-1)/N | all-reduce correct"
)
for n in (2, 4, 8, 16):
    data = [rng.integers(-5, 5, ELEMS).astype(np.float32) for _ in range(n)]
    total = sum(data)
    m_bytes = data[0].nbytes
    chunks = [list(np.split(x.copy(), n)) for x in data]
    rs = ring_reduce_scatter(chunks)
    owned_ok = all(
        np.allclose(chunks[r][(r + 1) % n], np.split(total, n)[(r + 1) % n])
        for r in range(n)
    )
    ag = ring_all_gather(chunks)
    ar_ok = owned_ok and all(
        np.allclose(np.concatenate(chunks[r]), total) for r in range(n)
    )
    print(
        f"{n:2d} | {rs.max() / m_bytes:9.4f}  {(n - 1) / n:7.4f}"
        f" | {ag.max() / m_bytes:9.4f}"
        f" | {(rs + ag).max() / m_bytes:9.4f}  {2 * (n - 1) / n:8.4f}"
        f" | {ar_ok}"
    )
```
```text
 N | RS sent/M  (N-1)/N | AG sent/M | AR sent/M  2(N-1)/N | all-reduce correct
 2 |    0.5000   0.5000 |    0.5000 |    1.0000    1.0000 | True
 4 |    0.7500   0.7500 |    0.7500 |    1.5000    1.5000 | True
 8 |    0.8750   0.8750 |    0.8750 |    1.7500    1.7500 | True
16 |    0.9375   0.9375 |    0.9375 |    1.8750    1.8750 | True
```

For every $$N$$ the byte counts match the formulas to four decimals and every GPU ends with exactly the direct sum, `total`. At $$N = 16$$ an all-reduce sends $$1.875M$$ per GPU, only 7% more than the $$1.75M$$ at $$N = 8$$. Widening a ring costs almost nothing in bytes. What it does cost is steps, $$2(N-1)$$ of them, and each step waits for its slowest hop.

That matters when a ring spans two nodes. In a ring of 16 GPUs over two nodes, only 2 of the 16 hops cross between nodes. NCCL runs several rings at once, one per NIC, and in the ideal case each GPU's NIC carries the crossing hop of exactly one of them. Each GPU then sends 1/8 of its bytes over InfiniBand and 7/8 over NVLink, and the cross-node share rides on all eight NICs at once, 8 × 50 = 400 GB/s. That is the third line in cell 1, 89% of NVLink. So a ring across nodes loses little bandwidth. It loses time on latency instead: every one of its 30 steps now waits for a cross-node hop, 8 µs instead of 3 µs in this model.

## Placing each dimension on the roofline

With the ring volumes in hand, each dimension's ratio is one division. I use a full training step (forward plus backward) throughout, a micro-batch of $$b$$ sequences of length $$s$$, hidden size $$h$$, and BF16 activations at 2 bytes per value.

For tensor parallelism, a transformer layer costs about $$24bsh^2$$ FLOPs forward and twice that backward, $$72bsh^2$$ in all, ignoring the attention-score term that grows with $$s^2$$. [Narayanan et al. (2021)](https://arxiv.org/abs/2104.04473) count $$96bslh^2$$ per iteration for $$l$$ layers because they include recomputing the forward pass; without recomputation that is $$72bsh^2$$ per layer. TP splits those FLOPs over $$t$$ GPUs, and Megatron's layout needs two all-reduces forward and two backward, each on the $$b \times s \times h$$ activation of $$2bsh$$ bytes ([Shoeybi et al., 2019](https://arxiv.org/abs/1909.08053)). With the ring volume:

$$\frac{\text{FLOPs}}{\text{bytes}} = \frac{72bsh^2/t}{4 \cdot 2\frac{t-1}{t} \cdot 2bsh} = \frac{4.5\,h}{t-1}$$

The batch dimensions $$b$$ and $$s$$ cancel. A bigger batch adds compute and activation bytes in the same proportion, so nothing you do to the batch moves TP's ratio; only the width $$h$$ and the degree $$t$$ do. Worse, every one of these all-reduces sits on the critical path. The next matrix multiply needs the summed activation, so the communication can't hide behind compute the way the roofline assumes.

![Megatron tensor-parallel MLP: every GPU multiplies the full input X by its column slice of A, applies GeLU, multiplies by its row slice of B to get a partial output, and an all-reduce of 2bsh bytes sums the partials before the next layer can start]({{ '/assets/blog/communication-roofline-3d-parallelism/tp-mlp.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Every GPU gets the same input X but a different slice of the layer's weights, so each one computes a partial answer on its own. The red all-reduce adds the partial answers together. The next layer can't start until that sum arrives, which is why this communication can't hide behind computation.*

Sequence parallelism swaps each all-reduce for a reduce-scatter plus an all-gather of the same total size ([Korthikanti et al., 2022](https://arxiv.org/abs/2205.05198)), which leaves the ratio alone. The $$3h/t$$ you may see in some write-ups counts the forward pass only and rounds the ring factor $$(t-1)/t$$ to 1.

For pipeline parallelism, a stage holds $$L/p$$ layers. Per micro-batch it receives one $$2bsh$$-byte activation going forward and sends one gradient of the same size going back:

$$\frac{72bsh^2 \cdot L/p}{2 \cdot 2bsh} = 18\,\frac{L}{p}\,h$$

A whole stage of compute against one tensor of bytes makes this ratio enormous.

For data parallelism, a training step costs about $$6\Psi$$ FLOPs per token for a model with $$\Psi$$ parameters ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)). If each GPU processes $$n_{\text{tok}}$$ tokens per step, DDP all-reduces $$\Psi$$ BF16 gradients once and FSDP moves 1.5x as much:

$$\text{DDP: } \frac{6\,n_{\text{tok}}\,\Psi}{2\frac{d-1}{d} \cdot 2\Psi} \approx 1.5\,n_{\text{tok}}$$

$$\text{FSDP: } \frac{6\,n_{\text{tok}}\,\Psi}{3\frac{d-1}{d} \cdot 2\Psi} \approx n_{\text{tok}}$$

Here $$\Psi$$ cancels and $$n_{\text{tok}}$$ survives, so the ratio grows with the tokens each GPU processes between synchronizations, which you set through micro-batch size and gradient accumulation. The communication also overlaps well. DDP fires gradient all-reduces bucket by bucket while backward is still running, and FSDP prefetches the next layer's all-gather while the current one computes ([Zhao et al., 2023](https://arxiv.org/abs/2304.11277)).

Cell 4 evaluates the four formulas for the 70B-class model ($$h = 8192$$, $$L = 80$$) against the ceiling of the link each would run over.

```python
H, L_LAYERS = 8192, 80  # a 70B-class transformer: hidden size, layers

def tp_ratio(h: int, t: int) -> float:
    """Megatron TP per token per layer: 72h^2/t FLOPs (fwd+bwd) vs
    4 ring all-reduces of h BF16 values."""
    flops = 72 * h**2 / t
    comm = 4 * 2 * (t - 1) / t * BYTES_BF16 * h
    return flops / comm  # = 4.5 h / (t - 1): batch and sequence cancel

def pp_ratio(h: int, layers: int, p: int) -> float:
    """PP per token: L/p layers of compute vs one activation sent
    forward and one gradient sent back."""
    return 72 * (layers / p) * h**2 / (2 * BYTES_BF16 * h)  # 18 (L/p) h

def ddp_ratio(tokens: float, d: int) -> float:
    """DDP: 6 * tokens * Psi FLOPs vs a ring all-reduce of Psi BF16
    gradients (Psi cancels)."""
    return 6 * tokens / (2 * (d - 1) / d * BYTES_BF16)

def fsdp_ratio(tokens: float, d: int) -> float:
    """FSDP / ZeRO-3: 2 all-gathers + 1 reduce-scatter = 1.5x DDP."""
    return 6 * tokens / (3 * (d - 1) / d * BYTES_BF16)

rows = [
    ("TP t=8 (in node)", tp_ratio(H, 8), NVLINK),
    ("TP t=16 (2 nodes)", tp_ratio(H, 16), IB_RING),
    ("PP p=8, L=80", pp_ratio(H, L_LAYERS, 8), IB),
    ("DDP d=16, 8k tok", ddp_ratio(8192, 16), IB),
    ("DDP d=16, 32k tok", ddp_ratio(32768, 16), IB),
    ("FSDP d=16, 8k tok", fsdp_ratio(8192, 16), IB),
    ("FSDP d=16, 32k tok", fsdp_ratio(32768, 16), IB),
]
print(
    f"{'dimension':18s} {'FLOPs/B':>9} {'link':>6} {'ceiling':>7}"
    f" {'ratio/ceil':>10}  verdict"
)
for label, ratio, link in rows:
    ceiling = C_PEAK / link.bandwidth
    if ratio >= ceiling:
        verdict = "compute-bound"
    else:
        verdict = f"comm, {ceiling / ratio:.1f}x short"
    print(
        f"{label:18s} {ratio:9,.0f} {link.name.split()[0]:>6}"
        f" {ceiling:7,.0f} {ratio / ceiling:10.2f}  {verdict}"
    )

print()
for t, link in ((8, NVLINK), (16, IB_RING)):
    # TP cannot overlap, so comm time adds to compute time
    exposed = C_PEAK / link.bandwidth / tp_ratio(H, t)
    print(
        f"TP t={t:<2d}: comm = {exposed:.2f}x compute, no overlap"
        f" -> at most {100 / (1 + exposed):.0f}% of peak"
    )
ceiling_ib = C_PEAK / IB.bandwidth
print(
    f"tokens/GPU to clear IB at d=16:"
    f" DDP {ceiling_ib / ddp_ratio(1, 16):,.0f}"
    f" | FSDP {ceiling_ib / fsdp_ratio(1, 16):,.0f}"
)
```
```text
dimension            FLOPs/B   link ceiling ratio/ceil  verdict
TP t=8 (in node)       5,266 NVLink   2,198       2.40  compute-bound
TP t=16 (2 nodes)      2,458   Ring   2,472       0.99  comm, 1.0x short
PP p=8, L=80       1,474,560 NDR400  19,780      74.55  compute-bound
DDP d=16, 8k tok      13,107 NDR400  19,780       0.66  comm, 1.5x short
DDP d=16, 32k tok     52,429 NDR400  19,780       2.65  compute-bound
FSDP d=16, 8k tok      8,738 NDR400  19,780       0.44  comm, 2.3x short
FSDP d=16, 32k tok    34,953 NDR400  19,780       1.77  compute-bound

TP t=8 : comm = 0.42x compute, no overlap -> at most 71% of peak
TP t=16: comm = 1.01x compute, no overlap -> at most 50% of peak
tokens/GPU to clear IB at d=16: DDP 12,362 | FSDP 18,544
```

TP at $$t = 8$$ inside a node scores 5,266 FLOPs/byte against an NVLink ceiling of 2,198, 2.4x of headroom, which the roofline calls compute-bound. The roofline assumes overlap, though, and TP gets almost none. The first line under the table gives the real bill. The all-reduces add 0.42x of the compute time, which caps TP = 8 at 71% of peak, and that is most of why the single-node baseline later runs at 67% MFU rather than close to 100%.

Push TP to $$t = 16$$ across two nodes and the ratio drops to 2,458. Doubling the GPUs roughly halves the ratio, since each GPU's share of the compute halves while the bytes it sends stay almost the same. The ring now spans both nodes and runs at an effective 400 GB/s, a ceiling of 2,472, so on bandwidth alone TP sits right at the knee. Because TP can't overlap, though, its communication now takes as long as its compute (1.01x on the second line under the table), which caps TP = 16 at 50% of peak before any latency is counted.

PP with $$p = 8$$ sits at 1.47 million FLOPs/byte, 74x above the InfiniBand ceiling. Bandwidth hardly touches PP. Its cost is idle time, which gets its own section.

DDP and FSDP at $$d = 16$$ depend on tokens per GPU. At 8k tokens both are communication-bound (1.5x and 2.3x short); at 32k both clear the ceiling. The break-even points are 12,362 tokens per GPU for DDP and 18,544 for FSDP, three to five 4,096-token sequences per GPU per step, which is an ordinary batch.

Cell 5 draws the same numbers as a roofline. The horizontal axis is the ratio and the vertical axis is the FLOP/s a GPU can reach, $$\min(C,\ \text{ratio} \times B_{\text{link}})$$. Each link gives a bent line that rises, then flattens at the GPU's peak. TP gets a curve of its own. Its communication adds to its compute instead of hiding behind it, so with ratio $$r$$ the reachable rate is $$C \cdot rB/(rB + C)$$, the dashed purple line, which bends gradually and never quite reaches the peak.

```python
intensity = np.logspace(2, 7, 400)
PURPLE, BLUE, ORANGE = "tab:purple", "tab:blue", "tab:orange"
points = (  # (label, ratio, link, color, overlaps compute?)
    [(f"TP t={t}", tp_ratio(H, t), NVLINK, PURPLE, False) for t in (4, 8)]
    + [
        (f"TP t={t}", tp_ratio(H, t), IB_RING, PURPLE, False)
        for t in (16, 32)
    ]
    + [
        (f"FSDP {k}k tok", fsdp_ratio(k * 1024, 16), IB, BLUE, True)
        for k in (4, 8, 32, 128)
    ]
    + [("PP p=8", pp_ratio(H, L_LAYERS, 8), IB, ORANGE, True)]
)

def attainable(ratio: float, link: Link, overlap: bool) -> float:
    """min(C, ratio*B) if comm hides behind compute (the roofline);
    C*rB/(rB + C) if comm and compute add up, as in TP."""
    comm_rate = ratio * link.bandwidth
    if overlap:
        return min(C_PEAK, comm_rate)
    return C_PEAK * comm_rate / (comm_rate + C_PEAK)

fig, ax = plt.subplots(figsize=(9, 5))
for link, color in ((NVLINK, "tab:green"), (IB, "tab:red")):
    roof = np.minimum(C_PEAK, intensity * link.bandwidth) / 1e12
    ax.loglog(intensity, roof, color=color, label=link.name)
    ax.axvline(C_PEAK / link.bandwidth, color=color, ls=":", lw=0.8)
no_overlap = [attainable(x, NVLINK, False) / 1e12 for x in intensity]
ax.loglog(
    intensity, no_overlap, color=PURPLE, ls="--", lw=1,
    label="NVLink, no overlap (TP)",
)
print(f"{'point':14s} | {'FLOPs/byte':>10} | link   | overlap | % of peak")
for label, ratio, link, color, overlap in points:
    flops = attainable(ratio, link, overlap)
    ax.scatter(ratio, flops / 1e12, color=color, zorder=3)
    ax.annotate(
        label, (ratio, flops / 1e12), textcoords="offset points",
        xytext=(4, -10), fontsize=7,
    )
    print(
        f"{label:14s} | {ratio:10,.0f} | {link.name.split()[0]:6s}"
        f" | {'yes' if overlap else 'no':7s} | {100 * flops / C_PEAK:8.1f}%"
    )
ax.set_xlabel("compute-to-communication ratio (FLOPs per byte sent)")
ax.set_ylabel("attainable TFLOP/s per GPU")
ax.set_title("Communication roofline: NVLink and InfiniBand ceilings (H100)")
ax.legend(loc="lower right", fontsize=8)
plt.show()
```
```text
point          | FLOPs/byte | link   | overlap | % of peak
TP t=4         |     12,288 | NVLink | no      |     84.8%
TP t=8         |      5,266 | NVLink | no      |     70.6%
TP t=16        |      2,458 | Ring   | no      |     49.8%
TP t=32        |      1,189 | Ring   | no      |     32.5%
FSDP 4k tok    |      4,369 | NDR400 | yes     |     22.1%
FSDP 8k tok    |      8,738 | NDR400 | yes     |     44.2%
FSDP 32k tok   |     34,953 | NDR400 | yes     |    100.0%
FSDP 128k tok  |    139,810 | NDR400 | yes     |    100.0%
PP p=8         |  1,474,560 | NDR400 | yes     |    100.0%
```
![Output of cell 5]({{ '/assets/blog/communication-roofline-3d-parallelism/cell-05-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Each solid line is a link. A GPU's speed climbs with the ratio until it hits the flat roof at 989 TFLOP/s. Green is NVLink; red is InfiniBand, which needs 9x more math per byte to reach the roof. Data and pipeline parallelism (blue and orange) hide their communication behind compute, so they ride the red line and reach the roof once their ratio is big enough. Tensor parallelism (purple) can't hide its communication, so it sits on the dashed curve and loses speed every time its degree doubles.*

On the slope the GPU waits on the wire; on the flat roof it waits on its own tensor cores. The InfiniBand line bends 9x further right than the NVLink one.

TP's points slide left and down the dashed curve as $$t$$ grows: 84.8% of peak at $$t = 4$$, 70.6% at $$t = 8$$, 49.8% at $$t = 16$$ across two nodes and 32.5% at $$t = 32$$. Crossing the node boundary barely moves the curve; the damage comes from sliding left along it. FSDP's points slide right as tokens per GPU grow: 22.1% of peak at 4k tokens, 44.2% at 8k, on the roof by 32k. PP sits far off to the right.

That picture is the placement rule. FSDP and PP sit on the InfiniBand line, the steep one, but each has a setting that pushes its point past the red knee: tokens per GPU for data parallelism, layers per stage for pipeline parallelism. Tensor parallelism has no such setting. Its point only moves left as you add GPUs, and because its communication never hides, every step left costs throughput.

![Communication roofline with NVLink and InfiniBand ceilings: as tokens per GPU grow from 2k to 128k the FSDP point climbs the InfiniBand slope onto the 989 TFLOP/s roof, while as the TP degree grows from 2 to 32 the TP point slides down the dashed no-overlap curve from 94% to about a third of peak, with only a small extra drop when it crosses the node boundary after t = 8]({{ '/assets/blog/communication-roofline-3d-parallelism/roofline-point.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The blue dot is FSDP. As each GPU handles more tokens per step, it climbs the red InfiniBand line until it reaches the roof, so crossing nodes is fine once the batch is big enough. The purple dot is tensor parallelism. As more GPUs share each layer, it slides down its dashed curve, and moving into the second node after t = 8 adds only a small extra dip. The readout in the corner shows each one's share of peak speed.*

## The pipeline bubble is an area

PP passes the bandwidth test easily and pays in idle time instead. With $$p$$ stages, the first micro-batch crosses $$p-1$$ stages before the last one has any work, and the backward passes drain out the same way at the end. On a Gantt chart (time across, GPUs down) the idle time forms a triangle at each end, and the share of the chart they cover is the *bubble fraction*. For GPipe, which runs every forward and then every backward ([Huang et al., 2018](https://arxiv.org/abs/1811.06965)), it is

$$\text{bubble} = \frac{p-1}{m+p-1}$$

The one-forward-one-backward (1F1B) pattern comes from PipeDream ([Harlap et al., 2018](https://arxiv.org/abs/1806.03377)), which ran it asynchronously with no flush between steps. Megatron-LM runs the synchronous version, PipeDream-Flush, which drains the pipeline at the end of every step ([Narayanan et al., 2021](https://arxiv.org/abs/2104.04473)). That synchronous 1F1B is often credited with shrinking the bubble, and it doesn't. 1F1B starts each micro-batch's backward as early as it can, which frees that micro-batch's stored activations early. A stage then holds at most $$p$$ micro-batches of activations, where GPipe holds all $$m$$. The win is memory.

What does shrink the bubble is *interleaved* 1F1B ([Narayanan et al., 2021](https://arxiv.org/abs/2104.04473)). Each GPU holds $$v$$ smaller, non-adjacent chunks of layers; with 80 layers, 4 GPUs and $$v = 2$$, GPU 0 holds layers 1 to 10 and 41 to 50. Every fill or drain step then takes $$1/v$$ as long, so

$$\text{bubble}_{\text{interleaved}} = \frac{p-1}{vm+p-1}$$

It costs $$v$$ times as many point-to-point messages and, as the simulator shows, more activation memory than plain 1F1B.

Each schedule becomes a fixed order of operations per GPU. `F3` is the forward pass of micro-batch 3 and `B3` its backward. In the interleaved schedule, `F3.1` means micro-batch 3 running on the GPU's second chunk of layers. Cell 6 builds the three orders for $$p = 4$$ stages and $$m = 8$$ micro-batches and prints them for the first and last GPU.

```python
@dataclass(frozen=True)
class Op:
    kind: str  # "F" (forward) or "B" (backward)
    mb: int  # micro-batch index
    chunk: int  # model chunk on this device, 0..v-1

def gpipe_order(p: int, m: int, d: int) -> list[Op]:
    """All forwards, then all backwards."""
    fwd = [Op("F", j, 0) for j in range(m)]
    return fwd + [Op("B", j, 0) for j in range(m)]

def one_f_one_b_order(p: int, m: int, d: int) -> list[Op]:
    """Warm up with p-d-1 forwards, alternate F/B, then drain."""
    warm = min(p - d - 1, m)
    ops = [Op("F", j, 0) for j in range(warm)]
    f, b = warm, 0
    while f < m:
        ops += [Op("F", f, 0), Op("B", b, 0)]
        f, b = f + 1, b + 1
    return ops + [Op("B", j, 0) for j in range(b, m)]

def interleaved_order(p: int, m: int, d: int, v: int) -> list[Op]:
    """Megatron-LM interleaved 1F1B: v model chunks per device,
    micro-batches in groups of p."""
    total = m * v

    def fwd(k: int) -> Op:
        return Op("F", (k // (p * v)) * p + k % p, (k // p) % v)

    def bwd(k: int) -> Op:
        return Op("B", (k // (p * v)) * p + k % p, v - 1 - (k // p) % v)

    warm = min((p - d - 1) * 2 + (v - 1) * p, total)
    ops = [fwd(k) for k in range(warm)]
    f, b = warm, 0
    while f < total:
        ops += [fwd(f), bwd(b)]
        f, b = f + 1, b + 1
    return ops + [bwd(k) for k in range(b, total)]

def show_order(ops: list[Op], v: int, limit: int = 16) -> str:
    tags = [f"{o.kind}{o.mb}" + (f".{o.chunk}" if v > 1 else "") for o in ops]
    return " ".join(tags[:limit]) + (" ..." if len(tags) > limit else "")

P, M, V = 4, 8, 2
print(f"p={P} stages, m={M} micro-batches")
print("F3 = forward of micro-batch 3, B3 = its backward, F3.1 = on chunk 1")
for d in (0, P - 1):
    print(f"GPU {d} GPipe : {show_order(gpipe_order(P, M, d), 1)}")
    print(f"GPU {d} 1F1B  : {show_order(one_f_one_b_order(P, M, d), 1)}")
    inter = show_order(interleaved_order(P, M, d, V), V, limit=12)
    print(f"GPU {d} inter.: {inter}")
```
```text
p=4 stages, m=8 micro-batches
F3 = forward of micro-batch 3, B3 = its backward, F3.1 = on chunk 1
GPU 0 GPipe : F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
GPU 0 1F1B  : F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
GPU 0 inter.: F0.0 F1.0 F2.0 F3.0 F0.1 F1.1 F2.1 F3.1 F4.0 F5.0 F6.0 B0.1 ...
GPU 3 GPipe : F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
GPU 3 1F1B  : F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7
GPU 3 inter.: F0.0 F1.0 F2.0 F3.0 F0.1 B0.1 F1.1 B1.1 F2.1 B2.1 F3.1 B3.1 ...
```

Under GPipe, GPU 0 runs all eight forwards, then all eight backwards. Under 1F1B it runs four forwards ($$p-d-1 = 3$$ warm-up forwards plus the first steady-state one) and then alternates `B0 F4 B1 F5 ...`, so it never holds more than four micro-batches. GPU 3, the last stage, has nothing to wait for and alternates from the start: `F0 B0 F1 B1 ...`. The interleaved GPU 0 walks through micro-batches in groups of $$p = 4$$, first on chunk 0 and then on chunk 1.

Cell 7 runs those orders. An operation starts once its GPU is free and its input exists. A forward needs the same micro-batch's forward from the previous stage; a backward needs the gradient from the next stage. A forward costs 1 time unit and a backward 2, both divided by $$v$$ for interleaved chunks. The cell reports the makespan, the idle fraction and the peak number of micro-batches whose activations are stored at once.

```python
def simulate(
    orders: list[list[Op]], v: int, t_fwd: float = 1.0, t_bwd: float = 2.0
) -> list[list[tuple]]:
    """Run each device's fixed op order as early as dependencies allow.

    Virtual stage = chunk * p + device.
    """
    p, n_stages = len(orders), len(orders) * v
    done: dict[tuple, float] = {}
    ptr, free = [0] * p, [0.0] * p
    timeline: list[list[tuple]] = [[] for _ in range(p)]
    while any(ptr[d] < len(orders[d]) for d in range(p)):
        progressed = False
        for d in range(p):
            if ptr[d] == len(orders[d]):
                continue
            op = orders[d][ptr[d]]
            vs = op.chunk * p + d
            if op.kind == "F":
                dep = ("F", op.mb, vs - 1) if vs > 0 else None
            elif vs < n_stages - 1:  # backward waits for the next stage
                dep = ("B", op.mb, vs + 1)
            else:  # ...or, on the last stage, for its own forward
                dep = ("F", op.mb, vs)
            if dep is not None and dep not in done:
                continue
            start = max(free[d], done.get(dep, 0.0))
            end = start + (t_fwd if op.kind == "F" else t_bwd) / v
            done[(op.kind, op.mb, vs)] = end
            timeline[d].append((start, end, op))
            free[d], ptr[d], progressed = end, ptr[d] + 1, True
        if not progressed:
            raise RuntimeError("schedule deadlocked")
    return timeline

def schedule_stats(
    timeline: list[list[tuple]], v: int
) -> tuple[float, float, float]:
    """Makespan, bubble fraction (idle share of device-time) and peak
    in-flight activations, in micro-batches."""
    makespan = max(end for dev in timeline for _, end, _ in dev)
    busy = sum(end - start for dev in timeline for start, end, _ in dev)
    peak = 0.0
    for dev in timeline:
        live = 0.0
        for _, _, op in dev:
            live += (1.0 if op.kind == "F" else -1.0) / v
            peak = max(peak, live)
    return makespan, 1 - busy / (len(timeline) * makespan), peak

schedules = {
    "GPipe": (simulate([gpipe_order(P, M, d) for d in range(P)], 1), 1),
    "1F1B": (simulate([one_f_one_b_order(P, M, d) for d in range(P)], 1), 1),
    f"Interleaved (v={V})": (
        simulate([interleaved_order(P, M, d, V) for d in range(P)], V),
        V,
    ),
}
print(f"p={P} stages, m={M} micro-batches, backward = 2x forward")
print(
    f"{'schedule':17s} | makespan | bubble sim | formula"
    " | peak activations"
)
for name, (tl, v) in schedules.items():
    makespan, bubble, peak = schedule_stats(tl, v)
    formula = (P - 1) / (v * M + P - 1)
    print(
        f"{name:17s} | {makespan:8.1f} | {bubble:10.3f} | {formula:7.3f}"
        f" | {peak:5.1f} micro-batches"
    )
```
```text
p=4 stages, m=8 micro-batches, backward = 2x forward
schedule          | makespan | bubble sim | formula | peak activations
GPipe             |     33.0 |      0.273 |   0.273 |   8.0 micro-batches
1F1B              |     33.0 |      0.273 |   0.273 |   4.0 micro-batches
Interleaved (v=2) |     28.5 |      0.158 |   0.158 |   5.5 micro-batches
```

GPipe and 1F1B finish together at 33.0 units with the same bubble, 0.273, which is the formula's $$(4-1)/(8+4-1) = 3/11$$. 1F1B only shows up in the last column: 4.0 micro-batches of stored activations against GPipe's 8.0. Interleaving with $$v = 2$$ finishes at 28.5 units with a bubble of 0.158 $$= 3/19$$ and holds 5.5 micro-batches of activations, more than 1F1B. Cell 8 draws the three timelines.

```python
colors = {
    ("F", 0): "#4C72B0",
    ("F", 1): "#9DB9E0",
    ("B", 0): "#DD8452",
    ("B", 1): "#F2BE95",
}
fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
for ax, (name, (tl, v)) in zip(axes, schedules.items()):
    for d, dev in enumerate(tl):
        for start, end, op in dev:
            ax.broken_barh(
                [(start, end - start)], (d - 0.4, 0.8),
                color=colors[(op.kind, op.chunk)], edgecolor="white",
            )
            ax.text(
                (start + end) / 2, d, str(op.mb),
                ha="center", va="center", fontsize=6,
            )
    ax.set_yticks(range(P), [f"GPU {d}" for d in range(P)])
    ax.invert_yaxis()
    ax.set_title(
        f"{name}: blue = forward, orange = backward, white gaps = bubble",
        fontsize=9,
    )
    ax.grid(False)
axes[-1].set_xlabel("time (units of one full-stage forward)")
plt.tight_layout()
plt.show()
```
![Output of cell 8]({{ '/assets/blog/communication-roofline-3d-parallelism/cell-08-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Each row is one GPU and time runs left to right. Blue blocks are forward passes, orange blocks are backward passes, and the number in a block is the micro-batch. White gaps are time a GPU sits idle. GPipe and 1F1B have the same total white area. The interleaved schedule's gaps are smaller.*

GPipe's idle time is two clean staircases, one while forwards fill the pipe and a mirror image while backwards drain it. 1F1B shuffles the blue and orange blocks without removing any white space. Its idle time becomes a fill staircase plus small gaps between the last backwards on the early GPUs. Same total area. In the bottom panel every block is half as long, so both triangles are about half as wide.

![Gantt chart of 4 GPUs filling in over time: the 1F1B schedule leaves grey idle triangles at the start and end for a bubble of 0.273, and the interleaved schedule with v = 2 halves each block and shrinks the bubble to 0.158]({{ '/assets/blog/communication-roofline-3d-parallelism/pipeline-bubble.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The grid fills in as 4 GPUs work through 8 micro-batches. The grey triangles at the start and end are GPUs waiting, either for the first micro-batch to reach them or for the last backward passes to come back. The counter shows the share of time spent idle. When each GPU holds two smaller chunks of layers (interleaved), every block gets shorter, and the idle share falls from 27% to 16%.*

Cell 9 repeats the simulation for $$m$$ from 4 to 64 and plots it against both formulas.

```python
def bubble_of(order_fn, m: int, v: int = 1) -> float:
    """Simulated bubble fraction for one schedule at m micro-batches."""
    if v == 1:
        orders = [order_fn(P, m, d) for d in range(P)]
    else:
        orders = [order_fn(P, m, d, v) for d in range(P)]
    return schedule_stats(simulate(orders, v), v)[1]

ms = np.array([4, 8, 16, 32, 64])
sim = {
    "GPipe": [bubble_of(gpipe_order, m) for m in ms],
    "1F1B": [bubble_of(one_f_one_b_order, m) for m in ms],
    "interleaved": [bubble_of(interleaved_order, m, V) for m in ms],
}
formula_1 = (P - 1) / (ms + P - 1)
formula_v = (P - 1) / (V * ms + P - 1)

print(
    " m | GPipe sim | 1F1B sim | (p-1)/(m+p-1)"
    " | interleaved sim | (p-1)/(vm+p-1)"
)
for i, m in enumerate(ms):
    print(
        f"{m:2d} | {sim['GPipe'][i]:9.3f} | {sim['1F1B'][i]:8.3f}"
        f" | {formula_1[i]:13.3f} | {sim['interleaved'][i]:15.3f}"
        f" | {formula_v[i]:14.3f}"
    )

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ms, formula_1, color="tab:blue", label="(p-1)/(m+p-1)  GPipe = 1F1B")
ax.plot(
    ms, formula_v, color="tab:orange",
    label=f"(p-1)/(vm+p-1)  interleaved, v={V}",
)
ax.scatter(
    ms, sim["GPipe"], marker="o", s=60, facecolors="none",
    edgecolors="tab:blue", label="GPipe (sim)",
)
ax.scatter(ms, sim["1F1B"], marker="x", color="tab:blue", label="1F1B (sim)")
ax.scatter(
    ms, sim["interleaved"], marker="s", color="tab:orange",
    label="interleaved (sim)",
)
ax.set_xscale("log", base=2)
ax.set_xlabel("micro-batches per step (m)")
ax.set_ylabel("bubble fraction")
ax.set_title(f"Pipeline bubble vs micro-batches (p={P})")
ax.legend(fontsize=8)
plt.show()
```
```text
 m | GPipe sim | 1F1B sim | (p-1)/(m+p-1) | interleaved sim | (p-1)/(vm+p-1)
 4 |     0.429 |    0.429 |         0.429 |           0.273 |          0.273
 8 |     0.273 |    0.273 |         0.273 |           0.158 |          0.158
16 |     0.158 |    0.158 |         0.158 |           0.086 |          0.086
32 |     0.086 |    0.086 |         0.086 |           0.045 |          0.045
64 |     0.045 |    0.045 |         0.045 |           0.023 |          0.023
```
![Output of cell 9]({{ '/assets/blog/communication-roofline-3d-parallelism/cell-09-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The horizontal axis is how many micro-batches the batch is cut into, and the vertical axis is the share of time GPUs sit idle. Lines are the formulas, markers are the simulator, and they land on top of each other. Cutting the batch into more micro-batches means less idle time, and interleaving (orange) roughly halves it again.*

Every simulated point lands on its formula to three decimals, and GPipe and 1F1B agree at every $$m$$. That leaves two levers. Raising $$m$$ from 8 to 64 cuts the bubble from 0.273 to 0.045, but $$m$$ counts micro-batches per optimizer step, so the global batch you can afford caps it. Interleaving roughly halves the bubble at any $$m$$ (0.045 to 0.023 at $$m = 64$$) and pays in messages and memory. Both work through $$m$$ and $$v$$. Bandwidth never enters the formula. When PP crosses nodes, the only bandwidth bill is the point-to-point transfer, which is small next to the bubble.

## Back to the paradox: predicting the slowdown

Cell 10 puts it together for one optimizer step of 262,144 tokens (64 sequences of 4,096) on the same model, whose parameter count follows from the $$24bsh^2$$ FLOP count: $$\Psi = 12Lh^2 \approx$$ 64.4B. A configuration's step time is the sum of five terms:

- Compute: FLOPs per GPU divided by $$C$$, at 100% matrix-multiply efficiency.
- TP communication: 4 all-reduces per layer per micro-batch, never overlapped. They run over NVLink if $$t \le 8$$; for TP = 16 they run on the two-node ring at 400 GB/s, and each of the $$2(t-1)$$ ring steps also pays the latency of its slowest hop.
- The PP bubble: $$(p-1)/m$$ of the stage time, with one sequence per micro-batch.
- PP activation transfers over InfiniBand.
- Whatever part of the DP gradient all-reduce doesn't fit behind the backward pass, which takes two-thirds of compute.

The four configurations cover what you can do with 8 or 16 GPUs: the single-node baseline, TP stretched to 16, and TP = 8 with either DP = 2 or PP = 2.

```python
SEQ = 4096
GLOBAL_TOKENS = 64 * SEQ  # one optimizer step: 64 sequences of 4,096 tokens
PSI = 12 * L_LAYERS * H**2  # params implied by 24bsh^2 fwd FLOPs (~64B)
GPUS_PER_NODE = 8

def step_time(
    t: int, p: int, d: int, ring: Link = IB_RING
) -> dict[str, float]:
    """Seconds per optimizer step for TP=t, PP=p, DP=d (alpha-beta
    model, 100% GEMM efficiency). `ring` is TP's link across nodes."""
    tokens = GLOBAL_TOKENS / d  # tokens each model replica processes
    layers = L_LAYERS / p
    compute = 72 * layers * H**2 * tokens / t / C_PEAK
    tp_link = NVLINK if t <= GPUS_PER_NODE else ring
    m = tokens / SEQ  # micro-batches of one sequence
    # 4 all-reduces per layer per micro-batch, never overlapped
    tp_bytes = 16 * H * (t - 1) / t * layers * tokens
    tp_bw = tp_bytes / tp_link.bandwidth
    tp_lat = 4 * layers * m * 2 * (t - 1) * tp_link.latency  # 2(t-1) steps
    tp_comm = tp_bw + tp_lat
    bubble = (compute + tp_comm) * (p - 1) / m
    p2p = tokens * H * BYTES_BF16 / IB.bandwidth if p > 1 else 0.0
    dp_comm = 2 * (d - 1) / d * BYTES_BF16 * PSI / (t * p) / IB.bandwidth
    # the gradient all-reduce hides behind the backward pass (2/3 of compute)
    dp_exposed = max(0.0, dp_comm - 2 / 3 * compute)
    parts = {
        "compute": compute,
        "TP comm": tp_comm,
        "PP bubble": bubble,
        "PP p2p": p2p,
        "DP exposed": dp_exposed,
    }
    parts["step"] = sum(parts.values())
    parts["tp_latency"] = tp_lat
    ideal = 72 * L_LAYERS * H**2 * GLOBAL_TOKENS / C_PEAK
    parts["mfu"] = ideal / (t * p * d * parts["step"])
    return parts

configs = [
    ("TP=8, 1 node", 8, 1, 1),
    ("TP=16, 2 nodes", 16, 1, 1),
    ("TP=8 x DP=2", 8, 1, 2),
    ("TP=8 x PP=2", 8, 2, 1),
]
results = {name: step_time(t, p, d) for name, t, p, d in configs}
base = results[configs[0][0]]["step"]
print(
    f"h={H}, L={L_LAYERS}, Psi={PSI / 1e9:.1f}B,"
    f" {GLOBAL_TOKENS:,} tokens per step; times in seconds"
)
print(
    f"{'config':14s} | compute | TP comm |   PP |   DP |  step"
    " | speedup | MFU"
)
for name, r in results.items():
    pp = r["PP bubble"] + r["PP p2p"]
    print(
        f"{name:14s} | {r['compute']:7.2f} | {r['TP comm']:7.2f}"
        f" | {pp:4.2f} | {r['DP exposed']:4.2f} | {r['step']:5.2f}"
        f" | {base / r['step']:6.2f}x | {100 * r['mfu']:2.0f}%"
    )
tp16 = results[configs[1][0]]
tp16_bw = tp16["TP comm"] - tp16["tp_latency"]
print(
    f"TP=16 comm = {tp16_bw:.2f} s moving bytes"
    f" + {tp16['tp_latency']:.2f} s ring-step latency"
)
# same TP=16 if all of a node's cross-node traffic shares one 400 Gb/s NIC
one_nic = step_time(16, 1, 1, ring=IB)["step"]
print(f"TP=16 with 1 NIC per node: {one_nic:.2f} s ({base / one_nic:.2f}x)")

fig, ax = plt.subplots(figsize=(9, 3.8))
left = np.zeros(len(results))
part_colors = {
    "compute": "tab:blue",
    "TP comm": "tab:red",
    "PP bubble": "tab:orange",
    "PP p2p": "tab:brown",
    "DP exposed": "tab:gray",
}
for part, color in part_colors.items():
    vals = np.array([r[part] for r in results.values()])
    ax.barh(list(results), vals, left=left, color=color, label=part)
    left += vals
ax.invert_yaxis()
ax.set_xlabel("seconds per optimizer step (lower is better)")
ax.set_title("Same model, same batch: where 16 GPUs spend a step")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.show()
```
```text
h=8192, L=80, Psi=64.4B, 262,144 tokens per step; times in seconds
config         | compute | TP comm |   PP |   DP |  step | speedup | MFU
TP=8, 1 node   |   12.81 |    6.21 | 0.00 | 0.00 | 19.01 |   1.00x | 67%
TP=16, 2 nodes |    6.40 |   11.36 | 0.00 | 0.00 | 17.76 |   1.07x | 36%
TP=8 x DP=2    |    6.40 |    3.10 | 0.00 | 0.00 |  9.51 |   2.00x | 67%
TP=8 x PP=2    |    6.40 |    3.10 | 0.23 | 0.00 |  9.74 |   1.95x | 66%
TP=16 comm = 6.44 s moving bytes + 4.92 s ring-step latency
TP=16 with 1 NIC per node: 62.86 s (0.30x)
```
![Output of cell 10]({{ '/assets/blog/communication-roofline-3d-parallelism/cell-10-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Each bar is one optimizer step, split by what the GPUs spend it on. Blue is useful math and red is waiting on tensor-parallel communication. Going to TP = 16 halves the blue but nearly doubles the red, so the bar barely gets shorter. The two bottom layouts keep the red short, so the whole bar halves.*

Take the TP = 16 row. Compute halves from 12.81 s to 6.40 s, as it should. TP communication goes the other way, from 6.21 s to 11.36 s. Moving the bytes takes 6.44 s, because each GPU sends about as many bytes as before over a ring slightly slower than NVLink. The other 4.92 s is latency: a step runs 20,480 all-reduces (4 per layer, 80 layers, 64 micro-batches), each of 30 ring steps that wait on a cross-node hop. The step shrinks only from 19.01 s to 17.76 s, 1.07x the single-node throughput for twice the GPUs, and MFU falls from 67% to 36%. That long red bar is the paradox from the opening.

The other two layouts keep the per-layer traffic on NVLink. With TP = 8 × DP = 2, each node runs the single-node layout on half the batch, 6.40 s of compute plus 3.10 s of TP communication. The gradient all-reduce ships each GPU's share of the 64.4B parameters, about 16 GB, over InfiniBand in roughly 0.32 s. It vanishes inside the 4.27 s backward pass. The step takes 9.51 s, 2.00x the baseline, at the same 67% MFU.

TP = 8 × PP = 2 lands at 9.74 s, 1.95x the baseline. It has the same compute and TP split plus 0.23 s of bubble and activation transfers, and with 64 micro-batches the bubble is only $$1/64$$ of the stage time.

Same 16 GPUs in all three. Under TP = 16, every layer of every micro-batch waits on four all-reduces that span both nodes. Under DP = 2, gradients cross once per step, and under PP = 2, one activation crosses forward and one gradient crosses back per micro-batch, at a single boundary.

The last output line shows how much the cluster's wiring matters. If all of a node's cross-node traffic had to share one 400 Gb/s NIC, the same TP = 16 step would take 62.86 s, 0.30x of one node: twice the GPUs, more than three times slower. With one NIC per GPU, as on H100 servers built for training, TP = 16 avoids that collapse but still gains almost nothing.

All of this is modelled, not measured. Real runs lose time to sub-100% matrix-multiply efficiency, NCCL protocol overhead and imperfect overlap, and frameworks claw back some of TP's communication by overlapping it. The latencies are order-of-magnitude guesses and make up almost half of TP = 16's communication time, so read 1.07x as "roughly no gain", not as a precise figure. What holds up is the comparison between layouts: keeping TP inside a node gets close to 2x from the second node, and stretching TP across nodes doesn't. That matches what [Narayanan et al. (2021)](https://arxiv.org/abs/2104.04473) measured: throughput peaked when the tensor-parallel size equalled the GPUs in one server.

The model also leaves out memory. With mixed-precision Adam, a 64.4B-parameter model carries about 16 bytes per parameter of weights, gradients and optimizer state, roughly 1 TB, and one node has 640 GB of HBM. So the single-node baseline is a reference point for communication, not a run you could launch as is. In practice the second node would hold FSDP-style shards rather than a plain DDP copy. That moves 1.5x the cross-node bytes, about 0.48 s per step instead of 0.32 s, which still hides inside the 4.27 s backward pass.

## The placement rule, and where it bends

Here is everything above in one table, with bytes sent per GPU and ratios for a full training step.

| Dimension | Collective | Bytes per GPU | Compute-to-comm ratio | Overlaps compute? | Where it goes |
|---|---|---|---|---|---|
| Tensor (TP) | 4 all-reduces per layer (2 forward, 2 backward) | $$16bsh\frac{t-1}{t}$$ per layer | $$\frac{4.5h}{t-1}$$ | Barely: on the critical path | Inside the NVLink domain, $$t \le 8$$ |
| Pipeline (PP) | point-to-point sends | $$4bsh$$ per micro-batch per boundary | $$18\frac{L}{p}h$$ | Yes; pays in bubble $$\frac{p-1}{m+p-1}$$ instead | Across nodes |
| DDP | 1 all-reduce per step | $$4\Psi\frac{d-1}{d}$$ | $$\approx 1.5\,n_{\text{tok}}$$ | Yes: bucketed during backward | Across nodes, outermost |
| FSDP / ZeRO-3 | 2 all-gathers + 1 reduce-scatter | $$6\Psi\frac{d-1}{d}$$ | $$\approx n_{\text{tok}}$$ | Yes: layer prefetch | Across nodes if $$n_{\text{tok}} \gtrsim$$ 19k, otherwise HSDP |

The rule, in order:

1. Tensor parallelism stays inside the NVLink domain, at most the 8 GPUs of a node, and only as wide as the model needs to fit. [Narayanan et al. (2021)](https://arxiv.org/abs/2104.04473) reached the same conclusion from measurements: TP up to the GPUs in a server, pipeline parallelism across servers.
2. Pipeline parallelism spans nodes if the model still doesn't fit, with enough micro-batches ($$m \gg p$$) or interleaving to keep the bubble small.
3. Data parallelism or FSDP goes outermost. If it turns communication-bound on InfiniBand, raise tokens per GPU or switch to hybrid sharding (HSDP). HSDP shards within a node and replicates across nodes, so FSDP's 1.5x traffic stays on NVLink and only a DDP-style all-reduce crosses.

![Placement flowchart: use tensor parallelism inside the node (t at most 8) if the model does not fit on one GPU, pipeline parallelism across nodes if it does not fit in one node, then data parallelism or FSDP as the outermost dimension, raising tokens per GPU or switching to HSDP until tokens per GPU clear about 20k]({{ '/assets/blog/communication-roofline-3d-parallelism/placement-rule.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Start at the top with your model and cluster and answer each question. The coloured boxes say which kind of parallelism to add and where it goes: tensor parallelism stays inside one server, pipeline parallelism spans servers if the model still doesn't fit, and data parallelism goes on the outside. The dashed loop at the bottom means: keep raising the tokens each GPU handles until the network stops being the bottleneck.*

The rule bends in a few places.

- TP mostly doesn't overlap, which is why even TP = 8 pays 0.42x extra and cell 5 gives it its own curve. Async TP in [TorchTitan](https://arxiv.org/abs/2410.06511) splits each matrix multiply and its collective into a micro-pipeline so they can overlap. That narrows the gap, but it doesn't change how TP's ratio falls as $$t$$ grows, or the latency of a ring that crosses nodes.
- The $$s^2$$ attention term adds FLOPs that grow with $$s^2 h$$, so at long context TP's ratio is somewhat better than $$4.5h/(t-1)$$. Context parallelism, a fourth dimension, splits the sequence itself.
- Activation recomputation reruns the forward pass during backward and brings its two all-reduces back with it, lowering TP's ratio further.
- The model gives every ring step a fixed latency. At scale NCCL switches latency-bound collectives, like small gradient buckets or inference decoding, to tree algorithms, and in-switch reduction (NVLS, SHARP) changes the byte counts as well.
- The rule is really "TP inside the NVLink domain", and that domain is growing. A GB200 NVL72 rack puts 72 GPUs on one NVLink fabric, so TP and expert parallelism can go wider before they meet InfiniBand.

Before adding GPUs, check each dimension's compute-to-communication ratio against the ceiling of the link it will run over. TP's ratio is fixed by the model's width and the TP degree, while PP's and DP's are set by how you batch, and those are the two to send across nodes.

## Where the field is now

Frontier training runs follow the same ordering, and the newest ones push it further.

Llama 3 ([Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783)) uses "4D parallelism" on up to 16K H100s. Innermost to outermost, the order is TP, context parallelism, PP and FSDP, because the innermost dimension needs the highest bandwidth and lowest latency. The team reports 38 to 43% BF16 MFU at that scale, well under the toy model's 67%.

DeepSeek-V3 ([DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437)) skips tensor parallelism entirely. The mixture-of-experts model trains with 16-way PP, 64-way expert parallelism across 8 nodes, and ZeRO-1 data parallelism. Its DualPipe schedule overlaps computation and communication across the forward and backward passes, and custom all-to-all kernels split traffic between InfiniBand and NVLink. The follow-up hardware paper ([Zhao et al., 2025](https://arxiv.org/abs/2505.09343)) argues for multi-plane network topologies and for merging the scale-up and scale-out networks. Either would narrow the gap between the two ceilings.

[TorchTitan (Liang et al., 2024)](https://arxiv.org/abs/2410.06511) ships async TP as a composable option next to FSDP, PP and context parallelism. Compiler work such as [TileLink (Zheng et al., 2025)](https://arxiv.org/abs/2503.20313) goes lower, generating kernels that interleave GEMM tiles with communication. Both target the "barely overlaps" cell of the placement table. The 9x ratio between the links stays put.

With 72 GPUs in one NVLink domain, "inside the node" becomes "inside the rack". That brings a new cost. [Arfeen et al. (2025)](https://arxiv.org/abs/2504.06095) point out that in a 72-GPU NVLink domain a single GPU failure can stall tensor parallelism across far more GPUs, and they propose running the affected replica at a reduced TP degree.

## Further reading

- [Narayanan et al., 2021, "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"](https://arxiv.org/abs/2104.04473): the TP-inside, PP-across rule and the interleaved schedule.
- [Austin et al., 2025, "How to Scale Your Model"](https://jax-ml.github.io/scaling-book/): communication rooflines and sharding at book length. Written for TPUs, with a GPU chapter.
- [Tazi et al., 2025, "The Ultra-Scale Playbook: Training LLMs on GPU Clusters"](https://huggingface.co/spaces/nanotron/ultrascale-playbook): the same dimensions with real benchmark runs, which this post leaves out on purpose.
- [Rajbhandari et al., 2019, "ZeRO"](https://arxiv.org/abs/1910.02054) and [Zhao et al., 2023, "PyTorch FSDP"](https://arxiv.org/abs/2304.11277): where FSDP's 1.5x traffic and hybrid sharding come from.
- [Qi et al., 2023, "Zero Bubble Pipeline Parallelism"](https://arxiv.org/abs/2401.10241): splits the backward pass into input-gradient and weight-gradient work to fill the bubble 1F1B leaves.
- [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md): how to read measured collective bandwidth using the $$(N-1)/N$$ factors from this post.

Two dimensions this post skips come next: context parallelism, which splits the sequence, and expert parallelism, which adds all-to-all traffic for mixture-of-experts. The Llama 3 and DeepSeek-V3 reports above are the best real-world examples of each.
