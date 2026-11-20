---
layout: post
title: "Why Does an LLM Take So Long to Say the First Word, Then Answer Almost Instantly?"
date: 2026-11-21 00:30:00+0530
description: Why an LLM pauses before its first word and then streams the rest, worked through with FLOP counts, the roofline model and a 70B-on-H100 estimate.
tags: llm inference prefill decode roofline
categories: machine-learning
related_posts: false
toc:
  sidebar: left
---

<style>
  /* This post only: use the window's width so tables, equations and figures fit without sideways scrolling. */
  @media (min-width: 576px) {
    .container[role="main"], .navbar .container, footer .container { max-width: min(calc(100vw - 32px), 1520px); }
    .container[role="main"] > .row > .col-sm-3 { flex: 0 0 18%; max-width: 18%; }
    .container[role="main"] > .row > .col-sm-9 { flex: 0 0 82%; max-width: 82%; }
  }
  .post-content img { max-width: min(100%, 960px) !important; }
  .post-content p.caption { opacity: .8; max-width: 92%; margin: -.25rem auto 1.75rem; }
  .post-content table { width: 100%; }
  .post-content table th, .post-content table td { overflow-wrap: anywhere; }
  @media (max-width: 575px) {
    .post-content mjx-container[display] { font-size: 90% !important; }
    .post-content table { font-size: 75%; }
    .post-content table th, .post-content table td { padding: .3rem .35rem; }
  }
</style>

Paste a long document into a chatbot and hit Enter. For a moment, nothing. Then the answer pours out faster than you can read it.

Serving engineers track those two halves with separate numbers. Time to first token (TTFT) is the pause. Time per output token (TPOT) is the average gap between tokens after that (the gap between any two consecutive tokens is called inter-token latency).

The obvious guess is that the first token is expensive and the rest are cheap. The arithmetic says otherwise. Take a 70B-parameter model in FP16 on two H100 GPUs and a 4,000-token prompt (the estimate is worked out [below](#the-back-of-envelope-version)). Prefill gets through the whole prompt in about 0.56 s, roughly 0.14 ms per token. Decode then spends about 21 ms on every single token it writes.

So per token, reading your prompt is about 150 times cheaper than writing the answer. Prefill is the efficient phase. It only feels slow because you sit through all 4,000 tokens of it before the model can say anything, while decode just has to stay ahead of your eyes. At 21 ms a token, it does.

## Same weights, very different shapes

Both phases run the exact same weights. What differs is the shape of the work they hand the GPU.

In prefill, all $$N$$ prompt tokens go in together as one matrix $$X \in \mathbb{R}^{N \times d}$$, where $$d$$ is the hidden size (8,192 for a Llama-style 70B). Each linear layer multiplies that whole matrix by a weight matrix like $$W_Q \in \mathbb{R}^{d \times d}$$. One weight matrix, $$N$$ rows of input. That's a matrix-matrix multiply, or GEMM.

Decode feeds in a single row, $$x_t \in \mathbb{R}^{1 \times d}$$, for the one new token. Same weight matrix, one row of input, so now it's a matrix-vector multiply (GEMV).

![Prefill vs decode on the same weight matrix W: prefill multiplies an N x d prompt matrix by W loaded once (about N FLOPs per byte, compute-bound), decode multiplies a single 1 x d token vector by W loaded in full (about 1 FLOP per byte, memory-bound)]({{ '/assets/blog/llm-first-word-latency/gemm-vs-gemv.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Same weights, two kinds of work. In prefill (left), the GPU loads the weight matrix W once and uses it for all 4,000 prompt tokens, so it does a lot of math for every byte it fetches. In decode (right), it loads the same W to handle a single token, so most of the time goes into fetching W, not computing with it.*
{: .caption}

Attention splits the same way. Every layer computes

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

with $$Q = XW_Q$$, $$K = XW_K$$ and $$V = XW_V$$ the query, key and value projections and $$d_k$$ the size of one attention head. $$QK^\top$$ is an $$N \times N$$ grid of scores, one for every pair of tokens. Causal masking throws away the upper half, since a token can look back but never ahead, and prefill fills in the whole remaining triangle in a single pass.

![Causal attention grid: prefill fills the whole 8 x 8 lower triangle at once (blue), then each decode step adds a single new row (orange) for tokens 9, 10 and 11]({{ '/assets/blog/llm-first-word-latency/causal-triangle.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Each square is one attention score: how much one token (a row) looks at another (a column). Grey squares are masked because a token can't look at tokens that come after it. Prefill fills in every blue square in one go; decode then adds just one orange row for each new token.*
{: .caption}

Decode only ever needs the newest row of that triangle, and the **KV cache** is why. Under causal masking, a token's key and value are fixed the moment they're computed, because nothing that comes later can change them. So the model keeps $$K$$ and $$V$$ for every token it has seen. At step $$t$$ it computes $$q_t$$, $$k_t$$ and $$v_t$$ for the new token alone, appends $$k_t$$ and $$v_t$$ to the cache, and attends with one query:

$$o_t = \text{softmax}\left(\frac{q_t K_{1:t}^\top}{\sqrt{d_k}}\right)V_{1:t}$$

$$K_{1:t}$$ and $$V_{1:t}$$ are the cached keys and values for tokens $$1$$ through $$t$$, and $$o_t$$ is the new token's attention output.

![KV cache during decode: each new token appends one key row and one value row, and its query sweeps the cached keys once to produce attention weights and the output o_t]({{ '/assets/blog/llm-first-word-latency/kv-cache-growing.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The KV cache during decode. Each new token adds one row to the stored keys (K) and values (V). Its query q_t is compared with every stored key once (the yellow flashes), those scores become weights, and the weighted values give the output o_t. Nothing from earlier tokens is recomputed.*
{: .caption}

| | Prefill (first token) | Decode (every later token) |
|---|---|---|
| Input per step | all $$N$$ prompt tokens | 1 new token |
| Linear algebra | matrix × matrix (GEMM) | matrix × vector (GEMV) |
| Attention per layer | the full $$N \times N$$ triangle | one row of length $$t$$ |
| Bottleneck | compute (FLOP/s) | memory bandwidth (bytes/s) |
| What the user feels | TTFT | TPOT |

Most of that table falls straight out of the shapes. The bottleneck row doesn't, and it's the row that matters. Getting to it takes two steps: count the work, then look at what actually limits the GPU.

## Count the FLOPs first

A rule of thumb that holds up well is 2 FLOPs per parameter per token. Each weight does one multiply and one add for every token that passes through it. For a model with $$P$$ parameters,

$$
\begin{gathered}
\text{FLOPs}_{\text{prefill}} \approx 2PN \\
\text{FLOPs}_{\text{decode}} \approx 2P \ \text{per token}
\end{gathered}
$$

Attention adds its own work on top, and that part grows with the square of the prompt length. With $$L$$ layers and causal masking it's roughly

$$\text{FLOPs}_{\text{attention}} \approx 2N^2 d L$$

This is the term people blame when they say prefill is slow "because attention is quadratic." At ordinary prompt lengths it barely registers. For a 70B model ($$d = 8{,}192$$, $$L = 80$$) at $$N = 4{,}000$$, attention comes to about $$2.1 \times 10^{13}$$ FLOPs against $$5.6 \times 10^{14}$$ for the linear layers. Under 4%. The two only meet at

$$N^* \approx \frac{P}{dL}$$

which works out to roughly 107,000 tokens for this model, or about 53,000 if the kernel computes the masked half anyway. Below that, and once the prompt is longer than a few hundred tokens (the next section explains that threshold), prefill time grows almost linearly with prompt length, and most of it goes into the same weight matrices decode uses.

So prefill does about $$N$$ times the work of a decode step, which here means 4,000 times. If FLOPs told the whole story, a decode step would take $$0.56\ \text{s} / 4{,}000 = 0.14$$ ms and a 500-token answer would show up in 70 ms. Real decode is about 150 times slower than that. Arithmetic isn't what sets its pace.

## Why less math doesn't mean less time

A GPU can run short of two things: compute, or the bandwidth to keep that compute fed. The **roofline model** ([Williams, Waterman & Patterson, 2009](https://doi.org/10.1145/1498765.1498785)) puts both limits on one chart, and once you've seen it, the prefill/decode split stops being mysterious.

On the x-axis sits **arithmetic intensity**, the math a computation does per byte it pulls from GPU memory (HBM) into the cores:

$$\text{AI} = \frac{\text{FLOPs}}{\text{bytes moved}}$$

The y-axis is the FLOP/s you actually get, and two ceilings cap it. At low intensity the cores mostly wait for data, so you get bandwidth × AI. That's the sloped memory roof, and an H100 (the SXM version, which all the numbers here assume) moves 3.35 TB/s. At high intensity the cores stay busy and performance flattens out at peak, about 989 TFLOP/s of dense FP16 on an H100. ([NVIDIA](https://www.nvidia.com/en-us/data-center/h100/) advertises 1,979, but that's a with-sparsity number.) The slope meets the flat roof at the **ridge point**:

$$
\begin{aligned}
\text{AI}_{\text{ridge}} &= \frac{\text{peak FLOP/s}}{\text{bandwidth}} \\
&\approx \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx 295 \ \text{FLOP/byte}
\end{aligned}
$$

Anything left of the ridge is memory-bound, anything right of it compute-bound. To see where the two phases land, remember that an FP16 weight is 2 bytes and contributes 2 FLOPs for every token it touches.

Decode reads every weight to process one token: $$2P$$ FLOPs over $$2P$$ bytes, so $$\text{AI} \approx 1$$ FLOP/byte. Nowhere near the ridge. Each step goes on streaming 140 GB of weights while the tensor cores use well under 1% of what they could do. (Decode also re-reads the KV cache every step, which adds bytes as the context grows; I leave it out to keep the numbers simple, and it only pushes decode further left.)

Prefill reads each weight once and reuses it across all $$N$$ tokens: $$2PN$$ FLOPs over roughly $$2P$$ bytes, so $$\text{AI} \approx N$$. A few hundred tokens into the prompt it crosses the ridge, and from there the tensor cores run flat out.

![H100 roofline on log-log axes: as prompt length N grows from 1 to 4,000, prefill climbs the 3.35 TB/s memory slope and reaches the 989 TFLOP/s compute roof near the 295 FLOP/byte ridge, while decode stays at an arithmetic intensity of about 1]({{ '/assets/blog/llm-first-word-latency/roofline-slide.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Left to right is how much math the GPU does per byte it reads; bottom to top is the speed it reaches. Decode (orange) sits at about 1 FLOP per byte, stuck low on the sloped memory line and using a tiny fraction of the GPU. As the prompt grows, prefill (blue) climbs that slope and, after about 300 tokens, reaches the flat ceiling where the GPU computes at full speed.*
{: .caption}

Prefill takes visible time because there's a lot of math, but it does that math at full speed. Decode has almost nothing to compute and still waits, every single step, for the whole model to come out of memory.

This is also why providers batch. If $$B$$ users decode together, one pass over the weights serves all of them: $$2PB$$ FLOPs over $$2P$$ bytes, so $$\text{AI} \approx B$$. A batch of 64 gets 64 tokens for the weight traffic of one (each user's KV cache is still read separately). Your own tokens don't come any faster, but the GPU does far more useful work per byte, and the compute still sitting idle turns out to be handy for prefill (more on that at the end).

## The back-of-envelope version

A 70B model in FP16 is 140 GB of weights. One H100 has 80 GB. So I split the model across two GPUs with tensor parallelism, each holding half of every layer, and assume compute and bandwidth simply double, ignoring communication between the GPUs and letting real kernels reach about half of peak. That puts effective compute $$F_{\text{eff}}$$ at about 500 TFLOP/s per H100, so about 1,000 TFLOP/s for the pair, and memory bandwidth $$BW$$ at 3.35 TB/s each, 6.7 TB/s together.

Prefill is compute-bound, so its time is FLOPs over compute:

$$
\begin{aligned}
\text{TTFT} &\approx \frac{2PN}{F_{\text{eff}}} \\
&= \frac{2 \times (70 \times 10^9) \times 4{,}000}{10^{15}} \approx 0.56 \ \text{s}
\end{aligned}
$$

Decode is memory-bound, so its time is bytes over bandwidth:

$$
\begin{aligned}
\text{TPOT} &\approx \frac{\text{weight bytes}}{BW} = \frac{140 \times 10^9}{6.7 \times 10^{12}} \\
&\approx 21 \ \text{ms} \quad (\approx 48 \ \text{tokens/s})
\end{aligned}
$$

Now go per token. Prefill spends $$2P / F_{\text{eff}}$$ on each prompt token and decode spends $$2P / BW$$ on each output token, so

$$
\begin{aligned}
\frac{\text{decode time per token}}{\text{prefill time per token}} &= \frac{2P / BW}{2P / F_{\text{eff}}} \\
&= \frac{F_{\text{eff}}}{BW} \approx \frac{10^{15}}{6.7 \times 10^{12}} \approx 150
\end{aligned}
$$

The parameter count cancels, which to me is the nicest part of the whole calculation. What's left, the 150× gap, is the GPU's *effective* ridge point: how many FLOPs it can do in the time it takes to move one byte. Run an 8B model on one H100, or a 405B model across as many H100s as it takes to hold it, and the per-token ratio stays roughly the same. Only the absolute times change. (Roughly, because attention and KV-cache traffic don't scale with the parameter count, and small models don't always keep the GPU busy.)

And 48 tokens a second is roughly 35 words a second, at about three-quarters of an English word per token. A typical adult reads around 4. Once prefill is done the stream outruns you, so it looks instant, even though every token in it is the expensive kind.

## Why it's even slower in a real product

That 0.56 s is a floor. A real service piles more on top, in three ways.

Start with reasoning models. They decode a hidden chain of thought before the visible answer, and those tokens run at decode speed, so the wait you actually see is

$$
\begin{aligned}
\text{TTFT}_{\text{visible}} &= \text{TTFT}_{\text{prefill}} \\
&\quad + N_{\text{think}} \times \text{TPOT}
\end{aligned}
$$

At 21 ms per token, 1,500 thinking tokens add about 31 s. Prefill is a rounding error next to that.

Then there's the queue. A server runs many users at once, and a 10,000-token prefill done in one go would stall everyone else's stream for as long as it runs. Schedulers like [Sarathi-Serve](https://arxiv.org/abs/2403.02310) chop long prompts into chunks and run each chunk alongside the ongoing decode batch, soaking up the idle compute the roofline says is there. Everyone keeps streaming. Your prefill, though, is now spread over several passes, and you may wait for GPU memory before it starts at all.

Last comes prefix caching, or rather missing it. If the start of your prompt (a system prompt, earlier turns in the chat) matches something the server processed recently, engines like [vLLM](https://arxiv.org/abs/2309.06180) (automatic prefix caching) and [SGLang](https://arxiv.org/abs/2312.07104) (RadixAttention) reuse its cached K and V and only prefill the new part. The catch is that every token's K and V depend on everything before it. Change one early token, say a timestamp at the top of the system prompt, and the cache is invalid for everything after it.

![Timelines of the first-word wait: a standard model shows its first token after queue plus a 0.56 s prefill, while a reasoning model first spends about 31 s decoding 1,500 hidden reasoning tokens before its first visible token]({{ '/assets/blog/llm-first-word-latency/ttft-timeline.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Why the first word can take even longer in a real product. Top: a standard model waits in a queue, spends about 0.56 s reading your prompt, then streams tokens 21 ms apart. Bottom: a reasoning model also writes about 1,500 hidden thinking tokens first, which adds roughly 31 s before you see anything.*
{: .caption}

Put together, the first word waits on one compute-bound pass over your whole prompt, plus whatever the service stacks in front of it. Every word after that waits on a memory-bound sweep through the weights. Per token, the first wait is still the cheap one, by about 150×.

## Further reading

- [Pope et al., 2022, "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102), the standard analysis of prefill vs decode cost at scale, including batch size and how to split a model across chips.
- [kipply, 2022, "Transformer Inference Arithmetic"](https://kipp.ly/transformer-inference-arithmetic/), which works the 2-FLOPs-per-parameter rule and memory- vs compute-bound decoding through by hand.
- [Yuan et al., 2024, "LLM Inference Unveiled: Survey and Roofline Model Insights"](https://arxiv.org/abs/2402.16363), a survey that looks at inference techniques through the roofline model.
- [Williams, Waterman & Patterson, 2009, "Roofline: An Insightful Visual Performance Model for Multicore Architectures"](https://doi.org/10.1145/1498765.1498785), the original roofline paper (Communications of the ACM 52(4)).
- [Agrawal et al., 2024, "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"](https://arxiv.org/abs/2403.02310), for chunked prefill and stall-free scheduling in detail.
