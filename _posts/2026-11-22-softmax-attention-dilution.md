---
layout: post
title: "A 1M-token context window isn't 1M tokens of memory: the ln N tax of softmax attention"
date: 2026-11-22 15:00:00+0530
description: Why softmax attention needs ln N more logit margin to keep its focus on the one relevant token, why needle-in-a-haystack tests hide the cost, and what log-N scaling can and can't fix.
tags: attention softmax long-context transformers llm
categories: machine-learning
related_posts: false
toc:
  sidebar: left
---

Paste a 300-page contract into a model with a million-token window and ask where the termination clause is. The whole document fits, with plenty of room to spare. Sometimes the answer comes back confident, well written, and about the wrong clause.

Part of that is position: models handle the middle of a long prompt worse than its edges. This post is about a different part, a cost that softmax attention charges for every token in the context, relevant or not. Each token adds only a little, but a long context adds a lot of tokens, and once you work out the cost, a million-token window stops looking like a million tokens of memory.

I'll build it up in four passes: attention as a vote, then the geometry of random vectors in high dimensions, then a short bound with a $$\ln N$$ in it, and finally a NumPy simulation you can run on a laptop in under a minute. If you know scaled dot-product attention, you're set.

## The paradox: million-token windows, thousand-token memory

Retrieval has not kept up with window size. [Liu et al. (2023)](https://arxiv.org/abs/2307.03172) found that models answer best when the relevant document sits at the start or end of the prompt, and worse when it sits in the middle. [Chroma's context-rot report (Hong et al., 2025)](https://www.trychroma.com/research/context-rot) tested 18 models and found accuracy falling as input grows, even on simple tasks, and falling faster when the haystack holds passages that are topically related to the question but don't answer it.

Yet the same long-context models routinely score near-perfectly on the standard needle-in-a-haystack (NIAH) test ([Kamradt](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)). [RULER (Hsieh et al., 2024)](https://arxiv.org/abs/2404.06654) put the two observations side by side: almost all the models it tested reach near-perfect accuracy on vanilla NIAH, then show large drops as the context gets longer or the task gets harder.

I think much of that gap comes from one line of softmax arithmetic. A context window is *addressable* memory, in that the model can read every token. It isn't *effective* memory, because every token you add votes in the softmax denominator. To hold the relevant token's attention share constant, its logit must beat the average distractor by an amount that grows as $$\ln N$$, about 2.3 nats more for every 10× more context. By the end you should be able to:

1. derive that margin bound in three lines;
2. predict the context length at which a given needle drops below 50% attention, and check it against a simulation;
3. explain why NIAH passes while retrieval from a haystack of similar items fails;
4. explain why scaling logits by $$\ln N$$ fixes the first problem but not the second.

## Attention is a vote with a crowd

Take one attention head at one decoding step. The query $$q$$ scores every key $$k_j$$ with a logit $$s_j = q\cdot k_j/\sqrt{d_k}$$, and softmax turns the logits into weights $$\alpha_j = e^{s_j}/\sum_i e^{s_i}$$. Suppose exactly one key, the **needle**, holds the information the query needs. Call its logit $$s^*$$; the other $$N-1$$ keys are distractors. The needle's weight is

$$
\alpha^* = \frac{e^{s^*}}{e^{s^*} + \sum_{j=1}^{N-1} e^{s_j}}.
$$

Read it as an election. The numerator is the needle's vote; the denominator is that vote plus everyone else's. Each distractor votes weakly, because $$e^{s_j}$$ is small when $$s_j$$ is small, but nothing in the formula stops it from voting. Double the context and the weak votes double while the needle's vote stays exactly where it was. No single distractor has to beat the needle; together they outvote it.

Two numbers describe the crowd, the mean distractor logit $$\mu$$ and its spread $$\sigma$$, and the needle's advantage is its **logit margin** $$\Delta s = s^* - \mu$$. The rest of the post asks how big $$\Delta s$$ has to be as $$N$$ grows.

![Hand-drawn sketch: the query sends one strong vote to the needle key and many weak votes to a crowd of N - 1 distractors; the needle's weight is its own vote divided by its vote plus the whole crowd's.]({{ '/assets/blog/softmax-attention-dilution/attention-vote.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The query hands out its attention like votes. The needle gets one strong vote and each distractor gets a weak one; the needle's share is its vote divided by all the votes together. Adding distractors makes the bottom of that fraction bigger, even though no single distractor got any stronger.*

## The geometry: a haystack on a hypersphere

Write each logit as a length times an angle: $$s_j = \beta\cos(q, k_j)$$, where $$\beta = \lVert q\rVert\lVert k\rVert/\sqrt{d_k}$$ is the logit scale. Picture the keys as points on the unit sphere in $$\mathbb{R}^{d}$$ and the query as a direction: a key's logit is then its height along that direction, its projection onto $$q$$.

For a uniformly random unit vector, symmetry makes every coordinate's square average to the same value, and the squares sum to 1, so each coordinate has variance $$1/d$$. Nearly all of the sphere's area sits in a thin band around the "equator" perpendicular to $$q$$. With $$d = 128$$, a random key's cosine with the query is roughly $$\mathcal{N}(0, 1/128)$$, so a typical projection is about $$\pm 0.09$$. Our needle, at cosine 0.5, lies far above the band.

That gives the crowd its statistics, $$\mu = 0$$ and $$\sigma = \beta/\sqrt{d}$$, and each distractor alone is weak. The same picture holds a second, slower threat. Draw a cap around $$q$$: the set of directions within cosine $$t$$ of the query. Its share of the sphere is about $$e^{-d t^2/2}$$, so $$N$$ random keys put about $$N e^{-d t^2/2}$$ points inside it. Set that to 1 and you get, to leading order, the height of the highest distractor:

$$
\max_j \cos(q, k_j) \approx \sqrt{\frac{2\ln N}{d}}.
$$

As $$N$$ grows, random keys reach higher, the empty cap above them shrinks, and the best random impostor climbs toward the needle, slowly, like $$\sqrt{\ln N}$$. A long context therefore threatens the needle in two ways: the **sum** of many weak votes, and the **max**, one lucky key that happens to point almost where the query points. With random keys the sum wins first. The NIAH section shows when the max takes over.

![On a cross-section of the key sphere, random keys crowd the equator while the empty cap above the best impostor shrinks as N grows from 10^2 to 10^6; the leading-order estimate of the impostor's cosine climbs from 0.27 to 0.465, approaching but never reaching the needle at 0.5.]({{ '/assets/blog/softmax-attention-dilution/haystack-sphere.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Left: a slice through the sphere of possible key directions. The yellow arrow is the query, the orange arrow is the needle, and the blue dots are random keys, which pile up near the equator. The shaded cap is the area no random key has reached yet. As the number of keys N grows, the best impostor climbs and the cap shrinks, but the impostor stays below the needle. Right: the same story as a histogram of cosines, with the needle fixed at 0.5 and the dashed 'max of N keys' line creeping toward it.*

## The ln N tax

The bound needs two facts:

1. By the law of large numbers, the total vote of many distractors is close to its expected value, $$\sum_{j=1}^{N-1} e^{s_j} \approx (N-1)\,\mathbb{E}[e^{s_j}]$$.
2. The Gaussian moment-generating function: if $$s_j \sim \mathcal{N}(\mu, \sigma^2)$$, then $$\mathbb{E}[e^{s_j}] = e^{\mu + \sigma^2/2}$$.

So the crowd's vote is about $$(N-1)e^{\mu+\sigma^2/2}$$. Require the needle to keep at least a share $$c$$ of the attention, $$\alpha^* \ge c$$, solve for $$e^{s^*}$$, and take logs:

$$
\Delta s \;\ge\; \ln(N-1) + \frac{\sigma^2}{2} + \ln\frac{c}{1-c}.
$$

The first term, $$\ln(N-1)$$, is the tax. Going from 100 to 1,000,000 tokens adds $$\ln 10^4 \approx 9.2$$ nats to the margin the needle needs, or $$\ln 10 \approx 2.30$$ nats per 10×. The second, $$\sigma^2/2$$, is a spread penalty: a crowd whose logits vary more casts a bigger total vote at the same mean, because $$e^x$$ rewards its high tail, and it comes back in the NIAH section. The last, $$\ln\frac{c}{1-c}$$, is how much attention you want. At $$c = 1/2$$ it is zero.

Setting $$c = 1/2$$ and solving for $$N$$ gives the **drowning length**, the context size at which the needle's share falls to one half:

$$
N^* \approx 1 + e^{\Delta s - \sigma^2/2}.
$$

A common explanation blames bounded norms: LayerNorm and weight decay supposedly cap $$\lVert q\rVert\lVert k\rVert$$, so logits can't grow with $$N$$. That claims too much. LayerNorm gains and $$W_q, W_k$$ are learned, and nothing stops a model from using large logits. The version I'd defend is about calibration. The logit scale is learned at the training length and then frozen, so a model trained on 8K-token sequences learned margins that pay the tax for $$\ln 8000$$, not for $$\ln 10^6$$. The [Qwen technical report (Bai et al., 2023, §2.5)](https://arxiv.org/abs/2309.16609) acts on the same idea: to extend context at inference, it rescales attention by a factor that depends on the ratio of the context length to the training length.

## Build it: drowning a needle in NumPy

The simulation tests every number above, using the haystack from the geometry section: `D = 128` (the head dimension $$d$$), `BETA = 20` (the logit scale $$\beta$$), and a needle at cosine `RHO = 0.5`. That gives a needle logit $$s^* = \beta\rho = 10$$, a distractor spread $$\sigma = 20/\sqrt{128} = 1.77$$, and a predicted drowning length of $$1 + e^{10 - 1.56} \approx 4{,}600$$ tokens, in the middle of the 100 to 100,000 range we sweep.

Each simplification gives something up:

- Keys are i.i.d. uniform on the sphere rather than learned, so there is no real key structure or anisotropy.
- One head and one query means no redundancy across heads and layers, which real models use to recover.
- With no positions there is no RoPE and no attention sink, which is why the U-shape is out of scope.
- I set $$\beta$$ by hand, as a stand-in for a scale the model would have learned at its training length.

Cell 1 pins the two dependencies, fixes the seed and defines those three constants.

```python
%pip install -q "numpy==2.3.3" "matplotlib==3.10.8"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SEED = 0
np.random.seed(SEED)

# Fixed categorical order (blue, orange, aqua, yellow) + a gray for guides.
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
plt.rcParams.update({
    "figure.figsize": (7.5, 4.2), "figure.dpi": 110, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "lines.linewidth": 2, "font.size": 10,
})

D = 128          # key dimension of one attention head
BETA = 20.0      # logit scale: s = BETA * cos(q, k), the role of ||q|| ||k|| / sqrt(d)
RHO = 0.5        # cosine between the query and the needle key
print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__} | d={D}, beta={BETA}, rho={RHO}")
```
```text
numpy 2.3.3 | matplotlib 3.10.8 | d=128, beta=20.0, rho=0.5
```

Cell 2 is the core experiment. Each trial draws one random query and 99,999 random keys, with the needle's logit fixed at `NEEDLE` $$= s^* = 10$$. `needle_weight` evaluates $$\alpha^*$$ for every context length in `NS` at once: a running sum of $$e^{s_j}$$ gives the crowd's vote for the first $$N-1$$ distractors, so each longer context contains the shorter one, the way appending tokens to a prompt does. Watch for the context length at which the curve crosses the dashed 50% line.

```python
N_MAX, TRIALS = 100_000, 20
NS = np.unique(np.logspace(2, 5, 31).astype(int))   # context lengths 100 .. 100,000

def random_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """n points drawn uniformly on the unit sphere in R^d."""
    g = rng.standard_normal((n, d), dtype=np.float32)
    return g / np.linalg.norm(g, axis=1, keepdims=True)

def haystack_logits(n: int, rng: np.random.Generator) -> np.ndarray:
    """Logits of n random distractor keys against a random query (float64 for exp)."""
    q = random_unit(1, D, rng)[0]
    return BETA * (random_unit(n, D, rng) @ q).astype(np.float64)

def needle_weight(needle: float, crowd_logits: np.ndarray, ns: np.ndarray) -> np.ndarray:
    """alpha* when the context is the needle plus the first N-1 distractors, for each N."""
    crowd = np.cumsum(np.exp(crowd_logits))[ns - 2]     # sum over N-1 distractors
    return np.exp(needle) / (np.exp(needle) + crowd)

rng = np.random.default_rng(SEED + 2)
NEEDLE = BETA * RHO                                     # the needle's logit never changes
alphas = np.array([needle_weight(NEEDLE, haystack_logits(N_MAX - 1, rng), NS) for _ in range(TRIALS)])
alpha_mean = alphas.mean(axis=0)
n_star_sim = np.exp(np.interp(0.5, alpha_mean[::-1], np.log(NS)[::-1]))

plt.fill_between(NS, alphas.min(0), alphas.max(0), color=BLUE, alpha=0.2, lw=0)
plt.plot(NS, alpha_mean, color=BLUE, label=f"needle weight (mean of {TRIALS} haystacks)")
plt.axhline(0.5, color=GRAY, ls="--", lw=1)
plt.axvline(n_star_sim, color=GRAY, ls=":", lw=1)
plt.xscale("log"); plt.xlabel("context length N"); plt.ylabel(r"attention on needle $\alpha^*$")
plt.title("Same needle, bigger haystack"); plt.legend(frameon=False)
for n in (100, 1_000, 10_000, 100_000):
    print(f"N={n:>7,}  alpha*={alpha_mean[NS == n][0]:.3f}")
print(f"needle drops below 50% at N ~ {n_star_sim:,.0f}")
plt.show()
```
```text
N=    100  alpha*=0.980
N=  1,000  alpha*=0.822
N= 10,000  alpha*=0.320
N=100,000  alpha*=0.045
needle drops below 50% at N ~ 4,714
```
![Output of cell 2]({{ '/assets/blog/softmax-attention-dilution/cell-02-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Each point on the x-axis is one context length, on a log scale, so each labelled tick is 10× more tokens. The blue line is the needle's share of attention averaged over 20 random haystacks, and the pale band shows the range from the best to the worst haystack. The dashed line marks 50%, and the dotted vertical line marks where the needle crosses it.*

The needle holds 0.980 of the attention at $$N = 100$$, 0.822 at 1,000, 0.320 at 10,000 and 0.045 at 100,000, and it falls below one half at $$N \approx 4{,}714$$. Nothing broke. The needle's logit is 10 in every row and its cosine with the query never moved; only the number of weak votes in the denominator changed. The band around the mean is narrow, too. Twenty different random haystacks give almost the same curve, because a sum of thousands of votes barely fluctuates.

Cell 3 overlays the closed form from the previous section, $$\alpha^*(N) = 1/\big(1 + (N-1)e^{-(\Delta s - \sigma^2/2)}\big)$$, with `MARGIN` $$= \Delta s = 10$$ and `SIGMA` $$= \sigma$$. If the three-line argument is right, the orange dashed curve should sit on the blue one.

```python
SIGMA = BETA / np.sqrt(D)          # spread of distractor logits: cos(q, k) ~ N(0, 1/d)
MARGIN = NEEDLE - 0.0              # delta_s = s* - mu, and mu = 0 for random keys

def alpha_theory(n: np.ndarray, margin: float, sigma: float) -> np.ndarray:
    """Mean-field needle weight: crowd ~ (N-1) * exp(mu + sigma^2 / 2)."""
    return 1.0 / (1.0 + (n - 1) * np.exp(-(margin - sigma**2 / 2)))

n_star_theory = 1 + np.exp(MARGIN - SIGMA**2 / 2)
plt.plot(NS, alpha_mean, color=BLUE, label="simulation")
plt.plot(NS, alpha_theory(NS, MARGIN, SIGMA), color=ORANGE, ls="--", label="theory")
plt.axhline(0.5, color=GRAY, ls="--", lw=1)
plt.xscale("log"); plt.xlabel("context length N"); plt.ylabel(r"$\alpha^*$")
plt.title("The bound predicts the curve"); plt.legend(frameon=False)
print(f"sigma = beta/sqrt(d) = {SIGMA:.3f}  ->  sigma^2/2 = {SIGMA**2 / 2:.3f} nats")
print(f"N* predicted = {n_star_theory:,.0f} | simulated = {n_star_sim:,.0f} "
      f"| gap = {abs(n_star_sim / n_star_theory - 1):.1%}")
print(f"extra margin per 10x context = ln(10) = {np.log(10):.3f} nats")
print(f"margin needed for alpha*=0.5 at N=1e6: {np.log(1e6 - 1) + SIGMA**2 / 2:.2f} nats (we have {MARGIN:.2f})")
plt.show()
```
```text
sigma = beta/sqrt(d) = 1.768  ->  sigma^2/2 = 1.562 nats
N* predicted = 4,618 | simulated = 4,714 | gap = 2.1%
extra margin per 10x context = ln(10) = 2.303 nats
margin needed for alpha*=0.5 at N=1e6: 15.38 nats (we have 10.00)
```
![Output of cell 3]({{ '/assets/blog/softmax-attention-dilution/cell-03-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Blue is the simulation from the previous plot; orange dashed is the formula's prediction. The two curves nearly overlap, so a three-line calculation is enough to predict when the needle drowns.*

It does. The spread penalty is $$\sigma^2/2 = 1.562$$ nats, the predicted drowning length is 4,618 and the simulated one is 4,714, a 2.1% gap. Two small effects account for it, and both make the real crowd slightly weaker than the formula assumes: cosines on a sphere have lighter tails than a Gaussian, and a finite crowd rarely samples the far tail that the expectation $$e^{\mu+\sigma^2/2}$$ counts. The last two lines turn the bound into a budget. Each 10× of context costs $$\ln 10 = 2.303$$ nats, so holding $$\alpha^* = 0.5$$ at a million tokens would need a 15.38-nat margin, and this needle has 10. A model can't find those extra 5.4 nats at inference time. They would have had to be learned.

The attention share is one symptom. What the rest of the network actually receives is the head's output, $$o = \sum_j \alpha_j v_j$$, a weighted average of the value vectors. Cell 4 tracks two things as $$N$$ grows: the attention entropy $$H(\alpha) = -\sum_j \alpha_j \ln\alpha_j$$ divided by its maximum $$\ln N$$ (0 means all attention on one token, 1 means spread evenly), and the cosine between the output and the needle's value $$v^*$$. Real value vectors share a common direction, since hidden states are anisotropic, so `value_vectors` builds each one as a shared "average token" direction plus noise. With isotropic random values the crowd's contributions would cancel and hide the effect.

```python
rng = np.random.default_rng(SEED + 4)
mean_value = random_unit(1, D, rng)[0].astype(np.float64)   # shared "average token" direction

def value_vectors(n: int) -> np.ndarray:
    """Anisotropic values: a common direction plus noise (random isotropic values would cancel)."""
    return 0.6 * mean_value + 0.8 * random_unit(n, D, rng).astype(np.float64)

def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b))

ent, cos_needle, cos_mean = [], [], []
for _ in range(5):
    s = haystack_logits(N_MAX - 1, rng)
    w, w_star, v_star = np.exp(s), np.exp(NEEDLE), value_vectors(1)[0]
    Z = w_star + np.cumsum(w)[NS - 2]                              # softmax denominator per N
    ent.append(np.log(Z) - (w_star * NEEDLE + np.cumsum(w * s)[NS - 2]) / Z)   # H = ln Z - E[s]
    out = (w_star * v_star + np.cumsum(w[:, None] * value_vectors(N_MAX - 1), axis=0)[NS - 2]) / Z[:, None]
    cos_needle.append(cos(out, v_star)); cos_mean.append(cos(out, mean_value))
ent, cos_needle, cos_mean = (np.mean(x, axis=0) for x in (ent, cos_needle, cos_mean))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(NS, ent / np.log(NS), color=BLUE)
ax1.set(xscale="log", xlabel="context length N", ylabel=r"$H(\alpha)\,/\,\ln N$", title="Entropy heads for its maximum")
ax2.plot(NS, cos_needle, color=BLUE, label="cos(output, needle value)")
ax2.plot(NS, cos_mean, color=ORANGE, label="cos(output, average value)")
ax2.set(xscale="log", xlabel="context length N", ylabel="cosine", title="Output drifts to the average token")
ax2.legend(frameon=False)
for n in (100, 100_000):
    i = np.flatnonzero(NS == n)[0]
    print(f"N={n:>7,}  H/lnN={ent[i] / np.log(n):.2f}  cos(o, v*)={cos_needle[i]:.2f}  cos(o, v_avg)={cos_mean[i]:.2f}")
plt.show()
```
```text
N=    100  H/lnN=0.03  cos(o, v*)=1.00  cos(o, v_avg)=0.60
N=100,000  H/lnN=0.85  cos(o, v*)=0.64  cos(o, v_avg)=1.00
```
![Output of cell 4]({{ '/assets/blog/softmax-attention-dilution/cell-04-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Left: how spread out the head's attention is, from 0 (all on one token) to 1 (perfectly even). It climbs toward 1 as the context grows. Right: how much the head's output looks like the needle's value (blue) versus the average token (orange). As the context grows, the output stops looking like the needle and starts looking like the average.*

At $$N = 100$$ the entropy is 0.03 of its maximum and the output is the needle's value (cosine 1.00). By $$N = 100{,}000$$ the entropy has climbed to 0.85 of $$\ln N$$, the output's cosine with the needle has fallen to 0.64, and its cosine with the average value is 1.00. The head still "sees" the needle. What it passes on, though, is the average of everything in context. That is context rot inside a single head: the right token is in the window, and the head's output barely carries it.

Cell 5 checks the second threat from the geometry section, the best single impostor. It tracks the largest distractor logit as $$N$$ grows and compares it with the extreme-value prediction $$\sigma\sqrt{2\ln N}$$, plus a refined version with the next correction term. Drawing a million full 128-dimensional keys twenty times is slow, and unnecessary. By the geometry section's symmetry, only a key's coordinate along $$q$$ matters. That coordinate is $$g_1/\lVert g\rVert$$ for a Gaussian vector $$g$$, and $$\lVert g\rVert^2 - g_1^2$$ is a chi-squared variable with $$d-1$$ degrees of freedom, so the cell samples it directly.

```python
rng = np.random.default_rng(SEED + 5)
max_logit = np.mean([np.maximum.accumulate(haystack_logits(N_MAX - 1, rng))[NS - 2] for _ in range(TRIALS)], axis=0)

def gaussian_max(n: np.ndarray, sigma: float, refined: bool) -> np.ndarray:
    """Expected max of n i.i.d. N(0, sigma^2): leading order, optionally with the next EVT term."""
    a = np.sqrt(2 * np.log(n))
    return sigma * (a - (np.log(np.log(n)) + np.log(4 * np.pi)) / (2 * a) if refined else a)

# By symmetry only the coordinate along q matters: cos = g1 / ||g||, so 1e6 keys cost nothing.
g1 = rng.standard_normal((TRIALS, 1_000_000))
cos_1m = g1 / np.sqrt(g1**2 + rng.chisquare(D - 1, size=g1.shape))
max_cos_1m = cos_1m.max(axis=1).mean()

plt.plot(NS, max_logit, color=BLUE, label="largest distractor logit (simulated)")
plt.plot(NS, gaussian_max(NS, SIGMA, refined=False), color=ORANGE, ls="--", label=r"$\sigma\sqrt{2\ln N}$")
plt.plot(NS, gaussian_max(NS, SIGMA, refined=True), color=AQUA, ls=":", label="with next EVT term")
plt.axhline(NEEDLE, color=GRAY, ls="--", lw=1, label="needle logit")
plt.xscale("log"); plt.xlabel("context length N"); plt.ylabel("logit")
plt.title("The best impostor creeps up, slowly"); plt.legend(frameon=False)
print(f"max distractor cosine: N=1e5 -> {max_logit[-1] / BETA:.3f} | N=1e6 -> {max_cos_1m:.3f} "
      f"| sqrt(2 ln 1e6 / d) = {np.sqrt(2 * np.log(1e6) / D):.3f} | needle = {RHO}")
print(f"at N=1e6 the needle is still the single best key, yet alpha* = {float(alpha_theory(np.array(1e6), MARGIN, SIGMA)):.4f}")
plt.show()
```
```text
max distractor cosine: N=1e5 -> 0.388 | N=1e6 -> 0.408 | sqrt(2 ln 1e6 / d) = 0.465 | needle = 0.5
at N=1e6 the needle is still the single best key, yet alpha* = 0.0046
```
![Output of cell 5]({{ '/assets/blog/softmax-attention-dilution/cell-05-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Blue is the score of the strongest random distractor as the context grows; the dashed and dotted lines are two versions of the formula that predicts it. The gray dashed line is the needle's score. The best distractor creeps upward but never reaches the needle, so what beats the needle is the crowd as a whole, not any one rival.*

The best random impostor reaches cosine 0.388 at 100,000 keys and 0.408 at a million. The leading-order formula $$\sqrt{2\ln N/d}$$ gives 0.465 at a million; it overshoots at finite $$N$$, and the refined curve in the plot tracks the simulation closely. Either way the needle at 0.5 stays above every random key, which makes the last printed line the most important result so far: **at a million tokens the needle is still the single best-matching key in the context, and it gets 0.46% of the attention.** Argmax would pick it out at once. Softmax loses it to the crowd.

The same picture gives the real ceiling that the bounded-norms argument was reaching for. Raising the logit scale $$\beta$$ sharpens softmax toward argmax, and argmax picks the needle as long as the needle is the top key. To leading order, random keys reach the needle's height when $$\sqrt{2\ln N/d} = \rho$$, that is, at

$$
N_{\max} \approx e^{\rho^2 d/2}.
$$

Beyond that length no logit scale, learned or hand-set, can pick out a needle at cosine $$\rho$$ from random keys. For $$\rho = 0.5$$ and $$d = 128$$ that gives $$e^{16} \approx 9$$ million tokens, and it is an underestimate: the leading-order term overshoots the real maximum (cell 5 again), and the exact spherical tail, $$P(\cos > 0.5) \approx 8 \times 10^{-10}$$, puts the ceiling nearer $$10^9$$ random keys. Either way it sits far above the 4,700 where the crowd already drowned the needle. Real haystacks aren't random, though. That brings the ceiling down fast.

## The NIAH illusion

Everything so far used random keys, with spread $$\sigma = 1.77$$. Real contexts differ, and the difference lands exactly on the $$\sigma^2/2$$ term.

In a standard NIAH test the needle is a sentence that has nothing to do with the essays around it. Seen from the query, the haystack looks the same everywhere: none of it is relevant, so its logits sit in a tight bunch and $$\sigma$$ is small. Real questions look different. Finding the one indemnity clause in a 500-page contract means searching a haystack full of other clauses, some close to the query and some not. The mean logit can stay put while the spread grows, and $$\sigma^2/2$$ grows with the square of the spread.

There is a second effect. The log of the crowd's total vote, $$\ln\sum_j e^{s_j}$$ (the log-sum-exp), behaves in two different ways depending on $$\sigma$$. For small $$\sigma$$ it is the sum regime from the bound, $$\ln N + \mu + \sigma^2/2$$, with many voters and none dominant. For large $$\sigma$$ it is the max regime, $$\mu + \sigma\sqrt{2\ln N}$$, where one extreme key casts nearly all of the vote. The two meet at $$\sigma_c = \sqrt{2\ln N}$$. Statistical physicists know this as the freezing transition of the [random-energy model (Derrida, 1981)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.24.2613). Past $$\sigma_c$$ the crowd's vote is mostly one key's vote, and the needle's real competitor is that single most similar-looking distractor, which is what Chroma's topically related distractors look like.

Cell 6 holds $$N = 10{,}000$$ and the margin over the mean ($$\Delta s = 10$$, $$\mu = 0$$) fixed, and sweeps only the spread $$\sigma$$ from 0.25 to 6, sampling Gaussian logits directly. The left panel shows $$\alpha^*$$; the right panel shows the simulated log-sum-exp against the sum and max approximations. A real on-topic haystack would also raise $$\mu$$, which only makes things worse. Holding it fixed isolates what spread alone does.

```python
rng = np.random.default_rng(SEED + 6)
N_FIXED, SIGMAS = 10_000, np.linspace(0.25, 6.0, 24)
SIGMA_C = np.sqrt(2 * np.log(N_FIXED))       # sum-dominated below, max-dominated above

def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)

# Same needle, same mean distractor logit (mu = 0); only the spread sigma changes.
crowd = SIGMAS[:, None, None] * rng.standard_normal((len(SIGMAS), 100, N_FIXED - 1))
lse = logsumexp(crowd)                                           # ln(sum of crowd votes)
alpha_sim = (1 / (1 + np.exp(lse - NEEDLE))).mean(axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(SIGMAS, alpha_sim, color=BLUE, label="simulation")
ax1.plot(SIGMAS, alpha_theory(N_FIXED, MARGIN, SIGMAS), color=ORANGE, ls="--", label="mean-field theory")
ax1.axvline(SIGMA, color=GRAY, ls=":", lw=1)
ax1.set(xlabel=r"distractor spread $\sigma$", ylabel=r"$\alpha^*$", title=f"N = {N_FIXED:,}, same margin over the mean")
ax1.legend(frameon=False)
ax2.plot(SIGMAS, lse.mean(axis=1), color=BLUE, label="ln-sum-exp (simulated)")
ax2.plot(SIGMAS, np.log(N_FIXED - 1) + SIGMAS**2 / 2, color=ORANGE, ls="--", label=r"sum: $\ln N + \sigma^2/2$")
ax2.plot(SIGMAS, SIGMAS * SIGMA_C, color=AQUA, ls=":", label=r"max: $\sigma\sqrt{2\ln N}$")
ax2.axvline(SIGMA_C, color=GRAY, ls="--", lw=1)
ax2.set(xlabel=r"distractor spread $\sigma$", ylabel="log of crowd's total vote", ylim=(0, 40), title="From crowd to impostor")
ax2.legend(frameon=False)
for label, s in (("off-topic haystack (NIAH)", 0.5), ("random keys, cells 2-5", SIGMA), ("on-topic haystack", 3.0)):
    print(f"{label:<27} sigma={s:.2f}  alpha*={np.interp(s, SIGMAS, alpha_sim):.3f}")
print(f"sum -> max switch at sigma_c = sqrt(2 ln N) = {SIGMA_C:.2f}")
plt.show()
```
```text
off-topic haystack (NIAH)   sigma=0.50  alpha*=0.660
random keys, cells 2-5      sigma=1.77  alpha*=0.317
on-topic haystack           sigma=3.00  alpha*=0.027
sum -> max switch at sigma_c = sqrt(2 ln N) = 4.29
```
![Output of cell 6]({{ '/assets/blog/softmax-attention-dilution/cell-06-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Both panels keep the context length (10,000) and the average distractor fixed and change only how spread out the distractor scores are. Left: the needle's share of attention collapses as the spread grows; the dotted line marks the random-key haystack used earlier. Right: the crowd's total pull (blue) follows the 'many small votes' formula (orange dashed) at first. Near the dashed vertical line it peels away and runs just under the 'one big vote' formula (green dotted), because by then a single extreme distractor supplies most of the pull.*

The needle, the context length and the average distractor are identical in every row. With an off-topic haystack ($$\sigma = 0.5$$) the needle keeps 0.660 of the attention; with the random-key haystack from cells 2 to 5 ($$\sigma = 1.77$$) it keeps 0.317; with an on-topic haystack ($$\sigma = 3$$) it keeps 0.027. That is a 24× drop with no change to the needle or to $$N$$, and the mean-field theory curve matches the simulation closely up to $$\sigma pprox 3$$; beyond that both are near zero.

The right panel shows why the bound eventually needs replacing. The simulated log-sum-exp follows $$\ln N + \sigma^2/2$$ up to about $$\sigma = 3$$, then bends away toward the max-regime line, and the regimes meet near the marked $$\sigma_c = 4.29$$. (The simulated curve stays a little below the max line for the same reason cell 5's maximum did: $$\sigma\sqrt{2\ln N}$$ is only the leading-order term.)

So a benchmark whose haystack has tiny $$\sigma$$ tests the most favourable case of the bound, the tax on $$N$$ and almost nothing else. Scoring 99% there says little about retrieval from a context full of near-misses, which is what people actually fill a 1M window with.

## Pay the tax up front: log-N scaling

If the bound demands about $$\ln N$$ more margin, supply it. Multiply every logit by a factor that grows with $$\ln N$$. [Chiang & Cholak (2022)](https://arxiv.org/abs/2202.12172) proposed this, multiplying attention logits by $$\log n$$, as a fix for length generalization. [Scalable-Softmax (Nakanishi, 2025)](https://arxiv.org/abs/2501.19399) makes the factor learnable, $$\mathrm{SSMax}(z) = \mathrm{Softmax}\big((s\log n)\, z\big)$$ with a learned scalar $$s$$, and Qwen's LogN-Scaling applies the same idea at inference, relative to the training length.

Why it works: if every logit is multiplied by $$\kappa \propto \ln N$$, the needle's margin becomes $$\kappa\,\Delta s$$, which also grows like $$\ln N$$ and cancels the tax. The required *raw* margin becomes a constant instead of something that grows with $$N$$.

The limit follows from the previous section. Scaling sharpens every logit, the best impostor's included, and sharpening pushes softmax toward argmax, which is only as good as the ranking. If one distractor's raw score beats the needle, a sharper softmax hands that distractor more attention.

![The needle's attention share falls from 0.98 to 0.04 as N grows from 10^2 to 10^5 under plain softmax, then stays between 0.69 and 0.82 across the same range once logits are multiplied by ln N / ln 1000.]({{ '/assets/blog/softmax-attention-dilution/softmax-drowning.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Left: the needle's bar (orange) against everyone else's (blue); the number on top is the needle's share. Right: the same share as a curve, with a white dot marking the current context length. First, plain softmax runs from 100 to 100,000 tokens and the needle's bar shrinks. Then log-N scaling switches on (yellow) and the run goes back down, with the needle's bar staying high the whole way.*

Cell 7 calibrates the scale at a training length of `N_TRAIN` $$= 1{,}000$$, so the factor is $$\kappa = \ln N/\ln 1000$$: it equals 1 at the training length and grows at longer ones. It runs two experiments, the random haystack from before and the same haystack with one **impostor**, a key at cosine 0.6, slightly above the needle's 0.5.

```python
rng = np.random.default_rng(SEED + 7)
N_TRAIN, IMPOSTOR = 1_000, 0.6         # calibrated at 1k tokens; an impostor key at cosine 0.6 > 0.5

def scaled_weights(crowd: np.ndarray, n: int, impostor: float | None) -> tuple[float, float]:
    """(needle, impostor) attention at length n with logits multiplied by ln n / ln N_TRAIN."""
    k = np.log(n) / np.log(N_TRAIN)
    votes = np.exp(k * crowd[: n - 1]).sum()
    imp = np.exp(k * BETA * impostor) if impostor is not None else 0.0
    if impostor is not None:
        votes -= np.exp(k * crowd[0])          # the impostor takes the first distractor's seat
    z = np.exp(k * NEEDLE) + votes + imp
    return np.exp(k * NEEDLE) / z, imp / z

clean, needle_imp, impostor_w = [], [], []
for _ in range(TRIALS):
    crowd = haystack_logits(N_MAX - 1, rng)
    clean.append([scaled_weights(crowd, n, None)[0] for n in NS])
    pair = np.array([scaled_weights(crowd, n, IMPOSTOR) for n in NS])
    needle_imp.append(pair[:, 0]); impostor_w.append(pair[:, 1])
clean, needle_imp, impostor_w = (np.mean(x, axis=0) for x in (clean, needle_imp, impostor_w))

plt.plot(NS, alpha_mean, color=BLUE, label="needle, plain softmax")
plt.plot(NS, clean, color=AQUA, label=r"needle, logits $\times \ln N / \ln N_{train}$")
plt.plot(NS, needle_imp, color=ORANGE, ls="--", label="needle, scaled, impostor present")
plt.plot(NS, impostor_w, color=YELLOW, ls="--", label="impostor, scaled")
plt.axhline(0.5, color=GRAY, ls="--", lw=1)
plt.xscale("log"); plt.xlabel("context length N"); plt.ylabel("attention weight")
plt.title("log-N scaling pays the crowd tax, not the impostor tax"); plt.legend(frameon=False, fontsize=8)
print("      N | plain | scaled | scaled+impostor (needle / impostor)")
for n in (100, 1_000, 10_000, 100_000):
    i = np.flatnonzero(NS == n)[0]
    print(f"{n:>7,} | {alpha_mean[i]:.3f} | {clean[i]:.3f}  | {needle_imp[i]:.3f} / {impostor_w[i]:.3f}")
plt.show()
```
```text
      N | plain | scaled | scaled+impostor (needle / impostor)
    100 | 0.980 | 0.806  | 0.199 / 0.754
  1,000 | 0.822 | 0.826  | 0.116 / 0.859
 10,000 | 0.320 | 0.803  | 0.064 / 0.920
100,000 | 0.045 | 0.719  | 0.034 / 0.953
```
![Output of cell 7]({{ '/assets/blog/softmax-attention-dilution/cell-07-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Blue: the needle under plain softmax, drowning as before. Green: the same needle with logits scaled by ln N / ln 1000, holding steady. The two dashed lines show what happens when one distractor, the impostor, matches the query slightly better than the needle: scaling gives the impostor (yellow) more and more attention while the needle (orange) fades.*

With plain softmax the needle goes from 0.980 to 0.045 across the sweep. With log-N scaling it holds at 0.806, 0.826, 0.803 and 0.719. The slow sag at 100,000 tokens is the spread penalty coming back: scaling multiplies $$\sigma$$ by $$\kappa$$ as well, so the $$\sigma^2/2$$ term grows like $$\kappa^2$$ while the margin grows only like $$\kappa$$. Over these three decades the cost is small. SSMax learns its $$s$$ per head, so a model can at least tune this trade-off; the fixed factor here can't.

The impostor rows show the limit. With the impostor present and scaling on, the needle's share falls from 0.199 to 0.034 while the impostor's rises from 0.754 to 0.953. Scaling sharpened attention toward the best key, as designed, and the best key was the wrong one. Log-N scaling fixes the sum; no choice of temperature fixes the max.

## What this doesn't explain: the U-shape

Dilution predicts a decline with $$N$$ that is the same everywhere in the context. It doesn't predict the U-shaped curve from [Liu et al. (2023)](https://arxiv.org/abs/2307.03172), where information at the start and end of the prompt survives better than information in the middle. The usual suspects are two positional effects the simulation leaves out on purpose. The first is [attention sinks (Xiao et al., 2023)](https://arxiv.org/abs/2309.17453): because softmax weights must sum to 1, heads learn to park spare attention on the first few tokens. That paper shows the attention, not better retrieval, so the idea that information placed there rides along is a hypothesis. The second is that [RoPE (Su et al., 2021)](https://arxiv.org/abs/2104.09864) has a long-term decay, an upper bound on the attention score that shrinks with relative distance, which favours recent tokens. If those effects lift the two ends, tokens in the middle get neither boost and pay the full $$\ln N$$ tax. The tax would set how deep the U is, and position its shape.

## Other ways to dodge the tax

Other lines of work attack a different term of the bound, or its assumptions:

| Approach | What it changes | Which term it attacks |
|---|---|---|
| log-N scaling ([Chiang & Cholak](https://arxiv.org/abs/2202.12172), [SSMax](https://arxiv.org/abs/2501.19399)) | multiplies logits by $$\propto \ln N$$ | pays $$\ln(N-1)$$ directly; ranking errors survive |
| Top-$$k$$ / sparse attention | normalizes over $$k \ll N$$ selected keys | replaces $$\ln(N-1)$$ with $$\ln k$$, but the selection must rank the needle in its top $$k$$ (a max-regime problem) |
| [Differential attention (Ye et al., 2024)](https://arxiv.org/abs/2410.05258) | subtracts two softmax maps, $$\mathrm{softmax}_1 - \lambda\,\mathrm{softmax}_2$$ | cancels the attention both maps spread over distractors, i.e. the crowd's floor |
| [softmax1 (Miller, 2023)](https://www.evanmiller.org/attention-is-off-by-one.html) | adds 1 to the denominator | lets a head attend to nothing; meant to remove the need for sinks (Xiao et al. found it only partly does), not the $$\ln N$$ tax |
| Stateful memory ([Titans](https://arxiv.org/abs/2501.00663), [Mamba](https://arxiv.org/abs/2312.00752)) | compresses history into a fixed-size state | no softmax over all $$N$$ tokens (Titans keeps softmax attention over a short window; Mamba has none); trades exact token lookup for compressed, lossy recall |

![The margin bound split into its three terms (the ln(N - 1) tax, the sigma squared over 2 spread penalty, and the attention target), with arrows from top-k sparse attention, log-N scaling and differential attention to the terms they attack, softmax1 and stateful memory pointing at the setup as a whole, and a note that none of them fix a wrong ranking.]({{ '/assets/blog/softmax-attention-dilution/margin-anatomy.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The top row is the margin rule from earlier, split into its three parts: the cost of the crowd's size, the extra cost of a spread-out crowd, and how much attention you want the needle to get. The purple boxes are fixes that shrink one of those parts; the teal boxes change the setup instead. The red note is the catch: none of them helps when a distractor really does match the query better than the needle.*

## Conclusion

A million-token window lets the model read a million tokens. It doesn't let one attention head pick one of them out. The softmax charges $$\ln N$$ in logit margin to keep a needle's share constant, and a model pays that charge with margins it learned at its training length. When the context outgrows that budget, a needle that never moves, and is still the best key in the context, falls from 98% of the attention to 0.46%. NIAH misses this because its haystack is off-topic. Log-N scaling pays the tax back, but neither it nor any other temperature trick can fix a context where something looks more relevant than the answer.

So treat a 1M window as a scratchpad, not a database. If you need exact recall from a long, on-topic context, you'll get more from putting less in it, ranking before you attend (retrieval, reranking, top-$$k$$), or using a model trained with length-aware scaling than from appending another 100K tokens. To check the claims yourself, change `RHO`, `BETA` or `D` in cell 1 and see whether the drowning length moves to $$1 + e^{\beta\rho - \beta^2/2d}$$.

## Further reading

- [Liu et al., 2023, *Lost in the Middle*](https://arxiv.org/abs/2307.03172): the U-shaped retrieval curve that motivates this post.
- [Hong et al., 2025, *Context Rot* (Chroma)](https://www.trychroma.com/research/context-rot): 18 models degrading with input length, and the effect of topically related distractors.
- [Hsieh et al., 2024, *RULER*](https://arxiv.org/abs/2404.06654): why near-perfect vanilla NIAH scores don't transfer to harder long-context tasks.
- [Chiang & Cholak, 2022, *Overcoming a Theoretical Limitation of Self-Attention*](https://arxiv.org/abs/2202.12172): where multiplying logits by $$\log n$$ comes from.
- [Nakanishi, 2025, *Scalable-Softmax Is Superior for Attention*](https://arxiv.org/abs/2501.19399): the learnable version, with long-context experiments.
- [Derrida, 1981, *Random-energy model*](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.24.2613): the sum-to-max transition of a log-sum-exp, from statistical physics.
