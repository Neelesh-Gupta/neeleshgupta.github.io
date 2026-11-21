---
layout: post
title: "The high-dimensional soap bubble: why straight lines break inside latent space"
date: 2026-11-21 23:00:00+0530
description: Why high-dimensional Gaussian samples live on a thin shell, why a straight line between two of them cuts through the empty core and even climbs in likelihood, and why a small VAE barely notices while Stable Diffusion turns the midpoint into static, with every claim checked in code.
tags: latent-space interpolation slerp gaussian diffusion
categories: machine-learning
related_posts: false
toc:
  sidebar: left
---

<style>
  /* use the window's width, as in Part 1, so text, code, equations and figures fill the screen */
  @media (min-width: 576px) {
    .container[role="main"], .navbar .container, footer .container { max-width: min(calc(100vw - 32px), 1520px); }
    .container[role="main"] > .row > .col-sm-3 { flex: 0 0 18%; max-width: 18%; }
    .container[role="main"] > .row > .col-sm-9 { flex: 0 0 82%; max-width: 82%; }
  }
  .post-content img { max-width: min(100%, 960px) !important; }
  /* wide display equations scroll inside their own box instead of widening the page on phones */
  mjx-container[display="true"] { display: block; overflow-x: auto; overflow-y: hidden; max-width: 100%; }
  /* long code and output lines wrap inside the column instead of sliding sideways */
  article pre, article pre code { white-space: pre-wrap !important; overflow-wrap: anywhere; word-break: normal; overflow-x: hidden !important; }
</style>

If you've played with a generative model, you've probably made this demo. Take two random seeds, decode a handful of points between them, and watch one image melt into the other. The natural way to fill in the points in between is a straight line, $$(1-t)\,z_0 + t\,z_1$$, and it feels too obvious to question. It isn't. In a high-dimensional latent space that line wanders somewhere the model has never been, and in a diffusion model you can watch it happen: a clean lighthouse at each end of the walk, and coloured static in the middle.

This is Part 3 of a four-part series on the geometry of latent spaces. [Part 1](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/what-is-latent-space/) treated a latent space as a learned chart of the data. [Part 2](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/ghosts-and-hybrids/) found holes in that chart between the training codes, and showed how a variational autoencoder fills them with noise balls held to a $$\mathcal{N}(0, I)$$ prior. Every picture there was two-dimensional, where a Gaussian looks like a friendly hill with its peak in the middle. Real models work in hundreds or thousands of dimensions, and there the same Gaussian behaves in a way the 2-D pictures get backwards. This post follows that behaviour: where the samples actually are, why a straight line between two of them misses them, and why some decoders shrug it off while others fall apart.

As in the earlier parts, each idea arrives four ways: an intuition, a picture of the geometry, the maths, and a short code cell that checks the maths on real numbers. Everything runs on a laptop CPU. The core cells take about a minute and a half, with MNIST as the only dataset download; one optional cell uses Stable Diffusion Turbo (a 2.5 GB download, about five minutes the first time) and saves its results, so nothing else depends on it. A bonus box at the end looks inside GPT-2 small, which the code downloads (about 0.5 GB) if needed.

## Quick quiz: where do Gaussian samples live?

Part 2 ended on a question, so start there. Draw 100,000 samples from a standard Gaussian in 512 dimensions, $$z \sim \mathcal{N}(0, I_{512})$$. The density is highest at the origin and falls off in every direction, so how many samples land within distance 10 of the origin?

None. Cell 2 below draws all 100,000 and counts zero; the closest is 19.55 away. Every sample lands in a thin shell around radius $$\sqrt{512} \approx 22.6$$, so thin that a histogram of the distances looks like a spike. In high dimensions a Gaussian's samples sit on the skin of a soap bubble, and the inside, where the density is highest, is empty.

That empty middle is where this post spends its time. A straight line between two points on the bubble's skin has to pass through it, and the sections below follow that line into the core and ask what a decoder makes of what it finds there.

![A teal histogram of the distance from the origin for Gaussian samples slides right to the square root of d as the dimension d sweeps from 1 to 512, while its spread stays near 0.71 and the region near the coral origin stays empty]({{ '/assets/blog/high-dimensional-soap-bubble/soap-bubble-growth.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the animation shows: we draw random points from a bell curve and measure how far each one lands from the centre. As the number of dimensions d grows from 1 to 512, the hump of distances slides right to about √d but stays just as narrow. Takeaway: in high dimensions almost every point lands at nearly the same distance, on a thin shell. None land near the centre.*

## Where Part 2 left us

[Part 2](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/ghosts-and-hybrids/) ended with a VAE whose noise balls fill the disc where its $$\mathcal{N}(0, I)$$ prior puts its samples. The decoder trains on a ball around every training code rather than on the code alone, and the KL term acts as a spring that holds the balls near unit size and pulls them toward the origin until they overlap. ELBO surgery showed why the filling stays incomplete. The objective pays for information and for prior mismatch out of one budget, and that VAE spent 93% of it on information, so the codes the decoder trains on (the aggregate posterior) never quite match the prior it samples from.

All of that was drawn in two dimensions, where the prior is a round hill, the densest place is the middle, and a hole is a gap in the plane between clusters of codes. Real models use the same prior in many more dimensions: dozens or hundreds in a typical VAE, 64 for this post's MNIST model, 16,384 in Stable Diffusion's noise. Two tools come back from Part 2 unchanged. The digit-classifier judge scores how confidently a decoded frame reads as a digit. On Blob World, [Part 1](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/what-is-latent-space/)'s images of a single Gaussian blob, the squared pixel distance to the nearest real image measures how far a frame has left the data. The hole itself changes shape. In high dimensions it runs along the radius, at the centre of the prior, which is where the 2-D pictures put the densest codes.

## The soap bubble

Write the squared length of $$z$$ as a sum over coordinates, $$\|z\|^2 = z_1^2 + z_2^2 + \dots + z_d^2$$. Each $$z_j$$ is a standard normal, so each term averages 1 and the sum averages $$d$$: every coordinate you add pushes the length further out, and it grows like $$\sqrt{d}$$. The terms are independent, so their fluctuations partly cancel. The wobble of the sum grows only like $$\sqrt{d}$$, and after the square root it doesn't grow at all.

Picture the histogram of $$\|z\|$$ as $$d$$ grows. In one dimension it's half a bell curve piled against zero. By $$d = 10$$ it's a hump near 3, and by $$d = 512$$ a narrow spike at 22.6. The peak slides right, the width stays near 0.7, and the space around the origin empties. So the shell is thin relative to its radius, about 3% of it at $$d = 512$$, yet it is just as wide in 512 dimensions as in 10.

The maths is short. $$\|z\|^2$$ is a sum of $$d$$ independent squared standard normals, which makes it a chi-squared variable with $$d$$ degrees of freedom. Each term has mean 1 and variance $$\mathbb{E}[z_j^4] - 1 = 3 - 1 = 2$$, so

$$
\begin{gathered}
\|z\|^2 \sim \chi^2_d, \\
\mathbb{E}\,\|z\|^2 = d, \qquad \operatorname{Var}\|z\|^2 = 2d.
\end{gathered}
$$

By the central limit theorem, $$\|z\|^2 \approx d + \sqrt{2d}\,g$$ with $$g \sim \mathcal{N}(0, 1)$$. Take the square root and keep the first-order term (the delta method):

$$
\begin{aligned}
\|z\| &= \sqrt{d}\,\sqrt{1 + \sqrt{2/d}\;g} \\
&\approx \sqrt{d}\left(1 + \frac{g}{\sqrt{2d}}\right) = \sqrt{d} + \frac{g}{\sqrt{2}}.
\end{aligned}
$$

The radius is $$\sqrt{d}$$ and the spread around it is $$1/\sqrt{2} \approx 0.707$$, whatever $$d$$ is. The rigorous version, with tail bounds in place of a first-order expansion, is the Gaussian Annulus Theorem: for $$\beta \le \sqrt{d}$$, all but a fraction $$3e^{-c\beta^2}$$ of the mass lies within $$\beta$$ of radius $$\sqrt{d}$$, for a fixed constant $$c > 0$$ ([Blum, Hopcroft & Kannan, *Foundations of Data Science*, ch. 2](https://www.cs.cornell.edu/jeh/book.pdf)).

Cell 1 sets up; cell 2 runs the quiz and checks both formulas.

```python
# New in Part 3: transformers for the GPT-2 bonus cell, scipy for exact chi-squared tails
%pip install -q "transformers==5.22.2" "scipy==1.16.2"
# ---- Part 2's setup cell, copied verbatim ---------------------------------------------------
%pip install -q "numpy==2.3.3" "torch==2.13.0" "torchvision==0.28.0" "scikit-learn==1.6.1" "matplotlib==3.10.8"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Circle, Ellipse
from sklearn.neighbors import NearestNeighbors

SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)
TEAL, CORAL, SLATE = "#14b8a6", "#f97362", "#64748b"  # series colours: teal = data, coral = void
HOLE_CMAP = LinearSegmentedColormap.from_list("hole", ["#f8fafc", CORAL, "#7f1d1d"])
H = W = 28
xs, ys = np.meshgrid(np.arange(W), np.arange(H))

def blob(cx: float, cy: float, s: float = 2.0) -> np.ndarray:
    """A 28x28 Gaussian blob of width s centred at (cx, cy), peak intensity 1."""
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * s**2)).astype(np.float32)

def blob_world(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n blob images with centres uniform in [4, 24]^2. Returns images and the true chart (centres)."""
    C = rng.uniform(4, 24, size=(n, 2))
    return np.stack([blob(cx, cy) for cx, cy in C]), C

def holey_blob_world(n: int, rng: np.random.Generator, r0: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Blob World with no centres within r0 of the image centre (14, 14): the manifold is an annulus."""
    C = rng.uniform(4, 24, size=(4 * n, 2))
    C = C[np.linalg.norm(C - 14, axis=1) >= r0][:n]
    return np.stack([blob(cx, cy) for cx, cy in C]), C

def flat(X: np.ndarray) -> np.ndarray:
    """(n, 28, 28) images -> (n, 784) points in pixel space."""
    return X.reshape(len(X), -1)

def montage(imgs: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Tile (rows*cols, 28, 28) images into one (rows*28, cols*28) picture."""
    return imgs.reshape(rows, cols, H, W).transpose(0, 2, 1, 3).reshape(rows * H, cols * W)

# ---- Part 1's autoencoder, copied verbatim -------------------------------------------------
class AE(nn.Module):
    """MLP autoencoder 784 -> 256 -> 128 -> d -> 128 -> 256 -> 784.

    Smooth ELU activations: with ReLU, this AE tends to cut Blob World's square into separate flaps.
    """
    def __init__(self, d: int, D: int = H * W):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, d))
        self.dec = nn.Sequential(nn.Linear(d, 128), nn.ELU(), nn.Linear(128, 256), nn.ELU(), nn.Linear(256, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))

def train_ae(X: np.ndarray, d: int, epochs: int = 60, seed: int = SEED) -> tuple[AE, np.ndarray, float]:
    """Train AE(d) with MSE + Adam. Returns the model, the codes z = e(x) and the final MSE."""
    torch.manual_seed(seed)
    model, Xt = AE(d), torch.from_numpy(X)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xt)).split(128):
            loss = ((model(Xt[idx]) - Xt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Z = model.enc(Xt)
        mse = ((model.dec(Z) - Xt) ** 2).mean().item()
    return model, Z.numpy(), mse

# ---- The VAE: same body, a Gaussian encoder head, and switches for the ablation -------------
class VAE(nn.Module):
    """Part 1's AE body with an encoder head q(z|x) = N(mu, diag sigma^2). The decoder outputs pixel logits."""
    def __init__(self, d: int, D: int = H * W):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(D, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU())
        self.mu, self.logvar = nn.Linear(128, d), nn.Linear(128, d)
        self.dec = nn.Sequential(nn.Linear(d, 128), nn.ELU(), nn.Linear(128, 256), nn.ELU(), nn.Linear(256, D))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.dec(z))

def vae_loss(model: VAE, x: torch.Tensor, beta: float = 1.0, sample: bool = True,
             sigma_x: float | None = None) -> torch.Tensor:
    """Negative ELBO per image, summed over pixels. sigma_x=None: Bernoulli likelihood; else Gaussian with fixed sigma_x."""
    mu, logvar = model.encode(x)
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu  # reparameterisation (or no noise)
    logits = model.dec(z)
    if sigma_x is None:
        recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)
    else:
        recon = ((torch.sigmoid(logits) - x) ** 2).sum(1) / (2 * sigma_x**2)
    kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(1)  # closed-form KL(N(mu, sigma^2) || N(0, I))
    return (recon + beta * kl).mean()

def spread_ratio(mu: torch.Tensor, logvar: torch.Tensor) -> float:
    """Ball spacing / ball radius: spread of the centres over the RMS ball radius. Unchanged by z -> c z."""
    return (mu.var(0).sum().sqrt() / logvar.exp().sum(1).mean().sqrt()).item()

def train_vae(X: np.ndarray, d: int = 2, beta: float = 1.0, sample: bool = True, sigma_x: float | None = None,
              epochs: int = 60, seed: int = SEED, log_every: int = 0) -> tuple[VAE, list[tuple[int, float, float]]]:
    """Train a VAE with Adam (lr 1e-3, batch 128). Optionally log (epoch, centre spread, spacing/radius)."""
    torch.manual_seed(seed)
    model, Xt = VAE(d), torch.from_numpy(X)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(1, epochs + 1):
        for idx in torch.randperm(len(Xt)).split(128):
            loss = vae_loss(model, Xt[idx], beta, sample, sigma_x)
            opt.zero_grad(); loss.backward(); opt.step()
        if log_every and epoch % log_every == 0:
            with torch.no_grad():
                mu, logvar = model.encode(Xt)
                history.append((epoch, mu.var(0).sum().sqrt().item(), spread_ratio(mu, logvar)))
    return model, history

mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True)
X_mnist = mnist.data[:10000].numpy().reshape(-1, H * W).astype(np.float32) / 255  # same 10k images as Part 1
y_mnist = mnist.targets[:10000].numpy()
print(f"numpy {np.__version__} | torch {torch.__version__} | torchvision {torchvision.__version__}")
print(f"MNIST subset: {X_mnist.shape}, pixels in [{X_mnist.min():.0f}, {X_mnist.max():.0f}]")

# ---- Part 2's digit judge, copied verbatim (Part 2, cell 5) ----------------------------------
Xt_mnist = torch.from_numpy(X_mnist)
# A small digit classifier as the judge: confident on real digits, unsure on ghostly overlays
torch.manual_seed(SEED)
judge = nn.Sequential(nn.Linear(H * W, 128), nn.ReLU(), nn.Linear(128, 10))
opt = torch.optim.Adam(judge.parameters(), lr=1e-3)
y_t = torch.from_numpy(y_mnist).long()
for _ in range(5):
    for idx in torch.randperm(len(Xt_mnist)).split(128):
        loss = F.cross_entropy(judge(Xt_mnist[idx]), y_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
X_held = torch.from_numpy(mnist.data[10000:12000].numpy().reshape(-1, H * W).astype(np.float32) / 255)

def judge_samples(images: np.ndarray) -> tuple[float, int]:
    """Mean classifier confidence, and how many digit classes get at least 2% of the samples."""
    with torch.no_grad():
        p = judge(torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32))).softmax(1)
    counts = np.bincount(p.argmax(1).numpy(), minlength=10)
    return p.max(1).values.mean().item(), int((counts >= 0.02 * len(images)).sum())

with torch.no_grad():
    held_acc = (judge(X_held).argmax(1).numpy() == mnist.targets[10000:12000].numpy()).mean()
held_conf, held_classes = judge_samples(X_held.numpy())

# ---- New in Part 3 ----------------------------------------------------------------------------
from pathlib import Path
from scipy.stats import chi2

ORANGE, BLUE = "#f97316", "#3b82f6"  # series colours: orange = LERP, blue = SLERP

def lerp(z0: np.ndarray, z1: np.ndarray, t) -> np.ndarray:
    """Straight line. z0, z1: (..., d); t: (T,). Returns (T, ..., d)."""
    t = np.asarray(t, dtype=np.float64).reshape(-1, *([1] * z0.ndim))
    return (1 - t) * z0 + t * z1

def slerp(z0: np.ndarray, z1: np.ndarray, t) -> np.ndarray:
    """Great-circle arc (White 2016). z0, z1: (..., d); t: (T,). Returns (T, ..., d)."""
    t = np.asarray(t, dtype=np.float64).reshape(-1, *([1] * z0.ndim))
    cos_omega = (z0 * z1).sum(-1, keepdims=True) / (np.linalg.norm(z0, axis=-1, keepdims=True)
                                                     * np.linalg.norm(z1, axis=-1, keepdims=True))
    omega = np.arccos(np.clip(cos_omega, -1.0, 1.0))
    return (np.sin((1 - t) * omega) * z0 + np.sin(t * omega) * z1) / np.sin(omega)

def active_units(mu: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Burda et al. (2016): latent j is active if the encoder mean moves with the input, Var_x(mu_j) > 0.01."""
    return mu.var(0) > threshold

def contrast(images: np.ndarray) -> np.ndarray:
    """Pixel standard deviation of each image: low for flat, washed-out frames."""
    return images.reshape(len(images), -1).std(1)

print(f"judge: {held_acc:.1%} held-out accuracy, mean confidence on real digits {held_conf:.3f}")
```
```text
numpy 2.3.3 | torch 2.13.0+cpu | torchvision 0.28.0+cpu
MNIST subset: (10000, 784), pixels in [0, 1]
judge: 92.8% held-out accuracy, mean confidence on real digits 0.910
```

Everything above the `New in Part 3` line is Part 2's setup cell and digit judge, unchanged: same pins, same 10,000 MNIST images, same `VAE` class and training loop. The judge is back at 92.8% held-out accuracy and 0.910 mean confidence on real digits, and it's the yardstick for every decoded digit below. Under the line are four small helpers. `lerp` and `slerp` are the two paths this post compares, `active_units` counts the latent dimensions a VAE actually uses, and `contrast` measures how washed out an image is. There are two new colours as well: orange is always LERP and blue is always SLERP. Teal is still the data and now also the shell where samples live, and coral is still the void and now also the hollow core.

Cell 2 draws 100,000 samples at $$d = 512$$ and counts those within distance 10 of the origin, then draws 20,000 for each $$d$$ in {1, 2, 10, 100, 512} and compares the mean and spread of $$\|z\|$$ with $$\sqrt{d}$$ and $$0.707$$. The right-hand panel and the three `mass peak` lines are for the next section.

```python
rng = np.random.default_rng(SEED)
d = 512
norms = np.concatenate([np.linalg.norm(rng.standard_normal((10_000, d)), axis=1) for _ in range(10)])
print(f"d = {d}: {(norms < 10).sum()} of {len(norms):,} samples lie within distance 10 of the origin, the density's peak")
print(f"||z||: mean {norms.mean():.2f} vs sqrt(d) = {np.sqrt(d):.2f} | std {norms.std():.3f} vs 1/sqrt(2) = "
      f"{1 / np.sqrt(2):.3f} | smallest of 100,000: {norms.min():.2f}")

dims = [1, 2, 10, 100, 512]
radius = {k: np.linalg.norm(rng.standard_normal((20_000, k)), axis=1) for k in dims}
for k in dims:
    print(f"  d = {k:>3}: mean ||z|| {radius[k].mean():6.2f} (sqrt(d) = {np.sqrt(k):6.2f}), std {radius[k].std():.3f}")

# Mass at radius r = density e^{-r^2/2} x shell area r^{d-1}. Work in logs: r^{d-1} overflows.
r = np.linspace(1e-3, 13, 4000)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3))
colours = plt.cm.viridis(np.linspace(0.0, 0.85, len(dims)))
for k, c in zip(dims, colours):
    ax1.hist(radius[k], bins=260, range=(0, 26), density=True, histtype="stepfilled", alpha=0.6, color=c, label=f"d = {k}")
    ax1.axvline(np.sqrt(k), color=c, ls=":", lw=1)
ax1.set_xlabel(r"distance from the origin $\|z\|$"); ax1.set_ylabel("density of samples")
ax1.set_title(r"The peak slides to $\sqrt{d}$; its width stays about 0.7"); ax1.legend()
ax2.plot(r, np.exp(-r**2 / 2), color="k", ls="--", lw=1.5, label=r"density $e^{-r^2/2}$ (same for every d)")
for k, c in zip([2, 10, 100], colours[1:4]):
    log_area = (k - 1) * np.log(r)
    log_mass = log_area - r**2 / 2
    ax2.plot(r, np.exp(log_area - log_area.max()), color=c, lw=1, ls=":")
    ax2.plot(r, np.exp(log_mass - log_mass.max()), color=c, lw=2.5, label=f"mass, d = {k}")
    print(f"mass peak for d = {k:>3}: r = {r[np.argmax(log_mass)]:.3f} vs sqrt(d - 1) = {np.sqrt(k - 1):.3f}")
ax2.plot([], [], color="grey", ls=":", label=r"shell area $r^{d-1}$ (dotted)")
ax2.set_xlabel("radius r"); ax2.set_ylabel("scaled to peak 1")
ax2.set_title("Density falls, area grows: mass peaks far out"); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()
```
```text
d = 512: 0 of 100,000 samples lie within distance 10 of the origin, the density's peak
||z||: mean 22.61 vs sqrt(d) = 22.63 | std 0.708 vs 1/sqrt(2) = 0.707 | smallest of 100,000: 19.55
  d =   1: mean ||z||   0.81 (sqrt(d) =   1.00), std 0.600
  d =   2: mean ||z||   1.25 (sqrt(d) =   1.41), std 0.655
  d =  10: mean ||z||   3.09 (sqrt(d) =   3.16), std 0.696
  d = 100: mean ||z||   9.98 (sqrt(d) =  10.00), std 0.705
  d = 512: mean ||z||  22.62 (sqrt(d) =  22.63), std 0.704
mass peak for d =   2: r = 0.999 vs sqrt(d - 1) = 1.000
mass peak for d =  10: r = 3.001 vs sqrt(d - 1) = 3.000
mass peak for d = 100: r = 9.951 vs sqrt(d - 1) = 9.950
```
![Output of cell 2]({{ '/assets/blog/high-dimensional-soap-bubble/cell-02-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: left, how far random points land from the centre in 1, 2, 10, 100 and 512 dimensions. Each hump moves right, to its dotted line at √d, without getting wider. Right, the reason. The dashed curve (how crowded points are at each distance) falls, the dotted curves (how much space there is at that distance) rise, and their product, the solid curves, peaks far from the centre. Takeaway: points land where there is some crowding and a lot of room.*

Zero of the 100,000 samples come within 10 of the origin, and the closest is 19.55 away. The mean norm is 22.61 against $$\sqrt{512} = 22.63$$ and the spread 0.708 against the predicted 0.707. The table shows the width settling: 0.600 at $$d = 1$$, 0.696 at $$d = 10$$, 0.704 at $$d = 512$$. The mean tracks $$\sqrt{d}$$ more closely as $$d$$ grows, from 0.81 against 1.00 at $$d = 1$$ (where a first-order expansion has no business working) to 9.98 against 10.00 at $$d = 100$$.

## The most likely point nobody visits

Two things decide how many samples land at distance $$r$$ from the origin: how dense the distribution is there, and how much room there is. The density $$e^{-r^2/2}$$ peaks at the origin and falls with $$r$$. The room, the surface area of a sphere of radius $$r$$, grows like $$r^{d-1}$$, a very steep power when $$d = 512$$. Near the origin there's density and no room. Far out it's the reverse. Samples land where the product peaks, and you can watch the two curves multiply as $$d$$ grows:

![A fixed falling density curve and a rising room curve multiply into a teal mass curve whose peak moves out to the square root of d minus 1 as d grows from 2 to 100, while the densest point stays at the origin]({{ '/assets/blog/high-dimensional-soap-bubble/density-times-surface.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the animation shows: the dashed curve is how crowded points are at each distance from the centre. It never changes. The dotted curve is how much space there is at that distance, and it rises more steeply as d grows. Their product, the solid curve, is where points actually land, and its peak slides outward to √(d − 1). Takeaway: the centre is the most crowded spot, but it is so small that almost nothing lands there.*

To make this exact, add up the density over a thin shell of radius $$r$$ and thickness $$dr$$. The shell's volume is proportional to $$r^{d-1}\,dr$$, so the radius has the distribution

$$
p(r) \;\propto\; r^{d-1}\, e^{-r^2/2}.
$$

Setting $$\frac{d}{dr}\log p(r) = \frac{d-1}{r} - r$$ to zero puts the mode at $$r^* = \sqrt{d-1}$$, a hair inside $$\sqrt{d}$$ (by 0.022 at $$d = 512$$). Now compare the origin with a place where a typical sample lands. Under the prior $$\log p(z) = -\tfrac12\|z\|^2 + \text{const}$$, so

$$
\begin{aligned}
\log p(0) - \log p(z_{\text{typical}}) &= \tfrac12 \|z_{\text{typical}}\|^2 \\
&\approx \tfrac{d}{2},
\end{aligned}
$$

which is 256 nats at $$d = 512$$. The origin is $$e^{256} \approx 10^{111}$$ times denser than where samples actually go, and cell 2 still found none of 100,000 within distance 10 of it. The most probable point is atypical. Information theory calls the shell the *typical set*: the samples whose log-density is close to its average, $$-\tfrac{d}{2}\log(2\pi e)$$. Almost every sample is typical, and the single most probable point is not.

Plain volume does the same thing, no Gaussian required. Shrinking a ball by a factor $$1 - \varepsilon$$ multiplies its volume by $$(1-\varepsilon)^d$$, so the fraction of the volume within $$\varepsilon$$ of the surface is $$1 - (1-\varepsilon)^d \ge 1 - e^{-\varepsilon d}$$. At $$d = 512$$ and $$\varepsilon = 0.01$$ that's 99.4%. A 512-dimensional orange is nearly all peel.

> **"The middle of a high-dimensional Gaussian is empty" is half right.** The origin is the density's mode: per unit volume, no point is more probable. What it lacks is volume. A model that reports densities, such as the log-likelihood of a VAE or a normalising flow, will rate the centre of latent space as the most plausible place there is, and a decoder trained on samples has never been there.

The right panel of cell 2 checks the mode. The solid mass curves peak at 0.999, 3.001 and 9.951 for $$d$$ = 2, 10 and 100, against $$\sqrt{d-1}$$ = 1.000, 3.000 and 9.950.

## Every two seeds are perpendicular

Two random directions in 512 dimensions have almost nothing in common. Their dot product is a sum of 512 products $$z_{0j} z_{1j}$$, each as likely to be negative as positive, and the sum mostly cancels. Draw two samples as arrows from the origin and both end on the shell, nearly always at close to 90° to each other. In 2-D the angle between two random directions could be anything. In 512-D its distribution collapses onto a right angle.

The cosine of the angle is

$$
\cos\theta = \frac{\langle z_0, z_1\rangle}{\|z_0\|\,\|z_1\|} = \frac{\sum_j z_{0j}\, z_{1j}}{\|z_0\|\,\|z_1\|}.
$$

The numerator sums $$d$$ independent terms with mean 0 and variance 1, so it's approximately $$\mathcal{N}(0, d)$$, and the denominator is about $$\sqrt{d}\cdot\sqrt{d} = d$$. Dividing,

$$
\cos\theta \;\approx\; \mathcal{N}\!\left(0, \tfrac{1}{d}\right).
$$

At $$d = 512$$ the standard deviation is 0.044, so a typical pair is within about 2.5° of a right angle. Near-orthogonality is a standard consequence of concentration of measure ([Vershynin, *High-Dimensional Probability*, ch. 3](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-1.pdf)). For this post it means every interpolation between two random seeds starts from the same picture: two points on a sphere, at the same radius, a quarter turn apart. The first panel of cell 3, at the end of the next section, checks the $$1/\sqrt{d}$$ law.

## LERP cuts through the core

Push a straight needle through a soap bubble, from one point on its skin to another a quarter turn away, and it passes through the air inside. LERP does the same in latent space. Its midpoint averages two points on the shell, and the average of two nearly perpendicular vectors of length $$\sqrt{d}$$ has length about $$\sqrt{d/2}$$, deep inside the bubble.

In the plane through $$z_0$$, $$z_1$$ and the origin, the endpoints sit on a circle of radius $$\sqrt{d}$$ a quarter turn apart, and LERP is the chord between them. A chord across a quarter circle dips to $$\cos 45° \approx 0.707$$ of the radius at its middle. The next section draws this plane exactly, from real 512-dimensional samples.

For independent $$z_0, z_1 \sim \mathcal{N}(0, I)$$, the point $$z(t) = (1-t)\,z_0 + t\,z_1$$ is a linear combination of independent Gaussians, so it is Gaussian with mean 0 and covariance $$\big((1-t)^2 + t^2\big) I$$:

$$
\begin{gathered}
z(t) \sim \mathcal{N}\!\big(0,\; ((1-t)^2 + t^2)\, I\big), \\
\|z(t)\| \approx \sqrt{\big((1-t)^2 + t^2\big)\, d}.
\end{gathered}
$$

At $$t = \tfrac12$$ the variance is exactly $$\tfrac12$$. The midpoint is a sample from $$\mathcal{N}(0, I/2)$$, not from the prior, and at $$d = 512$$ its norm is about 16.0 against a shell at 22.6 with spread 0.707, roughly $$(22.6 - 16.0)/0.707 \approx 9.4$$ shell standard deviations inside. Cell 3 computes the exact chance that a genuine prior sample is that short, $$P(\chi^2_{512} \le 256)$$.

The likelihood moves the other way. Since $$\log p(z) = -\tfrac12\|z\|^2 + \text{const}$$, the endpoints have $$\|z\|^2 \approx d$$ and the midpoint $$\|z\|^2 \approx d/2$$, so

$$
\begin{aligned}
\log p\big(z(\tfrac12)\big) - \log p(z_0) &\approx -\frac{d}{4} + \frac{d}{2} \\
&= \frac{d}{4},
\end{aligned}
$$

which is 128 nats at $$d = 512$$. Per unit volume the midpoint is $$e^{128}$$ times *more* likely than either end. LERP climbs the likelihood all the way to the middle, into a region of high density where no sample ever lands. [Nalisnick et al. (2018)](https://arxiv.org/abs/1810.09136) met the same gap from the other side: deep generative models trained on CIFAR-10 give higher likelihood to SVHN, a different dataset, than to their own test images, and the authors' follow-up traced this to the difference between high density and typicality ([Nalisnick et al., 2019](https://arxiv.org/abs/1906.02994)). A high-likelihood point can be one the model would never produce.

Cell 3 checks the cosine law for $$d$$ = 2, 10, 100 and 512, then follows one LERP path at $$d = 512$$ and averages the midpoint's statistics over 10,000 more pairs.

```python
rng = np.random.default_rng(1)
cosines = {}
for k in [2, 10, 100, 512]:
    a, b = rng.standard_normal((2, 20_000, k))
    cosines[k] = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print(f"d = {k:>3}: cos(angle) std {cosines[k].std():.4f} vs 1/sqrt(d) = {1 / np.sqrt(k):.4f} | "
          f"angles within 90 +- 10 deg: {(np.abs(np.degrees(np.arccos(cosines[k])) - 90) < 10).mean():.1%}")

d = 512
z0, z1 = rng.standard_normal((2, d))           # the pair drawn in this cell and the next
t = np.linspace(0, 1, 101)
line = lerp(z0, z1, t)
pairs = rng.standard_normal((2, 10_000, d))    # 10,000 more pairs for the averages
mids = lerp(pairs[0], pairs[1], [0.5])[0]
shell = np.linalg.norm(pairs[0], axis=1)
mid_norm = np.linalg.norm(mids, axis=1)
print(f"\nLERP midpoint, 10,000 pairs at d = {d}: per-coordinate variance {mids.var():.4f} (exact: 0.5)")
print(f"  ||midpoint|| {mid_norm.mean():.2f} vs sqrt(d/2) = {np.sqrt(d / 2):.2f}; shell {shell.mean():.2f} +- {shell.std():.3f}"
      f" -> {(shell.mean() - mid_norm.mean()) / shell.std():.1f} shell standard deviations inside")
print(f"  P(a real sample is that short) = P(chi2_{d} <= {d // 2}) = 10^{chi2.logcdf(d / 2, d) / np.log(10):.1f}")
log_p = lambda z: -0.5 * (z**2).sum(-1) - 0.5 * z.shape[-1] * np.log(2 * np.pi)  # log N(z; 0, I)
gain = log_p(mids) - 0.5 * (log_p(pairs[0]) + log_p(pairs[1]))
print(f"  log p(midpoint) - log p(endpoints): {gain.mean():+.1f} nats on average (d/4 = {d / 4:.0f})")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
for k, c in zip([10, 100, 512], plt.cm.viridis([0.25, 0.55, 0.85])):
    ax1.hist(cosines[k], bins=120, range=(-1, 1), density=True, histtype="step", lw=2, color=c, label=f"d = {k}")
    u = np.linspace(-1, 1, 400)
    ax1.plot(u, np.exp(-u**2 * k / 2) * np.sqrt(k / (2 * np.pi)), color=c, ls="--", lw=1)
ax1.set_xlabel("cosine of the angle between two samples"); ax1.set_ylabel("density")
ax1.set_title(r"Random pairs are nearly perpendicular (dashed: $\mathcal{N}(0, 1/d)$)", fontsize=10); ax1.legend()
lo, hi = np.percentile(shell, [0.5, 99.5])
ax2.axhspan(lo, hi, color=TEAL, alpha=0.25, label="shell: 99% of samples")
ax2.plot(t, np.linalg.norm(line, axis=1), color=ORANGE, lw=2.5, label="LERP, one pair")
ax2.plot(t, np.sqrt(((1 - t)**2 + t**2) * d), "k--", lw=1, label=r"$\sqrt{((1-t)^2+t^2)\,d}$")
ax2.set_xlabel("t"); ax2.set_ylabel(r"$\|z(t)\|$"); ax2.set_ylim(0, 26); ax2.legend(loc="lower left", fontsize=8)
ax2.set_title("The straight line sags into the core")
ax3.plot(t, log_p(line), color=ORANGE, lw=2.5, label="LERP, one pair")
ax3.axhline(log_p(pairs[0]).mean(), color=TEAL, lw=1.5, ls="--", label="typical sample")
ax3.set_xlabel("t"); ax3.set_ylabel(r"$\log p(z(t))$ (nats)"); ax3.legend(fontsize=8)
ax3.set_title("...and climbs in likelihood on the way")
plt.tight_layout(); plt.show()
```
```text
d =   2: cos(angle) std 0.7065 vs 1/sqrt(d) = 0.7071 | angles within 90 +- 10 deg: 11.2%
d =  10: cos(angle) std 0.3180 vs 1/sqrt(d) = 0.3162 | angles within 90 +- 10 deg: 38.7%
d = 100: cos(angle) std 0.0996 vs 1/sqrt(d) = 0.1000 | angles within 90 +- 10 deg: 92.0%
d = 512: cos(angle) std 0.0444 vs 1/sqrt(d) = 0.0442 | angles within 90 +- 10 deg: 100.0%

LERP midpoint, 10,000 pairs at d = 512: per-coordinate variance 0.5002 (exact: 0.5)
  ||midpoint|| 15.99 vs sqrt(d/2) = 16.00; shell 22.63 +- 0.703 -> 9.4 shell standard deviations inside
  P(a real sample is that short) = P(chi2_512 <= 256) = 10^-22.8
  log p(midpoint) - log p(endpoints): +128.1 nats on average (d/4 = 128)
```
![Output of cell 3]({{ '/assets/blog/high-dimensional-soap-bubble/cell-03-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: left, the angle between two random points. In 10 dimensions it varies a lot; in 512 dimensions it is almost always a right angle. Middle, the distance from the centre along a straight line between two points: the line leaves the teal band where real points sit and sags to about 16 halfway. Right, how likely each point on that line looks to the bell curve: it goes up in the middle. Takeaway: the straight line passes through a region that looks likely but never actually gets sampled.*

The cosine spreads match $$1/\sqrt{d}$$ to three decimals (0.0444 against 0.0442 at $$d = 512$$). The share of pairs within 10° of a right angle climbs from 11.2% in 2-D to 100.0% at $$d = 512$$. Over 10,000 pairs the LERP midpoint has per-coordinate variance 0.5002 and norm 15.99 against a predicted 16.00: 9.4 shell standard deviations inside a shell at 22.63. A real sample that short has probability $$10^{-22.8}$$. The log-density gain is +128.1 nats, against $$d/4 = 128$$. In the middle panel the single path leaves the teal band within the first sixth of the way and follows the dashed $$\sqrt{((1-t)^2 + t^2)\,d}$$ curve down to about 16. In the right panel its log-density climbs from about −750 to −605 nats, far above a typical sample's −727, before falling back.

## SLERP rides the shell

The fix is to walk along the skin instead of through it: keep the length fixed and turn the direction from $$z_0$$ toward $$z_1$$ at a steady rate, like a clock hand sweeping a quarter turn.

Both LERP and SLERP produce combinations $$a\,z_0 + b\,z_1$$, so both paths lie entirely in the plane spanned by $$z_0$$ and $$z_1$$. A picture of that plane is an exact slice of 512-dimensional space, with LERP as the chord and SLERP as the arc:

![In the plane through two 512-dimensional samples a quarter turn apart, an orange dot on the straight line sinks through the hollow coral core to 0.707 of the shell radius while a blue dot on the arc stays on the teal shell, with live norm readouts]({{ '/assets/blog/high-dimensional-soap-bubble/chord-vs-arc.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the animation shows: a flat slice through 512-dimensional space. The teal ring is where samples live and the faint coral disc inside it is the empty core. Two dots travel between the same two samples. The orange one takes the straight line and sinks to 71% of the ring's radius halfway; the blue one follows the ring. The numbers on the right are their distances from the centre. Takeaway: the curved path stays where the data is, and the straight one leaves it.*

SLERP comes from computer graphics, where Shoemake (1985) used it to animate rotations, and [White (2016)](https://arxiv.org/abs/1609.04468) brought it to latent spaces for the reason in this post. With $$\Omega$$ the angle between the endpoints, $$\cos\Omega = \langle z_0, z_1\rangle / (\|z_0\|\,\|z_1\|)$$,

$$
\begin{aligned}
\operatorname{slerp}(z_0, z_1; t) = {} & \frac{\sin\big((1-t)\,\Omega\big)}{\sin\Omega}\, z_0 \\
& + \frac{\sin(t\,\Omega)}{\sin\Omega}\, z_1 .
\end{aligned}
$$

It turns at constant angular speed, and when the endpoints have equal length it keeps that length the whole way. Random seeds are about a quarter turn apart, so put $$\Omega = \pi/2$$. Then $$\sin\Omega = 1$$ and $$\sin\big((1-t)\tfrac{\pi}{2}\big) = \cos\tfrac{\pi t}{2}$$, and SLERP becomes

$$
z(t) = \cos\!\big(\tfrac{\pi t}{2}\big)\, z_0 + \sin\!\big(\tfrac{\pi t}{2}\big)\, z_1 .
$$

For independent $$z_0, z_1 \sim \mathcal{N}(0, I)$$, this $$z(t)$$ is Gaussian with covariance $$\big(\cos^2\tfrac{\pi t}{2} + \sin^2\tfrac{\pi t}{2}\big) I = I$$. Every point on the path is distributed exactly like a prior sample, for every $$t$$; the path is *variance-preserving*. LERP's coefficients $$(1-t)$$ and $$t$$ sum to 1, which preserves the mean. SLERP's coefficients have squares that sum to 1, which preserves the variance, and in high dimensions the variance sets the radius.

Encoded codes usually have different lengths, and then SLERP's guarantees loosen. It no longer turns at exactly constant speed (nearly so when the lengths are similar), and for endpoints at most 90° apart its length stays between the two endpoints' lengths. The other common fix, LERP followed by rescaling each point back to the shell, lands on the same arc but moves along it at uneven angular speed.

> **"SLERP is a hack" is a myth.** It looks like a trick from animation software, but at 90° it is the path whose every point has the prior's distribution, which is what you want for a decoder trained on prior samples. LERP is the one with the unjustified assumption, that averaging two typical points gives a typical point.

Cell 4 reuses cell 3's pair, builds an orthonormal basis for their plane with one step of Gram-Schmidt, and projects both paths onto it. It checks that the projection loses nothing, compares the real SLERP with the cos/sin form, and compares per-coordinate variances over cell 3's 10,000 pairs.

```python
# Both paths are combinations of z0 and z1, so they lie in span(z0, z1): this plane is exact, not a projection.
e1 = z0 / np.linalg.norm(z0)
e2 = z1 - (z1 @ e1) * e1
e2 /= np.linalg.norm(e2)
basis = np.stack([e1, e2])                       # Gram-Schmidt: an orthonormal basis of the plane
arc = slerp(z0, z1, t)
vp = np.cos(np.pi * t / 2)[:, None] * z0 + np.sin(np.pi * t / 2)[:, None] * z1  # cos/sin, variance-preserving
xy_line, xy_arc = line @ basis.T, arc @ basis.T
residual = max(np.abs(line - xy_line @ basis).max(), np.abs(arc - xy_arc @ basis).max())
omega = np.degrees(np.arccos(z0 @ z1 / (np.linalg.norm(z0) * np.linalg.norm(z1))))
print(f"angle between the endpoints: {omega:.1f} deg | largest coordinate off the plane: {residual:.1e}")
print(f"||z(t)|| along LERP:  min {np.linalg.norm(line, axis=1).min():.2f} = "
      f"{np.linalg.norm(line, axis=1).min() / np.sqrt(d):.3f} sqrt(d)")
print(f"||z(t)|| along SLERP: from {np.linalg.norm(arc, axis=1).min():.2f} to {np.linalg.norm(arc, axis=1).max():.2f}")
print(f"SLERP vs cos/sin form: largest gap {np.linalg.norm(arc - vp, axis=1).max():.2f} "
      f"({np.linalg.norm(arc - vp, axis=1).max() / np.sqrt(d):.1%} of the radius; exactly 0 at 90 deg)")
for s_t in [0.3, 0.5]:
    vp_batch = np.cos(np.pi * s_t / 2) * pairs[0] + np.sin(np.pi * s_t / 2) * pairs[1]
    print(f"per-coordinate variance at t = {s_t}: cos/sin {vp_batch.var():.4f} | LERP {lerp(pairs[0], pairs[1], [s_t])[0].var():.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
shell_sd = shell.std()
theta = np.linspace(0, 2 * np.pi, 400)
core = shell.mean() - 3 * shell_sd
ax1.fill(core * np.cos(theta), core * np.sin(theta), color=CORAL, alpha=0.08)  # the hollow core
for k, alpha in [(3, 0.18), (1, 0.4)]:
    outer, inner = shell.mean() + k * shell_sd, shell.mean() - k * shell_sd
    ax1.fill(np.r_[outer * np.cos(theta), inner * np.cos(theta[::-1])],
             np.r_[outer * np.sin(theta), inner * np.sin(theta[::-1])], color=TEAL, alpha=alpha, lw=0)
ax1.plot(*xy_line.T, color=ORANGE, lw=2.5, label="LERP: the chord")
ax1.plot(*xy_arc.T, color=BLUE, lw=2.5, label="SLERP: the arc")
ax1.plot(*xy_line[[0, -1]].T, "o", color="k", ms=6)
ax1.plot(*xy_line[50], "o", color=ORANGE, ms=8)
ax1.plot(0, 0, "+", color=CORAL, ms=12, mew=2)
ax1.text(1, -2.5, "origin: highest density,\nno samples", color=CORAL, fontsize=9)
ax1.text(*(xy_line[0] + [0.6, 0.6]), r"$z_0$"); ax1.text(*(xy_line[-1] + [0.6, 0.3]), r"$z_1$")
ax1.set_aspect("equal"); ax1.set_xlim(-26, 26); ax1.set_ylim(-26, 26)
ax1.set_xlabel(r"coordinate along $z_0$"); ax1.set_ylabel("perpendicular coordinate in span$(z_0, z_1)$")
ax1.set_title(f"The plane through two samples (d = {d}); teal = the shell", fontsize=10); ax1.legend(loc="lower left")
ax2.axhspan(*np.percentile(shell, [0.5, 99.5]), color=TEAL, alpha=0.25, label="shell: 99% of samples")
ax2.plot(t, np.linalg.norm(line, axis=1), color=ORANGE, lw=2.5, label="LERP")
ax2.plot(t, np.linalg.norm(arc, axis=1), color=BLUE, lw=2.5, label="SLERP")
ax2.set_xlabel("t"); ax2.set_ylabel(r"$\|z(t)\|$"); ax2.set_ylim(0, 26); ax2.legend(loc="lower left")
ax2.set_title("SLERP stays on the shell the whole way")
plt.tight_layout(); plt.show()
```
```text
angle between the endpoints: 88.0 deg | largest coordinate off the plane: 1.8e-15
||z(t)|| along LERP:  min 16.40 = 0.725 sqrt(d)
||z(t)|| along SLERP: from 21.97 to 23.71
SLERP vs cos/sin form: largest gap 0.40 (1.8% of the radius; exactly 0 at 90 deg)
per-coordinate variance at t = 0.3: cos/sin 1.0009 | LERP 0.5806
per-coordinate variance at t = 0.5: cos/sin 1.0003 | LERP 0.5002
```
![Output of cell 4]({{ '/assets/blog/high-dimensional-soap-bubble/cell-04-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: the same picture, drawn from two real 512-dimensional samples (black dots). Left, the straight line (orange) cuts through the pale empty core, while the arc (blue) stays on the teal ring. Right, the distance from the centre along each path: blue stays inside the teal band and orange dips well below it. Takeaway: this is an exact slice. What you see is what really happens in 512 dimensions.*

This pair is 88.0° apart. The paths leave the plane by at most $$1.8 \times 10^{-15}$$, rounding error, so the left panel really is an exact slice. The chord bottoms out at 16.40, or $$0.725\sqrt{d}$$, a little above $$0.707\sqrt{d}$$ because the angle is 88° rather than 90° and the endpoints differ slightly in length. SLERP stays between 21.97 and 23.71, the lengths of its two endpoints, inside the 99% band all the way. It differs from the idealised cos/sin path by at most 0.40, 1.8% of the radius, for the same 88° reason. Over 10,000 pairs, the cos/sin path holds the per-coordinate variance at 1.0009 at $$t = 0.3$$ and 1.0003 at $$t = 0.5$$, while LERP drops it to 0.5806 and 0.5002, matching $$(1-t)^2 + t^2 = 0.58$$ and $$0.50$$.

## The same shape inside diffusion

Diffusion models are built on the same structure, and it's where the name "variance-preserving" comes from. The forward process of a denoising diffusion model ([Ho, Jain & Abbeel, 2020](https://arxiv.org/abs/2006.11239)) noises a data point $$x_0$$ as

$$
\begin{gathered}
x_t = \sqrt{\bar\alpha_t}\; x_0 + \sqrt{1 - \bar\alpha_t}\;\epsilon, \\
\epsilon \sim \mathcal{N}(0, I),
\end{gathered}
$$

where $$\bar\alpha_t$$ falls from 1 (clean data) to nearly 0 (pure noise). The squares of the two coefficients sum to 1, as on the cos/sin path, so if $$x_0$$ has unit variance per coordinate, so does every $$x_t$$. Write $$\sqrt{\bar\alpha_t} = \cos\phi_t$$ and the forward process is a quarter turn from data to noise. The cosine schedule of [Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) makes this almost literal, since its $$\bar\alpha_t$$ is a squared cosine of time.

The sampler runs the process backwards, and DDIM ([Song, Meng & Ermon, 2020](https://arxiv.org/abs/2010.02502)) makes that run deterministic. With its noise parameter at zero, the same starting noise always produces the same image, so the sampler is a fixed map from noise to images. The map only ever trained on starting noise at the shell's radius, and the DDIM paper's own interpolation experiments move between seeds with SLERP. In Stable Diffusion the seed is a 16,384-dimensional Gaussian sample, well past 512 dimensions, and cell 7 goes there. First, a model where the bubble barely matters.

## What the decoder does with a short vector

So far this has all been about where the prior's samples are. Whether a short vector *matters* depends on the decoder, and here the story gets less tidy.

A decoder only listens to the latent dimensions it was trained to use, and a VAE has a cheap way to make a dimension useless: set its posterior equal to the prior, $$\mu_j(x) = 0$$ and $$\sigma_j(x) = 1$$ for every input, which brings that dimension's KL cost to zero. The dimension then carries pure noise, and the decoder learns to ignore it. LERP shrinks every coordinate, but shrinking one the decoder ignores changes nothing. Only the shrinkage in the dimensions it uses counts, and a sphere in 15 dimensions is far less hollow than one in 512.

The test here is a radius dial. Pick a direction in latent space and decode points along it from radius 0 out to $$1.5\sqrt{d}$$. If the decoder cares about length, the dial will show it at $$\sqrt{d/2}$$, where LERP midpoints sit.

Call a latent dimension *active* if its encoder mean moves with the input. [Burda, Grosse & Salakhutdinov (2015)](https://arxiv.org/abs/1509.00519) measure this with the variance of the mean across the dataset:

$$
\begin{gathered}
\begin{aligned}
A_j &= \operatorname{Var}_{x}\Big(\mathbb{E}_{q(z \mid x)}[z_j]\Big) \\
&= \operatorname{Var}_x\big(\mu_j(x)\big),
\end{aligned} \\
\text{dimension } j \text{ is active if } A_j > 0.01 .
\end{gathered}
$$

If $$k$$ of the $$d$$ dimensions are active, the decoder effectively sees a $$k$$-dimensional Gaussian, with its shell at $$\sqrt{k}$$ and the same 0.707 spread. LERP's midpoint still shrinks the active part by $$1/\sqrt{2}$$, but the gap is now $$\sqrt{k}\,(1 - 1/\sqrt{2}) \approx 0.29\sqrt{k}$$, which grows with the number of *active* dimensions rather than with $$d$$. For $$k = 15$$ the midpoint sits about 1.6 spreads inside the shell; for $$k = 512$$ it was 9.4. Part 2 adds a second softening. The decoder doesn't train on prior samples at all but on samples from $$q(z \mid x)$$, and because the aggregate posterior isn't the prior, the encoded means sit well inside the shell.

Cell 5 trains Part 2's VAE with $$d = 64$$, the only change from Part 2. It counts active units, measures norms, turns the radius dial over 1,000 random directions, and compares LERP and SLERP midpoints over 1,000 pairs of prior samples and 1,000 pairs of encoded digits.

```python
d = 64
vae64, _ = train_vae(X_mnist, d=d)  # Part 2's VAE, unchanged except d: Bernoulli, beta = 1, seed 0, 60 epochs
with torch.no_grad():
    mu, logvar = vae64.encode(Xt_mnist)
mu, sigma = mu.numpy(), (0.5 * logvar).exp().numpy()
active = active_units(mu)
z_q = mu + sigma * np.random.default_rng(2).standard_normal(mu.shape)  # codes the decoder trained on
print(f"active units: {active.sum()} of {d}; the other {(~active).sum()} have Var(mu) < 0.01 and mean sigma {sigma[:, ~active].mean():.2f}")
print(f"||mu|| {np.linalg.norm(mu, axis=1).mean():.2f} | ||z ~ q(z|x)|| {np.linalg.norm(z_q, axis=1).mean():.2f} | "
      f"prior shell sqrt(d) = {np.sqrt(d):.2f} | active part of mu: {np.linalg.norm(mu[:, active], axis=1).mean():.2f}")

def decode(z: np.ndarray) -> np.ndarray:
    """Decode a batch of latent codes with the d = 64 VAE; returns (n, 28, 28) images in [0, 1]."""
    with torch.no_grad():
        return vae64.decode(torch.from_numpy(np.asarray(z, dtype=np.float32))).numpy().reshape(-1, H, W)

# The radius dial: the same 1,000 directions decoded at radius 0 ... 1.5 sqrt(d)
rng = np.random.default_rng(3)
directions = rng.standard_normal((1000, d))
directions /= np.linalg.norm(directions, axis=1, keepdims=True)
radii = np.linspace(0, 1.5 * np.sqrt(d), 13)
dial = np.array([(contrast(decode(rad * directions)).mean(), judge_samples(flat(decode(rad * directions)))[0])
                 for rad in radii])
for rad, (con, conf) in zip(radii, dial):
    if rad in radii[[0, 4, 6, 8, 12]]:
        print(f"  radius {rad:5.2f} ({rad / np.sqrt(d):.2f} sqrt(d)): contrast {con:.3f}, judge confidence {conf:.3f}")

# LERP vs SLERP midpoints, averaged over 1,000 pairs of each kind
i, j = rng.choice(len(mu), 1000), rng.choice(len(mu), 1000)
endpoint_pairs = {"prior samples": tuple(rng.standard_normal((2, 1000, d))), "encoded digits (mu)": (mu[i], mu[j])}
for name, (za, zb) in endpoint_pairs.items():
    frames = {"endpoints": decode(np.concatenate([za, zb])), "LERP mid": decode(lerp(za, zb, [0.5])[0]),
              "SLERP mid": decode(slerp(za, zb, [0.5])[0])}
    norms_ = {"endpoints": np.linalg.norm(za, axis=1).mean(), "LERP mid": np.linalg.norm(lerp(za, zb, [0.5])[0], axis=1).mean(),
              "SLERP mid": np.linalg.norm(slerp(za, zb, [0.5])[0], axis=1).mean()}
    print(f"{name}:")
    for key, imgs in frames.items():
        print(f"  {key:<9}  ||z|| {norms_[key]:5.2f} | judge {judge_samples(flat(imgs))[0]:.3f} | contrast {contrast(imgs).mean():.3f}")

# One pair of each kind as frame strips: prior samples, and Part 2's walk-2 digits (a 2 and a 7)
t9 = np.linspace(0, 1, 9)
pa, pb = rng.standard_normal((2, d))
da, db = mu[np.flatnonzero(y_mnist == 2)[0]], mu[np.flatnonzero(y_mnist == 7)[0]]
strips = [("prior samples, LERP", lerp(pa, pb, t9), ORANGE), ("prior samples, SLERP", slerp(pa, pb, t9), BLUE),
          ("digits 2 -> 7, LERP", lerp(da, db, t9), ORANGE), ("digits 2 -> 7, SLERP", slerp(da, db, t9), BLUE)]
fig = plt.figure(figsize=(13, 10.5))
gs = fig.add_gridspec(6, 1, height_ratios=[2.2, 1.2, 1, 1, 1, 1], hspace=0.75)
ax = fig.add_subplot(gs[0])
ax.plot(radii, dial[:, 0] / dial[:, 0].max(), color=SLATE, lw=2.5, marker="o", label="pixel contrast (scaled to max 1)")
ax.plot(radii, dial[:, 1], color="k", lw=2.5, marker="s", label="judge confidence")
ax.axvline(np.sqrt(d / 2), color=ORANGE, ls="--", label=r"LERP midpoint $\sqrt{d/2}$")
ax.axvline(np.sqrt(d), color=TEAL, ls="--", label=r"prior shell $\sqrt{d}$")
ax.set_xlabel(r"radius $\|z\|$ along 1,000 random directions"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, ncol=2)
ax.set_title("The radius dial: same direction, shorter vector")
ax = fig.add_subplot(gs[1])
ax.imshow(montage(decode(radii[:, None] * directions[0]), 1, len(radii)), cmap="gray", vmin=0, vmax=1)
ax.set_xticks(np.arange(len(radii)) * W + W / 2, [f"{rad:.1f}" for rad in radii], fontsize=8); ax.set_yticks([])
ax.set_title("one direction, radius 0 to 1.5 sqrt(d)", loc="left", fontsize=10)
for row, (name, path, colour) in enumerate(strips, start=2):
    ax = fig.add_subplot(gs[row])
    ax.imshow(montage(decode(path), 1, len(t9)), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(t9)) * W + W / 2, [f"|z|={v:.1f}" for v in np.linalg.norm(path, axis=1)], fontsize=8)
    ax.set_yticks([]); ax.set_title(name, loc="left", fontsize=10, color=colour)
plt.show()
```
```text
active units: 15 of 64; the other 49 have Var(mu) < 0.01 and mean sigma 1.00
||mu|| 3.56 | ||z ~ q(z|x)|| 8.03 | prior shell sqrt(d) = 8.00 | active part of mu: 3.55
  radius  0.00 (0.00 sqrt(d)): contrast 0.263, judge confidence 0.684
  radius  4.00 (0.50 sqrt(d)): contrast 0.255, judge confidence 0.659
  radius  6.00 (0.75 sqrt(d)): contrast 0.256, judge confidence 0.717
  radius  8.00 (1.00 sqrt(d)): contrast 0.259, judge confidence 0.751
  radius 12.00 (1.50 sqrt(d)): contrast 0.267, judge confidence 0.781
prior samples:
  endpoints  ||z||  7.98 | judge 0.748 | contrast 0.259
  LERP mid   ||z||  5.62 | judge 0.699 | contrast 0.255
  SLERP mid  ||z||  7.99 | judge 0.740 | contrast 0.259
encoded digits (mu):
  endpoints  ||z||  3.55 | judge 0.912 | contrast 0.274
  LERP mid   ||z||  2.50 | judge 0.727 | contrast 0.255
  SLERP mid  ||z||  3.59 | judge 0.782 | contrast 0.261
```
![Output of cell 5]({{ '/assets/blog/high-dimensional-soap-bubble/cell-05-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: top, the radius dial. 1,000 random directions are decoded at lengths from 0 to 12. Image contrast (grey) stays flat, while the digit classifier's confidence (black) dips near the centre and rises further out. Below it, one direction decoded at each length: a blurry 8-like shape that firms up into a clean 9. The four strips are walks by straight line (orange) and by arc (blue), with each frame's distance from the centre underneath. Takeaway: in this small VAE the two paths look almost the same, because it only uses 15 of its 64 dimensions.*

Only 15 of the 64 dimensions are active. The other 49 have mean $$\sigma$$ of 1.00 and means that don't move with the input. They're switched off: their posterior is the prior, and the decoder ignores them. The encoded means have norm 3.56, almost all of it (3.55) in the 15 active dimensions, well inside a 64-D shell at 8.00. With the encoder's noise added, the codes the decoder trained on have norm 8.03, right on the shell, but most of that length is noise in dimensions the decoder doesn't read.

The radius dial shows that the decoder does care about length, a little. Pixel contrast barely moves (between 0.255 and 0.267 at the five radii printed), so short vectors don't decode to washed-out images. The judge notices, though. Its confidence is 0.684 at the origin, 0.659 at half the shell radius, 0.751 on the shell and 0.781 beyond it, and the strip under the dial shows why: at radius 0 the decoder draws a blurry average, part 8 and part 9, and this direction only becomes a clean 9 as the vector lengthens. In the active dimensions, length works like a confidence knob.

Over 1,000 pairs of prior samples, the LERP midpoint (norm 5.62, against 7.98 at the endpoints) costs the judge 0.049 of confidence, from 0.748 to 0.699, and SLERP at norm 7.99 is back at 0.740. That's a real effect, and a small one. Between encoded digits the LERP drop is bigger, 0.912 to 0.727, but most of it is Part 2's effect: a point halfway between a 2 and a 7 is neither, whichever path you take, and SLERP only gets it back to 0.782. The strips agree. The two walks between prior samples look nearly identical, and the two walks from 2 to 7 differ mainly in how faint the middle frames are.

A decoder that uses even fewer dimensions should care even less. Cell 6 trains the same VAE class with $$d = 16$$ on Part 2's Blob World, whose images have two true coordinates, the blob's position. It brings back Part 2's off-manifold test, the squared pixel distance $$d^2$$ from a decoded frame to the nearest real Blob World image (Part 2's pixel-average ghost scored 6.283), and it follows the decoded blob along one SLERP path.

```python
X_blob, C_blob = blob_world(5000, np.random.default_rng(SEED))  # Part 2's Blob World, same seed
vae_blob, _ = train_vae(flat(X_blob), d=16)                      # the same VAE class, d = 16
with torch.no_grad():
    mu_b = vae_blob.encode(torch.from_numpy(flat(X_blob)))[0].numpy()
active_b = active_units(mu_b)
print(f"Blob World VAE(d = 16): {active_b.sum()} active units; Var(mu_j) by dimension: "
      f"{np.sort(mu_b.var(0))[::-1][:4].round(3)} ... smallest {mu_b.var(0).min():.1e}")

# ---- Part 2's off-manifold test, copied verbatim (Part 2, cell 2) ----
grid_c = np.stack(np.meshgrid(np.arange(2, 26.01, 0.25), np.arange(2, 26.01, 0.25)), -1).reshape(-1, 2)
manifold_index = NearestNeighbors(n_neighbors=1).fit(flat(np.stack([blob(cx, cy) for cx, cy in grid_c])))

def dist2_to_manifold(frames: np.ndarray) -> np.ndarray:
    """Squared pixel distance from each frame to the nearest real Blob World image."""
    return manifold_index.kneighbors(flat(frames))[0][:, 0] ** 2

def centre_of_mass(frames: np.ndarray) -> np.ndarray:
    """Intensity-weighted centre (x, y) of each frame: where the blob is."""
    w = np.clip(frames, 0, None)
    return np.column_stack([(w * xs).sum((1, 2)), (w * ys).sum((1, 2))]) / w.sum((1, 2))[:, None]
# ----

def decode_blob(z: np.ndarray) -> np.ndarray:
    """Decode latent codes with the Blob World VAE; returns (n, 28, 28) images."""
    with torch.no_grad():
        return vae_blob.decode(torch.from_numpy(np.asarray(z, dtype=np.float32))).numpy().reshape(-1, H, W)

rng = np.random.default_rng(4)
za, zb = rng.standard_normal((2, 300, 16))
t11 = np.linspace(0, 1, 11)
d2 = {name: np.array([dist2_to_manifold(decode_blob(p)).mean() for p in path(za, zb, t11)])
      for name, path in [("LERP", lerp), ("SLERP", slerp)]}
print(f"300 prior pairs, d^2 to the nearest real blob image: endpoints {d2['LERP'][[0, -1]].mean():.3f} | "
      f"midpoint LERP {d2['LERP'][5]:.3f}, SLERP {d2['SLERP'][5]:.3f} (Part 2's pixel ghost: 6.283)")
act_norm = lambda z: np.linalg.norm(z[..., active_b], axis=-1).mean()
print(f"in the active dims: ||z|| endpoints {act_norm(za):.2f}, LERP midpoint {act_norm(lerp(za, zb, [0.5])[0]):.2f}"
      f" -- inside a {active_b.sum()}-D Gaussian's bulk, not a hollow core")

# Where the blob goes along one SLERP path: equal steps in t, unequal steps in the image
t41 = np.linspace(0, 1, 41)
track = centre_of_mass(decode_blob(slerp(za[0], zb[0], t41)))
steps = np.linalg.norm(np.diff(track, axis=0), axis=1)
print(f"one SLERP path, 40 equal steps in t: blob moves {steps.min():.2f} to {steps.max():.2f} pixels per step "
      f"(std / mean = {steps.std() / steps.mean():.2f}; 0 would be constant speed)")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 3.9))
order = np.argsort(mu_b.var(0))[::-1]
ax1.bar(np.arange(16), mu_b.var(0)[order], color=np.where(active_b[order], TEAL, SLATE))
ax1.axhline(0.01, color=CORAL, ls="--", lw=1); ax1.set_yscale("log")
ax1.set_xlabel("latent dimension (sorted)"); ax1.set_ylabel(r"Var$_x(\mu_j)$"); ax1.set_title("Active units (teal)")
ax2.plot(t11, d2["LERP"], color=ORANGE, lw=2.5, marker="o", label="LERP")
ax2.plot(t11, d2["SLERP"], color=BLUE, lw=2.5, marker="s", label="SLERP")
ax2.set_ylim(0, None); ax2.set_xlabel("t"); ax2.set_ylabel(r"mean $d^2$ to nearest real image"); ax2.legend()
ax2.set_title("LERP drifts off a little; SLERP does not")
ax3.plot(track[:, 0], track[:, 1], color=BLUE, lw=1)
ax3.plot(track[:, 0], track[:, 1], "o", color=BLUE, ms=4)
ax3.set_xlim(0, 27); ax3.set_ylim(27, 0); ax3.set_aspect("equal")
ax3.set_xlabel("x (pixels)"); ax3.set_ylabel("y (pixels)"); ax3.set_title("Blob centre, every 0.025 in t")
ax4.plot(t41[1:] - 0.0125, steps, color=BLUE, lw=2.5)
ax4.set_xlabel("t"); ax4.set_ylabel("pixels moved per step"); ax4.set_title("Equal steps in z, unequal in the image")
plt.tight_layout(); plt.show()
```
```text
Blob World VAE(d = 16): 4 active units; Var(mu_j) by dimension: [1.11  1.067 1.042 0.797] ... smallest 2.2e-04
300 prior pairs, d^2 to the nearest real blob image: endpoints 2.075 | midpoint LERP 2.746, SLERP 2.073 (Part 2's pixel ghost: 6.283)
in the active dims: ||z|| endpoints 1.81, LERP midpoint 1.32 -- inside a 4-D Gaussian's bulk, not a hollow core
one SLERP path, 40 equal steps in t: blob moves 0.08 to 0.61 pixels per step (std / mean = 0.40; 0 would be constant speed)
```
![Output of cell 6]({{ '/assets/blog/high-dimensional-soap-bubble/cell-06-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: first, how much each of the Blob World model's 16 code numbers is actually used. Four (teal) are used and twelve (grey) are switched off. Second, how far decoded frames are from a real blob image along straight (orange) and curved (blue) walks; the straight walk drifts a little in the middle. Third and fourth, where the blob is along one curved walk and how far it moves per step. Takeaway: with only four dimensions in use the empty core barely matters, but equal steps in code space still move the blob by unequal amounts.*

The Blob World VAE uses 4 of its 16 dimensions, each with $$\operatorname{Var}(\mu_j)$$ between 0.80 and 1.11; the other twelve sit roughly three orders of magnitude lower. That's two more than the blob's true coordinates, and still only a quarter of the space. Over 300 prior pairs, the LERP midpoint drifts from $$d^2 = 2.075$$ at the endpoints to 2.746, 32% further from the nearest real image, while SLERP's midpoint stays at 2.073. The drift is measurable and far below the ghost's 6.283, and the norms explain why. In the four active dimensions the endpoints have norm 1.81 and the LERP midpoint 1.32. A 4-D Gaussian's samples spread widely, with 90% of their lengths between 0.84 and 3.08, so 1.32 is an ordinary length. With four active dimensions there's no hollow core to fall into. The bubble only bites when a decoder uses many dimensions, and cell 7 finds one that uses all of them.

The last two panels show something the final section comes back to. They follow the decoded blob along one SLERP path in 40 equal steps of $$t$$. SLERP moves through latent space at constant angular speed by construction, yet the blob moves anywhere from 0.08 to 0.61 pixels per step, nearly stopping around $$t = 0.73$$ and then racing to the end. Equal steps in $$z$$ aren't equal steps in the image.

## Where every dimension counts: Stable Diffusion's noise

Stable Diffusion's starting noise for a 512 × 512 image is a 4 × 64 × 64 tensor: 16,384 numbers, all drawn from $$\mathcal{N}(0, I)$$, and the denoiser reads every one. No KL term switches dimensions off, and no encoder leaves codes inside the shell. In training the starting noise was always a fresh sample, so its norm was almost always within 2% of $$\sqrt{16384} = 128$$ (three spreads of 0.7 on a radius of 128). A LERP midpoint has norm about 90.5.

Cell 7 uses SD-Turbo, which its [model card](https://huggingface.co/stabilityai/sd-turbo) describes as Stable Diffusion 2.1 distilled with adversarial diffusion distillation ([Sauer et al., 2023](https://arxiv.org/abs/2311.17042)) so that it makes an image in a single step. With one step and no guidance it's a fixed function from noise to image, the same determinism DDIM gives a many-step sampler. On a laptop CPU it takes roughly 20 seconds per image. Two noise seeds, 11 and 42, are joined by LERP and by SLERP in seven frames each, all with the same prompt, and the cell scores every frame by roughness: the mean jump in brightness between horizontally neighbouring pixels, low for a smooth photograph and high for static.

The cell is optional. Recomputing it needs `diffusers`, the 2.5 GB model download and about five minutes on a CPU. It saves its results to `./assets/sd/`, later runs just load them, and nothing else in the post depends on it. SD-Turbo is released as a research model under the Stability AI Community License, which sets its own terms for commercial use ([model card](https://huggingface.co/stabilityai/sd-turbo)).

```python
# Optional cell. Recomputing needs: %pip install -q "diffusers==0.43.0" "accelerate==1.16.0"
# and downloads stabilityai/sd-turbo (~2.5 GB, Stability AI Community License) once; about 5 minutes on a laptop CPU.
# The results are saved to ./assets/sd/, and later runs just load them.
cache = Path("assets/sd/sd_turbo_strips.npz")
if not cache.exists():
    from diffusers import AutoPipelineForText2Image
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", variant="fp16", torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    shape = (4, 64, 64)                                   # SD's noise latent for a 512 x 512 image: 16,384 numbers
    seeds = [torch.randn(shape, generator=torch.Generator().manual_seed(s)).numpy() for s in (11, 42)]
    t7 = np.linspace(0, 1, 7)
    paths = np.stack([lerp(*[s.ravel() for s in seeds], t7), slerp(*[s.ravel() for s in seeds], t7)])
    images = []
    for path in paths:                                    # one step, no guidance: a fixed map from noise to image
        out = pipe(["a photo of a lighthouse on a rocky coast at sunset"] * len(path),
                   latents=torch.from_numpy(path.reshape(-1, *shape).astype(np.float32)),
                   num_inference_steps=1, guidance_scale=0.0, output_type="np").images
        images.append((torch.nn.functional.interpolate(torch.from_numpy(out).permute(0, 3, 1, 2), size=256,
                                                       mode="area").permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, images=np.stack(images), norms=np.linalg.norm(paths, axis=-1), t=t7)

sd = np.load(cache)
images, latent_norms, t7 = sd["images"], sd["norms"], sd["t"]   # images: (2 paths, 7 frames, 256, 256, 3)
dim = 4 * 64 * 64
print(f"SD noise latent: d = {dim:,} | shell sqrt(d) = {np.sqrt(dim):.1f} | LERP midpoint sqrt(d/2) = {np.sqrt(dim / 2):.1f}")
for name, k in [("LERP", 0), ("SLERP", 1)]:
    rough = np.abs(np.diff(images[k].astype(np.float64), axis=2)).mean((1, 2, 3)) / 255  # jumps between neighbouring pixels
    print(f"{name:<5} ||z||     {' '.join(f'{v:6.1f}' for v in latent_norms[k])}")
    print(f"{'':<5} roughness {' '.join(f'{v:6.3f}' for v in rough)}")

fig, axes = plt.subplots(2, len(t7), figsize=(15, 4.9))
for k, (name, colour) in enumerate([("LERP", ORANGE), ("SLERP", BLUE)]):
    for j in range(len(t7)):
        axes[k, j].imshow(images[k, j]); axes[k, j].set_xticks([]); axes[k, j].set_yticks([])
        axes[k, j].set_xlabel(f"t={t7[j]:.2f}, |z|={latent_norms[k, j]:.0f}", fontsize=8)
    axes[k, 0].set_ylabel(name, color=colour, fontsize=12)
plt.suptitle("SD-Turbo, one step: the same prompt from noise interpolated two ways", fontsize=11)
plt.tight_layout(); plt.show()
```
```text
SD noise latent: d = 16,384 | shell sqrt(d) = 128.0 | LERP midpoint sqrt(d/2) = 90.5
LERP  ||z||      127.7  108.5   95.3   90.5   95.6  109.2  128.5
      roughness  0.025  0.022  0.091  0.115  0.093  0.019  0.019
SLERP ||z||      127.7  127.8  127.9  128.1  128.3  128.5  128.5
      roughness  0.025  0.025  0.025  0.025  0.021  0.020  0.019
```
![Output of cell 7]({{ '/assets/blog/high-dimensional-soap-bubble/cell-07-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: an image generator turns random noise into a photo of a lighthouse. Each row walks between the same two noise starting points. Top, the straight-line walk: the middle frames turn into coloured static. Bottom, the arc walk: every frame is a clean lighthouse. The number under each frame is its distance from the centre, 128 on the shell and down to 91 on the straight line. Takeaway: a model that uses all 16,384 of its dimensions breaks as soon as you leave the shell.*

The norms come out as predicted. LERP falls from 127.7 to 90.5 at the midpoint, $$\sqrt{d/2}$$ to the decimal, while SLERP stays between 127.7 and 128.5. The images are the clearest result in the post. Every SLERP frame is a clean lighthouse, with rocks, sea and sky shifting from one composition to the other, and roughness stays between 0.019 and 0.025. The LERP frames already look blotchy at $$t = 0.17$$, though roughness hasn't risen yet (0.022). From $$t = 0.33$$ to $$t = 0.67$$ they're coloured speckle over a faint lighthouse silhouette, with roughness up to 0.115, more than four times the endpoints' 0.025 and 0.019.

The 2-D intuition predicts something washed out or blurry in the middle, like Part 2's ghosts. These frames are noisy instead. A likely explanation, though I haven't tested it: in one step the network estimates the noise in its input and removes it, and it only ever trained on inputs whose noise had the shell's size. A LERP midpoint is about 30% too short, so the removal no longer matches the input, and the image decoder renders what's left over as high-frequency speckle.

The geometry is the same as in cells 5 and 6. What differs is how many dimensions the generator listens to: 15 in the MNIST VAE, 4 in Blob World, all 16,384 here.

## Bonus: transformers keep resetting to the shell

<details markdown="1">
<summary><strong>Open the bonus box:</strong> GPT-2's LayerNorm puts every token back on a sphere of radius √768</summary>

Transformers meet the same shell, by construction. Every GPT-2 block starts by normalising its input, and rescaling a vector so its coordinates have mean 0 and variance 1 fixes its length at $$\sqrt{d}$$, the radius of the Gaussian shell. Whatever the residual stream is doing, each block reads it off the same sphere. Geometrically, that's a sphere of radius $$\sqrt{d}$$ cut by the hyperplane of vectors whose coordinates sum to zero. The learned gain then stretches the sphere into an ellipsoid, differently in every block, and the bias shifts it; [Riechers (2024)](https://arxiv.org/abs/2405.04134) works out the ellipsoid's axes.

LayerNorm ([Ba, Kiros & Hinton, 2016](https://arxiv.org/abs/1607.06450)) computes $$y = (x - m\mathbf{1})/s$$ before its gain and bias, where $$m$$ is the mean of $$x$$'s $$d$$ coordinates and $$s = \sqrt{\tfrac1d \sum_i (x_i - m)^2}$$. Then

$$
\begin{gathered}
\|y\|^2 = \sum_{i=1}^{d} \frac{(x_i - m)^2}{s^2} = \frac{d\, s^2}{s^2} = d, \\
\sum_{i=1}^{d} y_i = 0 .
\end{gathered}
$$

so $$\|y\| = \sqrt{d}$$ exactly, for every input. RMSNorm ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)), which most current LLMs use, skips the centring: $$y = x/\operatorname{RMS}(x)$$ with $$\operatorname{RMS}(x) = \|x\|/\sqrt{d}$$, so $$\|y\| = \sqrt{d}$$ again.

> **"LayerNorm makes activations Gaussian" is a myth.** It fixes two numbers per token, the mean and the length, and nothing else. Points on a sphere can bunch into a narrow cone, and GPT-2's do: [Ethayarajh (2019)](https://arxiv.org/abs/1909.00512) found its contextual representations highly anisotropic, a long way from the nearly perpendicular directions of Gaussian samples.

Cell 8 hooks 13 of GPT-2 small's LayerNorms, the one at the start of each of the 12 blocks plus the final one, and records their inputs and outputs for one sentence.

```python
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
from huggingface_hub.utils import logging as hub_logging
from transformers import AutoTokenizer, GPT2Model, logging as tf_logging
hub_logging.set_verbosity_error(); tf_logging.set_verbosity_error(); tf_logging.disable_progress_bar()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2 = GPT2Model.from_pretrained("gpt2").eval()   # GPT-2 small, 124M parameters, d_model = 768
text = "Draw a sample from a Gaussian in five hundred dimensions and it lands on a thin shell, far from the origin."
layer_norms = [block.ln_1 for block in gpt2.h] + [gpt2.ln_f]   # the LayerNorm in front of every block, and the last one
seen = []
hooks = [ln.register_forward_hook(lambda m, inp, out: seen.append((inp[0][0], out[0]))) for ln in layer_norms]
with torch.no_grad():
    gpt2(**tokenizer(text, return_tensors="pt"))
for h in hooks:
    h.remove()

d_model = 768
residual = torch.stack([x.norm(dim=-1) for x, _ in seen])            # (13 LayerNorms, tokens): what flows in
pre_gain = torch.stack([((x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + ln.eps)).norm(dim=-1)
                        for (x, _), ln in zip(seen, layer_norms)])  # LayerNorm before its learned gain
post_gain = torch.stack([y.norm(dim=-1) for _, y in seen])          # after gain and bias
print(f"{residual.shape[1]} tokens, {len(layer_norms)} LayerNorms | sqrt(d_model) = {np.sqrt(d_model):.4f}")
print(f"residual-stream norm, median over tokens 2..: block 0 {residual[0, 1:].median():.1f} -> final {residual[-1, 1:].median():.1f}"
      f" | first token: {residual[0, 0]:.1f} -> {residual[-1, 0]:.1f}")
print(f"after normalising, before the gain: min {pre_gain.min():.4f}, max {pre_gain.max():.4f} over every token and layer")
print(f"after the gain and bias: norms range from {post_gain.min():.1f} to {post_gain.max():.1f}")
x = torch.randn(d_model) * 3 + 1
print(f"RMSNorm check on a random vector: ||x / RMS(x)|| = {(x / x.pow(2).mean().sqrt()).norm():.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
layers = np.arange(len(layer_norms))
labels = [str(i) for i in range(12)] + ["final"]
ax1.plot(layers, residual[:, 1:].median(1).values, color=SLATE, lw=2.5, marker="o", label="median token")
ax1.plot(layers, residual[:, 0], color=CORAL, lw=1.5, marker="s", label="first token")
ax1.set_yscale("log"); ax1.set_xticks(layers, labels); ax1.set_xlabel("block")
ax1.set_ylabel(r"$\|x\|$ entering LayerNorm"); ax1.legend(); ax1.set_title("The residual stream grows with depth")
ax2.axhline(np.sqrt(d_model), color=TEAL, lw=2.5, label=r"before the gain: $\sqrt{768} = 27.71$")
ax2.plot(layers, post_gain[:, 1:].median(1).values, color="k", lw=2, marker="o", label="after the gain, median token")
ax2.fill_between(layers, post_gain[:, 1:].min(1).values, post_gain[:, 1:].max(1).values, color="k", alpha=0.12,
                 label="after the gain, all tokens")
ax2.set_yscale("log"); ax2.set_xticks(layers, labels); ax2.set_xlabel("LayerNorm in front of block")
ax2.set_ylabel(r"$\|\mathrm{LN}(x)\|$"); ax2.legend(fontsize=8); ax2.set_title("Every block starts on the same sphere")
plt.tight_layout(); plt.show()
```
```text
24 tokens, 13 LayerNorms | sqrt(d_model) = 27.7128
residual-stream norm, median over tokens 2..: block 0 5.0 -> final 408.5 | first token: 10.5 -> 368.9
after normalising, before the gain: min 27.7076, max 27.7128 over every token and layer
after the gain and bias: norms range from 2.0 to 324.4
RMSNorm check on a random vector: ||x / RMS(x)|| = 27.7128
```
![Output of cell 8]({{ '/assets/blog/high-dimensional-soap-bubble/cell-08-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: left, how long each token's vector is as it enters each of GPT-2's normalisation steps (log scale). Typical tokens (grey) grow through the network; the first token (coral) jumps far above them and drops back at the final step. Right, the same vectors after normalisation. Before the learned scaling every one has exactly the length √768 = 27.71 (teal line), and after it the lengths spread out. Takeaway: every block starts from the same sphere, whatever the vectors did before.*

The residual stream grows with depth: the median token's norm goes from 5.0 entering block 0 to 408.5 entering the final LayerNorm. The first token is an outlier, around 3,000 through the middle blocks in the left panel, which matches what [Sun et al. (2024)](https://arxiv.org/abs/2402.17762) report for GPT-2: "massive activations" on the starting token that behave like fixed bias terms and pull attention toward it. Before the gain, all 312 normalised vectors (24 tokens at 13 LayerNorms) have norm between 27.7076 and 27.7128, against $$\sqrt{768} = 27.7128$$; the tiny shortfall is LayerNorm's $$\epsilon$$ in the denominator. After the gain and bias, norms range from 2.0 to 324.4, with the learned gains shrinking the sphere inside the blocks and stretching it at the final LayerNorm. The RMSNorm check on a random vector also lands on 27.7128.

[nGPT (Loshchilov et al., 2024)](https://arxiv.org/abs/2410.01131) goes further. It normalises every embedding and hidden state to unit length and trains the whole transformer on the hypersphere. Model-merging tools such as [mergekit](https://github.com/arcee-ai/mergekit) offer SLERP between two models' weights, the same interpolation applied to parameters. None of this makes the residual stream Gaussian; it shares the radius with this post's bubble, and nothing more.

</details>

## Recent work

The geometry in this post is decades old. Taking it seriously inside modern generators is recent.

[Samuel et al. (2023)](https://arxiv.org/abs/2306.08687), at NeurIPS 2023, start from the observation behind cell 7: text-to-image diffusion models only see starting noise in a narrow band of norms. They show that interpolating or averaging seeds with Euclidean or plain spherical operations works poorly, and propose a norm-aware interpolation, with matching centroids, that defines a new metric on the seed space. Picking seeds with it improves generation of rare concepts on few-shot and long-tail benchmarks.

[Bodin et al. (2024)](https://arxiv.org/abs/2408.08558), at ICLR 2025, generalise the cos/sin fix. Their Latent Optimal Linear combinations (LOL) form any linear combination of latents, whether an interpolation, a centroid or a whole low-dimensional subspace, so that the result still matches the latent distribution the generative model assumes. The method works across diffusion, flow-matching and continuous normalising flow models without changing the network.

On the transformer side, [Riechers (2024)](https://arxiv.org/abs/2405.04134) gives the full geometry behind the bonus box: LayerNorm's outputs lie in an $$(N-1)$$-dimensional hyperplane inside a hyperellipsoid set by the gain, with typical inputs mapped near its surface. [nGPT](https://arxiv.org/abs/2410.01131) makes the sphere the architecture's working space rather than a side effect of normalisation.

## The wrong ruler

In high dimensions a Gaussian is a soap bubble. Its samples sit on a thin shell of radius $$\sqrt{d}$$, and the centre, the densest point, holds essentially none of them. A straight line between two samples cuts through that hollow core, dropping to $$0.707\sqrt{d}$$ at its midpoint and climbing $$d/4$$ nats in likelihood on the way. How much a decoder minds depends on how many dimensions it listens to: a VAE that has switched off 49 of its 64 barely notices, and SD-Turbo, which listens to all 16,384, turns the midpoint into static.

SLERP fixes where we walk, keeping every point on the shell and distributed like the samples the generator trained on. It doesn't fix how we measure the walk. Go back to cell 6: SLERP moved through latent space at constant angular speed, and the decoded blob crawled at 0.08 pixels per step in one place and raced at 0.61 in another. Equal steps in $$z$$ aren't equal steps in the data. Every distance in this series so far, from the length of a walk to the angle between two seeds, has been measured in the latent space's own coordinates, as if they were the data's. We've been measuring distance with the wrong ruler. **Part 4** builds the right one from the decoder itself.

## Further reading

- [Part 1](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/what-is-latent-space/) and [Part 2](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/ghosts-and-hybrids/) of this series: latent spaces as learned charts, then holes, noise balls and ELBO surgery in two dimensions.
- [Blum, Hopcroft & Kannan, *Foundations of Data Science*, ch. 2](https://www.cs.cornell.edu/jeh/book.pdf): high-dimensional geometry from scratch, with the Gaussian Annulus Theorem and the orange-peel argument proved properly.
- [Vershynin, *High-Dimensional Probability*](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-1.pdf): the general theory of concentration behind the shell and the nearly perpendicular seeds.
- [White, 2016](https://arxiv.org/abs/1609.04468): the paper that brought SLERP to generative models; short, and full of practical sampling advice.
- [Song, Meng & Ermon, 2020](https://arxiv.org/abs/2010.02502): DDIM, the deterministic sampler that turns noise seeds into a map you can interpolate on.
- [Nalisnick et al., 2019](https://arxiv.org/abs/1906.02994): likelihood against typicality, turned into a practical test for out-of-distribution inputs.
