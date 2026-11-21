---
layout: post
title: "Averaging pixels makes a ghost. Averaging latents makes a hybrid, until it doesn't."
date: 2026-11-21 18:00:00+0530
description: "Why averaging pixels leaves the data, why an autoencoder's latent space has holes, and how a VAE's noise balls and KL term fill them, with the ELBO derived and every claim checked in code."
tags: latent-space vae autoencoders interpolation elbo
categories: machine-learning
related_posts: false
toc:
  sidebar: left
---

<style>
  /* wide display equations scroll inside their own box instead of widening the page on phones */
  mjx-container[display="true"] { display: block; overflow-x: auto; overflow-y: hidden; max-width: 100%; }
</style>

Walking through latent space is one of the most familiar demos in generative modelling. You encode two examples, draw a straight line between their codes, decode every point along it, and watch one image turn smoothly into the other. When it works, the model looks as if it has understood the data. When it doesn't, the frames in the middle come out wrong, sometimes as mush and sometimes as something stranger, and nothing about the model tells you in advance which you'll get.

This is Part 2 of a four-part series on the geometry of latent spaces. [Part 1](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/what-is-latent-space/) described a latent space as a learned chart of the data and ended on a warning: the chart is only ever checked on the data itself. An interpolation spends almost all of its time between the data points, so this post is about that in-between space. It looks at what a decoder does there, why a plain autoencoder leaves it full of holes, and how a variational autoencoder fills them.

As in Part 1, each idea comes four ways: an intuition, a picture of the geometry, the maths, and a short code cell that checks the maths on real numbers. Everything runs on a laptop CPU in about five minutes, with MNIST as the only download. You'll want Part 1's picture of latent spaces as charts and the definition of KL divergence; the rest is built here.

## Ghosts and hybrids

Take two Blob World images, one with the blob near the top-left corner and one with it near the bottom-right, and average them pixel by pixel. You get two blobs at half brightness. No Blob World image looks like that, since every one of them has a single blob at full brightness 1. Call it a ghost. Now encode both images with the autoencoder from Part 1, average the two codes, and decode the result. You get one full-brightness blob, halfway across the image. Call that a hybrid: a new, valid image between the two you started with.

That's the trick latent spaces are famous for, and it doesn't always work. Pick the wrong pair of codes for a 2 and a 7 in an autoencoder trained on MNIST, walk between them, and the frames in the middle show a crisp 1, a digit neither end asked for, decoded from codes no training image has and drawn as confidently as any real digit.

This post explains both outcomes in three steps. First, why averaging pixels always leaves the data, while averaging codes leaves it only where the latent space has holes. Second, how a variational autoencoder fills those holes; the explanation turns out to be about scale, and the reparameterisation trick has little to do with it. Third, what the filling costs, and why it stays incomplete in practice.

## Where Part 1 left us

[Part 1](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/what-is-latent-space/) argued that a latent space is a learned coordinate chart for the low-dimensional manifold $$\mathcal{M}$$ the data lives on. An encoder $$e$$ assigns coordinates $$z = e(x)$$, a decoder $$g$$ maps them back, and if reconstruction is perfect on the data, $$g(e(x)) = x$$ for every $$x \in \mathcal{M}$$, then the encoder is forced to be a chart with $$g$$ as its inverse. On Blob World the autoencoder's two code numbers recovered each blob's true position with a held-out $$R^2$$ of 1.000.

The argument has a gap, and Part 1 ended by pointing at it. Everything is checked on the data and nowhere else, so the decoder's behaviour at a code that no training image maps to isn't constrained by anything we've written down. A straight walk between two codes visits exactly those codes. The colours carry over from Part 1: teal is data and the codes of training images, coral is the void.

## The ghost: averaging pixels is a chord

Averaging two images is a cross-fade. Each pixel moves halfway from one value to the other, so the first blob fades out where it is while the second fades in where it is. Nothing moves. A real blob sliding from one corner to the other would have to pass through the middle of the image, and the cross-fade never puts any brightness there.

Geometrically, the cross-fade is a straight line in pixel space, the chord

$$
x(t) = (1 - t)\,x_0 + t\,x_1, \qquad t \in [0, 1].
$$

The data manifold isn't straight, and Part 1's distance formula says how it bends. Two blobs more than about $$3s$$ apart are all nearly the same distance apart, close to $$\sqrt{2\pi s^2}$$, however far apart they sit in the image, and every blob image away from the border has the same length, $$\lVert I_c \rVert^2 = \pi s^2$$. Vectors that all have the same length and are all that far apart are nearly orthogonal points on a sphere of radius $$\sqrt{\pi}\,s$$. So Blob World curls around that sphere, turning through close to 90° between any two distant blobs, and the chord between two of them cuts through the inside. Its midpoint has length $$\sqrt{\pi s^2/2}$$, a factor of $$1/\sqrt{2}$$ short of the sphere. Hold on to that $$1/\sqrt{2}$$; it comes back in Part 3 with Gaussian noise in place of blobs.

![A 2-D cartoon of pixel space: the straight line between two real images cuts inside the curved teal arc of real images, reaching 1/sqrt(2) of its radius at the midpoint, while panels on the right show the pixel average fading between two blobs and the latent average sliding one blob across]({{ '/assets/blog/ghosts-and-hybrids/chord-leaves-manifold.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the animation shows: a flat cartoon of pixel space. The teal curve stands for all the real images, which lie along a bend rather than a straight line. Averaging pixels moves you along the straight coral line between two real images. That line cuts inside the curve, away from every real image, and halfway along you get the two-dim-blobs ghost (top right). Averaging codes moves you along the curve itself, so every step is a real image and you see one blob sliding (bottom right).*

The maths pins the ghost down with two numbers: how bright it is, and how far it sits from the nearest real image. At the centre of blob $$a$$, the midpoint image $$\tfrac12(I_a + I_b)$$ has brightness

$$
\tfrac12\left(1 + e^{-\delta^2/2s^2}\right),
$$

because blob $$b$$, $$\delta$$ pixels away, still contributes $$e^{-\delta^2/2s^2}$$ there, a term that vanishes once $$\delta \gtrsim 3s$$ and leaves a peak of ½. For the distance, subtract an endpoint: $$x(t) - I_a = t\,(I_b - I_a)$$, so

$$
\big\| x(t) - I_a \big\|^2 = t^2\,\big\| I_a - I_b \big\|^2 = t^2 \cdot 2\pi s^2\left(1 - e^{-\delta^2/4s^2}\right),
$$

using Part 1's formula for the last step. That's the distance to one particular real image. For blobs this far apart it's also the distance to the nearest one, since moving the blob somewhere else can't help when the ghost's brightness is split between two places and every real image has all of it in one. So the squared distance from the chord to $$\mathcal{M}$$ is $$\min(t, 1 - t)^2\,\lVert I_a - I_b \rVert^2$$, which peaks at the midpoint at

$$
\tfrac14 \cdot 2\pi s^2 = \frac{\pi s^2}{2} \approx 6.28 \qquad (s = 2).
$$

The midpoint of the chord is $$\sqrt{\pi s^2/2}$$ from the nearest real image (and further from all the others), exactly half of $$\sqrt{2\pi s^2}$$, the distance between two unrelated real images. The cell checks the "nearest one" step on a dense grid of real blobs rather than trusting the argument.

The setup cell copies Part 1's `AE` class and `train_ae` loop unchanged and adds what this post needs: Holey Blob World (used further down), the first 10,000 MNIST training images (the same subset as Part 1), and a `VAE` class with two switches the ablation in cell 5 uses. I'll explain those when we get there.

```python
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
```
```text
numpy 2.3.3 | torch 2.13.0+cpu | torchvision 0.28.0+cpu
MNIST subset: (10000, 784), pixels in [0, 1]
```

Cell 2 trains `AE(d=2)` on 5,000 Blob World images and walks between blobs centred at (6, 6) and (22, 22), which are $$\delta = 22.6$$ pixels apart, far past $$3s$$. It walks twice: along the pixel chord, which gives the ghost, and along the straight line between the two codes, which gives the hybrid. To measure distance from $$\mathcal{M}$$ it doesn't use the training images. It renders a real blob for every centre on a quarter-pixel grid, 9,409 images in all, so "nearest real image" means the nearest point on the true manifold. Watch the middle panel, where the ghost's measured distance should sit on the dashed formula.

```python
X_blob, C_blob = blob_world(5000, np.random.default_rng(SEED))
ae_blob, Z_blob, mse_blob = train_ae(flat(X_blob), d=2)

# The true manifold, densely sampled: a real blob image for every centre on a 0.25-pixel grid
grid_c = np.stack(np.meshgrid(np.arange(2, 26.01, 0.25), np.arange(2, 26.01, 0.25)), -1).reshape(-1, 2)
manifold_index = NearestNeighbors(n_neighbors=1).fit(flat(np.stack([blob(cx, cy) for cx, cy in grid_c])))

def dist2_to_manifold(frames: np.ndarray) -> np.ndarray:
    """Squared pixel distance from each frame to the nearest real Blob World image."""
    return manifold_index.kneighbors(flat(frames))[0][:, 0] ** 2

def centre_of_mass(frames: np.ndarray) -> np.ndarray:
    """Intensity-weighted centre (x, y) of each frame: where the blob is."""
    w = np.clip(frames, 0, None)
    return np.column_stack([(w * xs).sum((1, 2)), (w * ys).sum((1, 2))]) / w.sum((1, 2))[:, None]

s, a, b = 2.0, (6.0, 6.0), (22.0, 22.0)  # opposite corners: delta = 22.6 pixels, far beyond 3s
delta = np.hypot(b[0] - a[0], b[1] - a[1])
t = np.linspace(0, 1, 101)
ghost = (1 - t)[:, None, None] * blob(*a) + t[:, None, None] * blob(*b)  # straight chord in pixel space
with torch.no_grad():
    z_a, z_b = ae_blob.enc(torch.from_numpy(flat(np.stack([blob(*a), blob(*b)]))))
    T = torch.from_numpy(t.astype(np.float32))[:, None]
    hybrid = ae_blob.dec((1 - T) * z_a + T * z_b).numpy().reshape(-1, H, W)  # straight line in the latent

d2_ghost, d2_hybrid = dist2_to_manifold(ghost), dist2_to_manifold(hybrid)
d2_formula = np.minimum(t, 1 - t) ** 2 * 2 * np.pi * s**2 * (1 - np.exp(-delta**2 / (4 * s**2)))
com = centre_of_mass(hybrid)
speed = np.linalg.norm(np.diff(com, axis=0), axis=1) / np.diff(t)
print(f"AE(d=2) on Blob World: recon MSE {mse_blob:.5f}")
print(f"ghost  (pixel average): peak at t=0.5 = {ghost[50].max():.3f} | d^2 to nearest real image = {d2_ghost[50]:.3f}"
      f" (pi s^2 / 2 = {np.pi * s**2 / 2:.3f})")
print(f"hybrid (latent average): peak at t=0.5 = {hybrid[50].max():.3f} | max d^2 along the path = {d2_hybrid.max():.3f}")
print(f"hybrid blob centre: t=0 {com[0].round(2)}, t=0.5 {com[50].round(2)}, t=1 {com[-1].round(2)}")
print(f"hybrid blob speed (pixels per unit t): min {speed.min():.1f}, max {speed.max():.1f}, "
      f"straight-line constant speed would be {delta:.1f}")

fig = plt.figure(figsize=(13, 7.4))
gs = fig.add_gridspec(3, 9, height_ratios=[1, 1, 2.4], hspace=0.45)
show = np.linspace(0, 100, 9).astype(int)
for j, k in enumerate(show):
    for row, frames, name, color in [(0, ghost, "pixel average: ghost", CORAL), (1, hybrid, "latent average: hybrid", TEAL)]:
        ax = fig.add_subplot(gs[row, j])
        ax.imshow(frames[k], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"t={t[k]:.2f}", fontsize=8)
        if j == 0:
            ax.set_title(name, loc="left", fontsize=10, color=color)
ax1, ax2, ax3 = fig.add_subplot(gs[2, 0:3]), fig.add_subplot(gs[2, 3:6]), fig.add_subplot(gs[2, 6:9])
ax1.plot(t, ghost.max((1, 2)), color=CORAL, lw=2.5, label="ghost")
ax1.plot(t, hybrid.max((1, 2)), color=TEAL, lw=2.5, label="hybrid")
ax1.set_xlabel("t"); ax1.set_ylabel("peak intensity"); ax1.set_ylim(0, 1.15); ax1.legend()
ax1.set_title("Every real image has peak 1")
ax2.plot(t, d2_ghost, color=CORAL, lw=2.5, label="ghost (measured)")
ax2.plot(t, d2_formula, "k--", lw=1.2, label=r"$\min(t,1-t)^2\,\|I_a-I_b\|^2$")
ax2.plot(t, d2_hybrid, color=TEAL, lw=2.5, label="hybrid")
ax2.axhline(np.pi * s**2 / 2, color=SLATE, ls=":", lw=1)
ax2.text(0.02, np.pi * s**2 / 2 + 0.2, r"$\pi s^2/2$", color=SLATE)
ax2.set_xlabel("t"); ax2.set_ylabel(r"$d^2$ to nearest real image"); ax2.legend(fontsize=8, loc="center right")
ax2.set_title("Distance from the manifold")
ax3.plot([a[0], b[0]], [a[1], b[1]], color=SLATE, ls="--", lw=1, label="straight, constant speed")
ax3.plot(com[:, 0], com[:, 1], color=TEAL, lw=1.5)
ax3.plot(com[::5, 0], com[::5, 1], "o", color=TEAL, ms=4, label="hybrid centre, every 0.05 in t")
ax3.set_xlim(0, 27); ax3.set_ylim(27, 0); ax3.set_aspect("equal")
ax3.set_xlabel("x (pixels)"); ax3.set_ylabel("y (pixels)"); ax3.legend(fontsize=8, loc="lower left")
ax3.set_title("Where the hybrid's blob is")
plt.show()
```
```text
AE(d=2) on Blob World: recon MSE 0.00039
ghost  (pixel average): peak at t=0.5 = 0.500 | d^2 to nearest real image = 6.283 (pi s^2 / 2 = 6.283)
hybrid (latent average): peak at t=0.5 = 0.825 | max d^2 along the path = 0.594
hybrid blob centre: t=0 [7.6  7.43], t=0.5 [13.71 14.81], t=1 [20.45 20.45]
hybrid blob speed (pixels per unit t): min 4.8, max 70.0, straight-line constant speed would be 22.6
```
![Output of cell 2]({{ '/assets/blog/ghosts-and-hybrids/cell-02-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: two ways to get from a blob in the top-left to a blob in the bottom-right. Top row: averaging the pixels. The first blob fades out while the second fades in, and halfway there are two dim blobs, which no real image looks like. Second row: averaging the codes and decoding. One bright blob slides across. Bottom left: the brightness of the brightest pixel. Real images always reach 1; the pixel average sinks to 0.5. Bottom middle: how far each frame is from the closest real image. The pixel average drifts far away and comes back, while the code average stays close the whole time. Bottom right: where the sliding blob actually is at each step. The dots bunch up at the ends and spread out in the middle, so the blob speeds up halfway across.*

The ghost does exactly what the maths says. At $$t = 0.5$$ its peak is 0.500 and its squared distance to the nearest real image is 6.283, matching $$\pi s^2/2 = 6.283$$ to three decimals, and the measured curve lies on the dashed $$\min(t, 1-t)^2\,\lVert I_a - I_b \rVert^2$$ for the whole walk. The hybrid never gets further than 0.594 from the manifold, under a tenth of the ghost's worst case. It isn't a perfect renderer: its peak at the midpoint is 0.825, not 1, and its endpoints are the autoencoder's reconstructions, centred at (7.6, 7.43) and (20.45, 20.45) instead of (6, 6) and (22, 22), because this autoencoder reconstructs blobs near the corners of its training square slightly toward the middle. But every frame shows a single blob, and that's the whole difference between a hybrid and a ghost.

The bottom-right panel shows one more thing. The blob does travel the diagonal, but not at a steady pace. A blob moving at constant speed would cover 22.6 pixels per unit of $$t$$, and this one moves as slowly as 4.8 and as fast as 70.0, lingering near the corners and rushing through the middle. Equal steps in $$z$$ are not equal steps in the image, which is the problem Part 4 takes up.

## The hybrid, and when it fails

Blob World is kind to autoencoders: two knobs, a smooth renderer, and a code space the network can fill. MNIST is the harder test. Cell 3 trains the same `AE(d=2)` on 10,000 digits and asks how often a straight walk between two of them leaves the codes behind.

That needs a definition of "leaves". For each training code, take the distance to its nearest neighbouring code; 99% of codes have a neighbour closer than a threshold $$\tau$$. A point further than $$\tau$$ from every code is in a gap the training set left open. All distances are divided by the RMS radius of the code cloud, so the same test can later be applied to a model whose latent has a completely different scale.

Before showing any particular walk, the cell prints a base rate: out of 500 walks between random pairs of different digits, how many cross a gap wider than $$\tau$$? Then it shows two walks. Walk 1 runs between the two digits whose code centroids are furthest apart, each represented by its most central image. Walk 2 goes looking for trouble. It starts at the tip of one digit's arm of codes and ends at the tip of another's, picking the pair of tips whose straight line crosses the most empty space.

```python
ae_mnist, Z_ae, mse_ae = train_ae(X_mnist, d=2)

def hole_maps(encode, decode, Z: np.ndarray, lim: np.ndarray, n: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-training-code distance and cycle error ||e(g(z)) - z|| on an n x n latent grid.

    Both are divided by the code cloud's RMS radius, so models with different latent scales share one colour scale.
    """
    scale = np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean())
    g1, g2 = np.meshgrid(np.linspace(*lim[0], n), np.linspace(*lim[1], n))
    grid = np.column_stack([g1.ravel(), g2.ravel()]).astype(np.float32)
    nearest = NearestNeighbors(n_neighbors=1).fit(Z).kneighbors(grid)[0][:, 0]
    with torch.no_grad():
        G = torch.from_numpy(grid)
        cycle = (encode(decode(G)) - G).norm(dim=1).numpy()
    return nearest.reshape(n, n) / scale, cycle.reshape(n, n) / scale

def hole_threshold(Z: np.ndarray) -> float:
    """99th percentile of each code's distance to its nearest other code, in RMS-radius units."""
    scale = np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean())
    return np.quantile(NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1], 0.99) / scale

def padded_box(Z: np.ndarray, pad: float = 0.1) -> np.ndarray:
    lo, hi = Z.min(0), Z.max(0)
    return np.stack([lo - pad * (hi - lo), hi + pad * (hi - lo)], axis=1)

def walk(Z: np.ndarray, i: int, j: int, n: int = 10) -> np.ndarray:
    """n evenly spaced points on the straight latent line from code i to code j."""
    tt = np.linspace(0, 1, n)[:, None]
    return ((1 - tt) * Z[i] + tt * Z[j]).astype(np.float32)

def gaps_along(Z: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Distance from each point of a path to the nearest training code, in RMS-radius units."""
    scale = np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean())
    return NearestNeighbors(n_neighbors=1).fit(Z).kneighbors(path)[0][:, 0] / scale

scale_ae, tau_ae = np.sqrt(((Z_ae - Z_ae.mean(0)) ** 2).sum(1).mean()), hole_threshold(Z_ae)
box_ae = padded_box(Z_ae)
near_ae, cycle_ae = hole_maps(ae_mnist.enc, ae_mnist.dec, Z_ae, box_ae)

# Base rate: how often does a walk between two random images of different digits cross a gap wider than tau?
rng = np.random.default_rng(7)
pairs = [(i, j) for i, j in rng.integers(0, len(X_mnist), (600, 2)) if y_mnist[i] != y_mnist[j]][:500]
crosses = lambda Z, tau: np.mean([gaps_along(Z, walk(Z, i, j, 41)).max() > tau for i, j in pairs])

# Walk 1: the two classes whose code centroids are farthest apart, each represented by its most central image
centroids = np.stack([Z_ae[y_mnist == k].mean(0) for k in range(10)])
c0, c1 = np.unravel_index(np.linalg.norm(centroids[:, None] - centroids[None], axis=-1).argmax(), (10, 10))
i0, i1 = [np.flatnonzero(y_mnist == k)[np.linalg.norm(Z_ae[y_mnist == k] - centroids[k], axis=1).argmin()] for k in (c0, c1)]
# Walk 2: between the tips of two arms (each digit's code at its 99th-percentile radius), the pair crossing the most void
radius = np.linalg.norm(Z_ae - Z_ae.mean(0), axis=1)
tips = [np.flatnonzero(y_mnist == k)[np.argsort(radius[y_mnist == k])[int(0.99 * (y_mnist == k).sum())]] for k in range(10)]
j0, j1 = max(((p, q) for p in tips for q in tips if p < q), key=lambda pq: gaps_along(Z_ae, walk(Z_ae, *pq, 41)).max())
walks = {"walk 1": (i0, i1), "walk 2": (j0, j1)}

with torch.no_grad():
    frames_ae = {w: ae_mnist.dec(torch.from_numpy(walk(Z_ae, *ij))).numpy().reshape(-1, H, W) for w, ij in walks.items()}
gaps_ae = {w: gaps_along(Z_ae, walk(Z_ae, *ij)) for w, ij in walks.items()}
print(f"AE(d=2) on MNIST: recon MSE {mse_ae:.4f} | code cloud RMS radius {scale_ae:.2f}")
print(f"hole threshold tau = {tau_ae:.3f} RMS radii (99% of codes have a neighbour closer than this)")
print(f"latent bounding box farther than tau from every code: {(near_ae > tau_ae).mean():.0%}")
print(f"random walks between different digits that cross a gap wider than tau: {crosses(Z_ae, tau_ae):.0%} of {len(pairs)}")
for w, (i, j) in walks.items():
    print(f"{w}: digit {y_mnist[i]} -> digit {y_mnist[j]} | frames past tau {(gaps_ae[w] > tau_ae).sum()}/10 "
          f"| widest gap {gaps_ae[w].max() / tau_ae:.1f} tau")

HOLE_NORM = LogNorm(vmin=1e-3, vmax=2.0)  # shared with the VAE hole maps in a later cell
STYLE = {"walk 1": dict(color="k", ls="--", lw=1), "walk 2": dict(color="k", ls="-", lw=1.5)}
fig = plt.figure(figsize=(14, 7.6))
gs = fig.add_gridspec(3, 10, height_ratios=[3.2, 1, 1], hspace=0.35)
axs = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6]), fig.add_subplot(gs[0, 6:9])]
sc = axs[0].scatter(Z_ae[:, 0], Z_ae[:, 1], c=y_mnist, cmap="tab10", s=2, alpha=0.6)
axs[0].legend(*sc.legend_elements(), fontsize=6, ncol=2, loc="lower left", markerscale=0.7)
axs[0].set_title("AE codes by digit class")
for ax, M, title in [(axs[1], near_ae, "nearest-code distance"), (axs[2], cycle_ae, r"cycle error $\|e(g(z))-z\|$")]:
    im = ax.imshow(M, origin="lower", extent=box_ae.ravel(), aspect="auto", cmap=HOLE_CMAP, norm=HOLE_NORM)
    ax.scatter(Z_ae[:, 0], Z_ae[:, 1], s=0.3, color=TEAL, alpha=0.25)
    ax.set_title(title + " / RMS radius")
fig.colorbar(im, cax=fig.add_subplot(gs[0, 9]))
for ax in axs:
    for w, ij in walks.items():
        path = walk(Z_ae, *ij)
        ax.plot(path[:, 0], path[:, 1], marker="o", ms=3, label=w, **STYLE[w])
    ax.set_xlim(box_ae[0]); ax.set_ylim(box_ae[1]); ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
axs[1].legend(fontsize=8, loc="lower left")
for row, w in enumerate(walks, start=1):
    for j in range(10):
        ax = fig.add_subplot(gs[row, j])
        ax.imshow(np.clip(frames_ae[w][j], 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(CORAL if gaps_ae[w][j] > tau_ae else TEAL); spine.set_linewidth(3)
        if j == 0:
            ax.set_ylabel(w, fontsize=10)
plt.show()
```
```text
AE(d=2) on MNIST: recon MSE 0.0376 | code cloud RMS radius 7.67
hole threshold tau = 0.114 RMS radii (99% of codes have a neighbour closer than this)
latent bounding box farther than tau from every code: 69%
random walks between different digits that cross a gap wider than tau: 4% of 500
walk 1: digit 0 -> digit 7 | frames past tau 0/10 | widest gap 0.5 tau
walk 2: digit 2 -> digit 7 | frames past tau 6/10 | widest gap 1.6 tau
```
![Output of cell 3]({{ '/assets/blog/ghosts-and-hybrids/cell-03-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: on the left, each dot is one MNIST image placed at its two code numbers and coloured by which digit it is, and the two black lines are the two walks. The middle and right panels are maps of where the code space is empty: pale means close to real training codes, dark red means far from anything the decoder was trained on. The bottom rows are the images the decoder produces at ten evenly spaced stops along each walk. A teal border means that stop is near real codes, and a coral border means it sits in an empty gap. Walk 1 stays teal and passes through other digits; walk 2 crosses a gap and shows a sharp 1 that neither end asked for.*

The base rate is low. Only 4% of the 500 random walks cross a gap wider than $$\tau$$, and walk 1, from a 0 to a 7 across the whole cloud, never gets further than $$0.5\,\tau$$ from a code. It doesn't turn to mush either. It passes through a 6 and a 9 on the way, one of them a faint 6 at $$t = 0.56$$, because the line runs through other digits' territory. That's a hybrid of the wrong kind, but every frame is a digit.

Walk 2 is different. Six of its ten frames are past $$\tau$$, with the widest gap at $$1.6\,\tau$$, and the middle of the walk shows a clean, sharp 1. Neither endpoint is a 1. The decoder was never trained at those codes, and what it draws there is a confident, wrong digit. Off-distribution doesn't always look broken, which is arguably worse, since nothing in the image tells you to distrust it.

The heatmaps say where the trouble is. 69% of the latent bounding box is further than $$\tau$$ from every code. That empty space sits around the outside of the cloud and between the arms, away from the crowded middle where most walks run.

## Why the decoder has holes

The decoder was graded only at the training codes. Everywhere else is an exam question nobody marked, and the network's answer there is whatever its architecture happens to produce.

You can see the consequence in cell 3. The codes form arms radiating from a dense centre, roughly one per digit, at a scale nobody chose (RMS radius 7.67). Between and beyond the arms there's empty plane, and the two heatmaps give that emptiness a shape.

In symbols, write the autoencoder loss out:

$$
\mathcal{L}(e, g) = \frac{1}{N} \sum_{i=1}^{N} \big\| x_i - g(z_i) \big\|^2, \qquad z_i = e(x_i).
$$

The decoder enters only through its values at the $$N$$ points $$z_1, \dots, z_N$$. Two decoders that agree at those points have identical loss however differently they behave between them. The loss can't tell them apart. The scale of the codes is just as arbitrary: replace $$e$$ with $$c\,e$$ and $$g$$ with $$g(\cdot / c)$$ for any $$c > 0$$ and the loss doesn't change. Nothing connects the code cloud to any particular distribution, least of all $$\mathcal{N}(0, I)$$, so sampling $$z \sim \mathcal{N}(0, I)$$ lands wherever that distribution happens to overlap the codes. For this model that's the crowded centre where several digits' arms meet, and cell 5 shows what comes out: faint overlays of several digits at once.

The two heatmaps in cell 3 turn this into something you can check, and they're cheap enough to run on any latent space that has a decoder. The first is the distance from each grid point to the nearest training code. The second is the cycle error $$\lVert e(g(z)) - z \rVert$$: decode $$z$$, encode the result, and see how far you land from where you started. Both are small on the codes. In the holes the nearest-code distance grows by construction, and the cycle error lights up wherever the decoder produces something the encoder doesn't map back to the same place.

Cycle error is a heuristic, not a proof. A decoder can draw a plausible image in a hole that the encoder then maps straight back, and the map will show nothing there.

## A hole with a known answer

Blob World has one more experiment with a known answer. Holey Blob World removes every blob whose centre is within 5 pixels of the image centre (14, 14), so the true manifold is an annulus, a square with a disc cut out of it. A blob that slides from the left of the hole to the right while staying on the manifold has to go around. The straight latent line between the two codes has no such obligation.

Cell 4 trains `AE(d=2)` on 5,000 holey blobs and walks between blobs at (6, 14) and (22, 14), on opposite sides of the hole; a straight slide in the image would pass right through the centre. I couldn't predict what the decoder would do where the latent line crosses the codes' own hole. It might draw a confident blob in the middle of the image, where it has never seen one, or it might produce garbage. To find out, the cell tracks the brightest pixel of every decoded frame and counts how many blobs each frame shows.

```python
X_holey, C_holey = holey_blob_world(5000, np.random.default_rng(SEED))
ae_holey, Z_holey, mse_holey = train_ae(flat(X_holey), d=2)
train_index = NearestNeighbors(n_neighbors=1).fit(flat(X_holey))

def bright_spots(img: np.ndarray, level: float = 0.25) -> int:
    """Number of local maxima above `level`: how many blobs a frame shows."""
    p = np.pad(img, 1, constant_values=-np.inf)
    neighbours = np.stack([p[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    return int(((img > neighbours.max(0)) & (img > level)).sum())

a, b = (6.0, 14.0), (22.0, 14.0)  # opposite sides of the hole; in the image, the straight slide crosses the centre
t = np.linspace(0, 1, 101)
with torch.no_grad():
    z_a, z_b = ae_holey.enc(torch.from_numpy(flat(np.stack([blob(*a), blob(*b)]))))
    T = torch.from_numpy(t.astype(np.float32))[:, None]
    walk_holey = ((1 - T) * z_a + T * z_b).numpy()
    frames_holey = ae_holey.dec(torch.from_numpy(walk_holey)).numpy().reshape(-1, H, W)

brightest = np.array([np.unravel_index(f.argmax(), f.shape)[::-1] for f in frames_holey])  # (x, y) of the peak pixel
d2_holey = dist2_to_manifold(frames_holey)
worst = int(d2_holey.argmax())  # the frame farthest from every real Blob World image
spots = np.array([bright_spots(f) for f in frames_holey])
print(f"AE(d=2) on Holey Blob World: {len(X_holey)} images, no centre within 5 px of (14, 14); recon MSE {mse_holey:.5f}")
print(f"brightest pixel inside the hole (r < 5) in {(np.linalg.norm(brightest - 14, axis=1) < 5).sum()} of {len(t)} frames")
two = t[spots >= 2]
print(f"frames showing two blobs at once: {len(two)} of {len(t)}" + (f", from t={two.min():.2f} to t={two.max():.2f}" if len(two) else ""))
print(f"frame farthest from the manifold: t={t[worst]:.2f}, peak {frames_holey[worst].max():.3f}, {spots[worst]} blobs")
print(f"  d^2 to nearest real Blob World image: {d2_holey[worst]:.3f} (the pixel ghost's midpoint: pi s^2 / 2 = {np.pi * s**2 / 2:.3f})")
print(f"  d^2 to nearest training image       : {train_index.kneighbors(flat(frames_holey[[worst]]))[0][0, 0] ** 2:.3f}")

fig = plt.figure(figsize=(13, 6.4))
gs = fig.add_gridspec(2, 9, height_ratios=[3, 1], hspace=0.3)
ax1, ax2 = fig.add_subplot(gs[0, 0:4]), fig.add_subplot(gs[0, 5:9])
angle = np.arctan2(C_holey[:, 1] - 14, C_holey[:, 0] - 14)
ax1.scatter(Z_holey[:, 0], Z_holey[:, 1], c=angle, cmap="twilight", s=2)
ax1.plot(walk_holey[:, 0], walk_holey[:, 1], "k--", lw=1.2)
ax1.plot(walk_holey[worst, 0], walk_holey[worst, 1], "o", mfc="none", mec=CORAL, ms=12, mew=2)
ax1.set_title("AE codes (colour = true angle around the hole)"); ax1.set_xlabel(r"$z_1$"); ax1.set_ylabel(r"$z_2$")
ax2.scatter(C_holey[:, 0], C_holey[:, 1], s=1, color=TEAL, alpha=0.3)
ax2.add_patch(Circle((14, 14), 5, color=CORAL, alpha=0.25))
ax2.plot(brightest[:, 0], brightest[:, 1], "k.-", lw=0.8, ms=4, label="brightest pixel of each decoded frame")
ax2.plot(*brightest[worst], "o", mfc="none", mec=CORAL, ms=12, mew=2)
ax2.set_xlim(0, 27); ax2.set_ylim(27, 0); ax2.set_aspect("equal"); ax2.legend(fontsize=8, loc="lower left")
ax2.set_title("Image plane: training centres (teal), the hole (coral)")
for j, k in enumerate(np.linspace(0, 100, 9).astype(int)):
    ax = fig.add_subplot(gs[1, j])
    ax.imshow(np.clip(frames_holey[k], 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(f"t={t[k]:.2f}", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(CORAL if spots[k] >= 2 else TEAL); spine.set_linewidth(3)
plt.show()
```
```text
AE(d=2) on Holey Blob World: 5000 images, no centre within 5 px of (14, 14); recon MSE 0.00053
brightest pixel inside the hole (r < 5) in 0 of 101 frames
frames showing two blobs at once: 21 of 101, from t=0.53 to t=0.73
frame farthest from the manifold: t=0.55, peak 0.552, 3 blobs
  d^2 to nearest real Blob World image: 7.339 (the pixel ghost's midpoint: pi s^2 / 2 = 6.283)
  d^2 to nearest training image       : 7.334
```
![Output of cell 4]({{ '/assets/blog/ghosts-and-hybrids/cell-04-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: on the left is the code of every training image, coloured by where its blob sits around the hole. The codes form a ring with an empty cross in the middle, and the dashed line, the walk, cuts straight through it. On the right is the same walk seen in the image: teal dots are where training blobs sat, the pink disc is the forbidden middle, and the black line follows the brightest spot of each decoded frame. It goes around the top rather than through the hole, then jumps across, and the jump is where the decoder breaks. The bottom strip shows nine frames along the walk; the coral-bordered one shows two blobs at once, a ghost.*

Neither guess was right. The decoder never puts the blob inside the hole: the brightest pixel lands inside the coral disc in 0 of 101 frames. Instead the blob climbs the left side of the image toward the top-left corner, the brightest pixel then jumps to the top-right corner, and the blob comes down the right side. The jump is a tear. For 21 of the 101 frames, from $$t = 0.53$$ to $$t = 0.73$$, the decoded image shows two blobs at once, and the worst frame, at $$t = 0.55$$, shows three faint blobs with a peak of 0.552. Its squared distance to the nearest real Blob World image is 7.339, further out than the pixel ghost's 6.283.

So the latent hole decodes to a ghost, the very thing latent interpolation was supposed to avoid. The line passes through codes no training image has, and the decoder, with no instruction for that region, blends the images on either side of it. Part 4 comes back to paths like this one.

## The VAE, derived

Fixing the holes takes two things the autoencoder lacks: a reason for the decoder to care about codes between the training points, and a fixed shape for the code cloud, so that sampling has somewhere definite to sample from. The variational autoencoder ([Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114); [Rezende et al., 2014](https://arxiv.org/abs/1401.4082)) gets both from one objective.

Start from a generative model: draw $$z \sim p(z) = \mathcal{N}(0, I)$$, then $$x \sim p_\theta(x \mid z)$$, with the decoder $$g_\theta(z)$$ supplying the parameters of that distribution. We'd like to maximise $$\log p_\theta(x)$$, but that needs the posterior $$p_\theta(z \mid x)$$, which we can't compute. So an encoder $$q_\phi(z \mid x) = \mathcal{N}\big(\mu_\phi(x), \operatorname{diag}\sigma_\phi^2(x)\big)$$ stands in for it. For any such $$q$$,

$$
\log p(x) = \mathbb{E}_{q}\!\left[\log \frac{p(x, z)}{p(z \mid x)}\right]
= \underbrace{\mathbb{E}_{q}\!\left[\log \frac{p(x, z)}{q(z \mid x)}\right]}_{\text{ELBO}}
+ \underbrace{\mathbb{E}_{q}\!\left[\log \frac{q(z \mid x)}{p(z \mid x)}\right]}_{\mathrm{KL}\left(q(z \mid x)\,\|\,p(z \mid x)\right)\;\ge\;0}.
$$

The first equality holds because $$\log p(x)$$ doesn't depend on $$z$$, so averaging it over $$q$$ changes nothing, and $$p(x) = p(x, z)/p(z \mid x)$$. The second multiplies and divides by $$q$$. The ELBO is a lower bound on $$\log p(x)$$, and the gap is a KL divergence to the true posterior, so pushing the ELBO up either raises the likelihood or makes the encoder a better posterior. The usual derivation applies Jensen's inequality to $$\log \mathbb{E}_q[p(x, z)/q(z \mid x)]$$; it gets the same bound but hides what the gap is.

Split $$p(x, z) = p(x \mid z)\,p(z)$$ and the ELBO becomes the familiar two terms:

$$
\mathrm{ELBO}(x) = \underbrace{\mathbb{E}_{q(z \mid x)}\big[\log p(x \mid z)\big]}_{\text{reconstruction}} - \underbrace{\mathrm{KL}\big(q(z \mid x)\,\|\,p(z)\big)}_{\text{prior term}}.
$$

For a diagonal Gaussian against $$\mathcal{N}(0, I)$$ the prior term has a closed form. In each dimension $$\log q - \log p = -\tfrac12 \log \sigma^2 - \tfrac{(z - \mu)^2}{2\sigma^2} + \tfrac{z^2}{2}$$, and averaging over $$z \sim \mathcal{N}(\mu, \sigma^2)$$ gives $$-\tfrac12\log\sigma^2 - \tfrac12 + \tfrac12(\mu^2 + \sigma^2)$$. Summed over the $$d$$ dimensions,

$$
\mathrm{KL}\big(q(z \mid x)\,\|\,\mathcal{N}(0, I)\big) = \frac12 \sum_{j=1}^{d} \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right).
$$

Read it term by term, because the next section leans on it. $$\mu_j^2$$ is a spring pulling every code toward the origin. $$\sigma_j^2 - \log\sigma_j^2 - 1$$ is zero at $$\sigma_j = 1$$ and grows on both sides of it, so it holds each code's noise near unit size: shrinking $$\sigma$$ costs $$-\log \sigma^2$$, and growing it costs $$\sigma^2$$.

The reconstruction term needs a likelihood. For MNIST I use a Bernoulli on each pixel,

$$
\log p(x \mid z) = \sum_j x_j \log \hat{x}_j + (1 - x_j) \log(1 - \hat{x}_j),
$$

with $$\hat{x} = \operatorname{sigmoid}(g(z))$$. That's minus the binary cross-entropy, which is what `binary_cross_entropy_with_logits` computes in `vae_loss`. MNIST pixels are grey levels in [0, 1] rather than bits, so strictly this needs a normalising constant, the continuous Bernoulli of [Loaiza-Ganem & Cunningham (2019)](https://arxiv.org/abs/1907.06845). Leaving it out is standard practice. It changes the reconstruction term, and through it the trained model a little, but none of the arguments below depend on it, and the KL numbers don't involve the likelihood at all. `vae_loss` returns minus the ELBO with a weight $$\beta$$ on the KL, and `beta=1` is the true ELBO.

> **The reparameterisation trick is a gradient estimator, not a regulariser.** To train through the expectation over $$q$$, the code samples $$z = \mu + \sigma \odot \epsilon$$ with $$\epsilon \sim \mathcal{N}(0, I)$$, so the randomness lives in $$\epsilon$$ and gradients flow through $$\mu$$ and $$\sigma$$. That's how we get gradients of the objective; it doesn't change what the objective is. If a VAE's latent space is smoother than an autoencoder's, the reason has to be in the objective, and finding it is the next section's job.

![One VAE training step: the encoder maps image x to mu(x) and log sigma^2(x), random noise epsilon is added to pick z = mu + sigma * epsilon, the decoder rebuilds x' from z, and two orange loss bubbles score the reconstruction and the KL prior term; teal arrows mark the gradient path]({{ '/assets/blog/ghosts-and-hybrids/elbo-anatomy.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the diagram shows: one training step of a VAE. The encoder turns an image into a centre ($$\mu$$) and a spread ($$\sigma$$). A small random nudge picks a code near that centre, and the decoder rebuilds the image from that code. Two scores, the orange bubbles, are added up: how well the rebuilt image x′ matches the original, and how far the centre and spread are from a standard bell curve. The teal arrows show the path the training signal flows back along. The random nudge sits off to the side of that path, and that's the whole trick.*

## Noise balls and the KL spring

An autoencoder's decoder sees one point per image. A VAE's decoder sees a different random point near $$\mu(x)$$ each time the image comes round, so it has to decode a whole neighbourhood, a ball of radius about $$\sigma$$, to the same digit. Balls cover area and points don't, so if the balls are big enough and close enough to touch, there's nowhere left for a hole. The catch is that the encoder controls both how big the balls are and how far apart they sit, and it can make the noise harmless by spreading the balls out. The KL is what stops it. That's the usual story, and it's right, but the scale argument below shows it covers only one of the two ways the encoder can cheat.

Picture the codes as islands, as in cell 3. Inflate each one into a translucent ball. Then tie every ball to the origin with a spring, the $$\mu^2$$ term, and let the springs pull until the balls overlap and tile a disc about the size of the prior.

![Codes start as isolated white points in six islands on a coral void, inflate into teal noise balls, and are pulled toward the origin by yellow springs until the overlapping balls cover the prior's dashed 2-sigma circle with no gaps]({{ '/assets/blog/ghosts-and-hybrids/balls-tile-the-plane.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the animation shows: each white dot is one image's code. On its own, a dot leaves empty space all around it, space where the decoder has never been trained. The VAE turns each dot into a small cloud (the teal discs), and the KL term pulls every cloud toward the centre like a spring (the yellow zig-zags). The text on the left names each step as it happens. Once the clouds overlap enough to cover the dashed circle, where random samples land, no empty gaps are left inside it.*

The maths is an argument about scale. Neither network cares what units $$z$$ is measured in, because each can undo a rescaling made by the other. Use that freedom to see what each ingredient allows on its own.

*Noise alone* (the decoder sees $$z = \mu + \sigma\epsilon$$, but there's no KL). Stretch the codes by $$c > 1$$, $$\mu \to c\mu$$ with $$\sigma$$ fixed, and let the decoder shrink them back, $$g'(z) = g(z/c)$$. Then

$$
g'\big(c\mu + \sigma\epsilon\big) = g\big(\mu + (\sigma/c)\,\epsilon\big),
$$

so the decoder now sees noise of size $$\sigma/c$$. Less noise means better reconstruction, so the loss keeps falling as $$c$$ grows and the model drifts back toward a plain autoencoder: centres far apart, balls too small to touch. Shrinking $$\sigma$$ directly does the same. Noise alone is defeated by **inflation**.

*KL alone* (the KL term is on, but the decoder sees only $$\mu$$). Run the same move the other way. Shrink the codes, $$\mu \to c\mu$$ with $$c < 1$$, and let the decoder stretch them back, $$g'(z) = g(z/c)$$. The decoder's inputs are unchanged, so reconstruction is unchanged, while the spring term falls to $$c^2\mu^2$$. The $$\sigma$$ head never reaches the decoder, so it sits at $$\sigma = 1$$, where its KL term is zero. The lowest KL is at $$c \to 0$$. KL alone is defeated by **deflation**: the codes collapse toward the origin while the decoder magnifies them.

*Both.* Inflate the codes and the spring bills you $$c^2\mu^2$$. Deflate them at fixed $$\sigma$$ and the decoder sees noise of size $$\sigma/c$$, so neighbouring balls blur into each other and reconstruction suffers; shrink $$\sigma$$ to compensate and the $$-\log\sigma^2$$ term bills you instead. Each ingredient blocks the escape route the other leaves open, so in a VAE the scale of the latent is set by the objective rather than chosen freely. That's what fills the holes: balls of a size the encoder can't shrink, packed into a region it can't inflate.

![Three panels: with noise only the codes move outward and the prior's circle is left empty; with KL only the codes collapse to the centre with no balls; in the VAE overlapping teal balls fill the circle, held in place by opposing noise and KL forces]({{ '/assets/blog/ghosts-and-hybrids/scale-pinned.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the diagram shows: why a VAE needs both of its ingredients. In each panel the dashed circle is where random samples from the prior land, and coral shading marks empty space inside it. Left: with noise but no KL, the model escapes the noise by pushing its codes far apart. The clouds stop touching and gaps open up. Middle: with the KL but no noise, the model can squeeze its codes toward the centre for free, because the decoder just zooms back in. Right: with both, pushing apart costs KL and squeezing together blurs the images, so the codes settle at a size that fills the circle.*

Cell 5 switches the ingredients on one at a time. The `VAE` class has the same 784-256-128 ELU body as Part 1's autoencoder, with separate linear heads for $$\mu$$ and $$\log\sigma^2$$, and `train_vae` takes two switches: `sample` (does the decoder see $$\mu + \sigma\epsilon$$, or just $$\mu$$?) and `beta` (the weight on the KL, 0 to turn it off). Together with cell 3's autoencoder that makes four models. For each one the cell logs the centre spread, $$\sqrt{\sum_j \operatorname{Var}(\mu_j)}$$, every 10 epochs, since that's the number the scale argument says should drift. It also reports the ratio of that spread to the RMS ball radius, which doesn't change under rescaling, so a rescaling can't hide it.

To judge the samples, the cell trains a small digit classifier (784-128-10, 92.8% accurate on 2,000 held-out digits) and reports two things about 1,000 images decoded from $$z \sim \mathcal{N}(0, I)$$: the classifier's average confidence, and how many of the ten digits get at least 2% of the samples. I first tried the obvious alternative, the pixel distance from each sample to its nearest real digit. It scored the autoencoder's blurry overlays best, because a blurry average is close in squared distance to many digits at once, which is the same flaw in pixel distance that Part 1 started from.

```python
Xt_mnist = torch.from_numpy(X_mnist)
recipes = {  # (beta, sample): which ingredients of the VAE are switched on
    "noise only": (0.0, True),   # decoder sees noise balls, no KL
    "KL only": (1.0, False),     # KL on the encoder, decoder sees only the centres
    "VAE": (1.0, True),          # both: Bernoulli likelihood, beta = 1
}
models, logs = {}, {}
for name, (beta, sample) in recipes.items():
    models[name], logs[name] = train_vae(X_mnist, beta=beta, sample=sample, log_every=10)

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
z_prior = torch.from_numpy(np.random.default_rng(1).standard_normal((1000, 2)).astype(np.float32))
rows = {}
with torch.no_grad():
    rows["AE"] = dict(mu=Z_ae, sigma=None, prior=ae_mnist.dec(z_prior).clamp(0, 1).numpy(), mse=mse_ae)
    for name, m in models.items():
        mu, logvar = m.encode(Xt_mnist)
        rows[name] = dict(mu=mu.numpy(), sigma=(0.5 * logvar).exp().numpy(), prior=m.decode(z_prior).numpy(),
                          mse=((m.decode(mu) - Xt_mnist) ** 2).mean().item())

print(f"judge: {held_acc:.1%} accuracy on 2,000 held-out real digits")
print(f"{'model':10s} {'recon MSE':>9s} {'centre spread':>13s} {'mean sigma':>10s} {'spacing/radius':>14s}"
      f" {'judge confidence on prior samples':>33s} {'digits covered':>14s}")
for name, r in rows.items():
    conf, classes = judge_samples(r["prior"])
    spread = np.sqrt(r["mu"].var(0).sum())
    sig = "-" if r["sigma"] is None else f"{r['sigma'].mean():.4f}"
    ratio = "-" if name in ("AE", "KL only") else f"{spread / np.sqrt((r['sigma'] ** 2).sum(1).mean()):.1f}"
    print(f"{name:10s} {r['mse']:9.4f} {spread:13.3f} {sig:>10s} {ratio:>14s} {conf:33.2f} {classes:>11d}/10")
print(f"{'real digits':10s} {'':>9s} {'':>13s} {'':>10s} {'':>14s} {held_conf:33.2f} {held_classes:>11d}/10")
for name in recipes:
    print(f"centre spread during training, {name:10s}: " + ", ".join(f"ep{e} {sp:.3f}" for e, sp, _ in logs[name]))

fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
pick = np.random.default_rng(3).choice(len(X_mnist), 300, replace=False)
for j, (name, r) in enumerate(rows.items()):
    ax = axes[0, j]
    ax.scatter(r["mu"][:, 0], r["mu"][:, 1], c=y_mnist, cmap="tab10", s=1, alpha=0.4)
    if name in ("noise only", "VAE"):  # only these decoders were trained on the balls
        for i in pick:
            ax.add_patch(Ellipse(r["mu"][i], 2 * r["sigma"][i, 0], 2 * r["sigma"][i, 1], color=TEAL, alpha=0.12, lw=0))
    for rad in (1, 2):
        ax.add_patch(Circle((0, 0), rad, fill=False, ls="--", color=CORAL, lw=1.2))
    span = max(np.abs(r["mu"]).max(), 2.3) * 1.05
    ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_aspect("equal")
    note = {"AE": "no balls", "noise only": "1-sigma balls (teal)", "KL only": "no balls: decoder never sees noise",
            "VAE": "1-sigma balls (teal)"}[name]
    ax.set_title(f"{name}: codes, {note}\nprior 1- and 2-sigma circles (coral)", fontsize=9)
    axes[1, j].imshow(montage(r["prior"][:64].reshape(-1, H, W), 8, 8), cmap="gray", vmin=0, vmax=1)
    axes[1, j].set_title(f"{name}: decoded from z ~ N(0, I)", fontsize=9); axes[1, j].axis("off")
plt.tight_layout()
plt.show()
```
```text
judge: 92.8% accuracy on 2,000 held-out real digits
model      recon MSE centre spread mean sigma spacing/radius judge confidence on prior samples digits covered
AE            0.0376         7.672          -              -                              0.77           7/10
noise only    0.0358        18.018     0.0062          755.1                              0.91           4/10
KL only       0.0388         1.124     1.0002              -                              0.81          10/10
VAE           0.0385         1.818     0.0645           18.3                              0.80          10/10
real digits                                                                                0.91          10/10
centre spread during training, noise only: ep10 10.268, ep20 13.329, ep30 15.900, ep40 16.308, ep50 17.505, ep60 18.019
centre spread during training, KL only   : ep10 1.269, ep20 1.182, ep30 1.138, ep40 1.164, ep50 1.137, ep60 1.125
centre spread during training, VAE       : ep10 1.824, ep20 1.790, ep30 1.768, ep40 1.823, ep50 1.779, ep60 1.818
```
![Output of cell 5]({{ '/assets/blog/ghosts-and-hybrids/cell-05-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: four models, one per column: a plain autoencoder, one trained with noise only, one with the KL term only, and the full VAE. The top row shows where each model puts its training images in code space, coloured by digit. The dashed coral circles show where random samples from the prior land (most fall inside the larger circle). Teal blobs show the noise cloud around each code, drawn only for the two models whose decoder is trained on noise. Each panel has its own axis scale, so check the numbers. The bottom row shows 64 images each model makes from the same random codes. The autoencoder makes blurry mixtures of several digits, noise only makes almost nothing but 2s and 3s, and KL only and the VAE both make all ten digits.*

Both drifts show up within 60 epochs. Noise alone inflates: its centre spread climbs from 10.268 at epoch 10 to 18.019 at epoch 60, while its balls shrink to a mean $$\sigma$$ of 0.0062. Its spacing-to-radius ratio is 755.1, against 18.3 for the VAE. It even reconstructs slightly better than the plain autoencoder (MSE 0.0358 against 0.0376), which fits it having become an autoencoder in all but name, although the two are trained with different losses (BCE and MSE), so the comparison is rough. Its prior samples show the cost. $$\mathcal{N}(0, I)$$ covers a small patch in the middle of a cloud whose spread is 18, and most samples come out a 2 or a 3: the classifier is as confident as it is on real digits (0.91), but only four digits get 2% of the samples.

KL alone deflates. Its spread falls from 1.269 to 1.125, with a small bump at epoch 40, and its $$\sigma$$ head sits at 1.0002, untouched, exactly as predicted (which is why no balls are drawn for it). The VAE holds still, its spread wandering between 1.768 and 1.824 and ending at 1.818.

Then comes the result I didn't expect: KL alone samples well. Its prior samples cover all ten digits at classifier confidence 0.81, level with the VAE's 0.80, and they're visibly sharper than the VAE's, since its decoder never had to average over a ball. The deflation the maths allows is real but slow under Adam. After 60 epochs the codes still fill the prior's disc densely enough that the decoder's smoothness bridges the gaps, and in a side run that kept going to 200 epochs the spread had only fallen to 1.04. [Ghosh et al. (2020)](https://arxiv.org/abs/1903.12436) report a closely related result. They read the VAE's sampling step as noise injected into the input of a deterministic decoder, replace that noise with explicit regularisers, fit a density to the codes after training (an ex-post density estimation step), and get samples comparable to or better than a VAE's from the resulting deterministic autoencoders. So in this experiment the VAE wins on something other than sample quality. Its scale is held in place by the objective instead of by how slowly the optimiser drifts, and its decoder has been trained on the balls between the codes as well as on the codes themselves.

The plain autoencoder shows the failure the last section predicted. Its prior samples are overlays of several digits at once, ghosts in latent space, and the classifier is least sure about them: confidence 0.77, with seven digits getting 2% of the samples.

## The same walks, again

Back to the walks. Cell 6 runs both of cell 3's walks again, through the VAE from cell 5 and between the same four training images, and redraws the hole maps on the same colour scale.

One wrinkle first. The gap threshold $$\tau$$ asks how far a point is from the nearest *code*. That's the right question for an autoencoder, whose decoder was only trained at codes, and the wrong one for a VAE, whose decoder was trained on every point of every ball. So the cell also asks the VAE's question directly: is a point outside every 2-σ ball, that is, is $$\min_i \lVert (z - \mu_i)/\sigma_i \rVert > 2$$?

```python
vae = models["VAE"]
with torch.no_grad():
    Z_vae = vae.encode(Xt_mnist)[0].numpy()
    frames_vae = {w: vae.decode(torch.from_numpy(walk(Z_vae, *ij))).numpy().reshape(-1, H, W) for w, ij in walks.items()}
tau_vae, box_vae = hole_threshold(Z_vae), padded_box(Z_vae)
gaps_vae = {w: gaps_along(Z_vae, walk(Z_vae, *ij)) for w, ij in walks.items()}
near_vae, cycle_vae = hole_maps(lambda x: vae.encode(x)[0], vae.decode, Z_vae, box_vae)
with torch.no_grad():
    mu_vae, sd_vae = Z_vae, (0.5 * vae.encode(Xt_mnist)[1]).exp().numpy()

def outside_balls(points: np.ndarray, k: float = 2.0) -> np.ndarray:
    """True where a latent point lies outside every k-sigma ball: a hole for a decoder that was trained on balls."""
    mu, sd = torch.from_numpy(mu_vae), torch.from_numpy(sd_vae)
    out = [((((pc[:, None] - mu[None]) / sd[None]) ** 2).sum(-1).min(1).values > k**2)
           for pc in torch.from_numpy(points).split(1024)]
    return torch.cat(out).numpy()

walk_points = np.concatenate([walk(Z_vae, i, j, 41) for i, j in pairs])

print(f"same images as before: walk 1 = digit {y_mnist[i0]} -> {y_mnist[i1]}, walk 2 = digit {y_mnist[j0]} -> {y_mnist[j1]}")
print(f"{'':4s} {'tau':>6s} {'walk 1 past tau':>15s} {'walk 2 past tau':>15s} {'random walks past tau':>21s}"
      f" {'box past tau':>12s} {'median cycle error':>18s}")
for name, Z, gaps, tau, near, cyc in [("AE", Z_ae, gaps_ae, tau_ae, near_ae, cycle_ae),
                                      ("VAE", Z_vae, gaps_vae, tau_vae, near_vae, cycle_vae)]:
    print(f"{name:4s} {tau:6.3f} {(gaps['walk 1'] > tau).sum():>12d}/10 {(gaps['walk 2'] > tau).sum():>12d}/10"
          f" {crosses(Z, tau):21.0%} {(near > tau).mean():12.0%} {np.median(cyc):18.3f}")
print("The VAE's decoder was trained on balls, not points, so its real holes are outside every 2-sigma ball:")
print(f"  random-walk points outside every ball: {outside_balls(walk_points).mean():.1%} | "
      f"walk 2 frames outside every ball: {outside_balls(walk(Z_vae, j0, j1)).sum()}/10")

fig = plt.figure(figsize=(14, 7.6))
gs = fig.add_gridspec(3, 10, height_ratios=[1, 1, 2.6], hspace=0.35)
for row, (name, frames, gaps, tau) in enumerate([("AE", frames_ae, gaps_ae, tau_ae), ("VAE", frames_vae, gaps_vae, tau_vae)]):
    for j in range(10):
        ax = fig.add_subplot(gs[row, j])
        ax.imshow(np.clip(frames["walk 2"][j], 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(CORAL if gaps["walk 2"][j] > tau else TEAL); spine.set_linewidth(3)
        if j == 0:
            ax.set_ylabel(f"{name}\nwalk 2", fontsize=10)
maps = [("AE", near_ae, box_ae, Z_ae, "nearest code"), ("AE", cycle_ae, box_ae, Z_ae, "cycle error"),
        ("VAE", near_vae, box_vae, Z_vae, "nearest code"), ("VAE", cycle_vae, box_vae, Z_vae, "cycle error")]
for j, (name, M, box, Z, label) in enumerate(maps):
    ax = fig.add_subplot(gs[2, 2 * j + (j > 1): 2 * j + 2 + (j > 1)])
    ax.imshow(M, origin="lower", extent=box.ravel(), aspect="auto", cmap=HOLE_CMAP, norm=HOLE_NORM)
    ax.scatter(Z[:, 0], Z[:, 1], s=0.2, color=TEAL, alpha=0.2)
    for w, ij in walks.items():
        path = walk(Z, *ij)
        ax.plot(path[:, 0], path[:, 1], **STYLE[w])
    ax.set_title(f"{name}: {label}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
plt.show()
```
```text
same images as before: walk 1 = digit 0 -> 7, walk 2 = digit 2 -> 7
        tau walk 1 past tau walk 2 past tau random walks past tau box past tau median cycle error
AE    0.114            0/10            6/10                    4%          69%              0.262
VAE   0.066            0/10            0/10                    9%          53%              0.113
The VAE's decoder was trained on balls, not points, so its real holes are outside every 2-sigma ball:
  random-walk points outside every ball: 0.5% | walk 2 frames outside every ball: 0/10
```
![Output of cell 6]({{ '/assets/blog/ghosts-and-hybrids/cell-06-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: the top two rows are walk 2 from cell 3, decoded first by the plain autoencoder and then by the VAE, between the same two images. Coral borders mark stops in an empty gap. The autoencoder has six of them; the VAE has none, and the stray 1 is gone. Below are the "how empty is it here" maps for both models side by side, in the same colours, with both walks drawn on (walk 1 dashed, walk 2 solid). The VAE's right-hand map is lighter, which means its decoder and encoder agree with each other over more of the space.*

Walk 2's void closes. Six of its ten autoencoder frames were past $$\tau$$, and none of the VAE's are. The VAE's frames go 2, 8, 9, 7. They're blurrier than the autoencoder's, and they pass through other digits the way walk 1 did, but every frame is a digit and none comes from outside a ball. The hallucinated 1 is gone. Walk 1 was fine for both models (0/10 frames past $$\tau$$). The median cycle error over the latent box falls from 0.262 to 0.113, so the VAE's decoder and encoder agree with each other over far more of the plane, which is why its cycle-error map is lighter.

By the nearest-code test alone the VAE doesn't look better: 9% of its random walks cross a gap wider than its own $$\tau$$, against 4% for the autoencoder, and 53% of its bounding box is past $$\tau$$, against 69%. That's the wrinkle at work. The VAE's codes are packed so tightly that its $$\tau$$ is small (0.066 RMS radii, against 0.114), and its decoder never needed the codes to be close, because the balls are what it was trained on. Measured the right way, only 0.5% of the points on the 500 random walks lie outside every 2-σ ball, and none of walk 2's frames do.

Latent interpolation in a plain autoencoder doesn't always fail, then: walk 1 was fine, and so were 96% of the random walks. It fails in the gaps, and nothing in an autoencoder tells you where they are.

## The β dial is a noise dial

The weight $$\beta$$ on the KL term looks like an abstract knob. With a Gaussian likelihood it turns out to be concrete, the amount of pixel noise the model believes the data contains, and a model that thinks every pixel is noisy won't spend information encoding detail it reads as noise.

Turn the dial up and the balls get fatter, the code cloud packs into the prior's disc, and eventually the islands merge into one.

To see why, swap the Bernoulli for a Gaussian with a fixed standard deviation, $$p(x \mid z) = \mathcal{N}\big(g(z), \sigma_x^2 I\big)$$. Then

$$
-\log p(x \mid z) = \frac{\big\| x - g(z) \big\|^2}{2\sigma_x^2} + \frac{D}{2}\log\big(2\pi\sigma_x^2\big),
$$

and the $$\beta$$-VAE loss, multiplied through by $$2\sigma_x^2$$ (which doesn't move the minimum), becomes

$$
\big\| x - g(z) \big\|^2 + 2\beta\sigma_x^2\,\mathrm{KL}\big(q(z \mid x)\,\|\,p(z)\big) + \text{const}.
$$

Only the product $$\beta\sigma_x^2$$ appears. Doubling $$\beta$$ or doubling $$\sigma_x^2$$ trains the same model, so the $$\beta$$ of [Higgins et al. (2017)](https://mlanthology.org/iclr/2017/higgins2017iclr-beta/) and the decoder's assumed noise level are one knob. Plain MSE training quietly picks a $$\sigma_x$$ for you, which is why "$$\beta = 1$$" with an MSE loss doesn't mean anything until you say which one.

What does the knob trade? The mean KL is the price, in nats, of what each code says about its image. The last section splits it exactly, and most of it turns out to be mutual information. Turning up $$\beta\sigma_x^2$$ makes nats expensive, so the encoder keeps fewer of them.

Cell 7 trains Gaussian-likelihood VAEs with $$\sigma_x \in \{0.05, 0.2, 0.5, 1.0\}$$, a 400-fold range of KL weight, from $$2\sigma_x^2 = 0.005$$ to 2.0 against the summed squared error. To keep four models inside the two-minute budget they train for 40 epochs instead of 60.

```python
sigmas = [0.05, 0.2, 0.5, 1.0]
show_idx = np.random.default_rng(4).choice(len(X_mnist), 8, replace=False)
dial = {}
for sx in sigmas:
    m, _ = train_vae(X_mnist, sigma_x=sx, epochs=40)  # 40 epochs keeps four models inside the 2-minute budget
    with torch.no_grad():
        mu, logvar = m.encode(Xt_mnist)
        kl = 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(1).mean().item()
        dial[sx] = dict(mu=mu.numpy(), sigma=(0.5 * logvar).exp().numpy(), kl=kl,
                        mse=((m.decode(mu) - Xt_mnist) ** 2).mean().item(), recon=m.decode(mu[show_idx]).numpy())

print(f"{'sigma_x':>7s} {'KL weight vs SSE (2 sigma_x^2)':>30s} {'mean KL (nats)':>14s} {'recon MSE':>9s} {'mean sigma':>10s}")
for sx, r in dial.items():
    print(f"{sx:7.2f} {2 * sx**2:30.4f} {r['kl']:14.2f} {r['mse']:9.4f} {r['sigma'].mean():10.3f}")
print(f"reference: the plain AE (no KL at all) reaches recon MSE {mse_ae:.4f}")

fig = plt.figure(figsize=(15, 7.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1])
pick = np.random.default_rng(3).choice(len(X_mnist), 300, replace=False)
for j, (sx, r) in enumerate(dial.items()):
    ax = fig.add_subplot(gs[0, j])
    ax.scatter(r["mu"][:, 0], r["mu"][:, 1], c=y_mnist, cmap="tab10", s=1, alpha=0.4)
    for i in pick:
        ax.add_patch(Ellipse(r["mu"][i], 2 * r["sigma"][i, 0], 2 * r["sigma"][i, 1], color=TEAL, alpha=0.12, lw=0))
    ax.add_patch(Circle((0, 0), 2, fill=False, ls="--", color=CORAL, lw=1.2))
    span = max(np.quantile(np.abs(r["mu"]), 0.995), 2.3) * 1.1
    ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_aspect("equal")
    ax.set_title(rf"$\sigma_x$ = {sx}: KL weight {2 * sx**2:.3f}, {r['kl']:.1f} nats", fontsize=10)
strip = np.concatenate([X_mnist[show_idx]] + [r["recon"] for r in dial.values()]).reshape(-1, H, W)
ax = fig.add_subplot(gs[1, :])
ax.imshow(montage(strip, 1 + len(sigmas), 8), cmap="gray", vmin=0, vmax=1)
ax.set_yticks([H * (k + 0.5) for k in range(1 + len(sigmas))], ["data"] + [rf"$\sigma_x$={sx}" for sx in sigmas])
ax.set_xticks([])
plt.tight_layout()
plt.show()
```
```text
sigma_x KL weight vs SSE (2 sigma_x^2) mean KL (nats) recon MSE mean sigma
   0.05                         0.0050          33.21    0.0378      0.021
   0.20                         0.0800           8.96    0.0393      0.041
   0.50                         0.5000           5.39    0.0409      0.085
   1.00                         2.0000           3.22    0.0433      0.213
reference: the plain AE (no KL at all) reaches recon MSE 0.0376
```
![Output of cell 7]({{ '/assets/blog/ghosts-and-hybrids/cell-07-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: four VAEs, each told to expect a different amount of pixel noise ($$\sigma_x$$), from very little on the left to a lot on the right. The top row shows where each model puts its training images, with a teal noise cloud around a sample of them and the coral circle marking where most random samples land. As the expected noise grows, the clouds grow and the codes crowd into the circle. Below are eight real digits (top row) and how each model rebuilds them. The rebuilt rows look almost the same, so this dial changes how the code space is organised far more than how sharp the images are.*

At $$d = 2$$ the dial trades information, not sharpness. As $$\sigma_x$$ goes from 0.05 to 1.0, the mean KL falls tenfold, from 33.21 to 3.22 nats, and the mean ball size grows tenfold, from 0.021 to 0.213. Reconstruction error barely moves, from 0.0378 to 0.0433, and the plain autoencoder, with no KL at all, gets 0.0376. The reconstruction rows look almost alike, and they're all soft (the plain autoencoder's frames in cell 3 are just as soft), because two numbers can't hold a sharp digit.

That rewrites a common complaint. A blurry VAE isn't a bug, and at this size the blur doesn't even come from the VAE. It comes from the two-number bottleneck, which is why the autoencoder is just as blurry. What $$\beta$$ buys here is a latent shaped like the prior, paid for in nats. With more latent dimensions there's detail left to give up, and then the same dial does trade sharpness. Push it far enough and the encoder gives up every nat, the decoder learns to ignore $$z$$, and you get posterior collapse ([Bowman et al., 2016](https://arxiv.org/abs/1511.06349)). Even at $$\sigma_x = 1$$ we're at 3.22 nats, nowhere near it.

## Holes that survive

The KL term pays two bills at once. Part of it pays for information, how much each code says about its own image. The rest pays for mismatch, how far the codes taken together are from the prior. Any useful encoder has to keep some information, and the objective charges for information and for mismatch out of the same budget. Nothing in it forces the codes, taken together, to match the prior exactly, and in practice they don't.

To picture it, average the encoder's balls over the whole dataset. That gives the aggregate posterior,

$$
q(z) = \frac{1}{N} \sum_{i=1}^{N} q(z \mid x_i),
$$

which is the distribution of codes the decoder is actually trained on: a lumpy sum of 10,000 small Gaussians, where the prior is one smooth round one. Wherever the prior has mass and $$q(z)$$ doesn't, sampling $$z \sim p(z)$$ lands somewhere the decoder rarely saw, which is exactly what a hole is.

To split the bill, average the KL term over the data, then add and subtract $$\log q(z)$$ inside it:

$$
\mathbb{E}_{x}\,\mathrm{KL}\big(q(z \mid x)\,\|\,p(z)\big)
= \underbrace{\mathbb{E}_{x}\,\mathbb{E}_{q(z \mid x)}\!\left[\log \frac{q(z \mid x)}{q(z)}\right]}_{I_q(x;\,z)}
+ \underbrace{\mathbb{E}_{q(z)}\!\left[\log \frac{q(z)}{p(z)}\right]}_{\mathrm{KL}\left(q(z)\,\|\,p(z)\right)}.
$$

This is the "ELBO surgery" of [Hoffman & Johnson (2016)](https://www.cs.columbia.edu/~blei/fogm/2023F/readings/HoffmanJohnson2016.pdf). The first term is the mutual information between an image and its code under the encoder, and the second is how far the aggregate posterior is from the prior. Both are non-negative, and the objective penalises their sum, so a model can lower its bill by matching the prior better or by carrying less information about $$x$$; nothing forces it to drive the mismatch term to zero. With $$N$$ training images, $$I_q$$ can't exceed $$\log N$$, because the most $$z$$ can tell you is which of the $$N$$ images it came from. [Rosca et al. (2018)](https://arxiv.org/abs/1802.06847) show that trained VAEs consistently fail to match the latent marginal, and trace it to an objective that matches conditional distributions one image at a time.

In two dimensions every term can be computed almost exactly. Cell 8 uses the VAE from cell 5. The mean KL comes from the closed form. $$I_q$$ comes from Monte Carlo, three samples from each image's ball, each scored with $$\log q(z \mid x_i) - \log q(z)$$ against the full 10,000-component mixture. $$\mathrm{KL}(q(z)\,\Vert\,p(z))$$ comes from integrating on a 241 × 241 grid. The three are computed independently, so whether the first equals the sum of the other two is a real check.

```python
with torch.no_grad():
    mu, logvar = vae.encode(Xt_mnist)
    var = logvar.exp()
    mean_kl = (0.5 * (mu**2 + var - logvar - 1).sum(1)).mean().item()  # exact, closed form

    def log_q_mix(z: torch.Tensor) -> torch.Tensor:
        """log q(z) for the aggregate posterior q(z) = (1/N) sum_i N(z; mu_i, diag sigma_i^2), in chunks."""
        out = []
        for zc in z.split(2048):
            lp = -0.5 * (((zc[:, None] - mu[None]) ** 2) / var[None] + logvar[None] + np.log(2 * np.pi)).sum(-1)
            out.append(torch.logsumexp(lp, dim=1) - np.log(len(mu)))
        return torch.cat(out)

    # I_q(x; z) by Monte Carlo: z ~ q(z|x_i), average log q(z|x_i) - log q(z)
    gen = torch.Generator().manual_seed(5)
    eps = torch.randn((3,) + mu.shape, generator=gen)
    z_s = (mu + var.sqrt() * eps).reshape(-1, 2)
    log_q_cond = (-0.5 * (eps**2 + logvar + np.log(2 * np.pi)).sum(-1)).reshape(-1)
    mi = (log_q_cond - log_q_mix(z_s)).mean().item()

    # KL(q(z) || p(z)) by integrating on a grid
    n, L = 241, 4.5
    g = torch.linspace(-L, L, n)
    G = torch.stack(torch.meshgrid(g, g, indexing="xy"), -1).reshape(-1, 2)
    log_q = log_q_mix(G)
    log_p = -0.5 * (G**2).sum(1) - np.log(2 * np.pi)
    cell = (2 * L / (n - 1)) ** 2
    q, p = log_q.exp(), log_p.exp()
    kl_agg = (q * (log_q - log_p)).sum().item() * cell
    hole_mass = (p * (q < 0.1 * p)).sum().item() * cell

print(f"mean KL(q(z|x) || p(z))   = {mean_kl:.3f} nats   (closed form)")
print(f"I_q(x; z)                 = {mi:.3f} nats   (Monte Carlo; bounded by log N = {np.log(len(mu)):.3f})")
print(f"KL(q(z) || p(z))          = {kl_agg:.3f} nats   (grid integral; q integrates to {q.sum().item() * cell:.4f})")
print(f"I_q + KL(q || p)          = {mi + kl_agg:.3f} nats   vs mean KL {mean_kl:.3f}")
print(f"prior mass where q(z) < p(z)/10: {hole_mass:.1%}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
ext = [-L, L, -L, L]
for ax, M, title in [(axes[0], q, "aggregate posterior q(z)"), (axes[1], p, "prior p(z) = N(0, I)")]:
    im = ax.imshow(M.reshape(n, n).numpy(), origin="lower", extent=ext, cmap="Greys")
    ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.8)
diff = (q - p).reshape(n, n).numpy()
vmax = np.abs(diff).max()
im = axes[2].imshow(diff, origin="lower", extent=ext, cmap=LinearSegmentedColormap.from_list("qp", [CORAL, "#f8fafc", TEAL]),
                    vmin=-vmax, vmax=vmax)
axes[2].contour(g.numpy(), g.numpy(), (q < 0.1 * p).reshape(n, n).numpy().astype(float), levels=[0.5], colors="k", linewidths=0.8)
axes[2].set_title("q(z) - p(z): coral = prior mass the decoder rarely saw"); fig.colorbar(im, ax=axes[2], shrink=0.8)
for ax in axes:
    for radius in (1, 2):
        ax.add_patch(Circle((0, 0), radius, fill=False, ls="--", color=SLATE, lw=0.8))
    ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
plt.tight_layout()
plt.show()
```
```text
mean KL(q(z|x) || p(z))   = 6.291 nats   (closed form)
I_q(x; z)                 = 5.863 nats   (Monte Carlo; bounded by log N = 9.210)
KL(q(z) || p(z))          = 0.427 nats   (grid integral; q integrates to 1.0000)
I_q + KL(q || p)          = 6.290 nats   vs mean KL 6.291
prior mass where q(z) < p(z)/10: 3.7%
```
![Output of cell 8]({{ '/assets/blog/ghosts-and-hybrids/cell-08-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: three maps of the same 2-D code space. Left: where the VAE's training codes actually are, darker meaning more of them. Middle: where the VAE assumes they are when it samples, a single smooth round hill. Right: the difference between the two. Teal is where real codes are more common than the VAE assumes, and coral is where the VAE will sample but rarely saw a training code. Those coral areas are the holes that remain. The black outline marks the worst of them, places with less than a tenth of the codes the VAE expects.*

The decomposition closes. The mean KL is 6.291 nats; $$I_q$$ is 5.863 nats, below its ceiling of $$\log N = 9.210$$; and $$\mathrm{KL}(q(z)\,\Vert\,p(z))$$ is 0.427 nats, with $$q$$ integrating to 1.0000 on the grid. The two add up to 6.290, a thousandth of a nat away from the closed form. Of the 6.29 nats the KL term charges, 93% pays for information and 7% for mismatch with the prior.

A small mismatch on average still leaves holes. 3.7% of the prior's mass sits where $$q(z)$$ is less than a tenth of $$p(z)$$. Some of it is the low-density rim far from the origin, and some is the coral patches inside the 1-σ circle, thin gaps between digit clusters close to the origin, where the prior is densest. About one sample in 27 from $$\mathcal{N}(0, I)$$ lands in a region the decoder saw less than a tenth as often as the prior expects. Hoffman and Johnson draw the practical conclusion from the same decomposition: to improve the bound, improve the prior as well as the encoder and decoder.

## Recent work

The question this post keeps asking, what the decoder does away from the training codes, is now a practical one for the autoencoders inside latent diffusion models. Those models sample with a diffusion process rather than from a fixed Gaussian prior, so matching a prior matters less there, but they still need a latent space the generator can move around in.

Two ICML 2025 papers regularise how the decoder behaves under simple transformations of the latent. [Skorokhodov et al. (2025)](https://arxiv.org/abs/2502.14831) find high-frequency components in autoencoder latents that get in the way of coarse-to-fine diffusion, and add a scale-equivariance regulariser that cuts ImageNet 256² FID by 19%. [EQ-VAE (Kouzelis et al., 2025)](https://arxiv.org/abs/2502.09509) fine-tunes a pretrained autoencoder to be equivariant to scaling and rotation, keeps its reconstruction quality, and reports a 7× training speed-up for DiT-XL/2. [REPA-E (Leng et al., 2025)](https://arxiv.org/abs/2504.10483) trains the autoencoder and the diffusion transformer together, which only works with a representation-alignment loss, and finds the autoencoder's latent structure improves along the way. [Yao et al. (2025)](https://arxiv.org/abs/2501.01423), which came up in Part 1, align the latent with a pretrained vision model instead.

Older work went after interpolation directly. ACAI ([Berthelot et al., 2018](https://arxiv.org/abs/1807.07543)) trains a critic to guess how far along an interpolation a decoded image is, and trains the autoencoder to make that impossible, so that interpolants look like data. Adversarial autoencoders ([Makhzani et al., 2015](https://arxiv.org/abs/1511.05644)) match $$q(z)$$ to $$p(z)$$ with an adversary instead of a per-image KL, which targets the mismatch term in ELBO surgery and leaves the information term alone.

## Into higher dimensions

Averaging pixels cuts a chord through pixel space that leaves the data straight away, and on Blob World it puts the midpoint exactly $$\sqrt{\pi s^2/2}$$ from the nearest real image. Averaging codes stays on the data wherever the decoder was trained, and an autoencoder's decoder was trained only at its codes, so a line across a gap decodes to a confident wrong digit or, on Holey Blob World, a ghost of the decoder's own making. A VAE trains the decoder on balls instead of points and pins their scale from both sides. The balls fill the prior's disc. ELBO surgery shows why the filling stays incomplete in practice: the objective charges for information and for prior mismatch out of one budget, and our VAE spent 93% of it on information.

All of that happened in two dimensions, where everything can be drawn. Real VAEs use dozens or hundreds of latent dimensions with the same $$\mathcal{N}(0, I)$$ prior. So draw $$z \sim \mathcal{N}(0, I)$$ in 512 dimensions. Where is it? Not near the origin, where the density is highest, and nowhere your 2-D pictures suggest. **Part 3, the high-dimensional soap bubble**, is about where Gaussian samples actually live, and why a straight line between two of them still goes somewhere the decoder has never been.

## Further reading

- [Part 1 of this series](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/what-is-latent-space/): latent spaces as learned charts, and why the chart is only checked on the data.
- [Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114): the VAE paper, short and still the clearest statement of the ELBO and the reparameterisation.
- [Doersch, 2016](https://arxiv.org/abs/1606.05908): a long, gentle tutorial on VAEs, good for the probabilistic side this post skipped.
- [Hoffman & Johnson, 2016](https://www.cs.columbia.edu/~blei/fogm/2023F/readings/HoffmanJohnson2016.pdf): a short workshop paper that splits the KL term into information and prior mismatch.
- [Ghosh et al., 2020](https://arxiv.org/abs/1903.12436): deterministic autoencoders that sample as well as VAEs, and the best explanation of cell 5's surprise.
