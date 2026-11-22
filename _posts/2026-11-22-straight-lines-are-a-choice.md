---
layout: post
title: "Straight lines are a choice: how decoders bend latent space, and how flow matching straightens it"
date: 2026-11-22 09:00:00+0530
description: Why equal steps in a latent space aren't equal steps in the data, how the decoder's own pull-back metric and a rubber-band geodesic fix that until a hole or a blurry decoder offers a shortcut, and how flow matching trains the straight line instead, with every claim checked in code.
tags: latent-space geodesics flow-matching interpolation vae
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

Every latent interpolation demo makes the same quiet assumption. Draw a straight line between two codes, take equal steps along it, and you get equal steps in whatever the model draws. Parts 1 to 3 of this series poked at where that line goes. This post pokes at the ruler. Once you measure a walk in pixels instead of latent coordinates, "straight" stops being a property of the line. It's a choice of metric, and the decoder is the one making it.

This is the fourth and last part of a series on the geometry of latent spaces. [Part 1](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/what-is-latent-space/) treated a latent space as a learned chart, coordinates the model invents for its data manifold. [Part 2](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/ghosts-and-hybrids/) found holes in that chart where the decoder was never trained, and watched a VAE fill most of them. [Part 3](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/high-dimensional-soap-bubble/) went to high dimensions, where the Gaussian prior is a soap bubble and SLERP keeps a walk on its skin. All three were about *where* a path goes. Here the questions are how long a path is, which one is shortest, and what goes wrong when the shortest one runs somewhere the decoder never learned. At the very end, flow matching turns the whole thing upside down: instead of searching a trained model for good paths, it trains the model so that a very simple path is good.

Same format as before. Each idea gets an intuition, a picture, the maths, and a short code cell that checks the maths on real numbers. It all runs on a laptop CPU in about four minutes, and MNIST is the only download. Every model from Parts 1-3 comes back, retrained with its original seed and checked against the numbers those posts printed.

## A blob that won't keep pace

Part 3 ended on a blob that wouldn't keep pace. A Blob World VAE with a 16-dimensional latent walked a SLERP path between two prior samples in 40 equal steps of $$t$$. SLERP moves at constant angular speed. If latent steps were image steps, the blob would glide.

It lurched. One step moved it 0.08 pixels, another 0.61, and the spread of its step sizes was 0.40 of their mean.

Now join the same two blob positions with a geodesic, drawn by a decoder that renders sharp blobs. The blob moves between 0.23 and 0.28 pixels every step, in a straight line, spread 0.05. Same endpoints. Different idea of straight.

Getting from the first walk to the second takes three pieces: a ruler that measures latent steps in the data (the decoder's pull-back metric), a way to find the shortest path under it (a rubber band), and a decoder whose ruler you can trust. The third piece turned out to be the interesting one. Part 3's own decoder fails it, and Part 2's Holey Blob World shows what a geodesic does when the shortest path runs through a hole.

![On a latent map shaded by how fast the decoded image changes, an orange rubber band pinned at two codes relaxes into a green geodesic as its energy falls from 339 to 255, while nine sampled blob positions in the image go from bunched and lurching to evenly spaced along a straight line]({{ '/assets/blog/straight-lines-are-a-choice/rubber-band-geodesic.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the animation shows: on the left is the model's map of blob images, brighter where a small step changes the image a lot, with a rubber band pinned at two points that starts straight (orange) and relaxes into the shortest path as the model measures it (green). On the right is the blob each band draws, jerky for the straight one and steady for the relaxed one. The path that looks straight on the map isn't the one that moves the blob straight.*

## Where Part 3 left us

Quick recap. Part 1's chart is only checked at the data. Part 2's holes sit between the codes, where the decoder never trained, and a VAE plugs most of them with noise balls held to a prior. Part 3's prior is a thin shell in high dimensions, so a straight line between two samples sags through the empty middle, and SLERP keeps the walk on the shell.

Notice what none of that tells you: how long a path is. Walk lengths, the angle between two seeds, the "even" spacing of interpolation frames were all measured in latent coordinates, as if a unit step in $$z$$ meant the same change everywhere. It doesn't. Part 3's last cell showed it, and Part 2 had already hinted at it with a hybrid blob that slid at anything from 4.8 to 70 pixels per unit of $$t$$ along a straight latent line. Measuring a path in the data takes a ruler that lives in the data, and the only thing connecting latent to data is the decoder. So that's where the ruler has to come from.

## The decoder's ruler

Picture a paper map whose scale bar changes from place to place: a centimetre is a kilometre in one corner and a hundred metres near the middle. Two routes of the same length on paper can be very different on the ground, and the shortest route on the ground can look bent on paper. A latent space is that kind of map. The decoder $$g_\theta: \mathbb{R}^d \to \mathbb{R}^D$$ turns each point $$z$$ into an image, and how far the image moves when $$z$$ moves depends on where $$z$$ is.

Zoom in far enough on one point $$z$$ and the decoder is nearly linear. A tiny circle of latent steps around $$z$$ becomes a tiny ellipse of image changes around $$g(z)$$, and that ellipse is the local scale bar. Big means the image changes fast here. Thin means one direction matters far more than the other.

In symbols, it's a first-order Taylor expansion. With $$J(z) = \partial g / \partial z \in \mathbb{R}^{D \times d}$$ the decoder's Jacobian,

$$
\begin{gathered}
g(z + dz) \approx g(z) + J(z)\,dz \
\Longrightarrow\quad \|g(z + dz) - g(z)\|^2 \approx dz^\top J(z)^\top J(z)\, dz = dz^\top G(z)\, dz.
\end{gathered}
$$

$$G(z) = J(z)^\top J(z)$$ is a $$d \times d$$ symmetric matrix called the **pull-back metric**: ordinary Euclidean distance in pixel space, pulled back through the decoder into latent coordinates. That's the ruler. A path $$\gamma: [0, 1] \to \mathbb{R}^d$$ through the latent now has a length measured in the data,

$$
L[\gamma] = \int_0^1 \sqrt{\dot\gamma(t)^\top G(\gamma(t))\, \dot\gamma(t)}\; dt,
$$

which is just the length of the decoded path $$t \mapsto g(\gamma(t))$$ in pixel space. A **geodesic** is the shortest path between two endpoints. Length is awkward to minimise directly (the same route at any speed has the same length), so the usual trick is to minimise the energy instead:

$$
E[\gamma] = \frac{1}{2}\int_0^1 \dot\gamma^\top G(\gamma)\, \dot\gamma \; dt.
$$

Cauchy-Schwarz, with $$v(t) = \sqrt{\dot\gamma^\top G \dot\gamma}$$ the decoded speed, gives

$$
L^2 = \Big(\int_0^1 v\, dt\Big)^2 \le \int_0^1 v^2\, dt = 2E,
$$

with equality only when $$v$$ is constant. So minimising energy buys two things at once: the shortest route, and constant speed along it in the data. Constant speed is exactly what Part 3's blob was missing. I'll use the ratio $$L^2 / 2E$$ as a check below. It's 1 for a constant-speed path and drops as the speed wobbles.

Blob World is where we can work out the true ruler by hand. Each image is a blob $$I_c(p) = \exp\!\big(-\|p - c\|^2 / 2s^2\big)$$ at pixel $$p$$, with centre $$c = (c_x, c_y)$$ and width $$s$$, and the true decoder (the renderer) maps $$c \mapsto I_c$$. Its derivative is $$\partial I_c / \partial c_x = \frac{x - c_x}{s^2}\, I_c$$, so if we treat the pixel sum as an integral and write $$u = x - c_x$$, $$w = y - c_y$$,

$$
\begin{aligned}
G_{xx} &= \sum_p \Big(\frac{x - c_x}{s^2}\Big)^2 I_c(p)^2 \
&\approx \frac{1}{s^4} \int u^2 e^{-u^2/s^2}\, du \int e^{-w^2/s^2}\, dw \
&= \frac{1}{s^4}\cdot\frac{s^3\sqrt{\pi}}{2}\cdot s\sqrt{\pi} = \frac{\pi}{2}.
\end{aligned}
$$

The cross term $$G_{xy}$$ integrates an odd function, so it's zero. By symmetry $$G_{yy}$$ matches $$G_{xx}$$. The true metric is $$G_{\text{true}} = \frac{\pi}{2} I_2$$ wherever a blob fits inside the image, and it doesn't depend on the width $$s$$ at all.

Blob World is flat. A blob that slides $$\delta$$ pixels travels exactly $$\sqrt{\pi/2}\,\delta$$ in pixel space, which means any bumps in a learned decoder's metric come from the chart it picked, not from the data.

There's a second way between two blobs, and it's going to cause trouble. Don't slide; cross-fade. That's the straight pixel-space line $$(1 - t) I_a + t I_b$$, Part 2's ghost, with length $$\|I_a - I_b\|$$. [Part 1](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/what-is-latent-space/) worked it out in closed form:

$$
\|I_a - I_b\| = \sqrt{2\pi s^2\big(1 - e^{-\delta^2/4s^2}\big)}.
$$

For small $$\delta$$, $$1 - e^{-x} \approx x$$ and this becomes $$\sqrt{\pi/2}\,\delta$$, the slide length, so up close the two routes agree. For any $$\delta > 0$$, though, $$1 - e^{-x} < x$$, and the cross-fade is strictly shorter. Past about $$3s$$ it stops growing altogether, capped at $$\sqrt{2\pi s^2} \approx 5.01$$ for $$s = 2$$. A chord beats an arc. Hold on to that, because any decoder that can draw a ghost is handing the rubber band a shortcut.

Cell 1 is Part 3's setup cell copied verbatim (minus two packages this post doesn't need), plus Part 2's off-manifold tests and the new tools. Among them, `pullback_metric` computes $$G(z)$$ with forward-mode autograd, `geodesic` is the rubber band, and `blob_position` finds a decoded blob. That last one fixes a small bias in Part 3's `centre_of_mass`, which weighs every pixel: a decoder's faint background, about 0.02 per pixel, drags its estimate toward the image centre by up to 1.5 pixels. The new one only counts intensity above 0.1. I kept the old one around purely to reproduce Part 3's number.

```python
# ---- Parts 2-3's setup cell, copied verbatim (Part 3 adds transformers and scipy; Part 4 needs neither) ----
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

# ---- Part 3's helpers, copied verbatim (Part 3, cell 1) ----------------------------------------
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

# ---- Part 2's off-manifold tests, copied verbatim (Part 2, cells 2 and 4) -----------------------
grid_c = np.stack(np.meshgrid(np.arange(2, 26.01, 0.25), np.arange(2, 26.01, 0.25)), -1).reshape(-1, 2)
manifold_index = NearestNeighbors(n_neighbors=1).fit(flat(np.stack([blob(cx, cy) for cx, cy in grid_c])))

def dist2_to_manifold(frames: np.ndarray) -> np.ndarray:
    """Squared pixel distance from each frame to the nearest real Blob World image."""
    return manifold_index.kneighbors(flat(frames))[0][:, 0] ** 2

def centre_of_mass(frames: np.ndarray) -> np.ndarray:
    """Intensity-weighted centre (x, y) of each frame: where the blob is."""
    w = np.clip(frames, 0, None)
    return np.column_stack([(w * xs).sum((1, 2)), (w * ys).sum((1, 2))]) / w.sum((1, 2))[:, None]

def bright_spots(img: np.ndarray, level: float = 0.25) -> int:
    """Number of local maxima above `level`: how many blobs a frame shows."""
    p = np.pad(img, 1, constant_values=-np.inf)
    neighbours = np.stack([p[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    return int(((img > neighbours.max(0)) & (img > level)).sum())

# ---- New in Part 4: the decoder's ruler and the rubber band ------------------------------------
from typing import Callable
GREEN = "#22c55e"  # series colour: geodesic
Decoder = Callable[[torch.Tensor], torch.Tensor]  # (n, d) codes -> (n, D) images

def as_tensor(z) -> torch.Tensor:
    return torch.as_tensor(np.asarray(z, dtype=np.float32))

def jacobian(g: Decoder, z: np.ndarray) -> np.ndarray:
    """J(z) = dg/dz for a batch of codes, exact, by forward-mode autograd: one pass per latent direction. (n, D, d)."""
    z = as_tensor(z)
    columns = [torch.func.jvp(g, (z,), (torch.eye(z.shape[1])[i].expand_as(z),))[1] for i in range(z.shape[1])]
    return torch.stack(columns, -1).detach().numpy()

def pullback_metric(g: Decoder, z: np.ndarray) -> np.ndarray:
    """G(z) = J^T J: the decoder's ruler. A latent step dz moves the image by sqrt(dz^T G dz)."""
    J = jacobian(g, z)
    return np.einsum("nDi,nDj->nij", J, J)

def decoded_steps(g: Decoder, path: np.ndarray) -> np.ndarray:
    """Pixel distance between consecutive decoded points of a latent path."""
    with torch.no_grad():
        x = g(as_tensor(path))
    return (x[1:] - x[:-1]).norm(dim=1).numpy()

def path_length(g: Decoder, path: np.ndarray) -> float:
    """Data-space length of a latent path: the length of its decoded image path."""
    return float(decoded_steps(g, path).sum())

def geodesic(g: Decoder, z0: np.ndarray, z1: np.ndarray, n: int = 32, iters: int = 1500,
             ridge: float = 0.0, init: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The rubber band: fix both ends, move the n - 1 interior points to minimise the discrete energy
    E = (n/2) sum_i ||g(z_{i+1}) - g(z_i)||^2. Minimisers are constant-speed geodesics; autograd does the Jacobians.
    `ridge` adds ridge * (n/2) sum ||z_{i+1} - z_i||^2, so latent directions the decoder ignores stay put."""
    a, b = as_tensor(z0), as_tensor(z1)
    start = as_tensor(lerp(z0, z1, np.linspace(0, 1, n + 1)) if init is None else init)[1:-1]
    inner = start.clone().requires_grad_(True)
    opt = torch.optim.Adam([inner], lr=0.02 * (b - a).norm().item())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    energy_log = []
    for _ in range(iters):
        path = torch.cat([a[None], inner, b[None]])
        x = g(path)
        energy = 0.5 * n * ((x[1:] - x[:-1]) ** 2).sum() + 0.5 * n * ridge * ((path[1:] - path[:-1]) ** 2).sum()
        opt.zero_grad(); energy.backward(); opt.step(); sched.step()
        energy_log.append(energy.item())
    return torch.cat([a[None], inner.detach(), b[None]]).numpy(), np.array(energy_log)

def tissot(ax, centre: np.ndarray, G: np.ndarray, r: float, **style) -> None:
    """Draw {centre + v : v^T G v = r^2}: semi-axes r / sqrt(lambda_i) along G's eigenvectors."""
    lam, vec = np.linalg.eigh(G)
    angle = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
    ax.add_patch(Ellipse(centre, 2 * r / np.sqrt(lam[0]), 2 * r / np.sqrt(lam[1]), angle=angle, **style))

def blob_position(frames: np.ndarray, floor: float = 0.1) -> np.ndarray:
    """Where the blob is: centre of mass of the intensity above `floor`. Part 3's centre_of_mass weighs every pixel,
    so a decoder's faint background (~0.02 per pixel) drags it toward the image centre by up to 1.5 px."""
    w = np.clip(frames - floor, 0, None)
    return np.column_stack([(w * xs).sum((1, 2)), (w * ys).sum((1, 2))]) / w.sum((1, 2))[:, None]

def speed_report(track: np.ndarray) -> str:
    """How evenly a blob moves: pixel step sizes along a track, and their std / mean (0 = constant speed)."""
    steps = np.linalg.norm(np.diff(track, axis=0), axis=1)
    return f"steps {steps.min():.2f}-{steps.max():.2f} px, std/mean {steps.std() / steps.mean():.2f}"

print(f"judge: {held_acc:.1%} held-out accuracy | Part 4 helpers ready: pullback_metric, geodesic, path_length, tissot")
```
```text
numpy 2.3.3 | torch 2.13.0+cpu | torchvision 0.28.0+cpu
MNIST subset: (10000, 784), pixels in [0, 1]
judge: 92.8% held-out accuracy | Part 4 helpers ready: pullback_metric, geodesic, path_length, tissot
```

Cell 2 checks both results. It differentiates the true renderer by central differences, builds $$J^\top J$$ at the image centre for three blob widths, maps $$\sqrt{\det G}$$ over every centre position, and measures both routes for blobs from a quarter of a pixel to 20 pixels apart. The slide length is summed from 400 tiny pixel-space steps, so it isn't quietly assuming the formula it's meant to test.

```python
def render_jacobian(c: np.ndarray, s: float = 2.0, h: float = 1e-3) -> np.ndarray:
    """dI/dc of the true renderer c -> blob(c) by central differences: (784, 2)."""
    return np.stack([(flat(blob(*(c + h * e), s=s)[None]) - flat(blob(*(c - h * e), s=s)[None]))[0] / (2 * h)
                     for e in np.eye(2)], axis=1)

for s_ in (1.5, 2.0, 3.0):
    G_true = render_jacobian(np.array([14.0, 14.0]), s=s_).T @ render_jacobian(np.array([14.0, 14.0]), s=s_)
    print(f"s = {s_}: J^T J at the centre = [[{G_true[0, 0]:.4f}, {abs(G_true[0, 1]):.4f}], "
          f"[{abs(G_true[1, 0]):.4f}, {G_true[1, 1]:.4f}]]   (pi/2 = {np.pi / 2:.4f})")

centres = np.arange(-1, 28.01, 0.5)
mag_true = np.array([[np.sqrt(np.linalg.det(render_jacobian(np.array([cx, cy])).T @ render_jacobian(np.array([cx, cy]))))
                      for cx in centres] for cy in centres])
inside = (centres >= 4) & (centres <= 24)
print(f"sqrt(det G) over the training square [4, 24]^2: {mag_true[np.ix_(inside, inside)].min():.4f} to "
      f"{mag_true[np.ix_(inside, inside)].max():.4f} (pi/2 = {np.pi / 2:.4f}); at a corner of the image: {mag_true[0, 0]:.3f}")

# Two ways from blob a to blob b, delta pixels apart: slide the blob (on the manifold) or cross-fade (the pixel chord)
print("delta | slide length (on M) | sqrt(pi/2) delta | cross-fade length ||I_a - I_b||")
deltas = np.linspace(0.25, 20, 80)
slide, chord = [], []
for delta in deltas:
    track_c = np.column_stack([np.linspace(14 - delta / 2, 14 + delta / 2, 401), np.full(401, 14.0)])
    frames = flat(np.stack([blob(cx, cy) for cx, cy in track_c]))
    slide.append(np.linalg.norm(np.diff(frames, axis=0), axis=1).sum())
    chord.append(np.linalg.norm(frames[-1] - frames[0]))
for delta in (1, 4, 8, 16):
    k = np.argmin(np.abs(deltas - delta))
    print(f"{deltas[k]:5.1f} | {slide[k]:19.2f} | {np.sqrt(np.pi / 2) * deltas[k]:16.2f} | {chord[k]:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
im = ax1.imshow(mag_true, origin="upper", extent=[centres[0] - 0.25, centres[-1] + 0.25, centres[-1] + 0.25, centres[0] - 0.25],
                cmap="viridis", vmin=0, vmax=1.7)
ax1.add_patch(plt.Rectangle((4, 4), 20, 20, fill=False, ec=TEAL, lw=2, ls="--"))
ax1.set_xlabel("blob centre x (pixels)"); ax1.set_ylabel("blob centre y (pixels)")
ax1.set_title(r"True magnification $\sqrt{\det G}$: flat ($\pi/2$) inside the training square", fontsize=10)
plt.colorbar(im, ax=ax1, fraction=0.046)
ax2.plot(deltas, slide, color=TEAL, lw=2.5, label="slide the blob (stays on the manifold)")
ax2.plot(deltas, np.sqrt(np.pi / 2) * deltas, "k--", lw=1, label=r"$\sqrt{\pi/2}\,\delta$")
ax2.plot(deltas, chord, color=CORAL, lw=2.5, label=r"cross-fade (straight line in pixels)")
ax2.axhline(np.sqrt(2 * np.pi * 2.0**2), color=CORAL, ls=":", lw=1, label=r"$\sqrt{2\pi s^2}$")
ax2.set_xlabel(r"distance between blob centres $\delta$ (pixels)"); ax2.set_ylabel("length in pixel space")
ax2.set_title("The ghost route is always shorter", fontsize=10); ax2.legend(fontsize=8)
plt.tight_layout(); plt.show()
```
```text
s = 1.5: J^T J at the centre = [[1.5708, 0.0000], [0.0000, 1.5708]]   (pi/2 = 1.5708)
s = 2.0: J^T J at the centre = [[1.5708, 0.0000], [0.0000, 1.5708]]   (pi/2 = 1.5708)
s = 3.0: J^T J at the centre = [[1.5708, 0.0000], [0.0000, 1.5708]]   (pi/2 = 1.5708)
sqrt(det G) over the training square [4, 24]^2: 1.4855 to 1.5708 (pi/2 = 1.5708); at a corner of the image: 0.164
delta | slide length (on M) | sqrt(pi/2) delta | cross-fade length ||I_a - I_b||
  1.0 |                1.25 |             1.25 | 1.23
  4.0 |                5.01 |             5.01 | 3.99
  8.0 |               10.03 |            10.03 | 4.97
 16.0 |               20.05 |            20.05 | 5.01
```
![Output of cell 2]({{ '/assets/blog/straight-lines-are-a-choice/cell-02-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: on the left, how fast a blob image changes as the blob moves, for every spot the blob could sit. Inside the dashed square, where all the training blobs live, it's the same everywhere; it only drops near the edges, where part of the blob falls off the image. On the right are the two routes between blobs. Sliding (teal) keeps getting longer as they move apart, while fading (coral) stops growing at about 7 pixels. Blob World is flat, and fading is always the shorter trip.*

Both derivations hold. The renderer's metric is $$1.5708\,I = \frac{\pi}{2} I$$ at the centre for $$s = 1.5$$, 2 and 3, with zero off the diagonal. Across the training square $$[4, 24]^2$$, $$\sqrt{\det G}$$ stays between 1.4855 and 1.5708 (the small dip runs along the square's edges and is deepest at its corners, where a blob 4 pixels from the edge loses a sliver of tail), and it falls to 0.164 at an image corner. The route table matches to the last printed digit. The slide is 20.05 at $$\delta = 16$$, exactly $$\sqrt{\pi/2} \cdot 16$$, while the cross-fade has flattened out at 5.01. Even at $$\delta = 1$$ the two already differ, 1.25 against 1.23.

So the ground truth is a flat plane. Everything learned below is an approximation to it.

## Seeing the ruler: magnification and Tissot ellipses

Part 1 ended its section on charts by peeling a globe onto a flat map. On a Mercator map Greenland looks about the size of Africa, though Africa is roughly fourteen times bigger, because the projection stretches everything near the poles. Cartographers have a neat way to show this, Tissot's indicatrix: draw identical little circles all over the globe and see what they turn into on the map. Near the equator they stay round. Toward the poles they balloon. A decoder is a map projection running the other way, flat latent onto curved data, and the same trick works on it.

Take every latent step $$v$$ from a point $$z$$ that changes the image by the same small amount $$r$$. Those steps form an ellipse, $$\{z + v : v^\top G(z)\, v = r^2\}$$. Where the decoder is twitchy, a small step already reaches $$r$$ and the ellipse is small; where it's sluggish, the ellipse is large. Draw one at every point of a grid and the latent's scale bars sit side by side.

The maths: diagonalise $$G = Q\Lambda Q^\top$$ with eigenvalues $$\lambda_1 \le \lambda_2$$ and eigenvectors in the columns of $$Q$$. In rotated coordinates $$u = Q^\top v$$ the ellipse reads $$\lambda_1 u_1^2 + \lambda_2 u_2^2 = r^2$$, so its semi-axes are $$r/\sqrt{\lambda_1}$$ and $$r/\sqrt{\lambda_2}$$ along the eigenvectors. Its area is $$\pi r^2 / \sqrt{\lambda_1\lambda_2} = \pi r^2 / \sqrt{\det G}$$. So $$\sqrt{\det G(z)}$$ works as a one-number summary, the **magnification factor**: how much data-space area one unit of latent area covers. [Bishop, Svensén & Williams (1997)](https://www.microsoft.com/en-us/research/publication/magnification-factors-for-the-gtm-algorithm/) came up with it to read the distortion of the generative topographic mapping, a 1990s ancestor of today's decoders.

![Equal teal circles on a globe become unequal ellipses when peeled onto a flat map; then, on the autoencoder's star-shaped latent of blob codes, white Tissot ellipses inflate from points, small in the dense centre and long and thin out along the arms]({{ '/assets/blog/straight-lines-are-a-choice/tissot-field.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the animation shows: identical circles drawn on a globe come out as ellipses of different sizes on a flat world map, which is how mapmakers show where a map stretches. The same drawing on the model's map of blob images shows where the model stretches. Each ellipse marks the latent steps that change the image by the same amount, and they're nowhere near equal.*

On Blob World we can go a step further and predict the picture. Suppose the learned decoder draws perfect blobs, so $$g = R \circ h$$, where $$h: z \mapsto c$$ tells you which blob centre the code $$z$$ stands for and $$R: c \mapsto I_c$$ is the renderer. The chain rule gives $$J_g = J_R J_h$$, so

$$
\begin{aligned}
G_{\text{learned}}(z) &= J_h^\top J_R^\top J_R J_h \
&= J_h(z)^\top G_{\text{true}}\, J_h(z) = \frac{\pi}{2}\, J_h(z)^\top J_h(z).
\end{aligned}
$$

$$G_{\text{true}}$$ is a constant, so every bit of variation in the learned metric comes from $$J_h$$, the chart. Track where the decoded blob sits as $$z$$ moves and you can predict the decoder's metric without looking at a single pixel. Mathematicians have a name for the part of a metric that no change of coordinates can remove: the Riemann curvature tensor. On Blob World it's zero. A metric that looks curved here can only be the chart's doing.

That's a testable claim, and cell 3 tests it. It retrains Part 1's 2-D AE on the same 5,000 blobs with the same seed, maps $$\sqrt{\det G}$$ and a field of Tissot ellipses over the codes, and then compares. It computes $$h(z)$$ as the differentiable `blob_position` of the decoded image, gets $$J_h$$ by autograd, and lines up $$\frac{\pi}{2} J_h^\top J_h$$ against $$G_{\text{learned}}$$ on 1,000 codes whose true centres sit well inside the image.

```python
X_blob, C_blob = blob_world(5000, np.random.default_rng(SEED))  # Part 1's and Part 2's Blob World, same seed
ae_blob, Z_blob, mse_blob = train_ae(flat(X_blob), d=2)          # Part 1's AE(d = 2), same seed
g_blob = ae_blob.dec
print(f"AE(d=2) on Blob World: recon MSE {mse_blob:.5f} (Parts 1 and 2: 0.00039)")

# The learned ruler on a grid over the codes
lo, hi = Z_blob.min(0), Z_blob.max(0)
pad = 0.08 * (hi - lo)
g1, g2 = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], 90), np.linspace(lo[1] - pad[1], hi[1] + pad[1], 90))
grid = np.column_stack([g1.ravel(), g2.ravel()])
mag = np.sqrt(np.linalg.det(pullback_metric(g_blob, grid))).reshape(g1.shape)
code_index = NearestNeighbors(n_neighbors=1).fit(Z_blob)
spacing = np.median(NearestNeighbors(n_neighbors=2).fit(Z_blob).kneighbors(Z_blob)[0][:, 1])
on_data = (code_index.kneighbors(grid)[0][:, 0] < 6 * spacing).reshape(g1.shape)
print(f"learned sqrt(det G) over the codes: 5th-95th percentile {np.percentile(mag[on_data], 5):.3f} to "
      f"{np.percentile(mag[on_data], 95):.3f} (ratio {np.percentile(mag[on_data], 95) / np.percentile(mag[on_data], 5):.1f}x)")

# Chart, not manifold: predict G from where the decoded blob sits, h(z), using the true metric (pi/2) I
xs_t, ys_t = torch.from_numpy(xs.ravel().astype(np.float32)), torch.from_numpy(ys.ravel().astype(np.float32))
def blob_centre(z: torch.Tensor) -> torch.Tensor:
    """h(z): blob_position of the decoded image, differentiable. (n, 2) codes -> (n, 2) pixel coordinates."""
    w = F.relu(g_blob(z) - 0.1)
    return torch.stack([(w * xs_t).sum(1), (w * ys_t).sum(1)], 1) / w.sum(1, keepdim=True)

interior = np.flatnonzero(np.all((C_blob > 6) & (C_blob < 22), axis=1))[:1000]  # away from the clipped borders
G_learned = pullback_metric(g_blob, Z_blob[interior])
J_h = jacobian(blob_centre, Z_blob[interior])
G_pred = (np.pi / 2) * np.einsum("nci,ncj->nij", J_h, J_h)
rel_err = np.linalg.norm(G_learned - G_pred, axis=(1, 2)) / np.linalg.norm(G_learned, axis=(1, 2))
sd_learned, sd_pred = np.sqrt(np.linalg.det(G_learned)), np.sqrt(np.linalg.det(G_pred))
print(f"G_learned vs (pi/2) J_h^T J_h on {len(interior)} interior codes: median relative error {np.median(rel_err):.1%}, "
      f"90th percentile {np.percentile(rel_err, 90):.1%}")
print(f"  sqrt(det G): learned / predicted = {np.median(sd_learned / sd_pred):.3f} (median; 5th-95th percentile "
      f"{np.percentile(sd_learned / sd_pred, 5):.3f}-{np.percentile(sd_learned / sd_pred, 95):.3f}) | "
      f"correlation of the logs {np.corrcoef(np.log(sd_learned), np.log(sd_pred))[0, 1]:.3f}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
extent = [g1.min(), g1.max(), g2.min(), g2.max()]
im = ax1.imshow(np.where(on_data, mag, np.nan), origin="lower", extent=extent, cmap="viridis", norm=LogNorm(), aspect="auto")
ax1.scatter(*Z_blob[::10].T, s=1, color="white", alpha=0.5)
plt.colorbar(im, ax=ax1, fraction=0.046, label=r"$\sqrt{\det G(z)}$")
ax1.set_title("Magnification: how fast the image changes", fontsize=10); ax1.set_xlabel("$z_1$"); ax1.set_ylabel("$z_2$")
ax2.scatter(*Z_blob.T, s=1, color=TEAL, alpha=0.15)
cg1, cg2 = np.meshgrid(np.linspace(lo[0], hi[0], 13), np.linspace(lo[1], hi[1], 13))
cgrid = np.column_stack([cg1.ravel(), cg2.ravel()])
cgrid = cgrid[code_index.kneighbors(cgrid)[0][:, 0] < spacing * 2]
G_c = pullback_metric(g_blob, cgrid)
r_t = 0.35 * (hi - lo).min() / 13 * np.median(np.sqrt(np.linalg.eigvalsh(G_c)[:, 0]))  # ellipse size: one image change r
for c_, G_ in zip(cgrid, G_c):
    tissot(ax2, c_, G_, r_t, fc="none", ec="k", lw=1)
ax2.set_xlim(extent[:2]); ax2.set_ylim(extent[2:]); ax2.set_aspect("auto")
ax2.set_title("Tissot ellipses: latent steps that change the image equally", fontsize=10); ax2.set_xlabel("$z_1$")
ax3.loglog(sd_pred, sd_learned, ".", color=TEAL, ms=3, alpha=0.6)
lim = [min(sd_pred.min(), sd_learned.min()) * 0.8, max(sd_pred.max(), sd_learned.max()) * 1.2]
ax3.plot(lim, lim, "k--", lw=1, label="learned = predicted")
ax3.set_xlabel(r"predicted from the blob's motion: $\sqrt{\det(\frac{\pi}{2} J_h^\top J_h)}$")
ax3.set_ylabel(r"learned $\sqrt{\det G}$"); ax3.legend(); ax3.set_title("Flat manifold, distorted chart", fontsize=10)
plt.tight_layout(); plt.show()
```
```text
AE(d=2) on Blob World: recon MSE 0.00039 (Parts 1 and 2: 0.00039)
learned sqrt(det G) over the codes: 5th-95th percentile 0.208 to 1.604 (ratio 7.7x)
G_learned vs (pi/2) J_h^T J_h on 1000 interior codes: median relative error 11.5%, 90th percentile 21.9%
  sqrt(det G): learned / predicted = 0.911 (median; 5th-95th percentile 0.813-1.061) | correlation of the logs 0.996
```
![Output of cell 3]({{ '/assets/blog/straight-lines-are-a-choice/cell-03-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: on the left, the model's map of blob images, coloured by how fast the image changes as you move. It's fastest in the bright middle and slowest out along the four arms. In the middle, the same map with ellipses marking latent steps that change the image equally: small at the centre, long and thin on the arms. On the right, one dot per training image, with the stretching predicted from the blob's motion across and the stretching the model actually has up. They sit on a line, so the uneven stretching is the model's choice of map.*

Reconstruction error is 0.00039, as in Parts 1 and 2, so this is the same chart. And its ruler is all over the place. Across the codes, $$\sqrt{\det G}$$ runs from 0.208 to 1.604 between the 5th and 95th percentiles, a factor of 7.7.

The left panel shows why. The AE bent the square of blob centres into a four-armed star. The middle of the square got squeezed into a small bright patch, where a tiny latent step moves the blob a lot, and the corners got stretched out into arms, where a big step barely moves it. The ellipses add direction. Along an arm they're long and thin, because walking along the arm changes the image slowly and walking across it changes it fast.

The prediction holds up well. Built only from how the decoded blob moves, $$\frac{\pi}{2} J_h^\top J_h$$ tracks the learned $$\sqrt{\det G}$$ with a correlation of 0.996 between their logs, across the whole 7.7-fold range. Their median ratio is 0.911 (90% of codes fall between 0.813 and 1.061), and the median relative error on the full $$2 \times 2$$ matrix is 11.5%. Why a bit under 1? The decoder's blobs aren't perfect copies of the renderer's, so nudging one changes the image slightly less than nudging a real blob would. That leftover is small and roughly constant.

This is the question the latent-geometry papers keep circling. When a latent space looks curved, with a varying metric and bent geodesics, is the data manifold curved or is the chart distorted? On Blob World we know, and it's the chart. Checking the metric against the truth is enough to tell.

## Geodesics as rubber bands

Finding the shortest path has a nice physical picture. Pin a rubber band at two points on the latent map and let go. Tension pulls it toward whatever route is shortest by the decoder's measure, and since a stretched band spreads its tension evenly, its points end up evenly spaced in the data. On the magnification map you'd see the band peel off the straight latent line and bend toward cheap regions, wherever the detour costs less than it saves.

Computing it is easy. Chop the path into $$n$$ segments, $$z_0 = z_a, z_1, \dots, z_n = z_b$$, with $$\Delta t = 1/n$$. The velocity on segment $$i$$ is about $$n(z_{i+1} - z_i)$$, and the Taylor expansion says $$J\,(z_{i+1} - z_i) \approx g(z_{i+1}) - g(z_i)$$. Substitute into the energy integral:

$$
\begin{aligned}
E &\approx \frac{1}{2}\sum_{i=0}^{n-1} \frac{1}{n}\, n^2 \|g(z_{i+1}) - g(z_i)\|^2 \
&= \frac{n}{2}\sum_{i=0}^{n-1} \|g(z_{i+1}) - g(z_i)\|^2 .
\end{aligned}
$$

That's the whole rubber band: the sum of squared pixel-space gaps between neighbouring decoded frames. You never form a Jacobian, because autograd goes straight through the decoder. The `geodesic` function in cell 1 pins both endpoints, starts the $$n - 1$$ interior points on the straight line, and nudges them with Adam to shrink the sum. Twenty lines. Squaring the gaps punishes one big jump more than several small ones, which is why the frames come out evenly spaced (Cauchy-Schwarz again, in discrete form).

Cell 4 tries it on cell 3's AE with two pairs of blobs. The first is Part 2's ghost-versus-hybrid pair at opposite corners, $$(6, 6)$$ and $$(22, 22)$$. Part 2 noticed the hybrid blob moving unevenly along that straight latent line and left it as a loose end for this post. I added the second pair, $$(5, 6)$$ to $$(23, 12)$$, after seeing what the first one did; the reason is coming up. My prediction going in: if the decoder's ruler is close to the true one, the blob on a geodesic should slide in a straight line at constant pace and cover $$\sqrt{\pi/2}\,\delta$$ in pixel space.

```python
pairs_blob = {"Part 2's pair": ((6.0, 6.0), (22.0, 22.0)),   # opposite corners, along the chart's diagonal
              "off-axis pair": ((5.0, 6.0), (23.0, 12.0))}   # no symmetry to keep the band straight
n = 64
runs = {}
for label, (a, b) in pairs_blob.items():
    with torch.no_grad():
        z_a, z_b = ae_blob.enc(torch.from_numpy(flat(np.stack([blob(*a), blob(*b)])))).numpy()
    paths = {"LERP": lerp(z_a, z_b, np.linspace(0, 1, n + 1))}
    paths["geodesic"], energy_log = geodesic(g_blob, z_a, z_b, n=n, iters=1500)
    delta = np.hypot(b[0] - a[0], b[1] - a[1])
    print(f"{label}: blob {a} -> {b}, delta = {delta:.1f} px | true slide length sqrt(pi/2) delta = {np.sqrt(np.pi / 2) * delta:.2f}"
          f" | energy {energy_log[0]:.0f} -> {energy_log[-1]:.0f}")
    tracks = {}
    for name, path in paths.items():
        with torch.no_grad():
            frames = g_blob(as_tensor(path)).numpy().reshape(-1, H, W)
        tracks[name] = blob_position(frames)
        L, E = path_length(g_blob, path), 0.5 * n * (decoded_steps(g_blob, path) ** 2).sum()
        u, rel = np.subtract(b, a) / delta, tracks[name] - np.array(a)
        stray = np.abs(u[0] * rel[:, 1] - u[1] * rel[:, 0]).max()   # distance from the straight segment's line
        print(f"  {name:8s}: data-space length {L:5.2f} | L^2/2E {L**2 / (2 * E):.3f} | blob {speed_report(tracks[name])}"
              f" | strays {stray:.2f} px off the straight line")
    runs[label] = (paths, tracks, energy_log, a, b)

fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
for label, ls in [("Part 2's pair", "-"), ("off-axis pair", ":")]:
    axes[0].semilogy(runs[label][2], color=GREEN, lw=2, ls=ls, label=label)
axes[0].set_xlabel("Adam iteration"); axes[0].set_ylabel("discrete energy"); axes[0].legend(fontsize=8)
axes[0].set_title("The band relaxes", fontsize=10)
axes[1].imshow(np.where(on_data, mag, np.nan), origin="lower", extent=extent, cmap="viridis", norm=LogNorm(), aspect="auto")
for label, (paths, tracks, _, a, b) in runs.items():
    for name, colour in [("LERP", ORANGE), ("geodesic", GREEN)]:
        axes[1].plot(*paths[name].T, color=colour, lw=2.5, label=name if label == "off-axis pair" else None)
        axes[2].plot(*tracks[name].T, "o-", color=colour, ms=2.5, lw=1, label=name if label == "off-axis pair" else None)
    axes[2].plot(*np.array([a, b]).T, "k--", lw=0.8)
axes[1].legend(fontsize=8); axes[1].set_xlabel("$z_1$"); axes[1].set_title("In the latent: the band bends off-axis", fontsize=10)
axes[2].set_xlim(0, 27); axes[2].set_ylim(27, 0); axes[2].set_aspect("equal"); axes[2].legend(fontsize=8)
axes[2].set_xlabel("x (pixels)"); axes[2].set_ylabel("y (pixels)"); axes[2].set_title("In the image: the blob goes straight", fontsize=10)
paths, tracks, _, a, b = runs["Part 2's pair"]
for name, colour in [("LERP", ORANGE), ("geodesic", GREEN)]:
    axes[3].plot(np.linspace(0, 1, n), np.linalg.norm(np.diff(tracks[name], axis=0), axis=1), color=colour, lw=2.5, label=name)
axes[3].axhline(np.hypot(b[0] - a[0], b[1] - a[1]) / n, color="k", ls="--", lw=1, label="constant speed")
axes[3].set_xlabel("t"); axes[3].set_ylabel("pixels moved per step"); axes[3].legend(fontsize=8)
axes[3].set_title("Part 2's pair: same route, even pace", fontsize=10)
plt.tight_layout(); plt.show()
```
```text
Part 2's pair: blob (6.0, 6.0) -> (22.0, 22.0), delta = 22.6 px | true slide length sqrt(pi/2) delta = 28.36 | energy 553 -> 348
  LERP    : data-space length 26.42 | L^2/2E 0.631 | blob steps 0.13-1.17 px, std/mean 0.78 | strays 0.98 px off the straight line
  geodesic: data-space length 26.38 | L^2/2E 1.000 | blob steps 0.32-0.39 px, std/mean 0.04 | strays 0.11 px off the straight line
off-axis pair: blob (5.0, 6.0) -> (23.0, 12.0), delta = 19.0 px | true slide length sqrt(pi/2) delta = 23.78 | energy 339 -> 255
  LERP    : data-space length 24.10 | L^2/2E 0.857 | blob steps 0.14-0.57 px, std/mean 0.41 | strays 2.63 px off the straight line
  geodesic: data-space length 22.60 | L^2/2E 1.000 | blob steps 0.28-0.33 px, std/mean 0.04 | strays 0.55 px off the straight line
```
![Output of cell 4]({{ '/assets/blog/straight-lines-are-a-choice/cell-04-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: the rubber band's energy falling as it relaxes, for both pairs. Next to it, the model's map with the straight latent paths in orange and the relaxed ones in green; for the second pair, the green one swings up an arm. Then where the blob actually goes in the image, where the green paths follow the dashed straight lines and the orange ones wander. Last, how far the blob moves per step for the first pair. Orange crawls, spikes in the middle and crawls again, while green stays flat. The path that bends on the map is the one that moves the blob straight and evenly.*

Part 2's pair keeps its route and fixes its pace. It sits on the star's diagonal, a symmetry axis of the chart, so nothing tugs the band sideways. The geodesic stays on LERP's line (data-space length 26.38 against 26.42) and just redistributes its frames. LERP's blob moves 0.13 pixels on some steps and 1.17 on others, a spread of 0.78 of the mean, because the straight latent line dawdles in the slow outer arms and sprints through the fast middle. The geodesic's blob moves between 0.32 and 0.39 pixels every step, spread 0.04, and $$L^2 / 2E$$ goes from 0.631 to 1.000. Part 2's loose end is tied off: the hybrid's uneven speed was the chart's distortion all along.

That's also why I wanted a second pair. On the diagonal the route can't bend, so the only thing to see is pacing. Off-axis, the straight latent line cuts across the star at an angle, and the band bends up the nearer arm before coming back. In the latent that looks curved. In the image it's the straight one: LERP's blob strays 2.63 pixels off the segment between the two blob positions, the geodesic's only 0.55, and the decoded length drops from 24.10 to 22.60. The pace evens out here too, from a spread of 0.41 to 0.04.

One wrinkle. The geodesics' decoded lengths come in 5 to 7% under $$\sqrt{\pi/2}\,\delta$$ (22.60 against 23.78, 26.38 against 28.36), because the learned decoder's blobs sit slightly off the true manifold, as cell 3 found, and its ruler reads a little short. LERP's wandering off-axis path even comes out slightly longer than the true slide, 24.10 against 23.78. Measured with that same ruler, the geodesic beats LERP on everything.

So a geodesic that curves in latent coordinates tells you nothing, on its own, about whether the manifold is curved. Here the manifold is flat and the geodesic is a straight slide. The bend is a distorted chart drawing a straight line.

## Part 3's cliffhanger, solved

Back to the path that started all this. Cell 5 retrains Part 3's Blob World VAE (16 latent dimensions, same seed, same epochs) and rebuilds its exact SLERP path from the same pair of prior samples. Part 3 found this VAE only uses 4 of its 16 dimensions. The other 12 are switched off: the decoder ignores them, they cost nothing under its ruler, and a band could wander along them for free. The solver adds a tiny penalty, $$10^{-3}$$ times the squared latent step, to stop that without touching anything the decoder sees.

I expected this to be the easy section. Run the band, watch the blob even out, done. The cell also checks something Part 3 never did, though: what the frames along that path actually look like. Real Blob World images all peak at an intensity of 1, and Part 2's squared distance $$d^2$$ to the nearest real image says how far a frame is from being one.

```python
vae_blob, _ = train_vae(flat(X_blob), d=16)        # Part 3's Blob World VAE(d = 16): same class, seed and epochs
g_vae16 = vae_blob.decode
with torch.no_grad():
    mu_b = vae_blob.encode(torch.from_numpy(flat(X_blob)))[0].numpy()
active_b = active_units(mu_b)
za, zb = np.random.default_rng(4).standard_normal((2, 300, 16))   # Part 3's prior pairs; its path ran za[0] -> zb[0]
t41 = np.linspace(0, 1, 41)
part3 = {"SLERP": slerp(za[0], zb[0], t41)}
# The 12 switched-off dimensions cost nothing in G; a tiny ridge keeps the band from wandering along them
part3["geodesic"], _ = geodesic(g_vae16, za[0], zb[0], n=40, iters=2000, ridge=1e-3)
with torch.no_grad():
    frames3 = {k: g_vae16(as_tensor(v)).numpy().reshape(-1, H, W) for k, v in part3.items()}
print(f"Part 3's VAE(d = 16): {active_b.sum()} active units (Part 3: 4)")
print(f"Part 3's SLERP, Part 3's estimator: blob {speed_report(centre_of_mass(frames3['SLERP']))}  (Part 3: 0.08-0.61, 0.40)")
track3 = blob_position(frames3["SLERP"])
print(f"  what it was walking between: start peak {frames3['SLERP'][0].max():.2f}, d^2 to M {dist2_to_manifold(frames3['SLERP'][:1])[0]:.2f}"
      f" (the pixel ghost: 6.28) | end peak {frames3['SLERP'][-1].max():.2f} | blob x goes {track3[0, 0]:.1f} -> "
      f"{track3[:, 0].max():.1f} -> {track3[-1, 0]:.1f} (overshoot)")
for name, fr in frames3.items():
    print(f"  {name:8s} in Part 3's VAE: data-space length {path_length(g_vae16, part3[name]):.2f} | faintest frame peak "
          f"{fr.max((1, 2)).min():.2f} | blob {speed_report(blob_position(fr))}")

# The same two blob positions, drawn by a decoder that renders sharp blobs: Part 1's AE from cell 3
pa, pb = track3[0], track3[-1]
with torch.no_grad():
    z_a, z_b = ae_blob.enc(torch.from_numpy(flat(np.stack([blob(*pa), blob(*pb)])))).numpy()
sharp = {"LERP": lerp(z_a, z_b, t41)}
sharp["geodesic"], _ = geodesic(g_blob, z_a, z_b, n=40, iters=1500)
with torch.no_grad():
    frames_s = {k: g_blob(as_tensor(v)).numpy().reshape(-1, H, W) for k, v in sharp.items()}
gap = np.linalg.norm(pb - pa)
for name, fr in frames_s.items():
    tr = blob_position(fr)
    print(f"  {name:8s} in Part 1's AE: data-space length {path_length(g_blob, sharp[name]):5.2f} (sqrt(pi/2) x {gap:.1f} px = "
          f"{np.sqrt(np.pi / 2) * gap:.2f}) | faintest frame peak {fr.max((1, 2)).min():.2f} | blob {speed_report(tr)}")

fig = plt.figure(figsize=(16, 7.6))
gs = fig.add_gridspec(4, 9, height_ratios=[3.2, 1, 1, 1], hspace=0.45)
axes = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6]), fig.add_subplot(gs[0, 6:9])]
rows = [("Part 3's VAE, SLERP", frames3["SLERP"], BLUE, "-"), ("Part 3's VAE, geodesic", frames3["geodesic"], GREEN, ":"),
        ("sharp AE, geodesic", frames_s["geodesic"], GREEN, "-")]
for name, fr, colour, ls in rows:
    tr = blob_position(fr)
    axes[0].plot(*tr.T, color=colour, ls=ls, lw=2, marker="o", ms=2.5, label=name)
    axes[1].plot(t41, fr.max((1, 2)), color=colour, ls=ls, lw=2.5, label=name)
    axes[2].plot(t41[1:] - 0.0125, np.linalg.norm(np.diff(tr, axis=0), axis=1), color=colour, ls=ls, lw=2.5, label=name)
axes[0].set_xlim(0, 27); axes[0].set_ylim(27, 0); axes[0].set_aspect("equal"); axes[0].legend(fontsize=8)
axes[0].set_xlabel("x (pixels)"); axes[0].set_ylabel("y (pixels)"); axes[0].set_title("Where the blob goes", fontsize=10)
axes[1].axhline(1, color=TEAL, ls="--", lw=1, label="every real blob")
axes[1].set_ylim(0, 1.1); axes[1].set_xlabel("t"); axes[1].set_ylabel("peak intensity of the frame"); axes[1].legend(fontsize=8)
axes[1].set_title("Part 3's decoder draws faint blobs; its geodesic fades further", fontsize=10)
axes[2].set_xlabel("t"); axes[2].set_ylabel("pixels moved per step"); axes[2].legend(fontsize=8)
axes[2].set_title("Only a sharp decoder's geodesic moves evenly", fontsize=10)
for row, (name, fr, colour, _) in enumerate(rows, start=1):
    for j, k in enumerate(np.linspace(0, 40, 9).astype(int)):
        ax = fig.add_subplot(gs[row, j]); ax.imshow(fr[k], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(name.replace(", ", "\n"), color=colour, fontsize=8)
plt.show()
```
```text
Part 3's VAE(d = 16): 4 active units (Part 3: 4)
Part 3's SLERP, Part 3's estimator: blob steps 0.08-0.61 px, std/mean 0.40  (Part 3: 0.08-0.61, 0.40)
  what it was walking between: start peak 0.21, d^2 to M 8.42 (the pixel ghost: 6.28) | end peak 0.46 | blob x goes 11.4 -> 23.8 -> 21.5 (overshoot)
  SLERP    in Part 3's VAE: data-space length 2.01 | faintest frame peak 0.21 | blob steps 0.01-0.92 px, std/mean 0.87
  geodesic in Part 3's VAE: data-space length 1.58 | faintest frame peak 0.18 | blob steps 0.01-1.02 px, std/mean 0.93
  LERP     in Part 1's AE: data-space length 12.41 (sqrt(pi/2) x 10.1 px = 12.67) | faintest frame peak 0.82 | blob steps 0.16-0.35 px, std/mean 0.20
  geodesic in Part 1's AE: data-space length 12.26 (sqrt(pi/2) x 10.1 px = 12.67) | faintest frame peak 0.84 | blob steps 0.23-0.28 px, std/mean 0.05
```
![Output of cell 5]({{ '/assets/blog/straight-lines-are-a-choice/cell-05-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: three walks between the same two blob positions. Blue is Part 3's original curved walk, dotted green is the shortest path according to Part 3's model, and solid green is the shortest path according to the sharper model from cell 3. The panels show where the blob goes, how bright each frame is (real blobs have brightness 1), and how far the blob moves per step. Part 3's model draws faint blobs and its shortest path fades them further; only the sharp model's shortest path keeps an even pace.*

Part 3's number comes back exactly: with its estimator, SLERP's blob moves 0.08 to 0.61 pixels per step, spread 0.40. The frames are where it gets interesting. The walk starts at a prior sample that decodes to a faint smudge, peak intensity 0.21, sitting at $$d^2 = 8.42$$ from the nearest real image. That's further off the manifold than Part 2's pixel ghost, at 6.28. The end point is faint too, with peak 0.46. And with the corrected `blob_position`, the blob's horizontal position goes from 11.4 out to 23.8 and back to 21.5. It overshoots and comes back.

So Part 3's lurching was only partly a chart effect. The walk began somewhere the decoder never drew a real blob.

The rubber band doesn't rescue it, and the reason is the most useful thing in this section. Under this decoder's ruler the geodesic is shorter than SLERP, 1.58 against 2.01, and it gets there by fading: its faintest frame drops to 0.18, and its pace is as ragged as ever, spread 0.93. The band did exactly what it was asked. This decoder draws blurry, faint blobs, and between faint images fading is cheaper than sliding. That's cell 2's chord-versus-arc inequality biting.

So the last two lines of the output keep the endpoints and swap the decoder. Same two blob positions, read off Part 3's first and last frames, now drawn by cell 3's AE, which renders sharp blobs. Its LERP is already better (spread 0.20), and the geodesic brings it down to 0.05. The blob moves between 0.23 and 0.28 pixels every step, never dimmer than 0.84, along a decoded path of 12.26 against the true slide's 12.67. That's the walk from the top of the post.

Which brings back the line this whole series hangs on: a path through latent space is only as good as the region the decoder was trained on. SLERP fixed where Part 3's walk went. A geodesic fixes how it walks, but only if the decoder's ruler is measuring real images. If it isn't, the band takes whatever shortcut the decoder hands it. Fading was one. Ghosts are another.

## Holes become mountains

Hand a route planner a map where someone forgot to draw a lake and it'll send you straight across the water. The planner did nothing wrong; the map did. A decoder's metric can fail the same way. In a region with no training data it reports whatever the network happens to do there, and if that's cheap, the geodesic cuts right through.

[Part 2](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/ghosts-and-hybrids/) set up exactly this. Holey Blob World throws out every blob centred within 5 pixels of the image centre, so the manifold is an annulus, a square with a disc punched out of it. A 2-D AE maps it to a ring of codes with an empty cross in the middle. Part 2 walked a straight latent line from a blob at $$(6, 14)$$ to one at $$(22, 14)$$, right across the cross. The decoded blob never entered the hole in the image. It climbed around the top, then tore, showing two blobs at once for a stretch of frames.

Cell 2's inequality says what a rubber band will do here. On the manifold, the only way across is around. The blob's centre can't come within 5 pixels of $$(14, 14)$$, so the shortest slide hugs a tangent to that circle, runs around the arc and leaves on the other tangent: $$2\sqrt{8^2 - 5^2} + 5\big(\pi - 2\arccos\tfrac{5}{8}\big) \approx 19.2$$ pixels of travel, $$\sqrt{\pi/2} \times 19.2 \approx 24.1$$ in pixel space. A cross-fade costs at most 5.01. Somewhere in the latent hole a deterministic decoder blends the two sides, every such ghost frame is part of that shortcut, and a band that can only see the decoder's ruler should go straight for it.

[Arvanitidis, Hansen & Hauberg (2018)](https://arxiv.org/abs/1710.11379) have a fix: let the decoder say "I don't know". It outputs a mean and a standard deviation per pixel, $$g(z) = \mu(z) + \sigma(z) \odot \epsilon$$ with $$\epsilon \sim \mathcal{N}(0, I_D)$$, and $$\sigma(z)$$ is made large away from the data. Now the Jacobian is random, $$J = J_\mu + \operatorname{diag}(\epsilon)\, J_\sigma$$, and averaging the metric over the noise gives

$$
\begin{aligned}
\mathbb{E}_\epsilon\big[J^\top J\big] &= J_\mu^\top J_\mu + J_\sigma^\top \mathbb{E}\big[\operatorname{diag}(\epsilon)^2\big] J_\sigma \
&\quad + (\text{terms linear in } \epsilon) \
&= J_\mu^\top J_\mu + J_\sigma^\top J_\sigma,
\end{aligned}
$$

since $$\mathbb{E}[\epsilon_j] = 0$$ and $$\mathbb{E}[\epsilon_j^2] = 1$$. The second term is the new part. Where $$\sigma$$ climbs from "confident" to "no idea", $$J_\sigma$$ is big, so crossing that boundary is expensive. The hole turns into a mountain range, and the band goes around. Handily, the expected metric is also just the pull-back metric of the stacked map $$z \mapsto (\mu(z), \sigma(z))$$, so cell 6 needs no new machinery: it hands the same `geodesic` function a decoder that outputs the mean image and the $$\sigma$$ image side by side.

For $$\sigma(z)$$ the cell follows the paper's recipe, an RBF network on the precision $$\beta = 1/\sigma^2$$:

$$
\begin{gathered}
\beta(z) = W\, v(z) + \zeta, \qquad v_k(z) = \exp\!\big(-\lambda_k \|z - c_k\|^2\big), \
\sigma(z) = \beta(z)^{-1/2}.
\end{gathered}
$$

The 256 centres $$c_k$$ come from k-means on the codes, each bump is as wide as its cluster, and the weights $$W \ge 0$$ are fitted by the Gaussian negative log-likelihood of the frozen AE's reconstruction errors. $$\zeta$$ sets the ceiling. Near the codes the bumps add up to a high precision and $$\sigma$$ is small; far from every code they all fade, $$\beta \to \zeta$$, and $$\sigma \to \zeta^{-1/2}$$. The paper keeps $$\zeta$$ tiny, just enough to avoid dividing by zero. I set $$\zeta = 1$$, so "no idea" means a standard deviation of 1 per pixel on images in $$[0, 1]$$, which is about as clueless as a decoder can get.

One practical note. Energy minimisation is local, and a band that starts straight across a mountain can get stuck on the ridge. So the cell starts three bands (straight across, bowed over the top of the ring, bowed under the bottom) and keeps the one with the lowest final energy. Both metrics get all three starts, so the plain decoder has every chance to go around if that's cheaper.

```python
X_holey, C_holey = holey_blob_world(5000, np.random.default_rng(SEED))  # Part 2's Holey Blob World, same seed
ae_holey, Z_holey, mse_holey = train_ae(flat(X_holey), d=2)
g_holey = ae_holey.dec
a, b = (6.0, 14.0), (22.0, 14.0)  # Part 2's walk: opposite sides of the hole
with torch.no_grad():
    z_a, z_b = ae_holey.enc(torch.from_numpy(flat(np.stack([blob(*a), blob(*b)])))).numpy()
print(f"AE(d=2) on Holey Blob World: recon MSE {mse_holey:.5f} (Part 2: 0.00053)")

# Decoder uncertainty (Arvanitidis et al. 2018): a precision that is high near the codes and falls to zeta far away.
# beta(z) = W v(z) + zeta with RBF features v_k(z) = exp(-lambda ||z - c_k||^2); sigma(z) = beta(z)^(-1/2)
from sklearn.cluster import KMeans
km = KMeans(256, n_init=4, random_state=SEED).fit(Z_holey)
rbf_c = torch.from_numpy(km.cluster_centers_.astype(np.float32))
h_k = np.array([np.linalg.norm(Z_holey[km.labels_ == k] - km.cluster_centers_[k], axis=1).mean() for k in range(256)])
lam = torch.from_numpy((0.5 / h_k**2).astype(np.float32))   # each bump as wide as its cluster
zeta = 1.0                                                   # far from every code, sigma -> 1 per pixel: "no idea"
rbf = lambda z: torch.exp(-lam * torch.cdist(z, rbf_c) ** 2)
log_W = torch.full((256, H * W), 5.0, requires_grad=True)
sigma_net = lambda z: (rbf(z) @ log_W.exp() + zeta) ** -0.5
Zt, Xt_h = torch.from_numpy(Z_holey), torch.from_numpy(flat(X_holey))
with torch.no_grad():
    resid2 = (g_holey(Zt) - Xt_h) ** 2                  # the decoder mean is frozen; only the variance is fitted
fit = torch.from_numpy(np.random.default_rng(SEED).choice(len(Zt), 2000, replace=False))  # 2,000 codes are plenty
opt = torch.optim.Adam([log_W], lr=0.05)
for _ in range(300):                                   # Gaussian negative log-likelihood of the residuals
    prec = rbf(Zt[fit]) @ log_W.exp() + zeta
    nll = 0.5 * (prec * resid2[fit] - torch.log(prec)).sum(1).mean()
    opt.zero_grad(); nll.backward(); opt.step()
log_W.requires_grad_(False)
g_stoch = lambda z: torch.cat([g_holey(z), sigma_net(z)], 1)   # E[G] = J_mu^T J_mu + J_sigma^T J_sigma
with torch.no_grad():
    along = sigma_net(as_tensor(lerp(z_a, z_b, np.linspace(0, 1, 65)))).mean(1)
    print(f"fitted sigma: {sigma_net(Zt).mean():.4f} per pixel on the codes (residual RMS {resid2.mean().sqrt():.4f}) | "
          f"up to {along.max():.3f} where the straight line crosses the latent hole")

# The band starts from three places: straight across, and bowed over the top or bottom of the ring; keep the lowest energy
top, bottom = [Z_holey[np.argmin(np.linalg.norm(C_holey - np.array(c), axis=1))] for c in ((14, 6), (14, 22))]
n = 48
tt = np.linspace(0, 1, n + 1)[:, None]
bow = lambda mid: np.where(tt < 0.5, (1 - 2 * tt) * z_a + 2 * tt * mid, (2 - 2 * tt) * mid + (2 * tt - 1) * z_b)
inits = {"straight": lerp(z_a, z_b, tt[:, 0]), "over the top": bow(top), "under the bottom": bow(bottom)}
holey = {"LERP": inits["straight"]}
for label, g_ in [("geodesic, deterministic", g_holey), ("geodesic, with uncertainty", g_stoch)]:
    runs = {k: geodesic(g_, z_a, z_b, n=n, iters=700, init=v) for k, v in inits.items()}
    best = min(runs, key=lambda k: runs[k][1][-1])
    print(f"{label}: final energy " + ", ".join(f"{k} {runs[k][1][-1]:.1f}" for k in runs) + f" -> keep '{best}'")
    holey[label] = runs[best][0]

for name, path in holey.items():
    with torch.no_grad():
        fr = g_holey(as_tensor(path)).numpy().reshape(-1, H, W)
    peak = np.array([np.unravel_index(f.argmax(), f.shape)[::-1] for f in fr])
    spots = np.array([bright_spots(f) for f in fr])
    print(f"{name:27s}: length {path_length(g_holey, path):5.2f} | brightest pixel in the hole {(np.linalg.norm(peak - 14, axis=1) < 5).sum():2d}/{n + 1}"
          f" | frames with 2+ blobs {(spots >= 2).sum():2d}/{n + 1} | max d^2 to M {dist2_to_manifold(fr).max():.3f}")
around = 2 * np.sqrt(8**2 - 5**2) + 5 * (np.pi - 2 * np.arccos(5 / 8))  # tangent, arc of the r0 = 5 circle, tangent
print(f"for reference: shortest slide around the hole on M = sqrt(pi/2) x {around:.1f} px = {np.sqrt(np.pi / 2) * around:.2f} | "
      f"cross-fade {np.linalg.norm(blob(*a) - blob(*b)):.2f}")

fig = plt.figure(figsize=(15, 8.2))
gs = fig.add_gridspec(4, 9, height_ratios=[3.4, 1, 1, 1], hspace=0.4)
ax1, ax2 = fig.add_subplot(gs[0, 0:4]), fig.add_subplot(gs[0, 5:9])
lo_h, hi_h = Z_holey.min(0) - 0.1 * np.ptp(Z_holey, 0), Z_holey.max(0) + 0.1 * np.ptp(Z_holey, 0)
hg1, hg2 = np.meshgrid(np.linspace(lo_h[0], hi_h[0], 120), np.linspace(lo_h[1], hi_h[1], 120))
with torch.no_grad():
    sig_map = sigma_net(as_tensor(np.column_stack([hg1.ravel(), hg2.ravel()]))).mean(1).numpy().reshape(hg1.shape)
im = ax1.imshow(sig_map, origin="lower", extent=[lo_h[0], hi_h[0], lo_h[1], hi_h[1]], cmap=HOLE_CMAP, norm=LogNorm(), aspect="auto")
plt.colorbar(im, ax=ax1, fraction=0.046, label=r"decoder uncertainty $\sigma(z)$")
ax1.scatter(*Z_holey[::4].T, s=1, color=TEAL, alpha=0.4)
styles = {"LERP": (ORANGE, "-"), "geodesic, deterministic": ("k", "--"), "geodesic, with uncertainty": (GREEN, "-")}
for name, path in holey.items():
    ax1.plot(*path.T, color=styles[name][0], ls=styles[name][1], lw=2.5, label=name)
ax1.legend(fontsize=8, loc="lower right"); ax1.set_xlabel("$z_1$"); ax1.set_ylabel("$z_2$")
ax1.set_title("The latent: codes (teal) and the uncertainty mountain", fontsize=10)
ax2.add_patch(Circle((14, 14), 5, color=CORAL, alpha=0.25))
ax2.scatter(*C_holey[::4].T, s=1, color=TEAL, alpha=0.4)
for row, (name, path) in enumerate(holey.items(), start=1):
    with torch.no_grad():
        fr = g_holey(as_tensor(path)).numpy().reshape(-1, H, W)
    peak = np.array([np.unravel_index(f.argmax(), f.shape)[::-1] for f in fr])
    ax2.plot(*peak.T, color=styles[name][0], ls=styles[name][1], lw=2, marker="o", ms=2.5, label=name)
    for j, k in enumerate(np.linspace(0, n, 9).astype(int)):
        ax = fig.add_subplot(gs[row, j]); ax.imshow(fr[k], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if bright_spots(fr[k]) >= 2:
            for sp in ax.spines.values():
                sp.set_color(CORAL); sp.set_linewidth(3)
        if j == 0:
            ax.set_ylabel(name.replace("geodesic, ", ""), color=styles[name][0], fontsize=9)
ax2.set_xlim(0, 27); ax2.set_ylim(27, 0); ax2.set_aspect("equal"); ax2.legend(fontsize=8, loc="lower right")
ax2.set_title("The image: brightest pixel of each frame; the hole in coral", fontsize=10)
plt.show()
```
```text
AE(d=2) on Holey Blob World: recon MSE 0.00053 (Part 2: 0.00053)
fitted sigma: 0.0157 per pixel on the codes (residual RMS 0.0229) | up to 0.775 where the straight line crosses the latent hole
geodesic, deterministic: final energy straight 280.3, over the top 366.6, under the bottom 378.7 -> keep 'straight'
geodesic, with uncertainty: final energy straight 2908.5, over the top 406.3, under the bottom 425.0 -> keep 'over the top'
LERP                       : length 25.08 | brightest pixel in the hole  0/49 | frames with 2+ blobs 10/49 | max d^2 to M 6.232
geodesic, deterministic    : length 23.67 | brightest pixel in the hole  0/49 | frames with 2+ blobs 32/49 | max d^2 to M 9.848
geodesic, with uncertainty : length 27.49 | brightest pixel in the hole  0/49 | frames with 2+ blobs  0/49 | max d^2 to M 0.474
for reference: shortest slide around the hole on M = sqrt(pi/2) x 19.2 px = 24.12 | cross-fade 5.01
```
![Output of cell 6]({{ '/assets/blog/straight-lines-are-a-choice/cell-06-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: on the left, the model's map of blob images with a hole in the middle. Teal dots are training images, and the shading is the model's uncertainty, pale where it has seen data and dark red where it hasn't. Three walks cross it: the straight line (orange), the shortest path by the plain ruler (black dashes) and the shortest path once uncertainty counts (green). On the right are the same walks in the image, traced by their brightest pixel, with the empty disc in pink. The strips show nine frames of each, two-blob ghosts outlined in coral. With the plain ruler the shortest path runs through ghosts; with uncertainty it goes around.*

The AE reproduces Part 2's model (reconstruction error 0.00053) and Part 2's tear: 10 of LERP's 49 frames show two blobs, and its worst frame is at $$d^2 = 6.23$$.

Then the plain geodesic does exactly what cell 2 predicted, and it's worse than LERP. Its straight start wins, at an energy of 280 against 367 and 379 for the two routes around. In it goes, straight through the hole. Its decoded path is shorter than LERP's, 23.67 against 25.08, and it pays for that in ghosts: 32 of 49 frames show two blobs, the worst at $$d^2 = 9.85$$. By the decoder's own ruler this is the best path between the two images, and two thirds of its frames aren't blob images.

Uncertainty fixes it. The fitted $$\sigma$$ is 0.016 per pixel on the codes, close to the AE's actual per-pixel error (an RMS of 0.023), and it climbs to 0.775 where the straight line crosses the latent hole. The straight band's energy jumps to 2,909, against 406 over the top, so the band goes around. Not one frame on that path shows two blobs, the worst sits at $$d^2 = 0.474$$, and the decoded length is 27.49.

That's about 14% more than the shortest possible slide around the hole (24.12), partly because the band keeps its distance from the mountain: in the image, its brightest pixel traces a wider arc than the tight tangent-and-arc route. On every path, ghostly ones included, the brightest pixel never lands inside the hole itself, just as in Part 2.

That closes the loop Part 2 left open. A plain decoder's metric only measures how fast the image changes, and across a hole a blend changes it slowly, so the metric can't see the hole at all. Give the decoder a sense of where it was trained and the hole gets expensive. Then the shortest path is the one that stays on the data.

## MNIST: do geodesics matter on real data?

There's no renderer for handwritten digits and no known flat chart to check against, so on MNIST the questions get practical. How much shorter is a geodesic than the straight latent line? And does the shorter path look any better?

The papers disagree. [Arvanitidis et al. (2018)](https://arxiv.org/abs/1710.11379) argue that a decoder distorts its latent space enough for straight lines to be poor paths, and that the decoder's uncertainty belongs in the metric. [Shao, Kumar & Fletcher (2017)](https://arxiv.org/abs/1711.08014) measured geodesics in trained generators and found them close to straight, so linear interpolation was a decent approximation in their models. Both can be right, since they're talking about different models and different metrics. Which is a good argument for measuring your own.

Cell 7 brings back Part 2's MNIST VAE (2-D latent, Bernoulli likelihood, $$\beta = 1$$, seed 0, the same 10,000 images), retrained and checked against the four numbers Part 2 published. The decoder $$g$$ is its mean image, the sigmoid of its logits. The cell puts Part 2's nearest-code hole map next to this model's magnification map, runs the band between 20 random pairs of different digits, and compares decoded lengths. It also asks Part 2's digit-classifier judge how confidently the in-between frames read as digits.

```python
vae, _ = train_vae(X_mnist, beta=1.0, sample=True)   # Part 2's frozen VAE: d = 2, Bernoulli, beta = 1, seed 0, 60 epochs
g_mnist = vae.decode
with torch.no_grad():
    mu_m, logvar_m = vae.encode(Xt_mnist)
    recon = ((g_mnist(mu_m) - Xt_mnist) ** 2).mean().item()
kl_m = (0.5 * (mu_m**2 + logvar_m.exp() - logvar_m - 1).sum(1)).mean().item()
Z_vae = mu_m.numpy()
print(f"Part 2's VAE reproduced: recon MSE {recon:.4f} (0.0385) | centre spread {np.sqrt(Z_vae.var(0).sum()):.3f} (1.818) | "
      f"mean sigma {(0.5 * logvar_m).exp().mean():.4f} (0.0645) | mean KL {kl_m:.3f} (6.291)")

# ---- Part 2's hole map, copied verbatim (Part 2, cell 3) ----
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

def padded_box(Z: np.ndarray, pad: float = 0.1) -> np.ndarray:
    lo, hi = Z.min(0), Z.max(0)
    return np.stack([lo - pad * (hi - lo), hi + pad * (hi - lo)], axis=1)
# ----
box = padded_box(Z_vae)
near_vae, _ = hole_maps(lambda x: vae.encode(x)[0], g_mnist, Z_vae, box)
mg1, mg2 = np.meshgrid(np.linspace(*box[0], 100), np.linspace(*box[1], 100))
mag_m = np.sqrt(np.linalg.det(pullback_metric(g_mnist, np.column_stack([mg1.ravel(), mg2.ravel()])))).reshape(mg1.shape)
with torch.no_grad():
    mag_codes = np.sqrt(np.linalg.det(pullback_metric(g_mnist, Z_vae[::20])))
print(f"magnification on the codes: 5th-95th percentile {np.percentile(mag_codes, 5):.0f} to {np.percentile(mag_codes, 95):.0f}")

# 20 pairs of different digits: data-space length of the straight line vs the geodesic
rng = np.random.default_rng(8)
pairs = [(i, j) for i, j in rng.integers(0, len(X_mnist), (40, 2)) if y_mnist[i] != y_mnist[j]][:20]
results = []
for i, j in pairs:
    straight = lerp(Z_vae[i], Z_vae[j], np.linspace(0, 1, 33))
    geo, _ = geodesic(g_mnist, Z_vae[i], Z_vae[j], n=32, iters=1000)
    conf = [judge_samples(g_mnist(as_tensor(p)).detach().numpy()[1:-1])[0] for p in (straight, geo)]
    results.append((path_length(g_mnist, straight), path_length(g_mnist, geo), *conf, straight, geo))
ratio = np.array([r[0] / r[1] for r in results])
print(f"LERP length / geodesic length over {len(pairs)} pairs: median {np.median(ratio):.3f}, "
      f"range {ratio.min():.3f} to {ratio.max():.3f}; {(ratio > 1.1).sum()} pairs above 1.1")
print(f"judge confidence on the in-between frames: LERP {np.mean([r[2] for r in results]):.3f} | "
      f"geodesic {np.mean([r[3] for r in results]):.3f} (real digits {held_conf:.3f})")

k = int(ratio.argmax())
i, j = pairs[k]
fig = plt.figure(figsize=(15, 7.6))
gs = fig.add_gridspec(3, 3, height_ratios=[3.6, 1, 1], hspace=0.35, wspace=0.45, width_ratios=[1, 1, 0.8])
ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
ext = [*box[0], *box[1]]
im = ax1.imshow(near_vae, origin="lower", extent=ext, cmap=HOLE_CMAP, aspect="auto")
plt.colorbar(im, ax=ax1, fraction=0.046, label="nearest-code distance")
ax1.set_title("Part 2's hole map", fontsize=10)
im = ax2.imshow(mag_m, origin="lower", extent=ext, cmap="viridis", norm=LogNorm(), aspect="auto")
plt.colorbar(im, ax=ax2, fraction=0.046, label=r"$\sqrt{\det G}$")
ax2.set_title("Part 4's magnification map, same model", fontsize=10)
for ax in (ax1, ax2):
    ax.plot(*results[k][4].T, color=ORANGE, lw=2); ax.plot(*results[k][5].T, color=GREEN, lw=2)
    ax.set_xlabel("$z_1$")
ax3.hist(ratio, bins=10, color=GREEN, alpha=0.8); ax3.axvline(1, color="k", ls="--", lw=1)
ax3.set_xlabel("LERP length / geodesic length"); ax3.set_ylabel("digit pairs")
ax3.set_title(f"{len(pairs)} pairs: how much a geodesic saves", fontsize=10)
for row, (name, idx, colour) in enumerate([("LERP", 4, ORANGE), ("geodesic", 5, GREEN)], start=1):
    ax = fig.add_subplot(gs[row, :])
    with torch.no_grad():
        frames = g_mnist(as_tensor(results[k][idx][np.linspace(0, 32, 11).astype(int)])).numpy().reshape(-1, H, W)
    ax.imshow(montage(frames, 1, 11), cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}: digit {y_mnist[i]} -> {y_mnist[j]}, data-space length {results[k][idx - 4]:.1f}",
                 loc="left", fontsize=10, color=colour)
plt.show()
```
```text
Part 2's VAE reproduced: recon MSE 0.0385 (0.0385) | centre spread 1.818 (1.818) | mean sigma 0.0645 (0.0645) | mean KL 6.291 (6.291)
magnification on the codes: 5th-95th percentile 20 to 114
LERP length / geodesic length over 20 pairs: median 1.012, range 1.000 to 1.169; 4 pairs above 1.1
judge confidence on the in-between frames: LERP 0.810 | geodesic 0.795 (real digits 0.910)
```
![Output of cell 7]({{ '/assets/blog/straight-lines-are-a-choice/cell-07-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: on the left, Part 2's map of where the digit model has training data (white) and where it doesn't (red). In the middle, the same model coloured by how fast its images change, fastest in the crowded centre, with one pair's straight walk (orange) and shortest path (green) drawn on both maps. On the right, how much the shortest path saves for each of 20 digit pairs. Most bars sit right at 1, meaning no saving at all. The strips show the pair with the biggest saving.*

Part 2's model is back exactly: reconstruction error 0.0385, centre spread 1.818, mean $$\sigma$$ 0.0645, mean KL 6.291 nats. Across the codes its magnification runs from 20 to 114 (5th to 95th percentile), highest in the dense centre, where the codes are packed tightest and a small step can change the digit.

On the debate, this model lands mostly with Shao. The median ratio of LERP length to geodesic length is 1.012, so a typical geodesic saves about 1%. The range is 1.000 to 1.169, and 4 of the 20 pairs come in above 1.1. The strips show the biggest one, a 6 and a 5 whose reconstructions look more like a 0 and a 3. The straight walk passes through something like a 6 and then a 2 (length 23.5). The geodesic (20.2) just cross-fades the 0 into the 3.

Shorter isn't prettier, though. The judge is 0.810 confident on LERP's in-between frames and 0.795 on the geodesic's, against 0.910 on real digits. That's the pixel metric's blind spot, the same one Part 2 hit when it threw out pixel distance as a judge for rewarding blurry overlays. In pixel space a blend of two digits is close to both, so a geodesic under the plain pull-back metric drifts toward blends whenever they're cheap. It's cell 5's lesson, on real data.

Why is this VAE's metric so tame? Part 2 has the answer. A VAE trains its decoder on noise balls that overlap until they tile the prior's disc, which leaves few holes and a smooth decoder, and a smooth decoder barely bends geodesics. Filling the holes flattened the ruler too. In Arvanitidis et al.'s experiments it's the uncertainty term that makes sparse regions expensive and pushes geodesics around them, and this cell leaves it out. For this model, under this ruler, LERP is fine.

## Two latent spaces inside Stable Diffusion

A quick detour before the last idea, to place all this in a model people actually use. A latent diffusion model like Stable Diffusion ([Rombach et al., 2022](https://arxiv.org/abs/2112.10752)) has two latent spaces, and different parts of this series cover each.

The first belongs to the autoencoder. An encoder squeezes a $$512 \times 512 \times 3$$ image into a $$64 \times 64 \times 4$$ code and a decoder maps it back. There is a KL term, but a tiny one: the released configurations weight it at $$10^{-6}$$ ([CompVis config](https://github.com/CompVis/latent-diffusion/blob/main/configs/autoencoder/autoencoder_kl_32x32x4.yaml)). That makes it much closer to Part 1's plain autoencoder than to Part 2's VAE, and its codes are nowhere near $$\mathcal{N}(0, I)$$. Stable Diffusion 1.x multiplies them by 0.18215 to get roughly unit variance before diffusion ([v1 inference config](https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml)). FLUX.1's autoencoder uses 16 channels instead of 4 and normalises with a scale of 0.3611 and a shift of 0.1159 ([Black Forest Labs' reference code](https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py)). Interpolating between two images' codes here is Part 2's problem and this post's: a learned chart with its own ruler and maybe some holes.

The second is the noise the sampler starts from, $$\mathcal{N}(0, I)$$ in the same $$4 \times 64 \times 64 = 16{,}384$$ dimensions. That's Part 3's soap bubble, and interpolating between two seeds there means staying on the shell.

![A latent diffusion pipeline: a blue noise seed of 16,384 numbers flows through the denoiser or flow into a green image latent, which the VAE decoder turns into a 512 by 512 image, while the VAE encoder maps real images into that latent; a blue note marks the noise seed as Part 3's soap bubble for SLERP, and a green note marks the image latent as Parts 2 and 4's learned chart with holes and its own ruler]({{ '/assets/blog/straight-lines-are-a-choice/two-latents-in-sd.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the diagram shows: an image generator has two hidden spaces. The noise it starts from is a soap-bubble shell (Part 3), and the compressed image space it decodes from is a learned map with its own distances and possibly holes (Parts 2 and 4). Before you interpolate, check which one you're in.*

## The twist: flow matching trains the straight line

Up to now the recipe has been: take a trained model, then hunt for paths that stay where it was trained. Flow matching flips that. Pick the simplest path you can, a straight line from noise to data, and train the model on that path. Wherever the path goes, the model has already been.

Picture a segment from every noise sample $$\epsilon$$ to a data point $$x_0$$. The model learns a velocity field (which way to move at each point $$x$$ and time $$t$$) and gets supervised at every point along every segment. Liu et al. prove that the learned flow keeps the same distribution as the training interpolation at every time, so the points a sampler visits are distributed just like the points it trained on. The series' spine sentence holds for free.

In this series' notation, with data $$x_0$$ at $$t = 0$$ and noise $$\epsilon \sim \mathcal{N}(0, I)$$ at $$t = 1$$, the rectified-flow interpolation ([Liu, Gong & Liu, 2022](https://arxiv.org/abs/2209.03003); flow matching, [Lipman et al., 2022](https://arxiv.org/abs/2210.02747)) is

$$
x_t = (1 - t)\, x_0 + t\,\epsilon, \qquad \frac{dx_t}{dt} = \epsilon - x_0,
$$

and the network $$v_\theta(x, t)$$ is trained by plain regression onto that velocity:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t,\, x_0,\, \epsilon}\,\big\|\, v_\theta(x_t, t) - (\epsilon - x_0) \,\big\|^2 .
$$

To generate, start at $$x(1) = \epsilon$$ and integrate $$dx/dt = v_\theta(x, t)$$ backwards to $$t = 0$$. [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) trains its transformer with this objective, sampling $$t$$ non-uniformly so the middle of the path gets more weight.

The catch is that the training lines cross. Two segments, from different noise samples to different data points, can pass through the same $$x$$ at the same $$t$$ heading different ways, and $$v_\theta(x, t)$$ only gets to give one answer. The answer that minimises the loss is the average of every direction through that point,

$$
v^*(x, t) = \mathbb{E}\big[\,\epsilon - x_0 \;\big|\; x_t = x\,\big],
$$

and an ODE driven by an averaged field can't follow any one straight line. Its trajectories never cross each other, so where training lines cross, trajectories bend instead. Trained on straight lines, the model samples along curves. And curves cost integration steps: a single Euler step $$x_0 \approx \epsilon - v_\theta(\epsilon, 1)$$ is only exact if the trajectory is straight.

Liu et al.'s fix is **reflow**. Draw fresh noise, run the trained ODE to see where each $$\epsilon$$ lands, and retrain on those pairs, $$(\hat x_0, \epsilon)$$ with $$\hat x_0 = \text{ODE}(\epsilon)$$, instead of random ones. Each noise sample is now tied to one endpoint by a flow whose paths never cross, so the new straight segments cross far less often, and there's much less left to average. They score the result with a straightness measure,

$$
S = \int_0^1 \mathbb{E}\,\big\|\, (x_1 - x_0) - \dot x_t \,\big\|^2\, dt,
$$

the mean squared gap between a trajectory's velocity and its end-to-end displacement. It's 0 exactly when every trajectory is a straight line run at constant speed, which makes straightness something you can actually train for.

![Straight grey training lines from twelve noise points to random points on two moons cross each other; at two crossings the two directions merge into one averaged green arrow, the model's learned paths bend, and after reflow they straighten into short direct strokes as the Euler steps needed drop from 16 to 1]({{ '/assets/blog/straight-lines-are-a-choice/crossing-lines-curve.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the animation shows: the model learns by drawing straight lines from random noise to random data. The lines cross, and at a crossing the model can only learn the average direction, so the paths it actually follows bend. Retraining on the model's own start and end points removes the crossings, and the paths straighten out after a single round.*

Cell 8 runs the whole thing on two moons in 2-D, where you can see every trajectory. A three-layer MLP takes $$(x, t)$$ and returns a velocity, trained for 3,000 Adam steps on random noise-data pairs; that's the 1-rectified flow. Reflow pairs 20,000 fresh noise samples with wherever that flow sends them and retrains, starting from the first model's weights. Samples are scored by sliced Wasserstein distance to 5,000 real moons points (project both point clouds onto 128 random directions, sort each projection, take the root-mean-square gap) for Euler samplers with 1 to 100 steps. Part 3's guardrail closes the cell.

```python
import copy
from sklearn.datasets import make_moons
def moons(n: int, seed: int) -> torch.Tensor:
    """Two moons, centred and scaled to roughly unit spread: the 'data' x0."""
    X = make_moons(n, noise=0.05, random_state=seed)[0]
    return torch.from_numpy(((X - [0.5, 0.25]) * [1.15, 2.0]).astype(np.float32))

def velocity_net() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))

def train_flow(model: nn.Module, x0: torch.Tensor, eps: torch.Tensor | None, steps: int = 3000) -> nn.Module:
    """Rectified flow: x_t = (1 - t) x0 + t eps, regress v(x_t, t) onto eps - x0. eps=None draws fresh noise (no coupling)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(x0), (1024,))
        e = torch.randn(1024, 2) if eps is None else eps[idx]
        t = torch.rand(1024, 1)
        xt = (1 - t) * x0[idx] + t * e
        loss = ((model(torch.cat([xt, t], 1)) - (e - x0[idx])) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def sample(model: nn.Module, eps: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Euler steps of dx/dt = v from t = 1 (noise) to t = 0 (data); returns the whole trajectory (n_steps + 1, n, 2)."""
    x, traj = eps.clone(), [eps.clone()]
    for k in range(n_steps, 0, -1):
        t = torch.full((len(x), 1), k / n_steps)
        x = x - model(torch.cat([x, t], 1)) / n_steps
        traj.append(x.clone())
    return torch.stack(traj)

def sliced_w2(a: torch.Tensor, b: torch.Tensor, n_proj: int = 128) -> float:
    """Sliced Wasserstein-2 distance between two equal-size 2-D point clouds."""
    dirs = torch.randn(n_proj, 2, generator=torch.Generator().manual_seed(0))
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    pa, pb = (a @ dirs.T).sort(0).values, (b @ dirs.T).sort(0).values
    return ((pa - pb) ** 2).mean().sqrt().item()

def straightness(traj: torch.Tensor) -> float:
    """Liu et al. (2022): mean squared gap between each step's velocity and the trajectory's end-to-end displacement."""
    n_steps = len(traj) - 1
    v = (traj[1:] - traj[:-1]) * n_steps
    return ((v - (traj[-1] - traj[0])[None]) ** 2).sum(-1).mean().item()

torch.manual_seed(SEED)
x0 = moons(20_000, seed=SEED)
flow1 = train_flow(velocity_net(), x0, eps=None)                       # 1-rectified flow: random (noise, data) pairs
eps_r = torch.randn(20_000, 2)
x0_r = sample(flow1, eps_r, 100)[-1]                                   # reflow: pair each noise with where the ODE sends it
flow2 = train_flow(copy.deepcopy(flow1), x0_r, eps=eps_r)              # 2-rectified flow, warm-started from the first

ref, eps_test = moons(5000, seed=1), torch.randn(5000, 2, generator=torch.Generator().manual_seed(2))
print("Euler steps | sliced W2 to real moons: 1-rectified, 2-rectified (reflow)")
step_grid = [1, 2, 4, 8, 16, 100]
sw = {name: [sliced_w2(sample(m, eps_test, k)[-1], ref) for k in step_grid] for name, m in [("1-RF", flow1), ("2-RF", flow2)]}
for k, s1, s2 in zip(step_grid, sw["1-RF"], sw["2-RF"]):
    print(f"{k:11d} | {s1:.3f}  {s2:.3f}")
target = 1.5 * sw["1-RF"][-1]
need = {name: next((k for k, v in zip(step_grid, s) if v <= target), None) for name, s in sw.items()}
print(f"Euler steps to get within 1.5x of the 100-step result ({target:.3f}): 1-rectified {need['1-RF']}, 2-rectified {need['2-RF']}")
print(f"straightness S (0 = perfectly straight): 1-rectified {straightness(sample(flow1, eps_test, 100)):.4f}, "
      f"2-rectified {straightness(sample(flow2, eps_test, 100)):.4f}")

# The guardrail: flow matching never trained on noise -> noise lines. Two seeds in SD's 16,384-d noise space still sag.
s0, s1 = np.random.default_rng(11).standard_normal((2, 4 * 64 * 64))
print(f"two noise seeds, d = 16,384: ||seed|| {np.linalg.norm(s0):.1f}, {np.linalg.norm(s1):.1f} | LERP midpoint "
      f"{np.linalg.norm(lerp(s0, s1, [0.5])[0]):.1f} | SLERP midpoint {np.linalg.norm(slerp(s0, s1, [0.5])[0]):.1f} (Part 3)")

fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
show = torch.arange(60)
pair_x0 = x0[torch.randperm(len(x0), generator=torch.Generator().manual_seed(3))[:60]]
for e, d_ in zip(eps_test[show], pair_x0):
    axes[0].plot([e[0], d_[0]], [e[1], d_[1]], color=SLATE, lw=0.7, alpha=0.7)
axes[0].set_title("Training: straight noise -> data lines cross", fontsize=10)
for ax, m, name in [(axes[1], flow1, "1-rectified: the paths it follows curve"), (axes[2], flow2, "after one reflow: nearly straight")]:
    traj = sample(m, eps_test[show], 100)
    for p in range(len(show)):
        ax.plot(traj[:, p, 0], traj[:, p, 1], color=GREEN, lw=0.8, alpha=0.8)
    ax.set_title(name, fontsize=10)
for ax in axes[:3]:
    ax.scatter(*ref[:1500].T, s=1, color=TEAL, alpha=0.3); ax.scatter(*eps_test[show].T, s=6, color="k")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
for name, colour in [("1-RF", SLATE), ("2-RF", GREEN)]:
    axes[3].semilogx(step_grid, sw[name], "o-", color=colour, lw=2.5, label={"1-RF": "1-rectified", "2-RF": "after reflow"}[name])
axes[3].set_xlabel("Euler steps"); axes[3].set_ylabel("sliced W2 to real moons"); axes[3].legend()
axes[3].set_title("Straighter paths need fewer steps", fontsize=10)
plt.tight_layout(); plt.show()
```
```text
Euler steps | sliced W2 to real moons: 1-rectified, 2-rectified (reflow)
          1 | 0.907  0.054
          2 | 0.445  0.052
          4 | 0.213  0.051
          8 | 0.106  0.051
         16 | 0.066  0.051
        100 | 0.051  0.051
Euler steps to get within 1.5x of the 100-step result (0.076): 1-rectified 16, 2-rectified 1
straightness S (0 = perfectly straight): 1-rectified 0.8909, 2-rectified 0.0009
two noise seeds, d = 16,384: ||seed|| 128.1, 127.6 | LERP midpoint 90.4 | SLERP midpoint 127.8 (Part 3)
```
![Output of cell 8]({{ '/assets/blog/straight-lines-are-a-choice/cell-08-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: first the training data, straight grey lines from random noise points (black) to random points on the moons (teal), crossing everywhere. Next, the paths the trained model actually follows from the same noise points, which curve. Then the same after one round of retraining on its own start and end points, now nearly straight. Last, how close the generated points are to the real moons (lower is better) against the number of sampler steps. The retrained model is as good in one step as the original is in a hundred.*

The first model needs its steps. One Euler step gives a sliced Wasserstein distance of 0.907, and each doubling roughly halves it (0.445, 0.213, 0.106) until 16 steps reach 0.066, against 0.051 with 100. After one round of reflow, a single step gives 0.054, and from 4 steps on it's 0.051. If "within 1.5 times the 100-step score" counts as good enough, the first model needs 16 steps and the reflowed one needs 1. The straightness score says why: 0.891 before reflow, 0.0009 after.

The middle panel shows where the bends come from. At $$t = 1$$, $$x_t$$ is pure noise and carries no information about which data point it was paired with, so $$v^*(x, 1) = x - \mathbb{E}[x_0]$$. Every trajectory starts off aimed at the mean of the data, which sits between the moons, and has to swerve toward one of them later. After reflow each noise sample has a partner of its own, and its trajectory heads straight there.

That's the series thesis, inverted. Parts 2 and 3, and this post up to here, kept finding that a path is only as good as the region the decoder was trained on, and kept hunting for paths that stay in it. Flow matching picks the path first and trains the model on it. The model's own paths still bend wherever training lines cross, and reflow, by making straightness a training target, buys most of that back.

**The guardrail.** Flow matching makes exactly one straight line valid: from a noise sample to the data point the model sends it to. It says nothing about the line between two noise samples. The model never trained there, and Part 3's geometry still applies. Two seeds in Stable Diffusion's 16,384 dimensions have norms 128.1 and 127.6. Their LERP midpoint has norm 90.4, deep inside the shell where the model expects its inputs, and SLERP's midpoint sits at 127.8, right on it. If you're interpolating seeds in a rectified-flow model like SD3, use SLERP, same as Part 3. Flow matching straightened the paths it trained on, and nothing else.

## Recent work

The two halves of this post, finding good paths and training on simple ones, have been drifting toward each other.

On the geometry side, [Arvanitidis et al. (2022)](https://arxiv.org/abs/2106.05367) (AISTATS 2022) swap the Euclidean distance between decoded images for the information geometry of the decoder's output distribution. The metric then respects whatever likelihood the decoder uses, Bernoulli pixels included, instead of treating every decoder as a mean image with Gaussian noise.

On the flow side, [Tong et al. (2024)](https://arxiv.org/abs/2302.00482) (TMLR) go after cell 8's crossings at training time. They pair noise and data within each minibatch by optimal transport instead of at random. Flows come out straighter and sampling gets cheaper, with no separate reflow stage. [Liu et al. (2024)](https://arxiv.org/abs/2309.06380) (ICLR 2024) take reflow itself to text-to-image scale: InstaFlow reflows Stable Diffusion, then distils it into a one-step text-to-image generator.

[Kapuśniak et al. (2024)](https://arxiv.org/abs/2405.14780) bring the two together. Metric Flow Matching keeps the flow-matching recipe but replaces the straight noise-to-data segments with interpolants that approximate geodesics under a data-dependent metric, so intermediate points stay near the data manifold. Think of it as cell 4's rubber band used as a training target. And for seed interpolation, [Bodin et al. (2024)](https://arxiv.org/abs/2408.08558) (ICLR 2025), from Part 3's recent work, generalise SLERP to any linear combination of seeds that keeps the distribution the model expects.

## The end of the series

Four posts, one sentence: **a path through latent space is only as good as the region the decoder was trained on.** Each part found a new way for a path to wander out of that region, and a new way to bring it back.

Part 1 called a latent space a learned chart, coordinates the model invents for the low-dimensional manifold its data lives on, checked only at the data. Part 2 walked between the data, found holes, and watched a VAE plug most of them with noise balls and a KL spring. Part 3 found the high-dimensional prior hollow, a soap bubble of radius $$\sqrt{d}$$ whose empty middle a straight line falls into and SLERP steps around. This post measured paths with the decoder's own ruler, the pull-back metric. A geodesic under it walks evenly through the data, as long as the decoder was trained where the path goes. Otherwise blurry decoders reward fading and ghost-drawing decoders reward ghosts, until uncertainty turns the holes into mountains. And flow matching flipped the problem, training the model on the path.

![Four panels under the sentence 'A path through latent space is only as good as the region the decoder was trained on': Part 1, a curved shape mapped to a flat grid; Part 2, scattered codes with a coral hole filled by teal noise balls; Part 3, a thin ring with an orange chord cutting inside and a blue arc on the ring; Part 4, a green geodesic bending across a shaded map above short straight noise-to-data arrows; arrows between panels ask what is between the codes, what about high d, and how to measure a path]({{ '/assets/blog/straight-lines-are-a-choice/series-map.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the diagram shows: the four parts of the series in order, each answering the question the one before left open, all under one rule: a walk through a model's hidden space only works where the model has seen data.*

If you keep only a few things from all four posts:

- Interpolating noise seeds in a diffusion or flow model? Use SLERP, or the cos/sin form when the seeds are near-perpendicular. A straight line sags into the shell's empty middle, and no training objective fixes that (Part 3, and cell 8's guardrail).
- Interpolating data codes in an autoencoder's latent? A straight line is often fine for nearby points and for a well-regularised VAE, which saved a median 1% in cell 7. Reach for a geodesic when the path is long, when even pacing matters, or when the chart is visibly distorted (cell 4).
- Before trusting any path, look at where it runs. Part 2's nearest-code and cycle-error maps are cheap, and so is cell 5's faintest-frame check. A geodesic under a plain decoder metric will take any shortcut the decoder offers, so add decoder uncertainty if there are holes (cell 6).
- With flow models, remember which straight line was the trained one: noise to data, yes; seed to seed, no.

None of the paths in this series were straight in any absolute sense. LERP is straight in latent coordinates, SLERP is straight on the shell, the geodesic is straight under the decoder's ruler, and the rectified-flow segment is straight under its training objective. Straight lines are a choice. The useful question is which ruler made yours straight.

## Further reading

- [Part 1](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/what-is-latent-space/), [Part 2](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/ghosts-and-hybrids/) and [Part 3](https://neelesh-gupta.github.io/neeleshgupta.github.io/blog/2026/high-dimensional-soap-bubble/) of this series: latent spaces as learned charts, holes and noise balls in two dimensions, and the soap bubble in many.
- [Arvanitidis, Hansen & Hauberg, 2018](https://arxiv.org/abs/1710.11379), "Latent Space Oddity": the pull-back metric for deep generative models, the stochastic-decoder metric and the RBF variance network behind cell 6.
- [Shao, Kumar & Fletcher, 2017](https://arxiv.org/abs/1711.08014): geodesics and parallel transport computed in trained generators, and the finding that they are close to straight.
- [Chen et al., 2018](https://arxiv.org/abs/1711.01204), "Metrics for Deep Generative Models": geodesics learned by a neural network instead of a discrete band, with a clear treatment of the pull-back metric.
- [Liu, Gong & Liu, 2022](https://arxiv.org/abs/2209.03003): rectified flow, reflow and the straightness measure used in cell 8.
- [Lipman et al., 2022](https://arxiv.org/abs/2210.02747): flow matching, the general framework that includes the straight-line path as one choice among many.
- [Bishop, Svensén & Williams, 1997](https://www.microsoft.com/en-us/research/publication/magnification-factors-for-the-gtm-algorithm/): magnification factors, the idea behind cell 3's heatmap, in its original setting.
