---
layout: post
title: "What is Latent Space? Everyone talks about it, no one really explains it"
date: 2026-11-21 09:00:00+0530
description: "What a latent space actually is: a learned coordinate chart on the data manifold, worked through with intrinsic dimension, PCA, autoencoders and why a circle has to tear."
tags: latent-space autoencoders manifolds pca representation-learning
categories: machine-learning
related_posts: false
toc:
  sidebar: left
---

<style>
  /* This post only: use the window's width so code, equations and figures fit without sideways scrolling. */
  @media (min-width: 576px) {
    .container[role="main"], .navbar .container, footer .container { max-width: min(calc(100vw - 32px), 1520px); }
    .container[role="main"] > .row > .col-sm-3 { flex: 0 0 18%; max-width: 18%; }
    .container[role="main"] > .row > .col-sm-9 { flex: 0 0 82%; max-width: 82%; }
  }
  .post-content img { max-width: min(100%, 960px) !important; }
  .post-content p.caption { opacity: .8; max-width: 92%; margin: -.25rem auto 1.75rem; }
  .post-content pre.highlight, .post-content pre.highlight code { white-space: pre-wrap; overflow-wrap: anywhere; }
  .post-content pre, .post-content pre code { white-space: pre-wrap; overflow-wrap: anywhere; }
  .post-content mjx-container[display] { max-width: 100%; }
  @media (max-width: 575px) {
    .post-content mjx-container[display] { font-size: 90% !important; }
  }
</style>

Latent space is one of those terms that turns up everywhere in machine learning. Stable Diffusion generates its images inside one. A VAE samples from one. People talk about walking through a latent space to morph one face into another, and about an LLM's latent space as if it were a place you could visit. Ask what a latent space actually *is*, though, and you usually get a sentence: a compressed representation of the data, a lower-dimensional encoding.

That sentence is true as far as it goes. What it can't tell you is why some latent spaces are pleasant to move around in while others are full of garbage, why interpolating between two codes sometimes gives a clean blend and sometimes nonsense, or why people keep reaching for an odd-looking formula called SLERP when they interpolate. Answering those questions takes a geometric picture, and building that picture is what this four-part series is for. The thread running through all of it is one idea: a path through latent space is only as good as the region the decoder was trained on.

This first part asks the most basic question, what a latent space is, and answers it with coordinates for the thin, curved surface that real data lives on. Each idea comes four ways: an intuition you can hold in your head, a picture of the geometry, the maths that makes it precise, and a short code cell that checks the maths on real numbers. You'll want to be comfortable with neural networks, MSE training and the SVD. You won't need any differential geometry, because the little we use gets built from maps of the Earth.

## Static vs. blobs: almost every image is noise

Here are two 28×28 grayscale images. The first is TV static, every pixel drawn at random from [0, 1]. The second is a soft white blob on a black background. To your eye they have nothing in common. To a computer they're the same kind of thing: a list of 784 numbers, or one point in $$\mathbb{R}^{784}$$.

Now draw 28×28 images uniformly at random and count how many look like anything at all. None will. You could keep drawing for the rest of your life and get nothing but static, because the images we care about sit in a sliver of pixel space so thin that random sampling never lands in it. This post is about putting coordinates on that sliver.

For that I want a dataset where I already know the right answer. Blob World, which I'll use for the whole series, is about as simple as images get: a Gaussian blob of width $$s = 2$$ pixels whose centre $$(c_x, c_y)$$ is drawn uniformly from $$[4, 24]^2$$,

$$
I_c(u, v) = \exp\!\left(-\frac{(u - c_x)^2 + (v - c_y)^2}{2s^2}\right),
$$

where $$(u, v)$$ is the pixel position. Every image has 784 pixels, but two numbers are enough to make one. Those two numbers, $$(c_x, c_y)$$, are the ground truth we'll try to recover later. The setup cell defines `blob()`, two more toy datasets we'll need further on (Sized Blob and Clock Hand), and the `AE` class.

```python
%pip install -q "numpy==2.3.3" "torch==2.13.0" "torchvision==0.28.0" "scikit-learn==1.6.1" "matplotlib==3.10.8"
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)
TEAL, CORAL, SLATE = "#14b8a6", "#f97362", "#64748b"  # series colours: teal = data, coral = void
H = W = 28
xs, ys = np.meshgrid(np.arange(W), np.arange(H))

def blob(cx: float, cy: float, s: float = 2.0) -> np.ndarray:
    """A 28x28 Gaussian blob of width s centred at (cx, cy), peak intensity 1."""
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * s**2)).astype(np.float32)

def blob_world(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n blob images with centres uniform in [4, 24]^2. Returns images and the true chart (centres)."""
    C = rng.uniform(4, 24, size=(n, 2))
    return np.stack([blob(cx, cy) for cx, cy in C]), C

def sized_blob_world(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Blob World with a third knob: the width s in [1.5, 3]."""
    P = np.column_stack([rng.uniform(4, 24, size=(n, 2)), rng.uniform(1.5, 3.0, size=n)])
    return np.stack([blob(cx, cy, s) for cx, cy, s in P]), P

def clock_hand(theta: float, length: float = 9.0) -> np.ndarray:
    """Asymmetric clock hand: thin bar from the centre plus a fat dot at the tip (period 2*pi, not pi)."""
    bar = [blob(14 + r * np.cos(theta), 14 + r * np.sin(theta), 0.9) for r in np.linspace(0, length, 12)]
    tip = blob(14 + length * np.cos(theta), 14 + length * np.sin(theta), 1.8)
    return np.max(bar + [tip], axis=0)

def flat(X: np.ndarray) -> np.ndarray:
    """(n, 28, 28) images -> (n, 784) points in pixel space."""
    return X.reshape(len(X), -1)

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

print(f"numpy {np.__version__} | torch {torch.__version__} | torchvision {torchvision.__version__}")
```
```text
numpy 2.3.3 | torch 2.13.0+cpu | torchvision 0.28.0+cpu
```

The next cell builds 5,000 blob images and 500 static ones and asks a question about distance. How far is each static image from the nearest blob image, and how far is each blob from its nearest neighbouring blob?

```python
X_blob, C_blob = blob_world(5000, np.random.default_rng(SEED))  # C_blob is the true chart
static = np.random.default_rng(1).uniform(0, 1, size=(500, H * W)).astype(np.float32)

# How far is a random image from the nearest Blob World image, compared with blob-to-blob gaps?
index = NearestNeighbors(n_neighbors=2).fit(flat(X_blob))
d_static = index.kneighbors(static, n_neighbors=1)[0][:, 0]
d_blob = index.kneighbors(flat(X_blob[:500]))[0][:, 1]  # column 0 is the point itself
print(f"ambient dimension D = {X_blob[0].size}; Blob World knobs = 2 (blob centre x, y)")
print(f"static image -> nearest blob image : {d_static.mean():6.2f} (min {d_static.min():.2f})")
print(f"blob image   -> nearest other blob : {d_blob.mean():6.2f} (max {d_blob.max():.2f})")

fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
for i in range(8):
    axes[0, i].imshow(static[i].reshape(H, W), cmap="gray", vmin=0, vmax=1)
    axes[1, i].imshow(X_blob[i], cmap="gray", vmin=0, vmax=1)
for ax in axes.flat:
    ax.axis("off")
axes[0, 0].set_title("random pixels (static)", loc="left", fontsize=10)
axes[1, 0].set_title("Blob World", loc="left", fontsize=10)
plt.tight_layout()
plt.show()
```
```text
ambient dimension D = 784; Blob World knobs = 2 (blob centre x, y)
static image -> nearest blob image :  15.62 (min 14.92)
blob image   -> nearest other blob :   0.18 (max 0.56)
```
![Output of cell 2]({{ '/assets/blog/what-is-latent-space/cell-02-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: the top row is eight images of pure random noise, the bottom row eight Blob World images. Every one of them is a list of 784 numbers. Only the bottom row has structure, and you could describe each of those images with just two numbers, the blob's position.*
{: .caption}

Across the 500 blobs it checks, neighbours are 0.18 apart on average and never more than 0.56. The closest any of the 500 static images gets to a blob is 14.92, roughly 27 times that biggest gap. Static lives somewhere else entirely.

Both rows of the figure sit in the same space, $$D = 784$$. What differs is how many things can change independently. For static that's all 784 pixels; for Blob World it's two. That count is the intrinsic dimension, and getting it right is the first job of any latent space.

## Two meanings of "latent space"

The phrase means two different things, and mixing them up causes confusion later. In a probabilistic model, a latent variable is an unobserved $$z$$ with a prior $$p(z)$$, and the data is generated from it, $$x \sim p(x \mid z)$$. That's how people talk about VAEs and diffusion models. Everywhere else, "latent space" just means the vectors some hidden layer assigns to its inputs, like a classifier's penultimate features, a CLIP embedding or a transformer's residual stream.

Either way, the latent is a set of coordinates the model picked. They're everywhere. A classifier arranges its penultimate features so that a linear head can split the classes. An autoencoder squeezes its input through a bottleneck that has to hold enough to rebuild it, and a VAE adds a prior you can sample from (that's Part 2). CLIP puts images and captions in one shared space. Latent diffusion runs the whole diffusion process inside an autoencoder's latent, while an LLM keeps updating each token's residual-stream vector, layer after layer.

![Six model families drawn as pipelines; in each the latent block (teal) is the classifier's penultimate features, the autoencoder bottleneck z, the VAE code z, CLIP's joint embedding space, latent diffusion's 64x64x4 latent, or an LLM's residual stream]({{ '/assets/blog/what-is-latent-space/latent-in-pipelines.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Six familiar models drawn as left-to-right pipelines. In each row the teal box is the latent space, the internal list of numbers the model uses to describe its input. It sits at a different point in each model, but its job is the same everywhere.*
{: .caption}

I'm going to follow only the latent spaces you can decode from, where a decoder $$g$$ turns $$z$$ back into data. A decoder is what lets you ask geometric questions. Nudge $$z$$ and what happens to the image? What's sitting between two training points? Embedding spaces like CLIP's will come up now and then, but the thread runs through autoencoders and what grew out of them.

## Ambient and intrinsic dimension

The ambient dimension $$D$$ is how many numbers you store per example. The intrinsic dimension $$k$$ is how many knobs it takes to produce every example. Photograph one object spinning on a turntable and $$k = 1$$, whether the camera shoots 28×28 or 4K. When $$D$$ is much bigger than $$k$$, something strange happens to distance, and Blob World is simple enough that we can work out exactly what.

Take two blobs $$I_a$$ and $$I_b$$ whose centres are $$\delta$$ pixels apart. Treat them as continuous functions for now and expand the square:

$$
\|I_a - I_b\|^2 = \int I_a^2 + \int I_b^2 - 2\int I_a I_b .
$$

Each self-term is a Gaussian integral, $$\int e^{-r^2/s^2} = \pi s^2$$. The cross-term needs one small trick. Let $$m$$ be the midpoint between the two centres; then $$\lvert p - a\rvert^2 + \lvert p - b\rvert^2 = 2\lvert p - m\rvert^2 + \delta^2/2$$ for any pixel position $$p$$, and the integral collapses to a single Gaussian times a constant:

$$
\int I_a I_b = e^{-\delta^2/4s^2}\int e^{-|p - m|^2/s^2} = \pi s^2\, e^{-\delta^2/4s^2}.
$$

Putting the pieces back together gives

$$
\boxed{\;\|I_a - I_b\|_2^2 = 2\pi s^2\left(1 - e^{-\delta^2/4s^2}\right)\;}
$$

The two ends of this formula tell the story. When $$\delta$$ is small, $$1 - e^{-u} \approx u$$ gives $$\|I_a - I_b\| \approx \sqrt{\pi/2}\,\delta \approx 1.25\,\delta$$, so distance grows in step with how far the blob moved. Good. When $$\delta$$ is large the exponential dies and the distance flattens out at $$2\pi s^2 \approx 25.13$$. Beyond roughly $$3s$$, every pair of blobs is the same distance apart, neighbours and opposite corners alike.

I find the geometry hiding in that derivation more useful than the formula itself. Every blob has the same squared norm, $$\pi s^2$$, so apart from tiny edge effects all of Blob World sits on a sphere of radius $$\sqrt{\pi}\,s \approx 3.54$$. The cross-term is an inner product, which makes the cosine between two blobs $$e^{-\delta^2/4s^2}$$. At $$\delta = 3s$$ that's about 0.11, an angle of 84°. Blobs a few widths apart point in almost perpendicular directions.

So picture a 2-D sheet lying on that sphere, turning into a fresh perpendicular direction every time the blob moves a little. Up close it's flat. From further out it's curled up so tightly that "a bit apart" and "opposite corners" look the same. Draw a straight line between two distant blobs and it cuts through the inside of the sphere, through images that aren't blobs at all; we'll come back to that line in Part 2.

The next cell checks the formula on the real pixel grid. It slides a blob to the right, keeping it $$3s$$ from the border so none of it gets clipped.

```python
s = 2.0
deltas = np.linspace(0, 16, 161)
start = np.array([6.0, 14.0])  # slide right along y = 14, staying >= 3s from both borders
measured = np.array([((blob(*start) - blob(start[0] + d, start[1])) ** 2).sum() for d in deltas])
formula = 2 * np.pi * s**2 * (1 - np.exp(-(deltas**2) / (4 * s**2)))

for d in (1, 4, 8, 16):
    i = int(np.argmin(np.abs(deltas - d)))
    print(f"delta = {d:>2}: measured {measured[i]:6.3f} | formula {formula[i]:6.3f}")
print(f"limit 2*pi*s^2 = {2 * np.pi * s**2:.3f} | max |measured - formula| = {np.abs(measured - formula).max():.1e}")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(deltas, measured, color=TEAL, lw=3, label="measured on the 28x28 grid")
ax.plot(deltas, formula, "k--", lw=1.2, label=r"$2\pi s^2(1-e^{-\delta^2/4s^2})$")
ax.axvline(3 * s, color=SLATE, ls=":", lw=1)
ax.text(3 * s + 0.2, 3, r"$\delta = 3s$", color=SLATE)
ax.set_xlabel(r"blob separation $\delta$ (pixels)")
ax.set_ylabel(r"$\|I_a - I_b\|_2^2$")
ax.set_title("Pixel distance stops growing after ~3 blob widths")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
```
```text
delta =  1: measured  1.523 | formula  1.523
delta =  4: measured 15.887 | formula 15.887
delta =  8: measured 24.672 | formula 24.672
delta = 16: measured 25.132 | formula 25.133
limit 2*pi*s^2 = 25.133 | max |measured - formula| = 4.7e-04
```
![Output of cell 3]({{ '/assets/blog/what-is-latent-space/cell-03-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: the teal curve is the measured pixel distance between two blobs as one slides away from the other, and the dashed line is the formula above. They sit on top of each other. Past the dotted line, about 6 pixels apart, the curve goes flat: moving the blob further away hardly changes the pixel distance at all.*
{: .caption}

The formula is right to within $$4.7 \times 10^{-4}$$. The squared distance is 1.523 at $$\delta = 1$$ and 15.887 at $$\delta = 4$$, and by $$\delta = 8$$ it has reached 24.672, already within 2% of the ceiling. Moving the blob from 8 pixels away to 16 adds just 0.46.

That plateau is a real problem. Nearest-neighbour search, k-means and pixel-wise losses all run on this distance, and in Blob World it stops telling "a bit apart" from "opposite corners" once blobs are more than about 6 pixels apart. Distance along the sheet keeps growing like $$1.25\,\delta$$. Getting at it takes coordinates on the sheet.

## The manifold hypothesis

Blob World was built to have two knobs. The manifold hypothesis says real data is like that too: *the data distribution concentrates near a $$k$$-dimensional submanifold $$\mathcal{M} \subset \mathbb{R}^D$$, with $$k \ll D$$.* A manifold is anything that looks like flat $$\mathbb{R}^k$$ once you zoom in far enough, the way a field looks flat even though the Earth isn't.

The evidence comes from estimating intrinsic dimension on real datasets. [Pope et al. (2021)](https://arxiv.org/abs/2104.08894) put MNIST at 7 to 13 against its 784 pixels, CIFAR-10 at 13 to 26 against 3,072, and ImageNet at 26 to 43 against 150,528. They also found that datasets with lower intrinsic dimension are easier for networks to learn, while the ambient dimension, the pixel count, has little effect on generalisation. [Fefferman, Mitter & Narayanan (2016)](https://arxiv.org/abs/1310.0425) go further and show the hypothesis can be tested statistically.

I'd keep three caveats in mind. Real data is noisy, so it sits *near* $$\mathcal{M}$$ rather than exactly on it. It may live on several manifolds rather than one. And $$k$$ can change from one region of the data to another.

## Measuring intrinsic dimension with TwoNN

We know Blob World has two knobs because we built it. Nobody tells us how many MNIST has, and we've just seen that ambient distances can't be trusted at long range. TwoNN sidesteps both problems by only ever looking at a point's two nearest neighbours, which are close enough that the sheet still looks flat.

Pick a data point and find its nearest neighbour, at distance $$r_1$$, and its second-nearest, at $$r_2$$. If the data lives on a line, the second neighbour is often much farther away than the first. In many dimensions there's so much room nearby that the second neighbour is nearly always only a little farther, and the ratio $$\mu = r_2/r_1$$ bunches up just above 1. How $$\mu$$ is spread out tells you the dimension.

You can see why by drawing two balls around the point, one reaching each neighbour. On a $$k$$-dimensional sheet a ball of radius $$r$$ holds volume $$\propto r^k$$, so doubling the radius buys twice the room on a line, four times on a plane and $$2^k$$ times in general.

Making this precise takes one assumption, that the density is roughly constant over the distance to the second neighbour. Call the first ball's volume $$V_1 = \omega_k r_1^k$$ and the shell between the two balls $$V_2 = \omega_k (r_2^k - r_1^k)$$, where $$\omega_k$$ is the volume of a unit ball. For points scattered uniformly at random, the volumes you sweep through before hitting each successive point are independent exponentials with the same rate. So

$$
\mu^k = \frac{r_2^k}{r_1^k} = \frac{V_1 + V_2}{V_1} = 1 + \frac{V_2}{V_1},
$$

and for two i.i.d. exponentials, $$P(V_2 > c\,V_1) = 1/(1 + c)$$. Setting $$c = m^k - 1$$ gives

$$
\begin{gathered}
P(\mu > m) = m^{-k} \\
\Longrightarrow\quad F(\mu) = 1 - \mu^{-k}, \qquad \mu \ge 1 .
\end{gathered}
$$

The density has cancelled out, which is the lovely part. What's left is a Pareto law that needs nothing but two neighbours per point. [Facco et al. (2017)](https://arxiv.org/abs/1803.06992), who introduced TwoNN, read $$k$$ off it in two ways. Plot $$-\log(1 - F(\mu)) = k \log \mu$$ and you get a line through the origin with slope $$k$$, which they fit on the smallest 90% of ratios. Or maximise the log-likelihood $$N \log k - (k + 1)\sum_i \log\mu_i$$, which peaks at

$$
\hat k = \frac{N}{\sum_{i=1}^{N} \log \mu_i}.
$$

The cell implements the maximum-likelihood version in a few lines. It runs it on Blob World ($$k = 2$$), on Sized Blob, where the blob's width becomes a third knob ($$k = 3$$), and on 10,000 MNIST digits, and plots the straight-line view.

```python
def twonn(X: np.ndarray) -> tuple[float, np.ndarray]:
    """TwoNN (Facco et al. 2017): mu = r2/r1 ~ Pareto(k), so the MLE is k = N / sum(log mu)."""
    dist, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    mu = dist[:, 2] / dist[:, 1]  # column 0 is the point itself
    mu = mu[np.isfinite(mu) & (mu > 1)]  # drop exact duplicates
    return len(mu) / np.log(mu).sum(), np.sort(mu)

mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True)
X_mnist = mnist.data[:10000].numpy().reshape(-1, H * W).astype(np.float32) / 255
X_sized, _ = sized_blob_world(5000, np.random.default_rng(3))
datasets = {"Blob World": flat(X_blob), "Sized Blob": flat(X_sized), "MNIST (10k)": X_mnist}

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for (name, X), color in zip(datasets.items(), [TEAL, "#0e7490", SLATE]):
    k_hat, mu = twonn(X)
    F = np.arange(1, len(mu) + 1) / (len(mu) + 1)  # empirical CDF of mu
    keep = F < 0.9  # Facco et al. drop the top 10% of ratios (noisiest tail)
    ax.plot(np.log(mu[keep]), -np.log(1 - F[keep]), ".", ms=2, color=color, alpha=0.5)
    ax.plot([0, np.log(mu[keep]).max()], [0, k_hat * np.log(mu[keep]).max()], color=color, lw=2, label=f"{name}: k = {k_hat:.2f}")
    print(f"{name:12s} D = {X.shape[1]}  N = {len(X):5d}  TwoNN k = {k_hat:.2f}")
ax.set_xlabel(r"$\log \mu$  ($\mu = r_2 / r_1$)")
ax.set_ylabel(r"$-\log(1 - F(\mu))$")
ax.set_title("TwoNN: the slope of each line is the intrinsic dimension")
ax.legend()
plt.tight_layout()
plt.show()
```
```text
Blob World   D = 784  N =  5000  TwoNN k = 2.01
Sized Blob   D = 784  N =  5000  TwoNN k = 2.99
MNIST (10k)  D = 784  N = 10000  TwoNN k = 13.35
```
![Output of cell 4]({{ '/assets/blog/what-is-latent-space/cell-04-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: one colour per dataset. For each point, TwoNN compares how far away its second-closest neighbour is with how far away its closest one is, and plots the results so they should fall on a straight line. The steeper the line, the more dimensions the data has: about 2 for Blob World, 3 for Sized Blob and 13 for MNIST.*
{: .caption}

Blob World comes out at 2.01 and Sized Blob at 2.99. From raw 784-dimensional points, with no idea how they were made, the estimator counted the knobs, and in the plot each dataset hugs its own line.

MNIST gives 13.35. Facco et al. got 13.4 using only the "2"s, so the full set of digits lands in the same place, about 59 times fewer dimensions than pixels. I wouldn't read much into the decimals, though. Pope et al. get 7 to 13 with a different estimator and 15 from their own TwoNN run, and they point out that TwoNN sometimes ranks datasets in odd orders. It also drifts when points are scarce or noisy, which isn't a worry with 5,000 clean blobs.

## Charts and atlases

There's a precise word for "coordinates on a sheet", and it comes from mapmaking. The Earth's surface is 2-D but sits in 3-D, and you can't flatten the whole thing without tearing or stretching something. Any one region fits on a page of an atlas, though, and that page gives every point in it two numbers. Mathematicians call the page a chart. On a $$k$$-dimensional manifold $$\mathcal{M}$$ it's a map

$$
\begin{gathered}
\varphi : U \to \mathbb{R}^k, \\
U \subseteq \mathcal{M} \text{ an open patch},
\end{gathered}
$$

that's a *homeomorphism* onto its image, meaning it's continuous and invertible and its inverse is continuous too. In plainer terms, points that are close on the surface get close coordinates, and close coordinates come from close points. A collection of charts that covers all of $$\mathcal{M}$$ is an atlas. Where two pages overlap, a transition map converts one page's coordinates into the other's. That's as much differential geometry as this series needs.

Flattening always distorts something. Mercator keeps shapes but inflates area, which is why Greenland looks about as big as Africa when Africa is around 14 times larger. A plain latitude-longitude grid stretches east-west distances by $$1/\cos(\text{latitude})$$, so a small circle near the pole gets drawn as a wide ellipse. In Part 4 we'll measure exactly this sort of distortion inside a latent space.

![The front hemisphere of a globe peels open into a flat latitude-longitude chart; fifteen equal teal circles on the sphere become ellipses on the chart, stretched east-west more and more towards the poles.]({{ '/assets/blog/what-is-latent-space/globe-to-map.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The front half of a globe is peeled open into a flat map. The teal circles are all the same size on the globe, but on the flat map the ones near the top and bottom get stretched sideways. Every flat map of a curved surface distorts something, and the same will be true of latent spaces.*
{: .caption}

Here's the claim the rest of this post tests:

> **A latent space is a learned chart.** The encoder $$e$$ assigns coordinates $$z = e(x)$$ to points on the data manifold, and the decoder $$g$$ maps coordinates back to data.

TwoNN told us how many coordinates we need. Next we'll try a straight chart from PCA and a bent one from an autoencoder, and then look at a manifold that no chart of the right dimension can cover.

## PCA is a linear chart

PCA is the classic chart-maker, and the picture behind it is simple. Hold a flat sheet of paper inside the data cloud, tilt it until the data's shadow on it is as spread out as possible, and read off each point's position on the paper. If the data really is flat, that gives a perfect chart. If it's curved, flat paper has no way to follow it.

The maths is a truncated SVD. Stack the $$N$$ centred points as the rows of $$X \in \mathbb{R}^{N \times D}$$ and decompose it as $$X = U\Sigma V^\top$$. The columns of $$V$$ are orthonormal directions in pixel space, and the singular values $$\sigma_1 \ge \sigma_2 \ge \dots$$ say how much the data spreads along each one. The PCA chart projects onto the top $$k$$ directions:

$$
\begin{gathered}
z = V_k^\top x \in \mathbb{R}^k, \\
\hat x = V_k z = V_k V_k^\top x .
\end{gathered}
$$

Doing that for every point at once gives $$\hat X = X V_k V_k^\top = U_k \Sigma_k V_k^\top$$, which is just the SVD cut off after term $$k$$. The error is whatever got cut:

$$
\|X - \hat X\|_F^2 = \Big\|\sum_{i > k} \sigma_i u_i v_i^\top\Big\|_F^2 = \sum_{i > k} \sigma_i^2 ,
$$

The cross terms vanish because the rank-one pieces $$u_i v_i^\top$$ are mutually orthogonal, and the Eckart-Young theorem guarantees that no rank-$$k$$ approximation does better. The fraction you keep, $$\sum_{i \le k}\sigma_i^2 / \sum_i \sigma_i^2$$, is the "explained variance" every PCA library prints.

There's a neat link to neural networks here. [Baldi & Hornik (1989)](https://doi.org/10.1016/0893-6080(89)90014-2) proved that a *linear* autoencoder trained with MSE learns the same subspace as PCA, though its axes may be rotated or rescaled within it. Calling PCA a linear autoencoder is fair, then. Calling PCA components "what a neural net learns" isn't, because once the encoder has nonlinearities it's free to leave flat subspaces behind.

You can see flat paper's weakness on a swiss roll, a 2-D sheet rolled up in 3-D. If the roll is taller than it is wide, its best-fitting plane runs along the height, and projecting onto it stacks the layers on top of each other. A bent chart unrolls it.

![A swiss roll flattened two ways: on the left, projecting onto PCA's best-fitting plane stacks the layers so purple, teal and yellow points interleave; on the right, unrolling along the sheet gives a clean purple-to-yellow gradient.]({{ '/assets/blog/what-is-latent-space/swiss-roll-unfold.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Both halves start from the same rolled-up sheet, coloured from purple at the inside of the roll to yellow at the outside. On the left, PCA flattens it by squashing it onto its best-fitting plane, and colours from different layers end up mixed together. On the right the sheet is unrolled instead, so neighbouring colours stay next to each other. The right half shows the ideal bent chart, the kind an autoencoder tries to learn, rather than a trained model's output.*
{: .caption}

The cell asks PCA how many components it needs for 90, 95 and 99% of the variance on Blob World and on MNIST. It also checks the residual identity numerically and plots Blob World in its top two PCA coordinates, coloured by the true $$c_x$$.

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
for name, X, color in [("Blob World (k = 2)", flat(X_blob), TEAL), ("MNIST", X_mnist, SLATE)]:
    cum = np.cumsum(PCA().fit(X).explained_variance_ratio_)
    need = [int(np.searchsorted(cum, q)) + 1 for q in (0.90, 0.95, 0.99)]
    print(f"{name:18s} components for 90 / 95 / 99% variance: {need[0]:3d} / {need[1]:3d} / {need[2]:3d}")
    ax1.plot(np.arange(1, len(cum) + 1), cum, color=color, lw=2, label=name)

# PCA = truncated SVD: the reconstruction error is exactly the sum of the discarded sigma_i^2.
Xc = flat(X_blob) - flat(X_blob).mean(axis=0)
_, S, Vt = np.linalg.svd(Xc, full_matrices=False)
Z_pca = Xc @ Vt[:2].T  # the linear chart: coordinates on the best-fitting plane
err = ((Xc - Z_pca @ Vt[:2]) ** 2).sum()
print(f"PCA-2 squared error {err:.1f} = sum of discarded sigma_i^2 {(S[2:] ** 2).sum():.1f}")

ax1.axhline(0.95, color="k", ls=":", lw=1)
ax1.set_xscale("log")
ax1.set_xlabel("number of principal components")
ax1.set_ylabel("cumulative explained variance")
ax1.legend(loc="lower right")
sc = ax2.scatter(Z_pca[:, 0], Z_pca[:, 1], c=C_blob[:, 0], cmap="viridis", s=3)
plt.colorbar(sc, ax=ax2, label="true blob x")
ax2.set_title("Blob World in its top-2 PCA coordinates")
ax2.set_xlabel("PC 1"); ax2.set_ylabel("PC 2")
plt.tight_layout()
plt.show()
```
```text
Blob World (k = 2) components for 90 / 95 / 99% variance:  23 /  31 /  49
MNIST              components for 90 / 95 / 99% variance:  85 / 150 / 326
PCA-2 squared error 45534.5 = sum of discarded sigma_i^2 45534.5
```
![Output of cell 5]({{ '/assets/blog/what-is-latent-space/cell-05-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows. Left: how much of the data's variation PCA captures as it uses more flat directions. Blob World has only two knobs, yet PCA needs about 31 directions to reach the dotted 95% line. Right: Blob World squeezed into PCA's best two directions and coloured by each blob's true left-right position. Purple blobs (left side of the image) and yellow blobs (right side) end up in the same spikes near the middle, so PCA has folded the data over onto itself.*
{: .caption}

Blob World has two degrees of freedom, and PCA wants 23 components for 90% of the variance, 31 for 95% and 49 for 99%. MNIST needs 85, 150 and 326, where TwoNN said 13. PCA isn't miscounting so much as counting something else, namely how many flat directions the data spreads into, and a sheet that keeps turning into new perpendicular directions spreads into a great many. The residual identity checks out exactly, at 45534.5 on both sides.

The right-hand panel shows what that costs. In PCA coordinates Blob World turns into a ring with spikes pointing inward, and blobs from the left edge of the image (purple) end up crowded into those spikes alongside blobs from the right edge (yellow). Points far apart on the true square get the same PCA coordinates, which is the swiss roll's folding happening to our blobs. A straight chart of a curved sheet either needs far too many dimensions or folds the sheet over on itself.

## An autoencoder bends the chart

An autoencoder lets a network bend the paper. The encoder $$e$$, here an MLP $$784 \to 256 \to 128 \to d$$, assigns the coordinates, the decoder $$g$$ mirrors it to map them back, and training minimises the reconstruction error

$$
\min_{e,\, g}\; \mathbb{E}_{x}\,\big\|x - g(e(x))\big\|^2 .
$$

Nothing in that loss mentions charts, manifolds or coordinates. So why should training produce a chart?

Suppose training works perfectly, so $$g(e(x)) = x$$ for every $$x \in \mathcal{M}$$. If two data points share a code, $$e(x) = e(x')$$, apply $$g$$ to both sides and you get $$x = g(e(x)) = g(e(x')) = x'$$. That makes $$e$$ injective on the data, with $$g$$ undoing $$e$$ on the codes $$e(\mathcal{M})$$. Both networks are continuous, so $$e\vert_{\mathcal{M}}$$ is a continuous bijection whose inverse $$g$$ is also continuous. That's a homeomorphism, which is the definition of a chart. **Perfect reconstruction forces the encoder to be a chart of the data manifold, with the decoder as its inverse.**

Hold on to the words "on the data", because the rest of the series turns on them. The argument says nothing about what $$g$$ does at codes no training point maps to, or what $$e$$ does to images off $$\mathcal{M}$$.

![The encoder e maps the data manifold M in pixel space R^784 onto codes e(M) in a 2-D latent space, and the decoder g inverts it on e(M) only; the rest of the latent space (coral) gets no training signal]({{ '/assets/blog/what-is-latent-space/chart-pair.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The encoder (top arrow) takes the sheet of real data in pixel space and lays it out in the latent space as the teal region, and the decoder (bottom arrow) maps it back. Training only ever checks the teal region. The coral area around it is never tested, so nothing tells us what the decoder produces there.*
{: .caption}

If the encoder really is a chart of Blob World, its codes should look like the true $$(c_x, c_y)$$ square drawn on a rubber sheet. It can be stretched or bent but not torn or folded, so colouring the codes by $$c_x$$, or by $$c_y$$, should give smooth gradients.

The cell trains `AE(d=2)` on the 5,000 blobs, then tries to predict each blob's true centre from $$z$$ in two ways. A linear regression asks whether the codes are the true square up to a linear transformation (rotation, scaling, shear). A cross-validated 10-nearest-neighbour regression asks something weaker, whether $$z$$ pins down the centre through any continuous function at all, and any chart passes that. PCA-2 gets both tests for comparison.

```python
_, Z_ae, mse_ae = train_ae(flat(X_blob), d=2, epochs=60)

def chart_scores(Z: np.ndarray, C: np.ndarray) -> tuple[float, float]:
    """R^2 of predicting the true centre from z: linearly, and with 10-NN (any smooth function)."""
    linear = LinearRegression().fit(Z, C).score(Z, C)
    smooth = cross_val_score(KNeighborsRegressor(n_neighbors=10), Z, C, cv=5, scoring="r2").mean()
    return linear, smooth

print(f"AE(d=2) reconstruction MSE: {mse_ae:.5f}")
for name, Z in [("AE z", Z_ae), ("PCA-2", Z_pca)]:
    linear, smooth = chart_scores(Z, C_blob)
    print(f"{name:6s} -> true (x, y):  linear R^2 = {linear:.3f} | 10-NN R^2 = {smooth:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
panels = [(Z_ae, 0, "AE latent, coloured by true x"), (Z_ae, 1, "AE latent, coloured by true y"), (Z_pca, 0, "PCA-2, coloured by true x")]
for ax, (Z, j, title) in zip(axes, panels):
    ax.scatter(Z[:, 0], Z[:, 1], c=C_blob[:, j], cmap="viridis", s=3)
    ax.set_title(title)
    ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
plt.tight_layout()
plt.show()
```
```text
AE(d=2) reconstruction MSE: 0.00039
AE z   -> true (x, y):  linear R^2 = 0.925 | 10-NN R^2 = 1.000
PCA-2  -> true (x, y):  linear R^2 = 0.758 | 10-NN R^2 = 0.921
```
![Output of cell 6]({{ '/assets/blog/what-is-latent-space/cell-06-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows: each dot is one blob image, placed at its two autoencoder code numbers. In the first panel the colour is the blob's true left-right position, in the second its true up-down position. Each colour changes smoothly in a single direction, so the autoencoder has rebuilt the blob grid, stretched into a pincushion and turned on its side but not torn. The third panel is PCA's version for comparison, where the colours get mixed up.*
{: .caption}

Reconstruction MSE is 0.00039 per pixel, and $$z$$ predicts the true centre with a held-out 10-NN $$R^2$$ of 1.000. The codes know exactly where the blob is. The linear $$R^2$$ is lower, 0.925, and the plot shows why: the square came out as a pincushion, its corners pulled into points and its middle squashed, with $$c_x$$ running up $$z_2$$ and $$c_y$$ running along $$z_1$$. The network swapped the axes, too. None of this breaks anything, since a chart is only defined up to a smooth invertible change of coordinates (the transition map again). The autoencoder found a perfectly good chart; it just isn't the one we drew.

PCA-2 scores 0.758 and 0.921, and the second number is the telling one. Even a flexible regressor can't undo PCA's folds, which is roughly the difference between bending paper and creasing it.

I made one change from the obvious setup. The autoencoder uses ELU activations instead of ReLU, because in my runs a ReLU version cut the square into separate flaps on three seeds out of three while reconstructing just as well. A gap where no training image lands costs the loss nothing, so this is the "on the data" caveat in action. Swap in `nn.ReLU` if you want to see it happen.

## Compression is a side effect

The usual one-line definition of a latent space is "a compressed representation". JPEG compresses a blob image very well, but nudge one of its DCT coefficients and you get a faint ripple pattern inside one 8×8 block instead of a blob that moved half a pixel. Small steps in JPEG's coordinates take you off $$\mathcal{M}$$.

The autoencoder's two numbers are useful for the opposite reason. Because $$z$$ is a chart, continuous in both directions, moving $$e$$ a little moves the blob a little. That's what we actually want from a latent space, continuity and directions that mean something, and compression comes along for free once $$d$$ matches $$k$$.

It also means more latent dimensions aren't automatically better. A 784-dimensional latent can reconstruct perfectly with the identity map and tell you nothing about the sheet. Use too few, on the other hand, and a faithful chart can become impossible for reasons that have nothing to do with how much information you keep.

## Why a circle has to tear

Try laying a necklace along a ruler so that every bead gets its own spot and neighbouring beads stay neighbours. Somewhere you'll have to cut the string, and the two beads on either side of the cut end up at opposite ends of the ruler.

Clock Hand images are a necklace. Each one shows a thin bar running out from the centre with a fat dot at its tip, rotated to some angle $$\theta \in [0, 2\pi)$$. There's one knob, $$k = 1$$, but it wraps around, since 359° and 1° look almost the same. So the data manifold is a circle, $$S^1$$, sitting in $$\mathbb{R}^{784}$$. The hand has to point one way on purpose. A bar centred on the pivot, like a propeller, looks the same after half a turn, which would quietly halve the circle.

The necklace problem has a short proof. *There is no continuous injective map from a circle to a line.* Suppose $$f : S^1 \to \mathbb{R}$$ were one, and pick two points $$p \neq q$$. They split the circle into two arcs, and both arcs run continuously from $$f(p)$$ to $$f(q)$$. By the intermediate value theorem each arc passes through $$\tfrac12\big(f(p) + f(q)\big)$$, so two different points share an image. $$\blacksquare$$

An encoder with $$d = 1$$ is a continuous network, so it runs into exactly this wall, and it can fail in two ways. It can fold, reusing codes for different angles so the decoder can't tell them apart. Or it can tear, hiding a near-vertical cliff in the gap between two training angles where no data checks it. Give it $$d = 2$$ and the circle fits as a ring. Topology can therefore force $$d > k$$: a circle needs two latent dimensions, and a sphere or a torus needs three.

Picture the ring of Clock Hand images being cut at one point and straightened. The two images that sat side by side at the cut end up as far apart as they can be.

![Twelve Clock Hand images on a teal circle; the circle is cut at 345 degrees and straightened into a line, and the 0-degree and 330-degree hands, neighbours on the circle, end up outlined in coral at opposite ends of the line.]({{ '/assets/blog/what-is-latent-space/chart-tear.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Twelve clock-hand images sit around a circle, each next to the ones it most resembles. Straightening the circle into a line forces exactly one cut, and the two hands on either side of the cut (outlined in coral) end up at opposite ends of the line even though they looked almost the same. A latent space with only one number has to make a cut like this somewhere.*
{: .caption}

The cell renders 1,500 Clock Hand images and builds three charts, `AE(d=1)`, `AE(d=2)` and PCA-2. Then it walks round the circle in order of true angle, measures each chart's speed, $$\|\Delta z\| / \Delta\theta$$, between neighbouring images, and prints the peak as a multiple of the median along with where it happens. Smooth charts move at a steady pace. Tears spike.

```python
theta = np.random.default_rng(2).uniform(0, 2 * np.pi, 1500)
X_clock = flat(np.stack([clock_hand(t) for t in theta]))
order = np.argsort(theta)
gaps = np.diff(np.r_[theta[order], theta[order][0] + 2 * np.pi])  # angle between neighbours, loop closed

def tear_ratio(Z: np.ndarray) -> tuple[float, int]:
    """Fastest speed |dz/dtheta| around the circle / the median speed. A tear shows up as a huge spike."""
    Zs = Z[order]
    speed = np.linalg.norm(np.diff(np.vstack([Zs, Zs[:1]]), axis=0), axis=1) / gaps
    return speed.max() / np.median(speed), int(np.argmax(speed))

_, Z1, mse1 = train_ae(X_clock, d=1, epochs=60)
_, Z2, mse2 = train_ae(X_clock, d=2, epochs=60)
Zp = PCA(n_components=2).fit_transform(X_clock)
for name, Z, mse in [("AE(d=1)", Z1, mse1), ("AE(d=2)", Z2, mse2), ("PCA-2", Zp, None)]:
    ratio, i = tear_ratio(Z)
    a, b = np.degrees(theta[order][i]), np.degrees(theta[order][(i + 1) % len(theta)])
    extra = f" | recon MSE {mse:.5f}" if mse is not None else ""
    print(f"{name:8s} max speed = {ratio:6.1f} x median, between angles {a:5.1f} and {b:5.1f} deg{extra}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].scatter(np.degrees(theta), Z1[:, 0], c=theta, cmap="twilight", s=3)
axes[0].set_xlabel(r"true angle $\theta$ (deg)"); axes[0].set_ylabel("z"); axes[0].set_title("AE(d=1): z vs angle")
for ax, Z, title in [(axes[1], Z2, "AE(d=2) latent"), (axes[2], Zp, "PCA-2 of Clock Hand")]:
    ax.scatter(Z[:, 0], Z[:, 1], c=theta, cmap="twilight", s=3)
    _, i = tear_ratio(Z)
    ends = Z[order][[i, (i + 1) % len(theta)]]
    ax.plot(ends[:, 0], ends[:, 1], "o", mfc="none", mec=CORAL, ms=12, mew=2)  # where the chart moves fastest
    ax.set_title(title); ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
plt.tight_layout()
plt.show()
```
```text
AE(d=1)  max speed =   65.1 x median, between angles 273.8 and 274.0 deg | recon MSE 0.00593
AE(d=2)  max speed =    1.7 x median, between angles 147.2 and 147.3 deg | recon MSE 0.00031
PCA-2    max speed =    1.6 x median, between angles  26.2 and  27.2 deg
```
![Output of cell 7]({{ '/assets/blog/what-is-latent-space/cell-07-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*What the figure shows. Left: the 1-D autoencoder's single code number for every hand angle; the sudden drop just past 265° is the tear. Middle: with two code numbers, the codes form a closed loop that goes all the way round with no break. Right: PCA's two directions also give a closed loop. Colour shows the true angle, and the coral circles mark where each chart moves fastest.*
{: .caption}

The 1-D autoencoder tears, as it has to. Its speed spikes to 65.1 times the median between 273.8° and 274.0°, the steepest step of a cliff in the left panel where, over about 11 degrees from 267° to 278°, the code drops from about 20 to about −7. It folds as well. The code bottoms out at about −17 near 347° and climbs back up through 0°, so 310° and 1.4° end up with the same code (about −9.3), and the cliff itself crosses codes already used elsewhere, landing 274° on the same code as 179°. You can see the damage in its reconstruction error, 0.00593, which is 19 times the 2-D model's.

Give the autoencoder a second dimension and the problem goes away. With $$d = 2$$ its peak speed is 1.7 times the median, which is ordinary variation, and its codes form a closed ring with the colours running continuously all the way round. PCA-2 closes the ring too (1.6×), so what blocked the 1-D chart was the number of dimensions rather than linearity. The coral circles on the two rings only mark where they stretch a little faster.

Activation functions matter here as well. With ReLU, the 2-D autoencoder left the ring open on five seeds out of five in my runs and still reconstructed almost perfectly. A second dimension makes a tear-free chart possible, but nothing in the loss makes the network find it. The data's topology decides which charts exist, and a loss that only ever looks at data points decides which one you end up with.

## What latent spaces buy you

The most concrete payoff is compute. [Latent diffusion](https://arxiv.org/abs/2112.10752) runs the expensive generative model inside an autoencoder's latent space. [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) downsamples by a factor of 8 into 4 channels, which turns a 512×512×3 image (786,432 numbers) into a 64×64×4 latent (16,384) and leaves the denoiser 48 times fewer numbers to work with.

The geometry pays off as well. Because nearby codes decode to nearby data, directions in a good latent space are moves along the manifold, and nearest-neighbour search compares positions on the sheet instead of the saturated pixel distances we saw earlier. Train an autoencoder to undo noise, and encoding then decoding pulls a corrupted image back towards $$\mathcal{M}$$. Pick any point, decode it, and you get data, provided the decoder was trained somewhere near there.

> **A t-SNE or UMAP plot is not a latent space.** [t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html) and [UMAP](https://arxiv.org/abs/1802.03426) try to keep local neighbourhoods intact when they squash data into 2-D. That's all they promise. There's no decoder, and global distances aren't preserved (cluster sizes and the gaps between clusters can mean very little). Standard t-SNE can't even place a new point without re-running the optimisation; UMAP can, through a built-in transform, but still has no way back to the data. So a t-SNE plot of a network's features is a picture of neighbourhoods in its latent space, drawn with a different ruler.

## Recent work

If you read one thing after this post, make it [Loaiza-Ganem et al. (2024, TMLR)](https://arxiv.org/abs/2404.02954). It's a survey that sorts deep generative models by whether they account for data living on a low-dimensional manifold, and two-step models like latent diffusion, which learn a representation first and model the distribution inside it second, fit its framework naturally.

Diffusion models can estimate $$k$$ themselves. [Stanczuk et al. (ICML 2024)](https://arxiv.org/abs/2212.12611) show that a diffusion model's score at small noise levels points back towards the manifold, which reveals $$k$$. [Kamkari et al. (NeurIPS 2024)](https://arxiv.org/abs/2406.03537) turned that into FLIPD, a fast estimate for each individual data point, which makes the "dimension varies across the data" caveat something you can measure. Both do TwoNN's job at the scale of real image models.

Learned representations have intrinsic dimensions too. [Cheng et al. (ICLR 2025)](https://arxiv.org/abs/2405.15471) track it layer by layer in language models and find a distinct high-dimensional phase in the middle of the network. The shape of the chart also matters for generation: [Yao et al. (2025)](https://arxiv.org/abs/2501.01423) find that giving a tokenizer a higher-dimensional latent improves reconstruction but hurts generation at a fixed model size and training budget, and that aligning the latent with a pretrained vision model such as DINOv2 closes much of that gap.

## What happens between the data points?

So a latent space is a learned chart. The encoder gives every point on the data manifold a coordinate, and perfect reconstruction makes the decoder that chart's inverse. Along the way we measured the manifold's dimension (2.01 for Blob World, 13.35 for MNIST), watched PCA fold it, watched an autoencoder recover it ($$R^2 = 1.000$$), and watched a circle force a 1-D chart to tear.

Every one of those checks happened at the data points and nowhere else. Try this, then. Take two training images, encode them, and walk in a straight line between their codes. Halfway along you're at a code that belongs to no training image. What does the decoder show you there? Sometimes a perfectly good image, sometimes garbage, and working out why is the job of **Part 2: why averaging pixels gives you a ghost, while averaging latents gives you a hybrid.**

## Further reading

- [Pope et al., 2021](https://arxiv.org/abs/2104.08894): measured intrinsic dimensions of standard image datasets, and why they matter for learning.
- [Facco et al., 2017](https://arxiv.org/abs/1803.06992): the TwoNN paper, short, with derivation and robustness checks.
- [Loaiza-Ganem et al., 2024](https://arxiv.org/abs/2404.02954): generative models through the manifold hypothesis.
- [Bengio, Courville & Vincent, 2013](https://arxiv.org/abs/1206.5538): the classic representation-learning review that put manifolds at the centre.
