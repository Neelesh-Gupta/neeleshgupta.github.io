---
layout: post
title: "Why Batch Norm struggles inside transformers!"
date: 2026-11-20 09:00:00+0530
description: Why CNNs normalize over the batch and transformers normalize each token, explained through geometry, the math of one shared formula, and runnable PyTorch code.
tags: normalization batch-norm layer-norm transformers pytorch
categories: machine-learning
related_posts: false
toc:
  sidebar: left
---

Open a ResNet and there's a `BatchNorm2d` after nearly every convolution. Open a GPT-style model and you get `LayerNorm` (or its cousin `RMSNorm`) in front of every attention and MLP block. On paper the two do the same job: subtract a mean, divide by a standard deviation, apply a learned scale and shift. So why did the two families of models end up with different ones?

In an earlier #BehindTheBasics post I gave the short answer: it comes down to which axis you compute the statistics over. Batch Norm compares a feature with the same feature in the other samples of the batch. Layer Norm looks at one sample and nothing else. My analogy was grading. Batch Norm grades on a curve, so your mark depends on who else sat the exam; Layer Norm marks each paper on its own.

The axis part I'd still defend. Everything around it was too quick, and two of that post's claims don't hold up: Batch Norm in an LLM does not "divide by zero" at batch size 1, and Layer Norm in a CNN does not automatically wreck accuracy. The tidy rule "images get Batch Norm, text gets Layer Norm" has been broken from both ends, too. ViT and ConvNeXt run Layer Norm on images, and most open LLMs since LLaMA use RMSNorm instead of Layer Norm.

This is the long version. I'll go through the claims one at a time and look at each from four sides: where the norm sends a vector, what it makes the model blind to, the derivation, and a code cell you can run (every number quoted in the text is one those cells printed). The thread through all of it is that the five layers are one formula applied over different axes of the same tensor. Once you can see the geometry of those axes, the rule of thumb falls out, and so do its exceptions.

Here's the grading analogy stretched to cover four of the five norms:

![Four copies of one gradebook (4 students by 4 questions, each question split into 3 parts) showing which scores each norm compares: one question across the whole class for Batch Norm, one student's whole paper for Layer Norm, one student's single question for Instance Norm, and one student's group of questions for Group Norm]({{ '/assets/blog/batch-norm-transformers/grading-analogy.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

## One formula, five choices of axes

Everything starts from two tensor shapes. After a convolution, a CNN's activations have shape $$(N, C, H, W)$$: $$N$$ images, $$C$$ channels, and an $$H \times W$$ grid of positions. A transformer's have shape $$(B, T, D)$$: $$B$$ sequences, $$T$$ token positions, and a $$D$$-dimensional vector per token. A normalization layer picks some set $$S$$ of these axes, takes the mean and variance over them, and uses those two numbers to standardize every entry that shares them.

To write that down, let $$x$$ be one entry and $$S(x)$$ the group of entries it shares statistics with, meaning everything that matches $$x$$ on the axes *not* in $$S$$. Then

$$
\mu_S = \frac{1}{|S(x)|} \sum_{x' \in S(x)} x', \qquad
\sigma_S^2 = \frac{1}{|S(x)|} \sum_{x' \in S(x)} (x' - \mu_S)^2, \qquad
y = \gamma \, \frac{x - \mu_S}{\sqrt{\sigma_S^2 + \epsilon}} + \beta .
$$

$$\epsilon$$ is a small constant (PyTorch's default is $$10^{-5}$$) that keeps the division finite. $$\gamma, \beta$$ are a learned scale and shift, one per channel or feature, which let the network undo the normalization if it wants to. One detail matters if you ever check your own code against PyTorch: the variance is the biased one, divided by $$\lvert S(x)\rvert$$ and not $$\lvert S(x)\rvert - 1$$. Every library norm layer does it this way.

With the formula fixed, the five layers differ only in $$S$$:

| Layer | Tensor | $$S$$ (axes averaged over) | One mean and variance per |
|---|---|---|---|
| Batch Norm | $$(N, C, H, W)$$ | $$N, H, W$$ | channel |
| Layer Norm (CNN form) | $$(N, C, H, W)$$ | $$C, H, W$$ | image |
| Layer Norm (transformer) | $$(B, T, D)$$ | $$D$$ | token |
| Instance Norm | $$(N, C, H, W)$$ | $$H, W$$ | image and channel |
| Group Norm | $$(N, C, H, W)$$ | $$C/G, H, W$$ (one group of channels) | image and group |
| RMSNorm | $$(B, T, D)$$ | $$D$$, and no mean subtraction | token |

Only Batch Norm puts the batch axis in its $$S$$. The other four compute their statistics inside a single sample, and most of this post comes back to that one difference.

![Five copies of an (N, C, H, W) tensor drawn as a block of cubes, with the cells that share one mean and variance shaded: a whole channel slab across the batch for Batch Norm, one whole image for Layer Norm, one channel of one image for Instance Norm, two channels of one image for Group Norm, and the Layer Norm set in a different color for RMSNorm]({{ '/assets/blog/batch-norm-transformers/norm-cube-grid.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Here is the same formula with its parts labelled. Only the statistics change from layer to layer; the small constant and the learned scale and shift play the same role in all five.

![The normalization formula y = gamma (x - mu_S) / sqrt(sigma_S^2 + eps) + beta with each part labelled, and five boxes below showing Batch Norm, Layer Norm, Instance Norm, Group Norm and RMSNorm as different choices of the axis set S]({{ '/assets/blog/batch-norm-transformers/formula-anatomy.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

And the animated version, if you'd like to watch the shaded set move around while the formula stays put:

![A 4 x 6 x 8 activation tensor (batch N, channels C, positions H, W) where the cells sharing one mean and variance light up in turn: a whole channel slab for Batch Norm, a whole image for Layer Norm, one channel of one image for Instance Norm, two channels of one image for Group Norm, and the Layer Norm set again for RMSNorm, while the formula beside it never changes]({{ '/assets/blog/batch-norm-transformers/norm-cube-sweep.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Time for code. Cell 1 pins the versions I tested with, seeds every random generator and keeps everything on the CPU, so your printed numbers should match mine.

```python
%pip install -q "torch==2.13.0" "numpy==2.3.3" "matplotlib==3.10.8" "scikit-learn==1.6.1"
import math
import random
import statistics

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(1)  # every model here is tiny; one thread is faster and keeps numbers reproducible
torch.set_printoptions(precision=4, sci_mode=False)
device = "cpu"            # CPU on purpose: the printed numbers match for every reader
print(f"torch {torch.__version__} | numpy {np.__version__} | device={device}")
```
```text
torch 2.13.0+cpu | numpy 2.3.3 | device=cpu
```

Cell 2 turns that table into code. `normalize(x, axes)` is the formula with $$\gamma = 1, \beta = 0$$, and its argument `axes` is $$S$$. Group Norm needs one reshape first: split the $$C$$ channels into $$G$$ groups so that "one group of channels" becomes an axis you can average over. RMSNorm is the same call with `center=False`. Each line is checked against the matching PyTorch layer, and the cell prints the worst disagreement.

```python
def normalize(x: torch.Tensor, axes: tuple[int, ...], eps: float = 1e-5, center: bool = True) -> torch.Tensor:
    """(x - mean_S) / sqrt(var_S + eps), where S = `axes`. center=False gives RMSNorm."""
    mu = x.mean(dim=axes, keepdim=True) if center else 0.0
    var = ((x - mu) ** 2).mean(dim=axes, keepdim=True)  # biased variance, as every norm layer uses
    return (x - mu) / torch.sqrt(var + eps)

N, C, H, W, G = 8, 6, 5, 5, 3
x = torch.randn(N, C, H, W) * 3 + 1
ours = {
    "BatchNorm":    (normalize(x, (0, 2, 3)), "N, H, W"),
    "LayerNorm":    (normalize(x, (1, 2, 3)), "C, H, W"),
    "InstanceNorm": (normalize(x, (2, 3)), "H, W"),
    "GroupNorm":    (normalize(x.view(N, G, C // G, H, W), (2, 3, 4)).view_as(x), "C/G, H, W"),
    "RMSNorm":      (normalize(x, (1, 2, 3), center=False), "C, H, W (no centering)"),
}
reference = {
    "BatchNorm":    nn.BatchNorm2d(C, affine=False).train()(x),
    "LayerNorm":    nn.LayerNorm((C, H, W), elementwise_affine=False)(x),
    "InstanceNorm": nn.InstanceNorm2d(C)(x),
    "GroupNorm":    nn.GroupNorm(G, C, affine=False)(x),
    "RMSNorm":      nn.RMSNorm((C, H, W), eps=1e-5, elementwise_affine=False)(x),
}
print(f"{'norm':<14}{'averages over':<25}{'max |ours - torch|':>18}")
for name, (y, axes_text) in ours.items():
    print(f"{name:<14}{axes_text:<25}{(y - reference[name]).abs().max().item():>18.1e}")

tokens = torch.randn(4, 7, 16)  # (B, T, D): a transformer's activations
print("transformer LayerNorm == normalize over D only:",
      torch.allclose(normalize(tokens, (2,)), F.layer_norm(tokens, (16,)), atol=1e-6))
```
```text
norm          averages over            max |ours - torch|
BatchNorm     N, H, W                             4.8e-07
LayerNorm     C, H, W                             3.6e-07
InstanceNorm  H, W                                3.6e-07
GroupNorm     C/G, H, W                           4.8e-07
RMSNorm       C, H, W (no centering)              2.4e-07
transformer LayerNorm == normalize over D only: True
```

All five match PyTorch to within $$5 \times 10^{-7}$$ (the worst gap, `4.8e-07`, is for Batch Norm and Group Norm), which is just float32 rounding. I find this the most satisfying output in the post. Five library classes, each with its own constructor arguments, reduce to a four-line function with a single knob, `axes`. The last line covers the transformer case: `F.layer_norm` on a $$(B, T, D)$$ tensor is `normalize` over the last axis alone, and the check prints `True`. Every later cell reuses this function, so whenever two norms behave differently below, the `axes` tuple is the only thing that changed.

## Layer Norm is a projection onto a sphere

Take a single token's vector $$x \in \mathbb{R}^d$$ and run Layer Norm on it with $$\gamma = 1, \beta = 0$$ and, for now, $$\epsilon = 0$$. Two things happen, and both have a clean picture.

First, centering. It turns out to be a projection. Let $$\mathbf{1} = (1, 1, \dots, 1)$$ be the all-ones vector in $$\mathbb{R}^d$$, and note that the mean of $$x$$'s entries is $$\mu = \frac{1}{d}\mathbf{1}^\top x$$, so

$$
x - \mu \mathbf{1} = \Big(I - \tfrac{1}{d}\mathbf{1}\mathbf{1}^\top\Big)\, x = P x .
$$

Look at the matrix $$P = I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top$$. It's symmetric, it satisfies $$P^2 = P$$, and it sends $$\mathbf{1}$$ to zero ($$P\mathbf{1} = \mathbf{1} - \mathbf{1} = 0$$). That makes it an orthogonal projection onto the hyperplane $$\mathbf{1}^\perp = \{z : z_1 + \dots + z_d = 0\}$$. In words: centering slides $$x$$ along $$\mathbf{1}$$ until its coordinates add up to zero, then forgets how far it slid.

The second step, dividing by the standard deviation, lands the vector on a sphere. The variance is the mean squared entry of the centered vector, $$\sigma^2 = \frac{1}{d}\lVert P x \rVert^2$$, so $$\sigma = \lVert Px \rVert / \sqrt{d}$$ and

$$
\mathrm{LN}(x) = \frac{P x}{\sigma} = \sqrt{d}\, \frac{P x}{\lVert P x \rVert},
\qquad
\lVert \mathrm{LN}(x) \rVert = \sqrt{d} .
$$

Every output has length $$\sqrt{d}$$, however long the input was. Put the two steps together and Layer Norm drops $$x$$ onto the hyperplane $$\mathbf{1}^\perp$$, then pushes it out to the sphere of radius $$\sqrt{d}$$ inside that hyperplane, which is a $$(d-2)$$-dimensional sphere. Each token loses exactly two numbers, its mean and its scale, and keeps its direction within $$\mathbf{1}^\perp$$. The learned $$\gamma$$ and $$\beta$$ then stretch the sphere into an ellipsoid and move it, so the next layer always receives a fixed, learned shape instead of activations of whatever size happened to come in.

![Forty 3D points first slide along the all-ones direction onto the plane x + y + z = 0, then move outward along their own rays until every one sits on the red circle of radius sqrt(3) in that plane: Layer Norm as projection then rescaling]({{ '/assets/blog/batch-norm-transformers/ln-sphere-projection.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

RMSNorm skips the first step. [Zhang & Sennrich (2019)](https://arxiv.org/abs/1910.07467) argued that re-centering contributes little to Layer Norm, so they dropped it and divided by the root mean square in place of the standard deviation:

$$
\mathrm{RMS}(x) = \sqrt{\tfrac{1}{d}\textstyle\sum_i x_i^2} = \frac{\lVert x \rVert}{\sqrt{d}},
\qquad
\mathrm{RMSNorm}(x) = \frac{x}{\mathrm{RMS}(x)} = \sqrt{d}\,\frac{x}{\lVert x \rVert} .
$$

The length is still $$\sqrt{d}$$, but the output can now sit anywhere on the full $$(d-1)$$-dimensional sphere; Layer Norm's outputs were confined to the slice of it that lies in $$\mathbf{1}^\perp$$. RMSNorm throws away the scale and keeps the mean. It's cheaper too: one reduction (the sum of squares) against Layer Norm's two (the mean, then the variance).

![Side by side, the same points under RMSNorm move radially onto the sphere of radius sqrt(3) and stay off the plane, while under Layer Norm they first drop onto the plane x + y + z = 0 and end on the red circle where that plane cuts the sphere]({{ '/assets/blog/batch-norm-transformers/rms-vs-ln.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Cell 3 tests both claims on 300 points in $$d = 3$$. In three dimensions the "sphere inside the plane" is a circle of radius $$\sqrt{3} \approx 1.7321$$ in the plane $$x + y + z = 0$$, which is small enough to actually plot. I set `eps=0.0` because the epsilon is the only thing that would nudge the lengths away from $$\sqrt{3}$$.

```python
points = torch.randn(300, 3) * torch.tensor([2.0, 0.7, 1.2]) + torch.tensor([1.5, -1.0, 0.5])
ln_points = normalize(points, (1,), eps=0.0)                 # each point (row) on its own
rms_points = normalize(points, (1,), eps=0.0, center=False)  # RMSNorm: rescale only
for name, p in [("LayerNorm", ln_points), ("RMSNorm", rms_points)]:
    norms = p.norm(dim=1)
    print(f"{name:<10} max |x+y+z| = {p.sum(1).abs().max():.2e} | ||y|| in [{norms.min():.4f}, {norms.max():.4f}]")
print(f"sqrt(d) = sqrt(3) = {math.sqrt(3):.4f}")

u = torch.tensor([1.0, -1.0, 0.0]) / math.sqrt(2)  # orthonormal basis of the plane x+y+z=0
v = torch.tensor([1.0, 1.0, -2.0]) / math.sqrt(6)
t = torch.linspace(0, 2 * math.pi, 200)
circle = math.sqrt(3) * (torch.outer(t.cos(), u) + torch.outer(t.sin(), v))
fig = plt.figure(figsize=(14, 4.6))
panels = [("raw points", points), ("LayerNorm: plane x+y+z=0, radius sqrt(3)", ln_points),
          ("RMSNorm: radius sqrt(3), off the plane", rms_points)]
for k, (title, p) in enumerate(panels, 1):
    ax = fig.add_subplot(1, 3, k, projection="3d")
    ax.scatter(*p.T, s=6, alpha=0.6)
    if k > 1:
        ax.plot(*circle.T, color="crimson", lw=2)
    lim = 4.5 if k == 1 else 2.0
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), xlabel="x", ylabel="y", zlabel="z", title=title)
    ax.set_box_aspect((1, 1, 1))
plt.tight_layout()
plt.show()
```
```text
LayerNorm  max |x+y+z| = 1.61e-06 | ||y|| in [1.7321, 1.7321]
RMSNorm    max |x+y+z| = 2.99e+00 | ||y|| in [1.7321, 1.7321]
sqrt(d) = sqrt(3) = 1.7321
```
![Output of cell 3]({{ '/assets/blog/batch-norm-transformers/cell-03-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Both predictions hold to float precision. After Layer Norm, no point's coordinates sum to more than `1.61e-06` in absolute value, so all 300 points are in the plane, and every length falls in `[1.7321, 1.7321]`. In the middle panel, the stretched, off-center cloud from the left has collapsed onto the red circle. RMSNorm also gives every point length `1.7321`, but its coordinate sums go as high as `2.99`, and in the right panel the points are spread over the sphere instead of sitting on the circle. Layer Norm took two numbers from each point (mean and scale); RMSNorm took one.

The sphere shows up in the backward pass as well. Since $$\mathrm{LN}(c\,x) = \mathrm{LN}(x)$$ for any $$c > 0$$, and $$\mathrm{LN}(x + b\mathbf{1}) = \mathrm{LN}(x)$$ for any $$b$$, nudging the input along $$x$$ or along $$\mathbf{1}$$ doesn't move the output. Differentiate those two identities and you find that the gradient of the loss with respect to $$x$$ is orthogonal to both $$x$$ and $$\mathbf{1}$$, so it lies in the tangent plane of the output sphere. Its size goes as $$1/\lVert Px \rVert$$, which means inputs with large activations get proportionally smaller gradients. [Xu et al. (2019)](https://arxiv.org/abs/1911.07013) found that these backward-pass effects (the gradients flowing through $$\mu$$ and $$\sigma$$) matter more for training than the forward normalization itself. I'm skipping the derivation for this one and just showing the result:

![A point LN(x) on the radius-sqrt(3) circle in the plane x + y + z = 0, with dashed arrows along x and along the all-ones vector (directions that leave the output unchanged) and a solid arrow tangent to the circle where the gradient lies]({{ '/assets/blog/batch-norm-transformers/ln-gradient-tangent.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

## Batch Norm reshapes the cloud

Now take a whole batch: $$B$$ samples with $$d$$ features each, stacked into a matrix $$X \in \mathbb{R}^{B \times d}$$ with one sample per row and one feature per column. Layer Norm works row by row, applying the projection-and-sphere map to each sample on its own. Batch Norm works column by column. For feature $$j$$ it computes

$$
\mu_j = \frac{1}{B}\sum_{i=1}^{B} X_{ij}, \qquad
\sigma_j^2 = \frac{1}{B}\sum_{i=1}^{B} (X_{ij} - \mu_j)^2, \qquad
\hat X_{ij} = \frac{X_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} .
$$

The picture I find most useful is a cloud of $$B$$ points in $$\mathbb{R}^d$$. Batch Norm slides the whole cloud so its centroid is at the origin, then stretches or squeezes it along each coordinate axis until the spread along every axis is 1. That's a single affine map, $$x \mapsto D^{-1}(x - \mu)$$ with $$D = \mathrm{diag}(\sigma_1, \dots, \sigma_d)$$, applied to every sample alike.

Because everyone gets the same map, two things survive it. Relative positions survive: a sample far out to the right of the cloud is still far out to the right afterwards, so Batch Norm erases nothing about an individual sample and only re-expresses each feature as "standard deviations from the batch average". Correlations survive too, because Batch Norm scales the axes and never rotates them. Removing correlations would take the full inverse square root of the covariance matrix, $$\Sigma^{-1/2}$$ (whitening), and Batch Norm only has the diagonal $$D^{-1}$$. A tilted, stretched cloud comes out still tilted.

![The 200-sample batch from cell 4 shifts its centroid to the origin and rescales each feature axis until its mean plus or minus one standard deviation box becomes the unit square, while its tilt, its correlation of 0.874 and the order of three highlighted samples stay the same]({{ '/assets/blog/batch-norm-transformers/bn-cloud-reshape.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Training and evaluation are different stories. During training, $$\mu_j$$ and $$\sigma_j$$ come from the current batch, so a sample's output depends on which other samples it happened to share a batch with. Batch Norm also keeps running averages, updated after each training batch as $$\hat\mu_j \leftarrow (1 - m)\,\hat\mu_j + m\,\mu_j$$ with momentum $$m = 0.1$$ in PyTorch (and the same for the variance). In evaluation mode (`model.eval()`) it switches to those running statistics and becomes a fixed per-feature affine map, $$y_j = \gamma_j (x_j - \hat\mu_j)/\sqrt{\hat\sigma_j^2 + \epsilon} + \beta_j$$, which doesn't look at the batch at all. Hold on to this one, because it comes back later: a trained Batch Norm model can run inference on one sample at a time without any trouble.

Cell 4 runs both norms on the same 200-sample, 2-feature batch. Two dimensions is a deliberately extreme case for Layer Norm. With $$d = 2$$ the hyperplane $$\mathbf{1}^\perp$$ is the line $$x_1 + x_2 = 0$$, and the "sphere of radius $$\sqrt{2}$$" inside a line is just two points, $$(1, -1)$$ and $$(-1, 1)$$. Keep an eye on the three colored samples.

```python
gen = torch.Generator().manual_seed(0)
mixing = torch.tensor([[2.0, 0.0], [1.2, 0.6]])  # correlated features with very different scales
cloud = torch.randn(200, 2, generator=gen) @ mixing.T + torch.tensor([2.0, 1.0])
bn_cloud = normalize(cloud, (0,), eps=0.0)  # columns: one feature across the batch
ln_cloud = normalize(cloud, (1,), eps=0.0)  # rows: one sample on its own

def corr(z: torch.Tensor) -> float:
    return torch.corrcoef(z.T)[0, 1].item()

print(f"correlation: raw {corr(cloud):.3f} | after BN {corr(bn_cloud):.3f}")
print(f"after BN: column means {bn_cloud.mean(0).abs().max():.1e}, column stds {bn_cloud.std(0, unbiased=False).tolist()}")
print("after LN: distinct outputs", sorted({tuple(round(v, 4) + 0.0 for v in row) for row in ln_cloud.tolist()}))

marked = [0, 1, 2]  # follow three samples through both norms
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, (title, z) in zip(axes, [("raw batch (rows = samples)", cloud), ("BatchNorm: per column", bn_cloud),
                                 ("LayerNorm: per row (d = 2)", ln_cloud)]):
    ax.scatter(z[:, 0], z[:, 1], s=10, alpha=0.4, color="gray")
    for i, c in zip(marked, ["crimson", "royalblue", "darkorange"]):
        ax.scatter(z[i, 0], z[i, 1], s=70, color=c, edgecolor="black", zorder=3)
    ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
    ax.set(title=title, xlabel="feature 1", ylabel="feature 2", aspect="equal")
axes[2].plot([-1.5, 1.5], [1.5, -1.5], "--", color="crimson", label="x1 + x2 = 0 (where LN outputs live)")
axes[2].set(xlim=(-1.6, 1.6), ylim=(-1.6, 1.6)); axes[2].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()
```
```text
correlation: raw 0.874 | after BN 0.874
after BN: column means 7.6e-08, column stds [1.0, 1.0]
after LN: distinct outputs [(-1.0, 1.0), (1.0, -1.0)]
```
![Output of cell 4]({{ '/assets/blog/batch-norm-transformers/cell-04-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Batch Norm behaves the way the column picture says it should. Each column ends up with mean `7.6e-08` and standard deviation `1.0`, and the correlation between the two features is `0.874` before and `0.874` after. In the middle panel the cloud has moved to the origin and changed units, but its shape hasn't changed, and the red, blue and orange points sit in the same order as on the left.

Layer Norm maps all 200 samples to just two outputs: `(-1.0, 1.0)` and `(1.0, -1.0)`. In two dimensions a sample keeps one bit, whether its first feature is bigger than its second, and all three colored samples land on the same point. That looks brutal, but it's what happens in every dimension, only easier to see here: Layer Norm strips each sample's own offset and scale. In $$d = 768$$ you still keep 766 degrees of freedom, which is plenty. The two you lose are gone for good, though, and whether that matters is the next question.

## What each norm makes your model blind to

I think the clearest way to tell these layers apart is by what they ignore, and one identity covers every case. Take the entries that share statistics, $$S(x)$$, and replace each one with $$a x' + b$$, using the same $$a > 0$$ and $$b$$ for the whole group. Then $$\mu_S \to a\mu_S + b$$ and $$\sigma_S \to a\sigma_S$$, so

$$
\frac{(a x + b) - (a\mu_S + b)}{a\,\sigma_S} = \frac{x - \mu_S}{\sigma_S} .
$$

With $$\epsilon = 0$$ the output doesn't move at all. So a norm layer is blind to any positive scale and any shift that's constant across its group $$S(x)$$. RMSNorm doesn't subtract a mean, which makes it blind to the scale $$a$$ but not to the shift $$b$$. With $$\epsilon > 0$$ the blindness is only approximate: once $$a\sigma_S$$ becomes comparable to $$\sqrt{\epsilon}$$, the output starts to change.

Read the groups off the first table and you get a concrete list:

| Layer | Blind to (in training) | Image example | Token example |
|---|---|---|---|
| Batch Norm | one scale and shift per channel, shared by the whole batch | the same exposure change applied to every image in the batch | none in practice |
| Layer Norm $$(C, H, W)$$ | one scale and shift per image | that image's overall brightness and contrast | n/a |
| Instance Norm | one scale and shift per image and channel | that image's color cast and per-channel contrast (its "style") | n/a |
| Group Norm | one scale and shift per image and group of channels | between Layer Norm and Instance Norm | n/a |
| Layer Norm $$(D)$$ / per pixel | one scale and shift per token or pixel vector | each pixel's channel vector length and offset | the token vector's length and mean |
| RMSNorm | one scale per token | n/a | the token vector's length |

Batch Norm is the odd one out, twice. In training it's blind only to changes that hit the *whole batch* identically, which almost never happens to a single input. In evaluation mode it's a fixed affine map and isn't blind to anything. Every other layer ignores some change to a single sample.

This also pins down what the original post meant when it said Layer Norm "suppresses intensity information" in a CNN. Layer Norm over $$(C, H, W)$$ sends a bright image and the same image at 30% brightness to identical activations, so the claim is true. What doesn't follow is that accuracy has to drop. That depends on whether the task needs absolute intensity, and a digit classifier doesn't; for it, this blindness works as robustness, which cell 7 measures. Instance Norm's blindness to per-image style is why [Ulyanov et al. (2016)](https://arxiv.org/abs/1607.08022) used it for fast style transfer, where throwing that information away is the whole point.

![A grid of check marks and crosses showing which input changes leave each norm's output unchanged: Batch Norm in training ignores only a change applied to the whole batch and in evaluation ignores nothing, Layer Norm, Group Norm and per-pixel Layer Norm ignore per-image brightness and offset but not a color cast, Instance Norm ignores all four, and per-pixel RMSNorm ignores brightness but not offset]({{ '/assets/blog/batch-norm-transformers/invariance-table.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Why should any of this help training? [Ioffe & Szegedy (2015)](https://arxiv.org/abs/1502.03167) pitched Batch Norm as a fix for *internal covariate shift*, the drift in each layer's input distribution as the layers before it train. [Santurkar et al. (2018)](https://arxiv.org/abs/1805.11604) put that story to a direct test. They injected random, time-varying noise after every Batch Norm layer, creating covariate shift on purpose, and the networks still trained about as well. Their measurements point somewhere else: Batch Norm makes the loss and its gradients change more smoothly, which tolerates larger learning rates. The invariances fit that account. A layer whose output ignores the scale of its input can't let activations blow up just by growing, and, as we saw with the sphere, the gradients shrink as the activations grow.

## Why CNNs took Batch Norm

Back to grading. A curve is fair when the class is big, everyone sits the same exam, and a score on one question means the same thing for every student. A CNN trained on ImageNet-style data ticks all three boxes, and Batch Norm brings two practical perks on top.

The scores are comparable because of weight sharing. A convolution applies the same filter at every position of every image, so channel $$c$$ means "how much of pattern $$c$$ is here" on every image in the dataset. Pooling its statistics across images and positions pools measurements of one well-defined quantity. That's why Batch Norm's $$S$$ for images contains $$H$$ and $$W$$ along with $$N$$.

The class is big because each statistic is built from a lot of numbers. With $$S = \{N, H, W\}$$, a channel's mean averages $$N \cdot H \cdot W$$ values, and a batch of 32 images at a $$56 \times 56$$ feature map gives $$32 \times 56 \times 56 \approx 100{,}000$$ values per statistic. Estimates from that many numbers are stable even at modest batch sizes and barely move from batch to batch, so the running averages used at test time match what the network saw during training.

Everyone sits the same exam because images get resized or cropped to one shape. Every entry in the batch is real data, and there's no padding to contaminate the statistics.

Now the perks. The first is that Batch Norm costs nothing at deployment. In evaluation mode it's a per-channel affine map, the convolution before it is linear, and the two fold into a single convolution with adjusted weights and bias, so a deployed ResNet pays nothing for its Batch Norm layers. The second is regularization. During training each image is normalized with statistics that depend on its batch-mates, which adds a small random wobble to every activation, and [Ioffe & Szegedy (2015)](https://arxiv.org/abs/1502.03167) reported that Batch Norm networks needed less Dropout.

History helped as well. Batch Norm arrived first, in 2015. Layer Norm came a year later, aimed at recurrent networks where Batch Norm was awkward to apply, and by then the standard CNN recipes were built around Batch Norm.

The weak spot is batch size. Detection and segmentation models often fit only one or two high-resolution images per GPU, and with batches that small the statistics get noisy and training and test stop matching. [Wu & He (2018)](https://arxiv.org/abs/1803.08494) introduced Group Norm for exactly this case: at batch size 2 on ImageNet, their ResNet-50 with Group Norm had 10.6% lower error than the Batch Norm version.

## Why transformers took Layer Norm

Run the same checklist on text and it fails on every line.

First you have to decide what Batch Norm even means for a $$(B, T, D)$$ tensor. The natural choice, and the one [Shen et al. (2020)](https://arxiv.org/abs/2003.07845) used, averages over the batch and time axes together, $$S = \{B, T\}$$: one mean and variance per feature, pooled over every position of every sequence. That choice runs into four separate problems.

The obvious one is padding. Sentences have different lengths, so to stack sentences of length 6, 4, 5 and 3 into one tensor you pad them all to a shared length $$T$$, usually with zeros (or a padding embedding), and a naive Batch Norm counts those positions as data. Say the batch has $$n = B \cdot T$$ positions, $$p$$ of them padding zeros, and call $$f = p/n$$ the padding fraction. Zeros add nothing to a sum, so for a feature whose real tokens have mean $$\mu_r$$ and mean square $$m_r = \mathbb{E}_r[x^2]$$,

$$
\mu_{\text{batch}} = (1 - f)\,\mu_r, \qquad
\sigma^2_{\text{batch}} = (1 - f)\,m_r - (1 - f)^2 \mu_r^2 .
$$

Both statistics depend on $$f$$, and $$f$$ depends on how long the *other* sentences in the batch are. So the same token gets a different value in a different batch. You can mask the padding out, but that only moves the problem: each feature's statistics then depend on how many real tokens the batch happens to contain.

Causality is the second problem. In a causal language model, position $$t$$ must never see tokens after $$t$$. With $$S = \{B, T\}$$, the mean and variance used at position $$t$$ include every later token in the batch, so training leaks information about the future back into the past through the statistics. At generation time the future tokens don't exist yet, and that information simply isn't there.

Then there's the data itself. [Shen et al. (2020)](https://arxiv.org/abs/2003.07845) measured the batch statistics of transformers trained on text and found they swing far more from batch to batch than in image models, with large outliers. Running averages built from statistics like that don't match any individual batch well, and the authors tied those swings to Batch Norm's poor results on translation.

The last problem is serving. A deployed language model batches requests from unrelated users and generates one token at a time with a key-value cache, and one user's output must not depend on who else is in the batch. Layer Norm guarantees that by construction: its $$S$$ is $$\{D\}$$, so each token is normalized using only its own vector.

Layer Norm sidesteps all four, because its statistics never cross a token boundary. Padding, sequence length, batch composition and causality can't reach it. To me that's the real reason transformers took Layer Norm. The layer has to ignore everything outside one token, and the per-token norms (Layer Norm and RMSNorm) are the only members of the family whose group $$S(x)$$ stays inside one token.

![Four sentences padded with more and more zero columns, from T = 6 to T = 64: the Batch Norm output of the watched token climbs from 0.540 to 3.511 as the padding fraction rises from 0.25 to 0.93, while its Layer Norm output stays at 0.120]({{ '/assets/blog/batch-norm-transformers/padding-drags-mean.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Cell 5 checks the padding formula numerically. Four sequences of lengths 6, 4, 5 and 3 (18 real tokens, every feature averaging about 2) get zero-padded to longer and longer lengths $$T$$, the way a tokenizer does with `padding="max_length"`. The cell follows one fixed token (position 0 of sequence 0, feature 0) and prints three things for it: its Batch Norm output with statistics over $$B$$ and $$T$$, what the formula predicts, and its Layer Norm output with statistics over $$D$$.

```python
D, lengths = 8, [6, 4, 5, 3]                          # 4 sequences, 18 real tokens in total
real = [torch.randn(L, D) + 2.0 for L in lengths]     # real token embeddings (mean 2, not 0)
watched = real[0][0, 0]                               # sequence 0, token 0, feature 0
r = torch.cat(real)[:, 0]
mu_real, meansq_real = r.mean(), (r ** 2).mean()

print(f"{'pad to T':>8}{'pad frac':>10}{'BN out':>9}{'formula':>9}{'LN out':>9}")
rows = []
for T in [6, 8, 12, 16, 24, 32, 48, 64]:
    batch = torch.zeros(len(lengths), T, D)           # padding positions are zeros
    for i, seq in enumerate(real):
        batch[i, :len(seq)] = seq
    bn_out = normalize(batch, (0, 1))[0, 0, 0].item() # BN in a transformer: stats over batch AND time
    ln_out = normalize(batch, (2,))[0, 0, 0].item()   # LN: stats over this token's own features
    f = 1 - sum(lengths) / (len(lengths) * T)         # padding fraction p / n
    mu = (1 - f) * mu_real                            # zeros drag the mean down by (n - p) / n
    predicted = ((watched - mu) / torch.sqrt((1 - f) * meansq_real - mu ** 2 + 1e-5)).item()
    rows.append((f, bn_out, predicted, ln_out))
    print(f"{T:>8}{f:>10.2f}{bn_out:>9.3f}{predicted:>9.3f}{ln_out:>9.3f}")

f_, bn_, pred_, ln_ = map(list, zip(*rows))
plt.figure(figsize=(7, 4))
plt.plot(f_, pred_, color="crimson", label="BN, formula")
plt.scatter(f_, bn_, color="crimson", zorder=3, label="BN, measured")
plt.plot(f_, ln_, "--", color="royalblue", label="LN (unchanged)")
plt.xlabel("fraction of the batch that is padding"); plt.ylabel("normalized value of the same token")
plt.title("Same token, same sentence: only the padding changed"); plt.legend()
plt.show()
```
```text
pad to T  pad frac   BN out  formula   LN out
       6      0.25    0.540    0.540    0.120
       8      0.44    0.834    0.834    0.120
      12      0.62    1.233    1.233    0.120
      16      0.72    1.533    1.533    0.120
      24      0.81    2.003    2.003    0.120
      32      0.86    2.382    2.382    0.120
      48      0.91    3.000    3.000    0.120
      64      0.93    3.511    3.511    0.120
```
![Output of cell 5]({{ '/assets/blog/batch-norm-transformers/cell-05-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Measurement and formula agree to all three printed decimals in every row. The Batch Norm output climbs from `0.540` when a quarter of the batch is padding ($$T = 6$$) to `3.511` when 93% of it is ($$T = 64$$), a 6.5-fold change for a token whose embedding never moved. The zeros drag the batch mean toward 0, so a token sitting near 2 looks further and further above "average". Layer Norm gives `0.120` in every row.

Now for the batch-size-1 claim. The original post said Batch Norm in an LLM fine-tuned at batch size 1 computes a variance of zero, divides by zero, and crashes. The zero variance happens only in one special case, and the division by zero never happens. There are really four situations, and cell 6 checks each of them.

With a single sample and no other axis in $$S$$, every feature's variance really is zero, and PyTorch refuses to continue in training mode; it raises an error. If you compute it by hand anyway, $$x - \mu = 0$$ exactly, and $$\epsilon$$ holds the denominator at $$\sqrt{\epsilon} > 0$$, so the result is $$\gamma \cdot 0 / \sqrt{\epsilon} + \beta = \beta$$ for every input. Nothing divides by zero and nothing explodes. The output just stops carrying any information about the input.

In a transformer, though, $$S = \{B, T\}$$, so a single sequence of 12 tokens still gives 12 values per feature and a variance that isn't zero. And at inference Batch Norm uses its running statistics, so batch size 1 is fine there too, unless those statistics were computed on padded batches.

To test that last case, the cell trains only the running statistics of a Batch Norm layer on 200 padded batches (8 sequences of random length 4 to 32, padded to 32, padding counted), then evaluates it on real tokens.

```python
D = 8
try:
    nn.BatchNorm1d(D).train()(torch.randn(1, D))      # one token, no other axis to average over
except ValueError as err:
    print("nn.BatchNorm1d on a batch of 1 ->", err)

gamma, beta = torch.full((D,), 2.0), torch.full((D,), 0.5)
one = torch.randn(1, D) * 3 + 1
manual = gamma * normalize(one, (0,)) + beta          # var = 0 and x - mu = 0: eps avoids 0/0, output is beta
print("manual BN at batch 1: every output equals beta ->", torch.allclose(manual, beta.expand_as(manual)))

sentence = torch.randn(1, 12, D) + 2.0                # ONE sequence of 12 tokens; BN over (B, T)
print(f"BN over (B, T) at B = 1: smallest per-feature variance = {sentence.var(dim=(0, 1), unbiased=False).min():.3f}")

bn = nn.BatchNorm1d(D, affine=False, momentum=None)   # momentum=None: running stats are an exact average
g = torch.Generator().manual_seed(0)
for _ in range(200):                                  # training: padded batches, padding counted as tokens
    batch = torch.zeros(8, 32, D)
    for i, L in enumerate(torch.randint(4, 33, (8,), generator=g).tolist()):
        batch[i, :L] = torch.randn(L, D, generator=g) + 2.0
    bn(batch.reshape(-1, D))
bn.eval()
real_tokens = torch.randn(500, D) + 2.0               # inference: real tokens only
out = bn(real_tokens)
print(f"real-token mean = {real_tokens.mean():.2f} | BN running mean = {bn.running_mean.mean():.2f}")
print(f"eval-mode BN output on real tokens: mean = {out.mean():.2f}, std = {out.std():.2f} (BN promised 0 and 1)")
```
```text
nn.BatchNorm1d on a batch of 1 -> Expected more than 1 value per channel when training, got input size torch.Size([1, 8])
manual BN at batch 1: every output equals beta -> True
BN over (B, T) at B = 1: smallest per-feature variance = 0.370
real-token mean = 2.00 | BN running mean = 1.12
eval-mode BN output on real tokens: mean = 0.71, std = 0.81 (BN promised 0 and 1)
```

All four situations show up in the output. PyTorch's `nn.BatchNorm1d` raises `Expected more than 1 value per channel when training`, an explicit error and not a numerical crash, and the hand-computed version returns $$\beta$$ for every feature (`True`). A single 12-token sequence has a smallest per-feature variance of `0.370`, comfortably above zero.

The last two lines are the failure that actually matters. The real tokens have mean `2.00`, but the running mean learned from padded batches is `1.12`. On average 56% of each training batch was real data, and $$0.56 \times 2 \approx 1.12$$, just as the padding formula says. Batch Norm promised outputs with mean 0 and standard deviation 1; on real tokens at inference it delivers mean `0.71` and standard deviation `0.81`. Every activation leaving this layer at test time is off by about 0.7 standard deviations from what the next layer saw in training. So Batch Norm in a language model doesn't crash. It computes the wrong statistics, and different wrong statistics in training and in inference.

## The swaps, and where the rule of thumb breaks

The original post made two predictions about swapping the layers: Layer Norm in a CNN loses accuracy, and Batch Norm in a transformer breaks. The last two cells try both swaps on problems small enough to train on a laptop CPU in under a minute. These are toys, and I want to be upfront about it. They can tell us whether a mechanism exists and which way it pushes, not how big the effect is on ImageNet or a real LLM.

### The CNN swap

For the CNN I used scikit-learn's digits dataset: 1,797 grayscale handwritten digits at $$8 \times 8$$ pixels. It ships with the library, so there's nothing to download. 1,400 images go to training and 397 to testing. The model is two $$3 \times 3$$ convolutions, each followed by a norm layer and a ReLU, then a linear classifier. The convolutions have no bias, which is standard when a norm layer comes next and also makes the first convolution exactly linear in the image (that will matter in a minute). I compared four norms:

- `BN`: `nn.BatchNorm2d`, $$S = \{N, H, W\}$$.
- `LN (C,H,W)`: Layer Norm over the whole feature map. `nn.GroupNorm(1, C)` computes exactly this, because one group holding all channels means $$S = \{C, H, W\}$$.
- `GN (4 groups)`: Group Norm, $$S = \{C/4, H, W\}$$.
- `LN per pixel`: the variant [ConvNeXt (Liu et al., 2022)](https://arxiv.org/abs/2201.03545) uses. It normalizes each pixel's $$C$$-channel vector on its own, the image version of a transformer's per-token Layer Norm.

Each norm trains twice, once at batch size 64 (150 steps) and once at batch size 1 (1,000 steps). The batch-64 models are then tested a second time on the test images dimmed to 30% brightness, which puts the "intensity" claim to a direct test.

```python
digits = load_digits()                                # 1,797 8x8 images bundled with scikit-learn
images = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
labels = torch.tensor(digits.target)
order = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))
train_idx, test_idx = order[:1400], order[1400:]

class PixelLayerNorm(nn.Module):
    """ConvNeXt-style LN: each pixel's C channels are normalized on their own."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

NORMS = {"BN": nn.BatchNorm2d, "LN (C,H,W)": lambda c: nn.GroupNorm(1, c),  # one group = LN over (C,H,W)
         "GN (4 groups)": lambda c: nn.GroupNorm(4, c), "LN per pixel": PixelLayerNorm}

def train_cnn(norm, batch_size: int, steps: int, lr: float) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, bias=False), norm(16), nn.ReLU(),
                          nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), norm(32), nn.ReLU(),
                          nn.Flatten(), nn.Linear(32 * 4 * 4, 10))
    opt, g = torch.optim.Adam(model.parameters(), lr), torch.Generator().manual_seed(1)
    for _ in range(steps):
        idx = train_idx[torch.randint(len(train_idx), (batch_size,), generator=g)]
        loss = F.cross_entropy(model(images[idx]), labels[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()

@torch.no_grad()
def accuracy(model: nn.Module, brightness: float = 1.0) -> float:
    return (model(brightness * images[test_idx]).argmax(1) == labels[test_idx]).float().mean().item()

results = {}
print(f"{'norm':<15}{'batch 64':>9}{'batch 1':>9}{'batch 64, images at 30% brightness':>36}")
for name, norm in NORMS.items():
    big, single = train_cnn(norm, 64, 150, 3e-3), train_cnn(norm, 1, 1000, 1e-3)
    results[name] = (accuracy(big), accuracy(single), accuracy(big, brightness=0.3))
    print(f"{name:<15}" + "".join(f"{a:>9.3f}" for a in results[name][:2]) + f"{results[name][2]:>36.3f}")

xs, width = np.arange(len(results)), 0.27
plt.figure(figsize=(9, 4))
for k, label in enumerate(["batch 64", "batch 1", "batch 64, dimmed to 30%"]):
    plt.bar(xs + (k - 1) * width, [v[k] for v in results.values()], width, label=label)
plt.xticks(xs, list(results)); plt.ylim(0, 1); plt.ylabel("test accuracy")
plt.title("Tiny CNN on 8x8 digits: which norm, which batch size, which brightness"); plt.legend(loc="lower right")
plt.show()
```
```text
norm            batch 64  batch 1  batch 64, images at 30% brightness
BN                 0.975    0.927                               0.826
LN (C,H,W)         0.972    0.924                               0.972
GN (4 groups)      0.975    0.932                               0.975
LN per pixel       0.982    0.927                               0.980
```
![Output of cell 7]({{ '/assets/blog/batch-norm-transformers/cell-07-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Some calibration before reading the table: with 397 test images, one image is worth 0.0025 of accuracy, so a gap under about 0.01 is a handful of images and I wouldn't read much into it.

At batch size 64 none of the norms wrecks the CNN. Batch Norm scores `0.975`, Layer Norm over $$(C, H, W)$$ `0.972`, Group Norm `0.975`, and per-pixel Layer Norm `0.982`. Swapping Batch Norm for Layer Norm costs one image. A tiny model on an easy dataset can't prove that Layer Norm matches Batch Norm on ImageNet, but that's roughly what [ConvNeXt](https://arxiv.org/abs/2201.03545) reported at scale, and [ViT](https://arxiv.org/abs/2010.11929) has used Layer Norm on images from the start.

At batch size 1, Batch Norm is fine. All four land between `0.924` and `0.932`. The original post would predict a Batch Norm failure here, and there isn't one, for the reason cell 6 gave. With $$S = \{N, H, W\}$$ and $$N = 1$$, each statistic still averages over $$8 \times 8 = 64$$ spatial positions. In training mode Batch Norm at batch size 1 behaves like Instance Norm, and its running averages turned out good enough for evaluation. Its small-batch problems are real, as the Group Norm result above shows, but they come from noisy statistics and train/test mismatch. A variance of zero has nothing to do with it.

The dimmed images turn the "intensity" argument around. At 30% brightness the Batch Norm model falls from `0.975` to `0.826`. Layer Norm holds at `0.972`, Group Norm at `0.975`, and per-pixel Layer Norm moves from `0.982` to `0.980`, a single image. That's what the invariance section predicted. With no bias in the first convolution, dimming the image just scales its output by 0.3, and every per-sample norm cancels that factor before the rest of the network sees it. Batch Norm in evaluation mode is a fixed affine map calibrated on full-brightness images, so it passes the dimming straight through and every activation shifts relative to $$\beta$$. Per-pixel Layer Norm drops one image, most likely because a digit's background pixels have almost no activation in any channel, so after dimming their variance is close to $$\epsilon$$ and the invariance stops being exact. So yes, Layer Norm is blind to intensity, just as the original post said. On a task that doesn't need intensity, that blindness lets the Layer Norm models read faint digits as well as normal ones.

### The transformer swap

The transformer task is synthetic, but it keeps the property that matters here: variable-length, padded batches. Each sequence has 2 to 32 tokens drawn from $$\{1, 2, 3\}$$, with one token twice as likely as the other two, and the label is whichever token appears most often. Batches of 32 sequences are zero-padded to their longest member. The model is a one-block pre-norm transformer: embedding plus learned positions, two-head self-attention, an MLP, a final norm, and a mean over the real tokens. Attention ignores padding through `key_padding_mask`, which leaves the norm layers as the only place padding can get in. The three versions differ only in the norm class:

- `BN`: `TokenBatchNorm`, a naive Batch Norm over $$S = \{B, T\}$$ that counts padding.
- `LN`: `nn.LayerNorm`, $$S = \{D\}$$.
- `RMSNorm`: `nn.RMSNorm`, $$S = \{D\}$$ without centering.

Each trains for 400 steps with Adam. The cell prints the mean and spread of the training loss over the last 50 steps, plus the accuracy on 500 fresh test sequences in evaluation mode.

```python
DIM, PAD = 32, 0                                       # tokens 1..3; label = the most frequent token

def make_batch(size: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lens = torch.randint(2, 33, (size,), generator=g)  # lengths 2..32, padded to the longest
    toks, y = torch.full((size, int(lens.max())), PAD), torch.zeros(size, dtype=torch.long)
    for i, L in enumerate(lens.tolist()):
        probs = torch.full((3,), 0.25); probs[torch.randint(3, (1,), generator=g)] = 0.5
        toks[i, :L] = torch.multinomial(probs, L, replacement=True, generator=g) + 1
        y[i] = torch.bincount(toks[i, :L], minlength=4)[1:].argmax()
    return toks, y, toks != PAD

class TokenBatchNorm(nn.Module):
    """Naive BN for (B, T, D): statistics over every position, padding included."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.reshape(-1, x.shape[-1])).reshape(x.shape)

class TinyTransformer(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
        self.emb, self.pos = nn.Embedding(4, DIM), nn.Embedding(64, DIM)
        self.norm1, self.norm2, self.norm_out = norm(DIM), norm(DIM), norm(DIM)
        self.attn = nn.MultiheadAttention(DIM, 2, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))
        self.head = nn.Linear(DIM, 3)

    def forward(self, toks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(toks) + self.pos(torch.arange(toks.shape[1]))
        h = self.norm1(x)
        x = x + self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)[0]
        x = self.norm_out(x + self.mlp(self.norm2(x)))
        m = mask.unsqueeze(-1).float()
        return self.head((x * m).sum(1) / m.sum(1))   # mean over real tokens only

curves = {}
for name, norm in {"BN": TokenBatchNorm, "LN": nn.LayerNorm, "RMSNorm": nn.RMSNorm}.items():
    torch.manual_seed(0); g = torch.Generator().manual_seed(1)
    model = TinyTransformer(norm); opt = torch.optim.Adam(model.parameters(), 3e-3)
    curves[name] = []
    for step in range(400):
        toks, y, mask = make_batch(32, g)
        loss = F.cross_entropy(model(toks, mask), y)
        opt.zero_grad(); loss.backward(); opt.step(); curves[name].append(loss.item())
    model.eval()
    with torch.no_grad():
        toks, y, mask = make_batch(500, torch.Generator().manual_seed(99))
        acc = (model(toks, mask).argmax(1) == y).float().mean().item()
    tail = curves[name][-50:]
    print(f"{name:<8} loss over last 50 steps: mean {statistics.mean(tail):.3f}, std {statistics.pstdev(tail):.3f} | test acc {acc:.3f}")

plt.figure(figsize=(8, 4))
for name, c in curves.items():
    plt.plot(np.convolve(c, np.ones(20) / 20, mode="valid"), label=name)  # 20-step moving average
plt.yscale("log"); plt.xlabel("training step"); plt.ylabel("loss (20-step moving average)")
plt.title("Tiny transformer, variable-length padded batches"); plt.legend()
plt.show()
```
```text
BN       loss over last 50 steps: mean 0.119, std 0.091 | test acc 0.976
LN       loss over last 50 steps: mean 0.055, std 0.054 | test acc 0.984
RMSNorm  loss over last 50 steps: mean 0.059, std 0.062 | test acc 0.986
```
![Output of cell 8]({{ '/assets/blog/batch-norm-transformers/cell-08-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

Batch Norm doesn't break the transformer, but it does hold it back. Over the last 50 steps its loss averages `0.119`, about twice Layer Norm's `0.055` and RMSNorm's `0.059`, and it's the noisiest of the three (standard deviation `0.091` against `0.054` and `0.062`). In the plot all three fall together for the first 20 or so steps. After that the Batch Norm curve flattens out around 0.12 to 0.15 while the other two keep going down. Test accuracy is `0.976` for Batch Norm against `0.984` and `0.986`, a gap of four to five sequences out of 500. The task is easy enough that all three solve it, which is why the difference shows up in the training curve. It's the same direction Shen et al. found on real translation tasks, where Batch Norm in a transformer worked, just worse.

Layer Norm and RMSNorm are hard to tell apart. Their curves almost overlap for all 400 steps, and their losses (`0.055` and `0.059`) and accuracies (`0.984` and `0.986`) are within noise of each other. Dropping the projection onto $$\mathbf{1}^\perp$$ cost nothing I could measure. That's [Zhang & Sennrich's](https://arxiv.org/abs/1910.07467) argument in miniature, and the same argument moved most LLMs to the cheaper layer.

So "images get Batch Norm, text gets Layer Norm" is a proxy for something else. What it really says is: use Batch Norm when statistics shared across the batch are reliable, which takes large, fixed-shape batches of comparable samples, and use a per-sample norm when they aren't. Images usually meet that bar and text usually doesn't. When images don't (tiny batches, detection, a ViT that treats an image as a sequence of patch tokens), the per-sample norms take over on images too. Here's the whole decision on one page:

![A flowchart for choosing a norm layer: large fixed-shape batches of comparable samples lead to Batch Norm, sequence models to RMSNorm or Layer Norm, images with small batches to Group Norm, ConvNeXt- or ViT-style image models to per-pixel or per-token Layer Norm, with Instance Norm as a side option for removing per-image style]({{ '/assets/blog/batch-norm-transformers/decision-cheatsheet.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

## Where the field is now

Language models have mostly moved to RMSNorm. [LLaMA (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) normalizes the input of every transformer sub-layer with RMSNorm, and most open-weight LLMs released since have copied that design. So the original post's "LLMs use Layer Norm" was true of GPT-2-era models and is now true only in spirit: still per token, just without the centering.

Image models use whatever their batch allows. Classic CNN recipes still use Batch Norm, for the reasons in the CNN section. [ViT](https://arxiv.org/abs/2010.11929) uses Layer Norm on each patch token, [ConvNeXt](https://arxiv.org/abs/2201.03545) swapped Batch Norm for per-pixel Layer Norm in a modernized ResNet, and Group Norm is the usual pick when a detection or segmentation model fits only a few images per device.

Some recent work pushes the sphere picture much further. [nGPT (Loshchilov et al., 2024)](https://arxiv.org/abs/2410.01131) keeps every hidden vector, and every embedding and weight vector, on the unit hypersphere throughout the network, and reports reaching the same accuracy in 4 to 20 times fewer training steps. Layer Norm uses that geometry inside one layer; nGPT makes it the design principle of the whole model.

Other work gets rid of the statistics altogether. [Transformers without Normalization (Zhu et al., 2025)](https://arxiv.org/abs/2503.10622) replaces every norm layer with Dynamic Tanh, $$\mathrm{DyT}(x) = \gamma \tanh(\alpha x) + \beta$$, an elementwise squashing function with a learned $$\alpha$$ and no mean or variance anywhere, and reports matching normalized transformers across several tasks. [Stronger Normalization-Free Transformers (Chen et al., 2025)](https://arxiv.org/abs/2512.10938) searches over functions like that and reports that $$\mathrm{erf}(\alpha x + s)$$ does better than Layer Norm, RMSNorm and DyT. On the CNN side, [NFNets (Brock et al., 2021)](https://arxiv.org/abs/2102.06171) reached strong ImageNet results without normalization by clipping gradients adaptively. If these results hold up at scale, the question this post started with stops being "which axis?" and becomes "do we need statistics at all?". I don't think that's settled yet.

Batch Norm's weak spots have their own fixes. [Batch Renormalization (Ioffe, 2017)](https://arxiv.org/abs/1702.03275) corrects batch statistics toward the running averages when batches are small or non-i.i.d., and [PowerNorm (Shen et al., 2020)](https://arxiv.org/abs/2003.07845) relaxes Batch Norm's zero-mean constraint and uses running statistics in the forward pass so that it works in transformers.

I've left out one related question: where the norm sits, as opposed to which axes it averages over. [Xiong et al. (2020)](https://arxiv.org/abs/2002.04745) show that putting Layer Norm before each sub-layer (Pre-LN) instead of after the residual addition (Post-LN, the original Transformer) lets transformers train without a learning-rate warm-up.

Here's the one-line version. All five layers compute the same formula, and picking a norm means picking $$S$$. Put the batch axis in $$S$$ only when the batch is a trustworthy sample of the data; otherwise keep $$S$$ inside one sample. The geometry tells you what that costs: two numbers per vector with Layer Norm, one with RMSNorm.

## Further reading

- [Ioffe & Szegedy (2015), "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"](https://arxiv.org/abs/1502.03167): the original Batch Norm paper, including the running statistics and the inference-time formula.
- [Ba, Kiros & Hinton (2016), "Layer Normalization"](https://arxiv.org/abs/1607.06450): introduces per-sample normalization and its invariances, written with recurrent networks in mind.
- [Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"](https://arxiv.org/abs/1910.07467): the case for dropping re-centering, which is the "skip the projection" step in this post.
- [Wu & He (2018), "Group Normalization"](https://arxiv.org/abs/1803.08494): the clearest treatment of the BN / LN / IN / GN family as choices of axes, with the small-batch experiments.
- [Shen et al. (2020), "PowerNorm: Rethinking Batch Normalization in Transformers"](https://arxiv.org/abs/2003.07845): measurements of why batch statistics misbehave on text.
- [Santurkar et al. (2018), "How Does Batch Normalization Help Optimization?"](https://arxiv.org/abs/1805.11604): the experiment that separated Batch Norm's benefit from internal covariate shift.
- [Zhu et al. (2025), "Transformers without Normalization"](https://arxiv.org/abs/2503.10622): the most direct current challenge to the idea that transformers need a norm layer at all.
