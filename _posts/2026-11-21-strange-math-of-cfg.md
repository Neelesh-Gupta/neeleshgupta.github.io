---
layout: post
title: "Why the “Correct” Setting Makes Worse Images: The Strange Math of Classifier-Free Guidance"
date: 2026-11-21 00:00:00+0530
description: Why the mathematically correct guidance setting makes the worst images, why classifier-free guidance misses the distribution it seems to target, and where exact toy models show it helping and hurting.
tags: diffusion guidance generative-models sampling manim
categories: machine-learning
related_posts: false
toc:
  sidebar: left
---

<style>
  /* This post only: use the window's width so code, tables and figures fit without sideways scrolling. */
  @media (min-width: 576px) {
    .container[role="main"], .navbar .container, footer .container { max-width: min(calc(100vw - 32px), 1520px); }
    .container[role="main"] > .row > .col-sm-3 { flex: 0 0 18%; max-width: 18%; }
    .container[role="main"] > .row > .col-sm-9 { flex: 0 0 82%; max-width: 82%; }
  }
  .post-content img { max-width: min(100%, 960px) !important; }
  .post-content pre.highlight, .post-content pre.highlight code { white-space: pre-wrap; overflow-wrap: anywhere; }
  .post-content table { width: 100%; }
  .post-content table th, .post-content table td { overflow-wrap: anywhere; }
  @media (max-width: 575px) {
    .post-content mjx-container[display] { font-size: 90% !important; }
    .post-content table { font-size: 75%; }
    .post-content table th, .post-content table td { padding: .3rem .35rem; }
  }
</style>

## The paradox: the "correct" sampler makes worse images

Most Stable Diffusion interfaces have a slider for guidance, whether it's called guidance scale, CFG scale or `guidance_scale`. Set it to 1 and the model does exactly what it was trained to do: it samples from $$p(x \mid c)$$, the distribution of training images that match the prompt $$c$$. The pictures are usually disappointing. Ask for a golden retriever and you get muddy colour and a dog-shaped blob. Move the slider to 7.5, the default in [diffusers](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img), and the same weights, starting from the same noise, give you a crisp dog you'd recognise anywhere. Push it to 20 and the image deep-fries, with neon saturation and anatomy bent the wrong way.

So the setting that is mathematically correct gives the worst pictures, and the best pictures come from a setting that is, strictly speaking, wrong. I wanted to know why, and the answer is more interesting than "guidance sharpens the distribution".

Part of it is the data. $$p(x\mid c)$$ is the distribution of web data, and captions like "a golden retriever in a park" are attached to a huge number of photos: blurry ones, badly lit ones, cropped ones, ones that barely match their caption. A faithful sampler has to reproduce all of them in proportion. That's what covering a distribution means. Nobody judges a generated image by how typical it is of the training set, though. We judge it by how clearly it shows what we asked for, and that rewards a sharp prototype over a representative draw.

The other part is the network. It isn't $$p(x\mid c)$$; it's an approximation, and it's least accurate where the data is thinnest, in the gaps between modes. At $$w=1$$ those errors leak into samples as washed-out textures and geometry that doesn't hold together.

CFG fixes both by cheating. It doesn't sample $$p(x\mid c)$$, and (the part most explanations skip) it doesn't sample the sharpened distribution it appears to target either. A better mental model is a sampler that nudges every step toward "more like the prompt than like anything else". That one nudge helps and hurts at once. It pushes samples away from the boundaries between classes and cancels some of the network's own error, which is good. It also overshoots the conditional mean, throws away diversity and, at large $$w$$, extrapolates clean off the curved data manifold, which is bad. The sweet spot on the slider is wherever the good still outweighs the bad.

To see each effect without a trained network muddying things, I use toy models small enough that every score is exact.

![As the guidance weight w sweeps from 1 to 15, the tilted target p(x)^(1-w) p(x given c)^w (green) barely moves while CFG samples from the reverse ODE (orange) slide past it and pile up far beyond the prompt's mode]({{ '/assets/blog/strange-math-of-cfg/gmm-tilt-hero.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Three blobs of data, one per class; the top one is the class your prompt asks for. The green shape is where guidance is supposed to send samples, and the orange dots are where it actually sends them. As the guidance weight w goes up, the green shape barely moves, but the orange dots race past it and keep going.*

## What classifier-free guidance is, from scratch

If you already know how a diffusion model is trained and sampled, skip ahead to the next section. This one covers the minimum the rest of the post leans on.

### Diffusion in two moves

A diffusion model learns to undo noise. The forward direction is fixed and dull: take a clean image $$x_0$$ and add Gaussian noise of standard deviation $$\sigma$$,

$$\begin{gathered} x_\sigma = x_0 + \sigma\,\epsilon, \\ \epsilon \sim \mathcal N(0, I). \end{gathered}$$

At small $$\sigma$$ the image is barely touched. At large $$\sigma$$ nothing of it survives, and $$x_\sigma$$ is essentially pure noise. (This is the variance-exploding form, which keeps the algebra simple. Stable Diffusion 1.x uses a variance-preserving schedule that also shrinks the image as noise goes in, $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$. The two are rescalings of each other, so nothing below depends on the choice.)

The network learns the reverse direction. Given a noisy image, its noise level and the prompt, it predicts the noise that was added, $$\epsilon_\theta(x_\sigma, \sigma, c)$$, trained by plain regression:

$$\begin{aligned} \mathcal L &= \mathbb E_{x_0,\,c,\,\sigma,\,\epsilon} \\ &\quad \big\|\epsilon_\theta(x_0 + \sigma\epsilon,\ \sigma,\ c) - \epsilon\big\|^2 . \end{aligned}$$

A noise prediction doubles as a guess of the clean image, $$\hat x_0 = x_\sigma - \sigma\,\epsilon_\theta(x_\sigma, \sigma, c)$$. It's also a score in disguise: the best possible noise prediction is $$\epsilon^\star = -\sigma\,\nabla_x \log p_\sigma(x_\sigma \mid c)$$, which is why the rest of the post can talk about scores and noise predictions interchangeably.

Sampling runs the film backwards. Start from pure noise at a large $$\sigma_{\max}$$, ask the network for its noise estimate, step to a slightly smaller $$\sigma$$, and repeat for a few dozen steps until $$\sigma = 0$$. Each step strips out a little noise, and the prompt decides which image the noise resolves into.

![Forward noising adds Gaussian noise until the three classes dissolve into one blob at sigma = 1.6; the reverse process, conditioned on the prompt, then denoises the same points and rebuilds only the prompt's class]({{ '/assets/blog/strange-math-of-cfg/diffusion-primer.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Each dot is one data point, and the blue ones belong to the prompt's class. First, noise is added until the three groups melt into a single cloud. Then the model runs the process backwards while being told the prompt, and every dot settles into the prompt's group.*

### Two ways to steer toward the prompt

Feeding the prompt to the network gives you samples from $$p(x\mid c)$$, which, as the paradox showed, is too non-committal. The first widely used fix was *classifier guidance*, popularised by [Dhariwal & Nichol (2021)](https://arxiv.org/abs/2105.05233). You train a separate classifier on noisy images, and at every sampling step you add its gradient $$\nabla_x \log p_\sigma(c \mid x)$$ to nudge the sample toward images the classifier calls $$c$$. It works. The catch is that it needs an extra model trained on noisy inputs, and for free-form text prompts it isn't obvious what that classifier would even be.

[Ho & Salimans (2022)](https://arxiv.org/abs/2207.12598) got rid of the classifier. During training, the prompt is replaced by an empty prompt $$\varnothing$$ some fraction of the time, so a single network learns two models at once: a conditional one, $$\epsilon_\theta(x_\sigma,\sigma,c)$$, and an unconditional one, $$\epsilon_\theta(x_\sigma,\sigma,\varnothing)$$. Nothing else about training changes.

![One CFG training step: for some fraction of training pairs the prompt is replaced by an empty prompt, noise is added to the image, and a single network is trained to predict that noise, so the same weights learn both the conditional and the unconditional model]({{ '/assets/blog/strange-math-of-cfg/cfg-training.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Training a CFG model is ordinary diffusion training plus one coin flip: some of the time the prompt is hidden. That is how a single network learns to make predictions both with and without the prompt.*

At sampling time, every step calls the network twice, once with the prompt and once with the empty prompt, and blends the two predictions with a guidance weight $$w$$:

$$\begin{aligned} \tilde\epsilon &= \epsilon_\theta(x_\sigma,\sigma,\varnothing) \\ &\quad + w\,\big[\epsilon_\theta(x_\sigma,\sigma,c) - \epsilon_\theta(x_\sigma,\sigma,\varnothing)\big]. \end{aligned}$$

The guided prediction goes into whatever solver step you'd have used anyway. It costs two network evaluations per step (usually batched into one call), and that's the whole recipe.

### The geometry: a point on a line

The formula has a simple picture. For a given noisy input the two predictions are two points, and $$\tilde\epsilon$$ always lies on the line through them. At $$w=0$$ it sits on the unconditional prediction, at $$w=1$$ on the conditional one, for $$0<w<1$$ it interpolates between them, and for $$w>1$$ it walks past the conditional prediction along $$\epsilon_\text{cond} - \epsilon_\text{uncond}$$, the direction of "what the prompt adds". Since $$\hat x_0 = x_\sigma - \sigma\epsilon$$ is affine in $$\epsilon$$, the same picture holds for the clean-image guesses: the guided guess $$\tilde x_0$$ sits on the line through the unconditional and conditional guesses, at the same $$w$$.

![The guided prediction always lies on the line through the unconditional and conditional predictions: as w goes from 0 to 1 it interpolates between them, and for every w above 1, including the common default of 7.5, it extrapolates past the conditional model]({{ '/assets/blog/strange-math-of-cfg/cfg-line-primer.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The grey and blue dots are the model's two guesses, without and with the prompt. CFG's final guess (orange) always sits on the line through them. Between the two dots it is a blend; beyond the blue dot, where everyday settings like 7.5 live, it is pushed further than the prompt-aware model itself would go.*

The scales people actually use (7.5 is the diffusers default) are well above 1, so in practice CFG is always extrapolating, pushing past what the conditional model itself believes. The rest of the post works out what that extrapolation does to the samples.

## Two arrows and a knob

A diffusion sampler starts from pure noise and keeps asking the network one question: from here, which way looks more like data? The answer is the score $$\nabla_x \log p_\sigma(x)$$, the gradient of the log-density of the data after it has been blurred by Gaussian noise of standard deviation $$\sigma$$. Add a prompt and you ask for the conditional score $$\nabla_x \log p_\sigma(x\mid c)$$ instead.

CFG asks both and compares. The unconditional arrow points toward "any natural image", which at high noise means the blurry average of everything the noisy state could turn into. The conditional arrow points toward "an image of $$c$$". The difference between them is the direction that makes the sample more like $$c$$ *compared with everything else*, and CFG takes the unconditional arrow and adds $$w$$ copies of that difference. That really is the whole method.

![CFG score field of the three-class toy at sigma = 0.7 as w goes 0 to 6: the unconditional score (grey) plus w times the implicit classifier (green) sums to the CFG score (orange), and the whole field turns from pointing into the overlap of the modes to pointing up and away from the other two classes]({{ '/assets/blog/strange-math-of-cfg/tug-of-war-field.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Each small arrow shows which way a noisy point gets pushed at that spot. The big arrows at the white dot show the recipe: grey (towards any image) plus w times green (towards the prompt and away from the other classes) gives orange (where CFG actually pushes). As w grows, the green part takes over and the whole field swings up and away from the other two classes.*

Writing it down takes three lines. Bayes' rule splits the conditional score into a prior term and a classifier term ($$\log p(c)$$ doesn't depend on $$x$$, so it vanishes from the gradient):

$$\begin{aligned} \nabla_x \log p_\sigma(x\mid c) &= \nabla_x \log p_\sigma(x) \\ &\quad + \nabla_x \log p_\sigma(c\mid x). \end{aligned}$$

Rearrange it, and a conditional model plus an unconditional one hand you a classifier gradient for free. This is the *implicit classifier*:

$$\begin{aligned} \nabla_x \log p_\sigma(c\mid x) &= \nabla_x \log p_\sigma(x\mid c) \\ &\quad - \nabla_x \log p_\sigma(x). \end{aligned}$$

Classifier guidance ([Dhariwal & Nichol, 2021](https://arxiv.org/abs/2105.05233)) scaled the gradient of a separately trained classifier. CFG scales the implicit one instead, by the guidance weight $$w$$:

$$\begin{aligned} \tilde s_w(x,\sigma) &= \nabla_x \log p_\sigma(x) \\ &\quad + w\,\big[\nabla_x \log p_\sigma(x\mid c) - \nabla_x \log p_\sigma(x)\big]. \end{aligned}$$

$$w=0$$ gives you the unconditional model, $$w=1$$ the plain conditional one, and anything with $$w>1$$ is guidance. In practice networks predict noise rather than scores, but for the variance-exploding process $$x_\sigma = x_0 + \sigma\epsilon$$ I use throughout, the two differ only by a factor, $$\hat\epsilon = -\sigma\,\nabla_x\log p_\sigma(x)$$, so the same rule becomes

$$\tilde\epsilon = \epsilon_\text{uncond} + w\,(\epsilon_\text{cond} - \epsilon_\text{uncond}),$$

which you'll find, almost word for word, inside the Stable Diffusion pipelines in diffusers. Both predictions come from the same network, trained with the prompt-dropout trick from the primer above ([Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598)). Watch the notation if you read that paper. Ho & Salimans write $$(1+w')\,\epsilon_\text{cond} - w'\,\epsilon_\text{uncond}$$, so their $$w'$$ is our $$w-1$$. I stick to the diffusers convention, where `guidance_scale = 1` means no guidance at all (diffusers only turns CFG on when `guidance_scale > 1`).

![One CFG sampling step: the same network is called with the prompt and with an empty prompt, the two noise predictions are combined as eps_uncond + w (eps_cond - eps_uncond), and a solver step produces a less noisy sample; the combine step is where the help (boundary repulsion, error correction) and the harm (mean overshoot, off-manifold extrapolation) enter]({{ '/assets/blog/strange-math-of-cfg/cfg-step-pipeline.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*One step of sampling with CFG: run the network twice, combine the two answers using the guidance weight, take a small denoising step, and repeat. The combine step is where both the benefits and the side effects of guidance come from.*

### A toy where every score is exact

I don't want a trained network sitting between us and the formula. So the toy has to be something whose scores can be written down exactly. Mine is a mixture of three 2D Gaussians at 90°, 210° and 330° on the unit circle, each with standard deviation 0.5, one per "class", and the prompt $$c$$ picks the top one. Adding noise of level $$\sigma$$ to a Gaussian mixture just widens each component's variance by $$\sigma^2$$, so $$p_\sigma(x)$$, $$p_\sigma(x\mid c)$$ and both scores stay closed-form at every noise level. Cell 1 pins the two dependencies and gives each role a colour that every figure reuses: grey for the unconditional model, blue for the conditional one, green for the tilted target (coming up shortly), orange for CFG and violet for autoguidance.

```python
%pip install -q "numpy==2.3.3" "matplotlib==3.10.8"
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SEED = 0
np.random.seed(SEED)

# One colour per *role*, used the same way in every figure of the post.
COL = {"uncond": "#52514e", "cond": "#2a78d6", "cfg": "#eb6834", "target": "#1baf7a", "auto": "#6250d6"}
TARGET_CMAP = LinearSegmentedColormap.from_list("target", ["#fcfcfb", COL["target"]])
plt.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e3",
    "grid.linewidth": 0.6, "lines.linewidth": 2, "font.size": 10,
})
print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__} | seed {SEED}")
```
```text
numpy 2.3.3 | matplotlib 3.10.8 | seed 0
```

Cell 2 builds the scores and checks the algebra before anything leans on it. `mixture_score` computes $$\nabla_x \log p_\sigma(x)$$ for any mixture, `score_uncond` and `score_cond` are the two models, and `cfg_score(w)` returns $$\tilde s_w$$. The first check compares the implicit classifier with the gradient of the true classifier $$\log p_\sigma(c\mid x)$$, which the cell gets independently by finite differences. The second confirms that the noise form and the score form of CFG really are one formula. Then it draws the arrows at a single noisy point.

```python
MEANS = np.stack([np.cos(np.deg2rad([90, 210, 330])), np.sin(np.deg2rad([90, 210, 330]))], axis=1)
VARS, UNIFORM, PROMPT = np.full(3, 0.5**2), np.full(3, 1 / 3), 0   # 3 classes; the prompt picks the top one
PROMPT_DIR = MEANS[PROMPT]                                         # unit vector toward the prompt's mode

def logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.max(axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis, keepdims=True))).squeeze(axis)

def component_logits(x: np.ndarray, sigma: float, means: np.ndarray, vars_: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """log w_k N(x; m_k, (v_k + sigma^2) I): VE noising just widens every component by sigma^2."""
    v = vars_[None, :] + sigma**2
    return np.log(weights)[None] - 0.5 * ((x[:, None] - means[None]) ** 2).sum(-1) / v - np.log(2 * np.pi * v)

def log_density(x, sigma, means, vars_, weights) -> np.ndarray:
    return logsumexp(component_logits(x, sigma, means, vars_, weights), axis=1)

def mixture_score(x, sigma, means, vars_, weights) -> np.ndarray:
    """Exact grad_x log p_sigma(x): responsibility-weighted pull toward each component mean."""
    logits = component_logits(x, sigma, means, vars_, weights)
    resp = np.exp(logits - logsumexp(logits, axis=1)[:, None])
    return -(resp[..., None] * (x[:, None] - means[None]) / (vars_[None, :, None] + sigma**2)).sum(1)

def score_uncond(x, s): return mixture_score(x, s, MEANS, VARS, UNIFORM)
def score_cond(x, s): return mixture_score(x, s, MEANS[[PROMPT]], VARS[[PROMPT]], np.ones(1))

def cfg_score(w: float, cond=score_cond, uncond=score_uncond):
    """CFG: unconditional score + w * (implicit classifier gradient). w = 1 is the plain conditional model."""
    return lambda x, s: uncond(x, s) + w * (cond(x, s) - uncond(x, s))

x, sigma, w = np.array([[1.6, 0.4]]), 0.7, 3.0
log_clf = lambda z: log_density(z, sigma, MEANS[[PROMPT]], VARS[[PROMPT]], np.ones(1)) + np.log(1 / 3) - log_density(z, sigma, MEANS, VARS, UNIFORM)
fd_grad = np.array([(log_clf(x + 1e-5 * e) - log_clf(x - 1e-5 * e)) / 2e-5 for e in np.eye(2)]).T
implicit = score_cond(x, sigma) - score_uncond(x, sigma)
eps_u, eps_c = -sigma * score_uncond(x, sigma), -sigma * score_cond(x, sigma)   # epsilon-prediction = -sigma * score
print(f"implicit classifier vs finite-diff grad log p(c|x): max |diff| = {np.abs(implicit - fd_grad).max():.1e}")
print(f"CFG eps-form vs score-form (w={w:g}):             max |diff| = {np.abs(eps_u + w * (eps_c - eps_u) + sigma * cfg_score(w)(x, sigma)).max():.1e}")

g = np.linspace(-2.5, 2.5, 120); G2 = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
fig, ax = plt.subplots(figsize=(5, 5))
ax.contour(g, g, np.exp(log_density(G2, sigma, MEANS, VARS, UNIFORM)).reshape(120, 120), levels=8, colors=COL["uncond"], linewidths=0.6)
u, S = score_uncond(x, sigma)[0], 0.3                          # S only shrinks arrows for display
for start, vec, c, lab in [(x[0], u, COL["uncond"], "unconditional score"), (x[0], score_cond(x, sigma)[0], COL["cond"], "conditional score"),
                           (x[0] + S * u, w * implicit[0], COL["target"], f"w x implicit classifier (w={w:g})"),
                           (x[0], cfg_score(w)(x, sigma)[0], COL["cfg"], "CFG score = their sum")]:
    ax.quiver(*start, *(S * vec), color=c, angles="xy", scale_units="xy", scale=1, width=0.012, label=lab)
ax.scatter(*MEANS.T, marker="x", color="#0b0b0b"); ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title(f"Score arrows at one noisy point ($\\sigma$={sigma})"); ax.legend(loc="lower left", fontsize=8)
plt.show()
```
```text
implicit classifier vs finite-diff grad log p(c|x): max |diff| = 5.2e-11
CFG eps-form vs score-form (w=3):             max |diff| = 4.4e-16
```
![Output of cell 2]({{ '/assets/blog/strange-math-of-cfg/cell-02-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The recipe from the animation above, frozen at w = 3 for one point: grey plus green gives orange. The grey contour lines show where the data is dense, and the black crosses mark the centres of the three classes.*

Both checks pass. The implicit classifier agrees with the finite-difference gradient of $$\log p_\sigma(c\mid x)$$ to $$5.2\times10^{-11}$$, which is just finite-difference error, and the two forms of CFG agree to $$4.4\times10^{-16}$$. The figure is the tug-of-war in miniature. At $$x=(1.6,\,0.4)$$, with $$\sigma=0.7$$, the grey unconditional arrow points back toward the bulk of the data. The green arrow, three times the implicit classifier drawn from the grey arrow's tip, points up and toward the prompt's mode, away from the class at 330°. Add the two and you get the orange CFG arrow, longer than the blue conditional one and leaning further away from the neighbouring class.

## The distribution CFG seems to aim for

The usual reading of the formula goes like this. If $$\tilde s_w$$ were the score of a single distribution, that distribution would be

$$\begin{aligned} \tilde p_w(x\mid c) &\propto p(x)\,p(c\mid x)^{w} \\ &\propto p(x)^{1-w}\,p(x\mid c)^{w}, \end{aligned}$$

where the second form uses $$p(c\mid x) = p(x\mid c)\,p(c)/p(x)$$. Raising the classifier to the power $$w$$ is like telling the model to be $$w$$ times as picky about the prompt. Points the classifier is sure about keep their density. Ambiguous points near a decision boundary, where $$p(c\mid x)$$ hovers around a half, get crushed by the extra factor $$p(c\mid x)^{w-1}$$. Picture the prompt's mode shrinking and edging away from its neighbours. I'll call $$\tilde p_w$$ the *tilted target*.

It's a tidy story, and the next section shows it's wrong, because sampling with $$\tilde s_w$$ doesn't produce this distribution. The target is still worth drawing, since everything later gets measured against it. Cell 3 evaluates $$p(x)^{1-w}p(x\mid c)^w$$ on clean data ($$\sigma=0$$) over an 80 × 80 grid for $$w \in \{1, 3, 6, 15\}$$, and reports where the target sits along the prompt's axis, how wide it is, and how much of its mass lies where the true classifier is less than 90% sure ($$p(c\mid x) < 0.9$$).

```python
GRID_LIM, GRID_N = 4.0, 80
EDGES = np.linspace(-GRID_LIM, GRID_LIM, GRID_N + 1)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
GRID = np.stack(np.meshgrid(CENTERS, CENTERS, indexing="ij"), -1).reshape(-1, 2)
LOGP_U = log_density(GRID, 0.0, MEANS, VARS, UNIFORM)
LOGP_C = log_density(GRID, 0.0, MEANS[[PROMPT]], VARS[[PROMPT]], np.ones(1))
POSTERIOR = np.exp(LOGP_C + np.log(UNIFORM[PROMPT]) - LOGP_U)              # p(c | x), the true classifier

def tilted_target(w: float) -> np.ndarray:
    """Grid bin masses of p(x)^(1-w) p(x|c)^w, i.e. p(x|c) p(c|x)^(w-1) up to normalisation."""
    logq = (1 - w) * LOGP_U + w * LOGP_C
    q = np.exp(logq - logq.max())
    return q / q.sum()

as_img = lambda v: v.reshape(GRID_N, GRID_N).T
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
for ax, w in zip(axes, [1, 3, 6, 15]):
    q = tilted_target(w)
    mean = q @ GRID
    std = np.sqrt(q @ (GRID - mean) ** 2).mean()
    print(f"w={w:>2}: mean along prompt axis {mean @ PROMPT_DIR:.2f} | std {std:.3f} | mass where p(c|x) < 0.9: {q[POSTERIOR < 0.9].sum():.3f}")
    ax.contourf(CENTERS, CENTERS, as_img(q), levels=np.linspace(0.05, 1, 9) * q.max(), cmap=TARGET_CMAP)
    ax.contour(CENTERS, CENTERS, as_img(np.exp(LOGP_U)), levels=5, colors=COL["uncond"], linewidths=0.5)
    ax.contour(CENTERS, CENTERS, as_img(np.exp(LOGP_C)), levels=[0.05, 0.3], colors=COL["cond"], linestyles=":", linewidths=1.2)
    ax.contour(CENTERS, CENTERS, as_img(POSTERIOR), levels=[0.5], colors="#0b0b0b", linestyles="--", linewidths=1)
    ax.set_title(f"w = {w}"); ax.set_aspect("equal"); ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1.8, 2.8); ax.set_xlabel("$x_1$")
axes[0].set_ylabel("$x_2$")
fig.suptitle("Tilted target $p(x)\\,p(c\\mid x)^{w}$ (green fill) vs true conditional (blue dotted); data density grey, boundary $p(c\\mid x)=0.5$ dashed")
plt.show()
```
```text
w= 1: mean along prompt axis 1.00 | std 0.500 | mass where p(c|x) < 0.9: 0.232
w= 3: mean along prompt axis 1.12 | std 0.447 | mass where p(c|x) < 0.9: 0.114
w= 6: mean along prompt axis 1.18 | std 0.429 | mass where p(c|x) < 0.9: 0.060
w=15: mean along prompt axis 1.25 | std 0.409 | mass where p(c|x) < 0.9: 0.014
```
![Output of cell 3]({{ '/assets/blog/strange-math-of-cfg/cell-03-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Each panel shows the sharpened distribution CFG seems to aim for (green) at a different guidance weight, next to the true prompt distribution (blue dotted). Turning up w makes the green blob a little tighter and nudges it away from the dashed boundary, but only slightly.*

The tilt does what it says, and it does it gently. As $$w$$ climbs from 1 to 15, the target drifts from 1.00 to 1.25 along the prompt's axis, narrows from 0.500 to 0.409, and its ambiguous mass falls from 0.232 to 0.014. In the plot, the green fill backs away from the dashed decision boundary and tucks itself just inside the blue dotted contours of the true conditional. Hold on to those numbers. They're what CFG would give you if it sampled the distribution it's usually said to sample.

## Why CFG never gets there

A diffusion sampler doesn't follow the score of one distribution. It follows a whole family of scores, one per noise level, and it only ends up sampling $$q$$ if every member of that family is the score of $$q$$ blurred to that level. Blurring and sharpening don't commute, though. Sharpen and then blur, or blur and then sharpen, and you get two different densities. CFG builds its scores the second way.

![Two routes from the same clean densities at w = 4 and sigma = 1: sharpening then blurring (green) and blurring then sharpening (orange, what CFG's scores correspond to) end at visibly different densities]({{ '/assets/blog/strange-math-of-cfg/noncommute-paths.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The same two operations done in two different orders. Top route: sharpen first, then blur, which is what a correct sampler would need. Left and bottom route: blur first, then sharpen, which is what CFG actually does. The green and orange curves at the end don't match, and that mismatch is why CFG can't land on the target it seems to aim for.*

To make that precise, write $$\mathcal N_\sigma *$$ for convolution with $$\mathcal N(0,\sigma^2 I)$$, which is what the forward process does to a density. A valid reverse process for the tilted target needs the scores of $$\mathcal N_\sigma * \tilde p_w$$. CFG doesn't give it those. It gives it the score of the tilted noisy marginals, and in general

$$\begin{gathered} \mathcal N_\sigma * \big(p_0^{1-w}\,p_0(\cdot\mid c)^{w}\big) \\ \not\propto\; p_\sigma^{1-w}\,p_\sigma(\cdot\mid c)^{w}. \end{gathered}$$

So the CFG scores at different noise levels aren't the noisy versions of any one clean distribution. The sampler is chasing a target that keeps moving as $$\sigma$$ shrinks. Two 1D Gaussians show what that does.

*Case A, equal variances.* Take $$p_0(x)=\mathcal N(0,s_0^2)$$ and $$p_0(x\mid c)=\mathcal N(\mu,s_0^2)$$. Noise keeps both Gaussian with variance $$v_\sigma=s_0^2+\sigma^2$$, and the CFG score collapses into the score of a single Gaussian:

$$\begin{aligned} \tilde s_w(x,\sigma) &= (1-w)\,\frac{-x}{v_\sigma} + w\,\frac{-(x-\mu)}{v_\sigma} \\ &= -\frac{x - w\mu}{v_\sigma}. \end{aligned}$$

That's exactly the score of $$\mathcal N(w\mu,\ s_0^2+\sigma^2)$$ at every $$\sigma$$, a perfectly valid family, so CFG samples $$\mathcal N(w\mu, s_0^2)$$. Here blurring and tilting happen to commute (the tilt is a pure shift), so CFG does land on the tilted target. Look at where that target is, though. The mean has moved from $$\mu$$ to $$w\mu$$, so CFG overshoots the conditional mean, linearly in $$w$$. For images, whatever separates "red apple" from "apple" gets multiplied by $$w$$, and out comes a neon apple.

![Case A with equal variances: as w goes from 1 to 6 the CFG sample density N(w mu, 0.5^2) (orange) keeps its width and slides away from the conditional N(mu, 0.5^2) (blue), its mean running from 1 to 6]({{ '/assets/blog/strange-math-of-cfg/mean-overshoot-1d.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*A one-dimensional example. Blue is the true distribution for the prompt and orange is what CFG produces. The orange curve keeps the same width, but its centre slides to w times the true centre, so at w = 6 it sits at 6 instead of 1.*

*Case B, unequal variances.* Now centre both at zero and make the conditional narrower: $$p_0(x)=\mathcal N(0,\sigma_u^2)$$ and $$p_0(x\mid c)=\mathcal N(0,\sigma_c^2)$$ with $$\sigma_c<\sigma_u$$. The CFG score is linear, $$\tilde s_w = -\tilde\tau_\sigma\,x$$, with precision

$$\tilde\tau_\sigma = \frac{w}{\sigma_c^2+\sigma^2} - \frac{w-1}{\sigma_u^2+\sigma^2}.$$

Any Gaussian that has been through the forward process has precision $$1/(a+\sigma^2)$$ for some fixed clean variance $$a$$. Put differently, the *implied clean variance* $$1/\tilde\tau_\sigma - \sigma^2$$ has to be the same at every $$\sigma$$. For CFG it isn't. At $$\sigma=0$$ it matches the tilted target's variance $$\big(w/\sigma_c^2-(w-1)/\sigma_u^2\big)^{-1}$$, and as $$\sigma\to\infty$$ it heads to $$w\sigma_c^2-(w-1)\sigma_u^2$$, which goes negative whenever $$w\sigma_c^2 < (w-1)\sigma_u^2$$. (You'll sometimes read that $$\tilde\tau_\sigma$$ blows up as $$\sigma\to0$$. It doesn't; it's finite there. What breaks is its shape across noise levels.)

The ODE is linear, so it integrates in closed form. Start at $$\mathcal N(0,\sigma_{\max}^2)$$, run down to $$\sigma=0$$, take the limit $$\sigma_{\max}\to\infty$$, and the samples end up with standard deviation

$$\text{std}_\text{ODE} = \frac{\sigma_c^{\,w}}{\sigma_u^{\,w-1}},$$

which is not the tilted target's $$\big(w/\sigma_c^2-(w-1)/\sigma_u^2\big)^{-1/2}$$.

[Bradley & Nakkiran (2024)](https://arxiv.org/abs/2408.09000) prove the general version: neither DDPM nor DDIM sampling with CFG produces the "gamma-powered" distribution $$p(x)^{1-w}p(x\mid c)^{w}$$, and the two samplers don't even agree with each other. They also say what CFG is instead. In the SDE limit it's a predictor-corrector method in the sense of [Song et al. (2020)](https://arxiv.org/abs/2011.13456), alternating a DDIM-style denoising step on the conditional distribution with a Langevin correction that sharpens toward a gamma-powered distribution at the current noise level. If you want one line for what the guidance slider does, "denoise, then sharpen, then repeat" is the most honest one I know.

### Two counterexamples you can run

Cell 4 adds the sampler that the rest of the post reuses. `sample` runs the reverse variance-exploding process from $$\mathcal N(0, 80^2)$$ down to $$\sigma=0$$ on the noise schedule from [Karras et al. (2022)](https://arxiv.org/abs/2206.00364), and it can do that three ways. `ode` integrates the probability-flow ODE $$dx/d\sigma = -\sigma\,\tilde s_w(x,\sigma)$$ with Heun's method. `ddim` uses plain Euler on the same ODE, which in this parameterisation is exactly DDIM with $$\eta=0$$. `sde` runs the reverse SDE and adds fresh Gaussian noise at every step. The cell then samples Case A at $$w=3$$ ($$s_0=0.5$$, $$\mu=1$$) and Case B at $$w\in\{1,3,6\}$$ ($$\sigma_c=0.5$$, $$\sigma_u=1$$), and plots the implied clean variance against $$\sigma$$.

```python
def karras_sigmas(steps: int, s_min: float = 0.002, s_max: float = 80.0, rho: float = 7.0) -> np.ndarray:
    ramp = np.linspace(0, 1, steps)
    return np.append((s_max ** (1 / rho) + ramp * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho, 0.0)

def sample(score_fn, n: int, dim: int, steps: int = 40, method: str = "ode", seed: int = SEED) -> np.ndarray:
    """Reverse VE diffusion from N(0, s_max^2 I).
    ode  = Heun on the probability-flow ODE dx/dsigma = -sigma * score
    ddim = Euler on the same ODE (exactly DDIM with eta = 0 in VE form)
    sde  = reverse SDE, Euler-Maruyama: fresh Gaussian noise every step."""
    rng = np.random.default_rng(seed)
    sig = karras_sigmas(steps)
    x = sig[0] * rng.standard_normal((n, dim))
    for s, s_next in zip(sig[:-1], sig[1:]):
        if method == "sde":
            g2 = s**2 - s_next**2
            x = x + g2 * score_fn(x, s) + np.sqrt(g2) * rng.standard_normal(x.shape)
            continue
        d = -s * score_fn(x, s)
        x_next = x + (s_next - s) * d
        if method == "ode" and s_next > 0:                       # Heun correction
            x_next = x + (s_next - s) * 0.5 * (d - s_next * score_fn(x_next, s_next))
        x = x_next
    return x

s0, mu, w = 0.5, 1.0, 3.0                                       # Case A: N(0, s0^2) vs N(mu, s0^2)
case_a = lambda x, s: (1 - w) * (-x / (s0**2 + s**2)) + w * (-(x - mu) / (s0**2 + s**2))
xa = sample(case_a, 20000, 1, steps=60)
print(f"Case A, w={w:g}: sample mean {xa.mean():.3f}, std {xa.std():.3f}   (conditional is N({mu:g}, {s0}^2); CFG gives N(w*mu = {w * mu:g}, {s0}^2))")

sc, su = 0.5, 1.0                                               # Case B: N(0, su^2) vs narrower N(0, sc^2)
cfg_precision = lambda s, w: w / (sc**2 + s**2) - (w - 1) / (su**2 + s**2)
sig_axis = np.logspace(-2, np.log10(80), 200)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for w, c in [(1, COL["cond"]), (3, COL["cfg"]), (6, "#a8401a")]:
    xo = sample(lambda x, s: -cfg_precision(s, w) * x, 20000, 1, steps=200)
    xs = sample(lambda x, s: -cfg_precision(s, w) * x, 20000, 1, steps=400, method="sde")
    print(f"Case B, w={w}: ODE std {xo.std():.4f} (predicted {sc**w / su**(w - 1):.4f}) | SDE std {xs.std():.4f} | tilted target std {cfg_precision(0, w) ** -0.5:.4f}")
    ax.plot(sig_axis, 1 / cfg_precision(sig_axis, w) - sig_axis**2, color=c, label=f"w = {w}")
ax.axhline(0, color="#0b0b0b", linewidth=0.8)
ax.set_xscale("log"); ax.set_xlabel("noise level $\\sigma$"); ax.set_ylabel("implied clean variance $1/\\tilde\\tau_\\sigma - \\sigma^2$")
ax.set_title("A valid diffusion keeps this flat; CFG makes it drift (and go negative)"); ax.legend()
plt.show()
```
```text
Case A, w=3: sample mean 2.984, std 0.500   (conditional is N(1, 0.5^2); CFG gives N(w*mu = 3, 0.5^2))
Case B, w=1: ODE std 0.4982 (predicted 0.5000) | SDE std 0.5076 | tilted target std 0.5000
Case B, w=3: ODE std 0.1247 (predicted 0.1250) | SDE std 0.2609 | tilted target std 0.3162
Case B, w=6: ODE std 0.0157 (predicted 0.0156) | SDE std 0.1762 | tilted target std 0.2294
```
![Output of cell 4]({{ '/assets/blog/strange-math-of-cfg/cell-04-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*If the sampler were following a real distribution, this curve would be a flat line, like the blue one at w = 1. With guidance (orange curves) it drifts and even drops below zero, a variance no real distribution can have. That is the moving target the sampler ends up chasing.*

Case A comes out where the algebra put it. The mean is 2.984, which is $$w\mu=3$$, three times the conditional mean, and the width hasn't changed (the 0.016 shortfall is there because the sampler starts at $$\sigma_{\max}=80$$ rather than infinity). Case B is the interesting one. At $$w=3$$ the ODE samples have a standard deviation of 0.1247, against 0.1250 from the formula and 0.3162 for the tilted target. CFG has squeezed the distribution to less than half the width that even the *sharpened* target asks for. At $$w=6$$ it's 0.0157 against 0.2294. The SDE sampler lands much closer (0.2609 and 0.1762), which is worth remembering for later.

The plot shows why. At $$w=1$$ the implied clean variance is a flat line at $$\sigma_c^2=0.25$$, as it should be for a real diffusion. At $$w=3$$ and $$w=6$$ it starts at the tilted target's variance for small $$\sigma$$ (0.1 and about 0.053), crosses zero at $$\sigma\approx0.45$$ for $$w=3$$ and $$\sigma\approx0.27$$ for $$w=6$$, and keeps falling toward −1.25 and −3.5 as $$\sigma$$ grows. No distribution has a negative variance, so at high noise the sampler is following scores that don't belong to any clean distribution at all.

## In 2D, samples run from the boundaries

Back in 2D both cases happen at once, and the mixture adds a third effect of its own. Near the origin all three modes overlap, so the unconditional density $$p_\sigma(x)$$ is high there. The CFG score contains a $$-(w-1)\,\nabla_x\log p_\sigma(x)$$ term, so whenever $$w>1$$ it actively climbs *down* the unconditional density. Wherever classes overlap, guidance shoves samples out of the overlap and through the far side of the prompt's mode. The cloud overshoots the mode centre and piles up on its outer rim, about as far from the other classes as it can get. [Chidambaram et al. (2024)](https://arxiv.org/abs/2409.13074) prove a version of this for mixtures of Gaussians and of compactly supported distributions: as guidance grows, samples increasingly favour the boundary of the conditional's support.

That one picture holds both the help and the harm. Being far from the decision boundary is what makes a sample unambiguous, a golden retriever nobody would mistake for a cat. Sitting on the far rim of the mode is what makes it exaggerated, a retriever that's somehow extra golden.

![Particles following the reverse probability-flow ODE with the CFG score for w = 1, 3, 6 and 15: at w = 1 they settle on the prompt's mode, at larger w they are pushed out of the overlap between classes and land beyond the tilted target (green), with final means 0.99, 1.65, 2.15 and 3.03 along the prompt's axis]({{ '/assets/blog/strange-math-of-cfg/trajectories-by-w.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Each panel starts from the same noise but uses a different guidance weight. Watch where the orange dots end up: at w = 1 they settle on the prompt's group, while at larger w they get pushed out of the crowded middle, past the green target, and pile up further and further away.*

### Sampling the mixture for real

Cell 5 draws 8,000 CFG samples for each $$w\in\{1,3,6,15\}$$, once with the ODE (40 Heun steps) and once with DDIM (25 Euler steps), and compares them with the tilted target from cell 3 using a histogram estimate of $$\mathrm{KL}(\text{samples}\,\|\,\text{target})$$ on the same grid. Histogram KL is biased upward with a finite sample, so the cell also scores 8,000 *exact* draws from the target. That's the floor a perfect sampler would hit.

```python
N_SAMPLES = 8000

def kl_to_grid(samples: np.ndarray, q: np.ndarray) -> float:
    """Histogram estimate of KL(samples || q) on the shared grid (q holds bin masses)."""
    h, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[EDGES, EDGES])
    p = h.ravel() / len(samples)
    m = p > 0
    return float((p[m] * np.log(p[m] / np.maximum(q[m], 1e-300))).sum())

def draw_from_grid(q: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    """Exact samples from the grid distribution q: the estimator's noise floor."""
    rng = np.random.default_rng(seed)
    cell = EDGES[1] - EDGES[0]
    return GRID[rng.choice(len(q), size=n, p=q)] + rng.uniform(-cell / 2, cell / 2, (n, 2))

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
for ax, w in zip(axes, [1, 3, 6, 15]):
    q = tilted_target(w)
    x_ode = sample(cfg_score(w), N_SAMPLES, 2, steps=40, method="ode")
    x_ddim = sample(cfg_score(w), N_SAMPLES, 2, steps=25, method="ddim")
    print(f"w={w:>2}: target mean {q @ GRID @ PROMPT_DIR:.2f} | ODE mean {(x_ode @ PROMPT_DIR).mean():.2f}, std {x_ode.std(0).mean():.3f}"
          f" | KL to target: ODE {kl_to_grid(x_ode, q):.2f}, DDIM {kl_to_grid(x_ddim, q):.2f}, exact-sample floor {kl_to_grid(draw_from_grid(q, N_SAMPLES), q):.2f}")
    ax.contour(CENTERS, CENTERS, as_img(q), levels=6, colors=COL["target"], linewidths=1)
    ax.scatter(x_ode[:1500, 0], x_ode[:1500, 1], s=3, color=COL["cfg"], alpha=0.35, linewidths=0)
    ax.contour(CENTERS, CENTERS, as_img(POSTERIOR), levels=[0.5], colors="#0b0b0b", linestyles="--", linewidths=1)
    ax.set_title(f"w = {w}"); ax.set_aspect("equal"); ax.set_xlim(-2.5, 4); ax.set_ylim(-2.5, 4); ax.set_xlabel("$x_1$")
axes[0].set_ylabel("$x_2$")
fig.suptitle("CFG samples from the reverse ODE (orange) vs the tilted target they supposedly follow (green contours)")
plt.show()
```
```text
w= 1: target mean 1.00 | ODE mean 0.99, std 0.504 | KL to target: ODE 0.07, DDIM 0.08, exact-sample floor 0.07
w= 3: target mean 1.12 | ODE mean 1.65, std 0.356 | KL to target: ODE 0.96, DDIM 1.27, exact-sample floor 0.05
w= 6: target mean 1.18 | ODE mean 2.15, std 0.316 | KL to target: ODE 2.77, DDIM 3.31, exact-sample floor 0.04
w=15: target mean 1.25 | ODE mean 3.03, std 0.284 | KL to target: ODE 8.39, DDIM 9.23, exact-sample floor 0.04
```
![Output of cell 5]({{ '/assets/blog/strange-math-of-cfg/cell-05-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Where CFG samples really land (orange dots) compared with the target they are supposed to follow (green contours). As w grows, the cloud gets tighter and drifts upward, well away from the green target.*

At $$w=1$$ everything lines up. The ODE mean is 0.99 against a target of 1.00, and its KL of 0.07 matches the floor, so both the sampler and the estimator check out. From $$w=3$$ on, the samples leave the target behind. The target barely moves (1.12, 1.18, 1.25) while the samples' mean runs off to 1.65, 2.15 and 3.03, and their spread shrinks from 0.504 to 0.284, well under the target's 0.41. By the top of the range KL has climbed to 8.39 for the ODE and 9.23 for DDIM, against floors of 0.04 to 0.05. In the figure the orange cloud at $$w=15$$ sits around $$x_2\approx3$$, about 1.8 units beyond the mean of the green target it's supposed to be following (1.25), so CFG overshoots even its own nominal target.

## Why the SDE sampler does better

Case B already showed the SDE sampler ending up much closer to the tilted target than the ODE. The difference comes down to what each one does with an error. A deterministic sampler commits to its trajectory, so whatever mismatch it picks up at high noise between the CFG scores and any valid family, it carries all the way to the end. A stochastic sampler injects fresh noise at every step and then partly removes it. That loop behaves like Langevin dynamics, and Langevin dynamics driven by $$\tilde s_w$$ pulls samples toward the distribution whose score $$\tilde s_w$$ actually is at the current noise level, $$p_\sigma(x)^{1-w}\,p_\sigma(x\mid c)^{w}$$. Step after step, the SDE re-equilibrates toward that moving density, so it finishes nearer the tilted target. That's the corrector in Bradley & Nakkiran's picture.

There's a practical consequence. The same `guidance_scale` does different things under a deterministic solver (DDIM, DPM-Solver-style ODE samplers) and a stochastic one (ancestral or DDPM-style), so a setting you tuned for one sampler won't carry over one-to-one to the other.

![Same scores, same w = 6, same starting noise: smooth ODE paths (left) overshoot the tilted target and end at KL 2.77, while noisy SDE paths (right) keep getting pulled back and end at KL 0.40 (sample means 2.15 and 1.52 against a target mean of 1.18)]({{ '/assets/blog/strange-math-of-cfg/ode-vs-sde.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Same model, same w = 6, same starting points, two ways of running the sampler. The smooth version (left) overshoots the green target. The noisy version (right) keeps getting nudged back and finishes much closer to it.*

### One guidance weight, three samplers

Cell 6 fixes $$w=6$$ and runs all three samplers from the same seed, with exact draws from the target as a reference row.

```python
w = 6
q = tilted_target(w)
runs = {
    "ODE  (Heun, 40 steps)": sample(cfg_score(w), N_SAMPLES, 2, steps=40, method="ode"),
    "DDIM (Euler, 25 steps)": sample(cfg_score(w), N_SAMPLES, 2, steps=25, method="ddim"),
    "SDE  (Euler-Maruyama, 250 steps)": sample(cfg_score(w), N_SAMPLES, 2, steps=250, method="sde"),
    "exact draws from the target": draw_from_grid(q, N_SAMPLES),
}
print(f"w = {w}: KL to the tilted target p(x)^(1-w) p(x|c)^w   [target mean along prompt axis {q @ GRID @ PROMPT_DIR:.2f}]")
for name, xs in runs.items():
    print(f"  {name:<34} KL {kl_to_grid(xs, q):5.2f} | mean along prompt axis {(xs @ PROMPT_DIR).mean():.2f}")
```
```text
w = 6: KL to the tilted target p(x)^(1-w) p(x|c)^w   [target mean along prompt axis 1.18]
  ODE  (Heun, 40 steps)              KL  2.77 | mean along prompt axis 2.15
  DDIM (Euler, 25 steps)             KL  3.31 | mean along prompt axis 2.20
  SDE  (Euler-Maruyama, 250 steps)   KL  0.40 | mean along prompt axis 1.52
  exact draws from the target        KL  0.04 | mean along prompt axis 1.18
```

The disagreement is large. With identical scores and the same $$w$$, the ODE ends at KL 2.77 and DDIM at 3.31, while the SDE reaches 0.40 against a floor of 0.04. The means say the same thing: the target sits at 1.18 along the prompt's axis, the SDE at 1.52, the ODE at 2.15 and DDIM at 2.20. Noise doesn't fix CFG, and 0.40 is still ten times the floor. It does cut the KL by about a factor of seven (2.77 to 0.40) and the mean overshoot by about a factor of three (0.97 to 0.34), which is what you'd expect if the injected noise is doing correction work the deterministic samplers never get to do.

## Looking at the clean image instead

Everything so far happened in noisy space. To see what CFG does to *images*, it helps to look at the network's guess of the clean image instead. Tweedie's formula ([Efron, 2011](https://doi.org/10.1198/jasa.2011.tm11181)) ties the score to the best possible denoiser, the posterior mean of the clean image given the noisy one:

$$\begin{aligned} \hat x_0(x_\sigma) &= \mathbb E[x_0\mid x_\sigma] \\ &= x_\sigma + \sigma^2\,\nabla_x\log p_\sigma(x_\sigma). \end{aligned}$$

Plug in the CFG score and the $$x_\sigma$$ terms cancel:

$$\begin{aligned} \tilde x_0 &= \hat x_0(x_\sigma) \\ &\quad + w\,\big[\hat x_0(x_\sigma, c) - \hat x_0(x_\sigma)\big]. \end{aligned}$$

In clean-image space, then, CFG draws a straight line from the unconditional guess through the conditional guess, and keeps going for another $$w-1$$ times that distance.

That line is why guidance helps. At high and medium noise $$\hat x_0$$ is a *mean*, the average of every clean image the noisy state could have come from. Natural images live on a thin, curved manifold, and the average of points on a curve sits inside the curve, off the manifold. That's the blur in a $$w=1$$ sample's early steps. The difference vector $$\hat x_0(x_\sigma,c) - \hat x_0(x_\sigma)$$ points from the global average toward the prompt's average. Part of it is normal to the prompt's patch of the manifold and pulls the guess back onto it; part of it is tangent and pushes the guess away from neighbouring classes.

It's also why guidance hurts. The manifold curves and the line doesn't, so once the line has crossed the manifold, more $$w$$ only carries it further out, into high-norm territory where the implicit classifier is extremely confident and no real image lives. In pixel space that's the deep-fried look: saturated colour as pixel norms grow, plus exaggerated edges. [Chidambaram et al. (2024)](https://arxiv.org/abs/2409.13074) arrive at the same place from the other side. With any nonzero score error, a large enough guidance weight pushes samples off the support of the data.

![CFG as straight-line extrapolation in clean-image space: the guided guess x0-tilde (orange) moves along the line from the unconditional to the conditional Tweedie guess, crosses the unit-circle manifold near w = 1.6 and then leaves it, ending about 2.0 off the manifold at w = 6 (shown for one noisy point)]({{ '/assets/blog/strange-math-of-cfg/manifold-overshoot.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*The circle stands for the set of real images, and the blue half is what the prompt asks for. CFG builds its guess (orange) by extending a straight line from the 'any image' guess through the 'prompt' guess. A little guidance puts the guess right on the circle; more guidance shoots it straight past, into territory where no real image exists. That is the deep-fried look.*

Off-manifold behaviour is exactly what [CFG++ (Chung et al., 2024)](https://arxiv.org/abs/2406.08070) targets. Its authors blame CFG's off-manifold behaviour for mode collapse, and they reformulate the guided step so it stays on the manifold.

### A circle for a manifold

Cell 7 swaps the three blobs for something curved: the unit circle, built from 128 narrow Gaussians (std 0.03), with the prompt picking the upper half. It draws 4,000 noisy points from the prompt's half at $$\sigma=1$$, computes both Tweedie denoisers exactly, and checks that score-space CFG and $$\hat x_0$$-space extrapolation really are the same operation. Then it turns up $$w$$ and tracks two things: how far $$\tilde x_0$$ lands off the circle (the harm), and how deep it sits inside the prompt's half, measured as the mean of $$\sin\theta$$, where $$\theta$$ is the angle of $$\tilde x_0$$ around the circle (the help; 1 is the top of the circle and 0 is the class boundary).

```python
K, RING_STD = 128, 0.03
theta = (np.arange(K) + 0.5) * 2 * np.pi / K
RING = np.stack([np.cos(theta), np.sin(theta)], axis=1)       # data manifold: the unit circle
UPPER = RING[:, 1] > 0                                         # the prompt selects the upper half

def ring_score(x, s, mask): return mixture_score(x, s, RING[mask], np.full(mask.sum(), RING_STD**2), np.full(mask.sum(), 1 / mask.sum()))
def denoiser(x, s, mask): return x + s**2 * ring_score(x, s, mask)   # Tweedie: E[x0 | x_t]

sigma, rng = 1.0, np.random.default_rng(SEED)
xt = RING[UPPER][rng.integers(0, UPPER.sum(), 4000)] + RING_STD * rng.standard_normal((4000, 2)) + sigma * rng.standard_normal((4000, 2))
x0_u, x0_c = denoiser(xt, sigma, np.ones(K, bool)), denoiser(xt, sigma, UPPER)
via_score = xt + sigma**2 * cfg_score(4.0, lambda x, s: ring_score(x, s, UPPER), lambda x, s: ring_score(x, s, np.ones(K, bool)))(xt, sigma)
print(f"Tweedie check, w=4: CFG in score space vs extrapolation in x0-hat space, max |diff| = {np.abs(via_score - (x0_u + 4.0 * (x0_c - x0_u))).max():.1e}")

ws = np.concatenate([np.linspace(1, 3, 9), [4, 5, 6, 7.5, 10, 15, 20]])
extrap = {w: x0_u + w * (x0_c - x0_u) for w in ws}
off = np.array([np.abs(np.linalg.norm(extrap[w], axis=1) - 1).mean() for w in ws])            # harm: distance off the circle
margin = np.array([(extrap[w][:, 1] / np.linalg.norm(extrap[w], axis=1)).mean() for w in ws])  # help: depth into the class (sin of angle)
for w in [1, 1.5, 2, 3, 7.5, 20]:
    i = int(np.argmin(np.abs(ws - w)))
    print(f"  w={w:>4}: off-manifold distance {off[i]:.3f} | class margin {margin[i]:.3f}")
print(f"off-manifold distance is smallest at w = {ws[off.argmin()]:g} (sigma = {sigma})")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
t = np.linspace(0, 2 * np.pi, 400)
axes[0].plot(np.cos(t), np.sin(t), color=COL["uncond"], linewidth=1)
axes[0].plot(np.cos(t[t < np.pi]), np.sin(t[t < np.pi]), color=COL["cond"], linewidth=3, label="prompt's sub-manifold")
for w, c in [(1, "#f5b393"), (1.5, COL["cfg"]), (3, "#a8401a")]:              # light -> dark = more guidance
    axes[0].scatter(*extrap[w][:250].T, s=6, color=c, alpha=0.7, linewidths=0, label=f"$\\tilde x_0$, w = {w:g}")
axes[0].set_aspect("equal"); axes[0].set_xlim(-2.6, 2.6); axes[0].set_ylim(-1.3, 2.8)
axes[0].legend(fontsize=7, loc="upper left", markerscale=2.5); axes[0].set_title(f"Extrapolated $\\tilde x_0$ at $\\sigma$ = {sigma}")
axes[1].plot(ws, margin, color=COL["cond"]); axes[1].set_xlabel("guidance w"); axes[1].set_ylabel("mean sin(angle)")
axes[1].set_title("Help: depth into the class (saturates)")
axes[2].plot(ws, off, color=COL["cfg"]); axes[2].set_xlabel("guidance w"); axes[2].set_ylabel("mean | ||x|| - 1 |")
axes[2].set_title("Harm: distance off the manifold")
axes[2].axvline(ws[off.argmin()], color="#0b0b0b", linestyle="--", linewidth=1)
plt.show()
```
```text
Tweedie check, w=4: CFG in score space vs extrapolation in x0-hat space, max |diff| = 8.9e-16
  w=   1: off-manifold distance 0.235 | class margin 0.843
  w= 1.5: off-manifold distance 0.087 | class margin 0.901
  w=   2: off-manifold distance 0.180 | class margin 0.931
  w=   3: off-manifold distance 0.538 | class margin 0.961
  w= 7.5: off-manifold distance 2.367 | class margin 0.992
  w=  20: off-manifold distance 7.560 | class margin 0.997
off-manifold distance is smallest at w = 1.5 (sigma = 1.0)
```
![Output of cell 7]({{ '/assets/blog/strange-math-of-cfg/cell-07-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*Left: CFG's guesses for many noisy points at three guidance weights, light orange for w = 1 through dark orange for w = 3. Middle: how firmly the guesses sit inside the prompt's half, which improves quickly and then levels off. Right: how far the guesses land from the circle, which dips at first and then climbs steadily. The dashed line marks the best setting for this toy.*

The identity holds to $$8.9\times10^{-16}$$, and the geometry plays out the way the argument said it would. At $$w=1$$ the conditional guess is a blurry average sitting 0.235 inside the circle. A little guidance pulls it *onto* the manifold: at $$w=1.5$$ it's only 0.087 off, 2.7 times closer. Then the distance climbs again, close to linearly, through 0.180 at $$w=2$$, 0.538 at $$w=3$$ and 2.367 at $$w=7.5$$, up to 7.560 at $$w=20$$. The class margin shoots up early and flattens out: 0.843 at $$w=1$$, 0.901 at $$w=1.5$$, 0.961 at $$w=3$$ and 0.997 at $$w=20$$. Nearly all the help shows up early. The harm keeps coming. The optimum here, $$w=1.5$$, is specific to this circle at this noise level, so don't read anything into the value itself. What carries over is the shape, a short descent followed by a long climb.

## Guidance also covers for a bad network

Everything above used exact scores, and with exact scores $$w=1$$ is already perfect. There's nothing for guidance to fix. A trained network isn't exact, and its errors aren't random: networks are least accurate in the low-density regions between modes, where they saw the least data. Two kinds of error matter most in practice. The model is *blurry*, with too much spread around each mode, and it *leaks*, putting some of the prompt's probability on neighbouring classes. The leak is just imperfect prompt-following under another name.

CFG goes straight at the leak. Mass that leaked to another class sits exactly where $$p(c\mid x)$$ is small, which is what guidance suppresses, so this is the error-correction half of the help. [Karras et al. (2024)](https://arxiv.org/abs/2406.02507) follow that idea to its end with *autoguidance*. If guidance works by extrapolating away from errors, guide the model with a deliberately worse version of *itself* (smaller or less trained, with the same conditioning) instead of an unconditional model. The difference vector then points away from the model's own mistakes without also pointing away from the other classes, and you get the correction without the exaggeration. On ImageNet they report FIDs of 1.01 at 64×64 and 1.25 at 512×512.

### Building a deliberately bad model

Cell 8 builds the imperfect model by hand from the same three-class mixture: every component 1.3 times too wide, with 20% of the prompt's mass leaked onto the other two classes. The "bad" model for autoguidance gets the same treatment at 1.8 times and 40%. Autoguidance is the CFG formula unchanged, with the bad conditional model sitting where the unconditional model used to be. This time every sampler is scored against the *true* conditional $$p(x\mid c)$$, since the question now is which one gets closest to the real thing.

```python
def weak_model(blur: float, leak: float):
    """An imperfect network, analytically: every component too wide (blur) and part of the
    prompt's mass leaking onto the other classes (imperfect prompt-following)."""
    v = VARS * blur**2
    w_c = np.full(3, leak / 2)
    w_c[PROMPT] = 1 - leak
    return (lambda x, s: mixture_score(x, s, MEANS, v, w_c)), (lambda x, s: mixture_score(x, s, MEANS, v, UNIFORM))

main_cond, main_uncond = weak_model(blur=1.3, leak=0.2)        # the model we ship
bad_cond, _ = weak_model(blur=1.8, leak=0.4)                   # a worse version of it, for autoguidance
q_true = tilted_target(1.0)                                    # w = 1 tilt is the true conditional p(x|c)
ws = [1, 1.5, 2, 2.5, 3, 4, 6, 8]
kl_cfg = [kl_to_grid(sample(cfg_score(w, main_cond, main_uncond), N_SAMPLES, 2), q_true) for w in ws]
kl_auto = [kl_to_grid(sample(cfg_score(w, main_cond, bad_cond), N_SAMPLES, 2), q_true) for w in ws]   # same formula, worse model in the "uncond" slot
kl_exact = kl_to_grid(sample(score_cond, N_SAMPLES, 2), q_true)

print(f"KL to the TRUE conditional p(x|c)   [exact model, w=1: {kl_exact:.2f}]")
for w, a, b in zip(ws, kl_cfg, kl_auto):
    print(f"  w={w:>3}: weak model + CFG {a:5.2f} | weak model + autoguidance {b:5.2f}")
print(f"best CFG: w={ws[int(np.argmin(kl_cfg))]} (KL {min(kl_cfg):.2f}) | best autoguidance: w={ws[int(np.argmin(kl_auto))]} (KL {min(kl_auto):.2f})")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(ws, kl_cfg, marker="o", color=COL["cfg"], label="weak model + CFG")
ax.plot(ws, kl_auto, marker="o", color=COL["auto"], label="weak model + autoguidance")
ax.axhline(kl_exact, color=COL["uncond"], linestyle="--", linewidth=1, label="exact model, w = 1")
ax.set_xlabel("guidance w"); ax.set_ylabel("KL to true $p(x\\mid c)$"); ax.set_yscale("log"); ax.legend()
ax.set_title("Guidance as error correction: help first, then harm")
plt.show()
```
```text
KL to the TRUE conditional p(x|c)   [exact model, w=1: 0.07]
  w=  1: weak model + CFG  1.15 | weak model + autoguidance  1.15
  w=1.5: weak model + CFG  0.63 | weak model + autoguidance  0.59
  w=  2: weak model + CFG  0.50 | weak model + autoguidance  0.29
  w=2.5: weak model + CFG  0.63 | weak model + autoguidance  0.23
  w=  3: weak model + CFG  0.90 | weak model + autoguidance  0.32
  w=  4: weak model + CFG  1.60 | weak model + autoguidance  0.80
  w=  6: weak model + CFG  3.18 | weak model + autoguidance  2.12
  w=  8: weak model + CFG  4.83 | weak model + autoguidance  3.31
best CFG: w=2 (KL 0.50) | best autoguidance: w=2.5 (KL 0.23)
```
![Output of cell 8]({{ '/assets/blog/strange-math-of-cfg/cell-08-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*How close each method gets to the true prompt distribution as guidance increases (lower is better). Plain CFG (orange) improves a little and then gets much worse. Autoguidance (violet) improves more before it turns around. The dashed line is what a perfect model scores.*

The weak model starts at KL 1.15 at $$w=1$$, where the exact model scores 0.07. A little guidance helps: CFG drops to 0.63 at $$w=1.5$$ and 0.50 at $$w=2$$. Then the harm takes over: 0.90 at $$w=3$$, then 3.18 at $$w=6$$ and 4.83 at $$w=8$$, far worse than not guiding at all (1.15). Autoguidance traces the same U, only lower, and bottoms out at 0.23 at $$w=2.5$$. That's less than half of CFG's best, because its extrapolation strips out the model's own blur and leak without also shoving samples away from the other classes. Neither gets back to 0.07, though. Guidance can't stand in for a better network, and here plain CFG at its best is still seven times worse than the exact model.

## Where the sweet spot comes from

Line up the measurements and the one knob turns out to drive two groups of effects:

| | Effect | Where we measured it | How it changes with $$w$$ |
|---|---|---|---|
| Help | Repulsion from decision boundaries | cell 3: ambiguous mass 0.232 → 0.014 | saturates: nothing is less ambiguous than certain |
| Help | Pull onto the prompt's manifold | cell 7: off-manifold 0.235 → 0.087 | only until the line crosses the curve |
| Help | Cancelling the model's leak and blur | cell 8: KL 1.15 → 0.50 | only until the error is cancelled |
| Harm | Mean overshoot | cells 4 and 5: 1 → $$w\mu$$; 0.99 → 3.03 | linear in 1D, sublinear in the 2D mixture |
| Harm | Variance collapse, lost diversity | cells 4 and 5: std 0.50 → 0.125 (1D, $$w=3$$); 0.504 → 0.284 (2D) | compounds with $$w$$ |
| Harm | Extrapolation off the curved manifold | cell 7: 0.087 → 7.560 | grows roughly linearly |

The help saturates while the harm keeps growing, so their sum has an optimum somewhere in the middle. On these toys it lands between $$w\approx1.5$$ and $$2.5$$. For Stable Diffusion on real prompts the shipped default is 7.5. The number depends on how far apart the classes are, how curved the manifold is and how bad the network's errors are, but the shape of the trade-off doesn't.

So CFG's scores aren't the noised scores of any one distribution, and it doesn't sample the distribution it appears to target. In the SDE limit it's a predictor-corrector that keeps pushing toward "more like the prompt than anything else", and that push helps right up until it doesn't.

### What these toys can't tell you

Everything here lives in two dimensions with exact scores, and dimension matters. [Lehman Pavasovic et al. (2025)](https://arxiv.org/abs/2502.07849) find a "blessing of dimensionality": in their high-dimensional analysis, CFG's distortions of the target shrink as dimension grows. [Ventura et al. (2026)](https://arxiv.org/abs/2602.00716) add a caveat, showing the distortions persist when the number of classes grows exponentially with dimension, which looks a lot more like open-ended text prompts. So I'd read numbers like KL 2.77, or a mean overshoot of 2.15 against 1.18, as illustrations of a mechanism. They predict nothing quantitative about a 64×64×4 latent, and this is the part of the argument I'm least sure transfers. Two other simplifications are worth knowing about. The noise process is variance-exploding, whereas Stable Diffusion 1.x and XL use variance-preserving schedules and newer models use flow matching (VP and VE are rescalings of each other, so the mechanism carries over; I haven't checked flow matching), and the "weak model" has the simplest error structure I could write down.

## What people are doing about the harm

Most of the recent work on guidance goes after one specific item on the harm list. For oversaturation and norm blow-up there's *CFG rescale* ([Lin et al., 2023](https://arxiv.org/abs/2305.08891)), which rescales the guided prediction toward the conditional prediction's standard deviation (blended back in with a factor $$\phi \approx 0.7$$) and so damps the high-norm drift from the clean-image section. *APG* ([Sadat et al., 2024](https://arxiv.org/abs/2410.02416)) splits the guidance update into a component parallel to the conditional prediction, which is the one that oversaturates, and an orthogonal one, which improves quality, then turns the parallel part down.

The *guidance interval* ([Kynkäänniemi et al., 2024](https://arxiv.org/abs/2404.07724)) targets lost diversity. It finds guidance harmful at high noise, mostly unnecessary at low noise and useful only in between, so it applies guidance only in that middle stretch (ImageNet-512 FID 1.81 → 1.40). That lines up with cell 4's plot, where the implied clean variance goes most wrong at large $$\sigma$$.

Off-manifold extrapolation is what *CFG++* ([Chung et al., 2024](https://arxiv.org/abs/2406.08070)) fixes, by recasting guidance as an inverse problem so the guided step stays on the manifold, and [Cai et al. (2026)](https://arxiv.org/abs/2601.21892) add a training-free manifold projection for flow-matching models. *Autoguidance* ([Karras et al., 2024](https://arxiv.org/abs/2406.02507)) handles class exaggeration by replacing the unconditional model with a worse copy of the conditional one, as in cell 8. [Moufad et al. (2025)](https://arxiv.org/abs/2505.21101) take the most direct route of all. They show CFG isn't a valid diffusion model because it drops a Rényi-divergence term that acts as a repulsive force, and they propose a Gibbs-like sampler that starts from a sample drawn without CFG and refines it toward the tilted target while preserving diversity.

## Optional: try it on Stable Diffusion

If you have a GPU, the cell below shows the same mechanism on a real model. You'll need CUDA, the `diffusers` stack and a one-time download of about 2.7 GB of half-precision Stable Diffusion v1.5 weights (from the [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) mirror of the original release). It renders one prompt at $$w\in\{1, 3, 7.5, 20\}$$ from the same starting noise, so the guidance weight is the only thing that changes from image to image. diffusers' `guidance_scale` is exactly our $$w$$.

```python
# Optional and NOT part of the CPU build. Needs a CUDA GPU and `pip install torch diffusers transformers accelerate`;
# the first run downloads ~2.7 GB of half-precision Stable Diffusion v1.5 weights. Flip RUN_SD to True to render the sweep.
import importlib.util

RUN_SD = False
PROMPT_TEXT = "a golden retriever in a sunlit park"
SD_SCALES = [1.0, 3.0, 7.5, 20.0]          # diffusers' guidance_scale is exactly our w (1 = plain conditional)
stack_ready = all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers"))

if not (RUN_SD and stack_ready):
    print(f"Skipped (RUN_SD={RUN_SD}, diffusers installed={stack_ready}). On a GPU machine set RUN_SD = True to render w = {SD_SCALES}.")
else:
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16").to("cuda")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    for ax, w in zip(axes, SD_SCALES):
        generator = torch.Generator("cuda").manual_seed(SEED)   # identical starting noise for every w
        image = pipe(PROMPT_TEXT, guidance_scale=w, num_inference_steps=30, generator=generator).images[0]
        ax.imshow(image); ax.set_title(f"w = {w:g}"); ax.axis("off")
    plt.show()
```
```text
Skipped (RUN_SD=False, diffusers installed=False). On a GPU machine set RUN_SD = True to render w = [1.0, 3.0, 7.5, 20.0].
```

In the CPU build this cell is switched off, which is why it prints "Skipped". With `RUN_SD = True` on a GPU you get the 1×4 grid. Each image maps back onto one of the toys. At $$w=1$$ you're sampling $$p(x\mid c)$$ from an imperfect network, so expect the hedged, washed-out look of cell 8's weak model at $$w=1$$. At 3 and 7.5 the boundary repulsion and error correction from cells 3, 7 and 8 are doing their job. At 20 you're out on the long climb of cell 7's harm curve, extrapolating off the manifold into high-norm pixel values that show up as oversaturated colour.

## Further reading

- [Ho & Salimans (2022), *Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598): the original method, short and readable. Mind the $$(1+w)$$ convention.
- [Bradley & Nakkiran (2024), *Classifier-Free Guidance is a Predictor-Corrector*](https://arxiv.org/abs/2408.09000): the proof that CFG doesn't sample the gamma-powered distribution, and what it does instead.
- [Chidambaram et al. (2024), *What does guidance do? A fine-grained analysis in a simple setting*](https://arxiv.org/abs/2409.13074): the mixture analysis behind the rim pile-up in cell 5.
- [Karras et al. (2024), *Guiding a Diffusion Model with a Bad Version of Itself*](https://arxiv.org/abs/2406.02507): autoguidance, and the clearest case that guidance is partly error correction.
- [Lehman Pavasovic et al. (2025), *Classifier-Free Guidance: From High-Dimensional Analysis to Generalized Guidance Forms*](https://arxiv.org/abs/2502.07849) and [Ventura et al. (2026), *Emergence of Distortions in High-Dimensional Guided Diffusion Models*](https://arxiv.org/abs/2602.00716): what changes once you leave two dimensions.
- [Dieleman (2022), *Guidance: a cheat code for diffusion models*](https://sander.ai/2022/05/26/guidance.html): a friendly tour of classifier and classifier-free guidance.
- For background, [Song et al. (2020)](https://arxiv.org/abs/2011.13456) on the SDE and probability-flow framework, [Karras et al. (2022)](https://arxiv.org/abs/2206.00364) for the noise schedule and Heun sampler used in the code, and [Fan et al. (2025), *CFG-Zero\**](https://arxiv.org/abs/2503.18886) for a fix specific to flow matching.
