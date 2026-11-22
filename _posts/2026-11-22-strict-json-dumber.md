---
layout: post
title: "Why Forcing an LLM to Output Valid JSON Can Secretly Make It Dumber"
date: 2026-11-22 16:00:00+0530
description: Why constrained decoding guarantees valid JSON but can cost reasoning accuracy, traced through a 30-line logit mask, the geometry of renormalisation, field order on a real model and the local-vs-global gap, with every claim checked in code.
tags: llm structured-outputs constrained-decoding json reasoning
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

Here's a small experiment. I gave Qwen2.5-0.5B-Instruct twenty arithmetic word problems and asked it to reply in JSON with two fields, `answer` and `reasoning`. Then I ran the same problems again with the two keys swapped. With `answer` first, it got none of them right. With `reasoning` first, it got all twenty. Both times, every single reply was valid JSON.

Nothing else changed: same model, same problems, same two fields. A 0.5B model on one kind of problem can't tell you how big this effect is in your own system, but it does show which way it points, and the reason sits inside the constrained decoder. So that's where this post goes. We'll write the logit mask ourselves in about 30 lines of Python and watch what it does to the model's probabilities, with pictures, a bit of math and code you can run on a laptop.

I'll assume you're comfortable with logits, softmax, autoregressive decoding, the KV cache and BPE.

## The paradox: JSON that always parses, answers that get worse

If you run an LLM in production, you have probably switched on structured outputs. The engine guarantees the response matches your schema, your `json.loads` stops throwing, and a whole class of 3 a.m. pages goes away. Then someone runs the eval suite and accuracy has moved the wrong way.

[Tam et al. (2024)](https://arxiv.org/abs/2408.02442) put numbers on this. On GSM8K, asking for JSON that follows a schema took Claude 3 Haiku from 86.51% (free text) to 23.44%, GPT-3.5-turbo from 75.99% to 49.25% and LLaMA-3-8B from 75.13% to 48.90%. Gemini 1.5 Flash barely moved (89.33% to 89.21%). Those rows come from format instructions in the prompt, not from a decoding constraint, and there is a rebuttal we'll get to near the end. Still, the direction is consistent enough to ask what is happening mechanically.

The model still knows JSON. What changes is the distribution it decodes from, and the damage shows up at three scales:

1. One token: masking invalid tokens and renormalising manufactures confidence the model never had.
2. One field: an answer-first schema deletes the scratchpad the answer depended on. Field order is computation order.
3. The whole output: masking token by token samples from a distribution that is not "the model's distribution, restricted to valid JSON".

By the end of the first code cell, a model that put 90% of its probability on hedging ("It depends...", "Maybe...") will emit `{"answer": true}` at 60% confidence.

## How constrained decoding works

Engines such as Outlines, vLLM, llama.cpp and XGrammar compile a JSON schema (or a regex, or a grammar) into an automaton over tokens: a finite state machine for regular constraints, as in [Willard & Louf (2023)](https://arxiv.org/abs/2307.09702), or a pushdown automaton for full context-free grammars, as in llama.cpp's GBNF and XGrammar. An FSM is enough for everything in this post. At decoding step $$t$$ the FSM is in a state $$s_t$$ that records how much of the JSON has been written. From $$s_t$$ it computes $$V_{\text{valid}}(s_t)$$: the tokens that keep the output a prefix of some valid document.

Before sampling, the engine edits the logit vector $$\mathbf{l} \in \mathbb{R}^{|V|}$$:

$$
l'_i = \begin{cases} l_i & i \in V_{\text{valid}}(s_t) \\ -\infty & \text{otherwise} \end{cases}
$$

so after the softmax every valid token $$i$$ gets

$$
P_{\text{masked}}(y_t = i) = \frac{p_i}{\sum_{j \in V_{\text{valid}}(s_t)} p_j},
$$

where $$p_i = \operatorname{softmax}(\mathbf{l})_i$$ is what the model would have said with no constraint. The denominator gets its own name, since the rest of the post keeps returning to it:

$$
M_t = \sum_{j \in V_{\text{valid}}(s_t)} p_j ,
$$

the valid mass at step $$t$$.

$$M_t$$ is the share of the model's belief that the grammar allows. The engine samples from the masked distribution, feeds the token to the FSM, moves to $$s_{t+1}$$, and repeats.

![One constrained decoding step as a loop: the model produces logits, the constraint engine masks every token outside the FSM's valid set (the discarded mass 1 − M_t is never seen downstream), decoding renormalises and picks y_t, and the FSM transition feeds y_t back into the model]({{ '/assets/blog/strict-json-dumber/mask-pipeline.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*How to read it: follow the loop. The model (middle) scores every one of its roughly 152,000 possible next tokens. The constraint engine (right) checks where we are in the JSON and blanks out every token that would break it, and whatever probability those tokens had is simply thrown away. Decode (left) rescales what is left so it adds up to 1 again, picks a token, and moves the JSON tracker one step forward. Then the loop runs again for the next token.*

### Setup

The first cell pins versions and seeds. Everything after it uses either hand-written toy distributions or a small open model on CPU. It was tested with Python 3.11.9, torch 2.13.0 and transformers 5.22.2, and the whole notebook runs in about 4 minutes on a laptop CPU, plus a one-time ~1 GB model download.

```python
%pip install -q "torch==2.13.0" "numpy==2.3.3" "matplotlib==3.10.8" "transformers==5.22.2"
import json, random, textwrap
import numpy as np
import torch
import matplotlib.pyplot as plt
import huggingface_hub
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
huggingface_hub.utils.logging.set_verbosity_error()
print(f"torch {torch.__version__.split('+')[0]} | numpy {np.__version__} | running on CPU")
```
```text
torch 2.13.0 | numpy 2.3.3 | running on CPU
```

### A 30-line FSM logit mask

Start with a toy. The vocabulary has 8 tokens and the FSM accepts exactly one schema, `{"answer": true|false}`, with states 0 to 5 (5 accepts). The "model" is a function from FSM state to logits. It knows JSON syntax perfectly: everywhere but state 3 it puts over 99.9% on the right structural token. At state 3, right after `{"answer":`, it has been asked whether a transaction is fraudulent and would rather hedge: 60% on "It" (as in "It depends"), 30% on "Maybe", and only 6% and 4% on `true` and `false`.

`mask_logits` is the formula above in code. Before running it, guess: does the mask just pick the best valid token, and how sure does it look doing so?

```python
VOCAB = ["{", '"answer"', ":", "true", "false", "}", "It", "Maybe"]
ID = {token: i for i, token in enumerate(VOCAB)}
# FSM for the schema {"answer": true|false}: state -> {allowed token: next state}
FSM = {0: {"{": 1}, 1: {'"answer"': 2}, 2: {":": 3}, 3: {"true": 4, "false": 4}, 4: {"}": 5}}

def toy_model(state: int) -> torch.Tensor:
    """Logits of a model that knows JSON syntax but, asked 'Is this transaction fraud?', wants to hedge."""
    if state == 3:  # right after {"answer": the model would rather write "It depends..." or "Maybe..."
        probs = {"It": 0.60, "Maybe": 0.30, "true": 0.06, "false": 0.04}
    else:           # everywhere else it confidently predicts the next piece of syntax
        probs = {next(iter(FSM[state])): 0.97}
    p = torch.full((len(VOCAB),), 1e-4)
    for token, v in probs.items():
        p[ID[token]] = v
    return (p / p.sum()).log()

def mask_logits(logits: torch.Tensor, state: int) -> torch.Tensor:
    """The whole trick: every token the FSM forbids in this state gets -inf before the softmax."""
    allowed = torch.tensor([ID[t] for t in FSM[state]])
    masked = torch.full_like(logits, float("-inf"))
    masked[allowed] = logits[allowed]
    return masked

def entropy_bits(p: torch.Tensor) -> float:
    p = p[p > 0]
    return max(0.0, float(-(p * p.log2()).sum()))  # max() avoids printing -0.00

state, output = 0, []
while state != 5:  # greedy constrained decoding
    logits = toy_model(state)
    p_raw, p_masked = logits.softmax(-1), mask_logits(logits, state).softmax(-1)
    token = VOCAB[int(p_masked.argmax())]
    print(f"state {state} | raw top {VOCAB[int(p_raw.argmax())]:>8} ({p_raw.max():.2f}) | "
          f"valid mass {p_raw[p_masked > 0].sum():.2f} | emit {token:>8} ({p_masked.max():.2f}) | "
          f"entropy {entropy_bits(p_raw):.2f} -> {entropy_bits(p_masked):.2f} bits")
    output.append(token)
    state = FSM[state][token]
print("\nJSON out:", "".join(output).replace(":", ": "))
```
```text
state 0 | raw top        { (1.00) | valid mass 1.00 | emit        { (1.00) | entropy 0.01 -> 0.00 bits
state 1 | raw top "answer" (1.00) | valid mass 1.00 | emit "answer" (1.00) | entropy 0.01 -> 0.00 bits
state 2 | raw top        : (1.00) | valid mass 1.00 | emit        : (1.00) | entropy 0.01 -> 0.00 bits
state 3 | raw top       It (0.60) | valid mass 0.10 | emit     true (0.60) | entropy 1.40 -> 0.97 bits
state 4 | raw top        } (1.00) | valid mass 1.00 | emit        } (1.00) | entropy 0.01 -> 0.00 bits

JSON out: {"answer": true}
```

On the syntax steps (states 0, 1, 2 and 4) the valid mass is 1.00, so the mask costs nothing. State 3 is where it bites. The model's top token is `It` at 0.60, the valid mass is 0.10, and the engine emits `true` at 0.60. Ninety percent of what the model believed was deleted, and `true` jumped tenfold, from 0.06 to 0.60, purely from dividing by $$M_t = 0.10$$.

Entropy fell too, from 1.40 to 0.97 bits, so the output is *more* certain than the model was. And the final string `{"answer": true}` keeps no trace of the hedge or the 0.10. Downstream code has nothing to inspect.

## Rung 1, one token: renormalisation manufactures confidence

Why does it behave like that? Write $$A_t$$ for the event "the next token is valid", $$\{y_t \in V_{\text{valid}}(s_t)\}$$. By Bayes' rule

$$
\begin{aligned}
P(y_t = i \mid A_t) &= \frac{P(y_t = i,\, A_t)}{P(A_t)} \\
&= \frac{p_i \,\mathbb{1}[i \in V_{\text{valid}}]}{M_t},
\end{aligned}
$$

which is exactly the masked distribution. For a single step, masking is conditioning on validity. Two things follow. Ratios among valid tokens survive: $$p_{\text{true}} / p_{\text{false}} = 0.06/0.04 = 1.5$$ before and $$0.60/0.40 = 1.5$$ after. And $$M_t$$ is divided out, so the output keeps no record of how much belief sat outside the valid set.

Geometry shows what gets lost. A next-token distribution is a point on the probability simplex $$\Delta^{|V|-1}$$ (non-negative coordinates summing to 1), and the valid tokens span one face $$F$$ of it. Masking zeroes the invalid coordinates and rescales:

$$
q = \frac{\Pi_F\, p}{\mathbf{1}^\top \Pi_F\, p},
$$

where $$\Pi_F$$ keeps only the valid coordinates. This is a radial projection: every point on the ray from the origin through $$\Pi_F p$$ lands on the same $$q$$. Collapse the vocabulary to three outcomes, {true, false, hedge}, and the simplex becomes a triangle with the true-false edge as its valid face. Masking slides each distribution along the straight line from the hedge corner, through $$p$$, down to that edge, and every point on that line ends up in the same place. The map is many-to-one. The coordinate it throws away is exactly $$p_{\text{hedge}} = 1 - M_t$$.

Entropy can move either way. In the cell above it fell. Take $$p = (0.05, 0.05, 0.90)$$ instead: $$H(p) \approx 0.57$$ bits, but the masked $$q = (0.5, 0.5)$$ has $$H(q) = 1$$ bit. The masked entropy describes the valid tokens and says nothing about how much the model wanted to leave.

![A confident hedger at (true, false, hedge) = (0.06, 0.04, 0.90) slides along the ray from the masked hedge corner onto the valid true-false edge and lands at (0.60, 0.40), the same point as a genuine 60/40 model]({{ '/assets/blog/strict-json-dumber/simplex-projection.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*How to read it: every point inside the triangle is one possible belief over three options: true, false, or hedge. The closer a point sits to a corner, the more weight that option gets. The red dot is a model that mostly wants to hedge (90%). Masking forbids hedging, so the dot slides straight down to the bottom edge, where only true and false exist, and lands at 60/40. The blue dot that then appears is a model that really was 60/40 all along. After masking, the two look exactly the same.*

The next cell draws this for four distributions.

```python
# Collapse the vocabulary to 3 outcomes: true, false, hedge (= everything the mask forbids).
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])  # true, false, hedge

def to_xy(p: np.ndarray) -> np.ndarray:
    """Barycentric (p_true, p_false, p_hedge) -> a 2D point inside the triangle."""
    return p @ CORNERS

def renormalise(p: np.ndarray) -> np.ndarray:
    """Mask 'hedge' and rescale. In 2D this slides p along the ray from the hedge corner onto the valid edge."""
    q = p.copy()
    q[2] = 0.0
    return q / q.sum()

dists = {"confident hedge": [0.06, 0.04, 0.90], "unsure model": [0.30, 0.20, 0.50],
         "genuine 60/40": [0.60, 0.40, 0.00], "hedge, leans false": [0.02, 0.08, 0.90]}
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.add_patch(plt.Polygon(CORNERS, fill=False, lw=1.2, color="0.4"))
ax.plot(*CORNERS[:2].T, lw=6, alpha=0.3, color="tab:green", label="valid face (true-false edge)")
for (name, p), color in zip(dists.items(), ["tab:red", "tab:orange", "tab:blue", "tab:purple"]):
    p = np.array(p)
    q = renormalise(p)
    start, end = to_xy(p), to_xy(q)
    if not np.allclose(start, end):
        ax.annotate("", end, start, arrowprops=dict(arrowstyle="->", color=color, lw=1.6))
    ax.scatter(*start, color=color, s=55, zorder=3, label=name)
    print(f"{name:>18}: (true, false, hedge) = {p.tolist()} -> masked (true, false) = ({q[0]:.2f}, {q[1]:.2f})")
for xy, label, dy in zip(CORNERS, ["true", "false", "hedge (masked)"], [-0.07, -0.07, 0.03]):
    ax.text(xy[0], xy[1] + dy, label, ha="center", fontsize=11)
ax.set_title("Masking = sliding every distribution onto the valid edge\n(three different models land on the same 60/40 point)", pad=28)
ax.set_aspect("equal"); ax.set_axis_off(); ax.legend(loc="upper left", fontsize=8, frameon=False)
plt.show()
```
```text
   confident hedge: (true, false, hedge) = [0.06, 0.04, 0.9] -> masked (true, false) = (0.60, 0.40)
      unsure model: (true, false, hedge) = [0.3, 0.2, 0.5] -> masked (true, false) = (0.60, 0.40)
     genuine 60/40: (true, false, hedge) = [0.6, 0.4, 0.0] -> masked (true, false) = (0.60, 0.40)
hedge, leans false: (true, false, hedge) = [0.02, 0.08, 0.9] -> masked (true, false) = (0.20, 0.80)
```
![Output of cell 3]({{ '/assets/blog/strict-json-dumber/cell-03-out-1.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*How to read it: the same triangle, now with four models. Each arrow shows where masking moves a model. The red, orange and blue models start in very different places (90%, 50% and 0% hedging) but all end on the same spot on the bottom edge. Only the purple model, whose small non-hedge share leans towards false, lands somewhere else. Masking keeps the ratio between true and false and throws everything else away.*

Three very different models print the same line. The confident hedger (0.06, 0.04, 0.90), the unsure one (0.30, 0.20, 0.50) and the calibrated 60/40 model all come out as (0.60, 0.40); only the fourth, which leans `false` inside its hedge, lands at (0.20, 0.80). Code that reads the JSON, or even the masked probabilities, cannot tell a model that is 60% sure from one that is 6% sure.

### What happens in a real model

Toy logits show the mechanism. Whether it shows up in practice is a separate question, so the next cell loads Qwen2.5-0.5B-Instruct and gives it word problems with four arithmetic operations ("A baker makes 9 trays of 18 cookies, sells 29, then bakes 5 more trays of 11..."). For each problem it measures two things:

- With no schema: what is the first token, and how much probability goes to a digit? That digit mass is $$M_1$$ if a schema forced the answer at the very first token.
- With an answer-first schema and `{"answer": ` already forced: how much probability goes to digits (the valid mass), and what does the masked first-digit distribution look like?

I expected the valid mass after `{"answer": ` to be small, as in the toy.

```python
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # ~1 GB download on first run; CPU is fine
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
DIGIT_IDS = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]  # Qwen tokenizes numbers digit by digit
rng = random.Random(SEED)

def make_problem() -> tuple[str, int]:
    """A four-operation word problem: multiply, subtract, multiply, add."""
    a, b, c, d, e = rng.randint(3, 9), rng.randint(12, 25), rng.randint(5, 30), rng.randint(2, 6), rng.randint(11, 19)
    q = (f"A baker makes {a} trays of {b} cookies. She sells {c} cookies, "
         f"then bakes {d} more trays of {e} cookies. How many cookies does she have now?")
    return q, a * b - c + d * e

PROBLEMS = [make_problem() for _ in range(20)]
ANSWER_FIRST = '{"answer": <integer>, "reasoning": <string>}'

def chat(user: str, schema: str | None) -> str:
    system = "You are a careful math assistant." + (f" Reply with JSON only, matching this schema: {schema}" if schema else "")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def next_token_probs(text: str) -> torch.Tensor:
    return model(**tok(text, return_tensors="pt")).logits[0, -1].softmax(-1)

print("answer | free-form 1st token | P(digit) at token 1 | P(digit) after {\"answer\": | masked: top digit, P(correct 1st digit), entropy")
for q, ans in PROBLEMS[:5]:
    p_free = next_token_probs(chat(q, None))                        # no schema: what does it want to say first?
    p_json = next_token_probs(chat(q, ANSWER_FIRST) + '{"answer": ')  # schema-forced prefix, answer comes next
    masked = p_json[DIGIT_IDS] / p_json[DIGIT_IDS].sum()             # the mask + renormalise step
    top = int(p_free.argmax())
    print(f"{ans:>6} | {tok.decode([top])!r:>8} ({p_free[top]:.2f})     | {p_free[DIGIT_IDS].sum():>19.4f} | "
          f"{p_json[DIGIT_IDS].sum():>25.4f} | {int(masked.argmax())}, {masked.max():.2f}, "
          f"{masked[int(str(ans)[0])]:.2f}, {entropy_bits(masked):.2f} bits")
```
```text
answer | free-form 1st token | P(digit) at token 1 | P(digit) after {"answer": | masked: top digit, P(correct 1st digit), entropy
   188 |     'To' (0.69)     |              0.0021 |                    0.9993 | 1, 0.31, 0.31, 2.90 bits
   155 |     'To' (0.53)     |              0.0015 |                    0.9997 | 1, 0.31, 0.31, 2.81 bits
   136 |     'To' (0.69)     |              0.0022 |                    0.9989 | 1, 0.30, 0.30, 2.89 bits
    85 |     'To' (0.59)     |              0.0016 |                    0.9998 | 1, 0.30, 0.06, 2.92 bits
   209 |     'To' (0.69)     |              0.0021 |                    0.9995 | 1, 0.33, 0.17, 2.84 bits
```

It isn't. After `{"answer": ` the valid mass is between 0.9989 and 0.9998. Once the model is inside an answer-first JSON object it has accepted that a number comes next, and the mask at that step throws away almost nothing.

The mass went missing earlier. With no schema, the first token is `To` (0.53 to 0.69), the start of a sentence like "To find the total...", and digits get between 0.0015 and 0.0022. So about 99.8% of what the model wanted to do at the start was something other than answering, mostly beginning to work the problem. An answer-first structure takes that option away before the mask ever touches an answer token. The cost was paid upstream, by forced syntax that put the model in a state where the answer has to come now.

Nor does the model look decisive. The masked first digit is `1` at 0.30 to 0.33, with entropy 2.81 to 2.92 bits against a maximum of $$\log_2 10 \approx 3.32$$ for ten digits. For 188, 155 and 136 the `1` happens to be right, but a model that always guesses `1` would be right for every answer from 100 to 199, so that says nothing about computation. For 85 the correct first digit gets 0.06.

Logging $$M_t$$ at answer tokens is still worth the one sum it costs, since it catches the toy-model failure. This cell shows its limit. A high $$M_t$$ says the model accepted the format and tells you nothing about whether it had done the work. That is the next rung.

## Rung 2, one field: field order is computation order

A transformer spends a fixed amount of computation per token: $$L$$ layers, one forward pass. Each token it writes buys another forward pass, and the intermediate results stay available to later tokens through the KV cache. Writing "9 * 18 = 162" before the answer is serial computation the model could not fit into one pass. [Feng et al. (2023)](https://arxiv.org/abs/2305.15408) prove that, under a standard complexity assumption (TC⁰ ≠ NC¹), bounded-depth transformers cannot directly produce correct answers to basic arithmetic problems unless their size grows super-polynomially with input length, while constant-size models solve them by generating chain-of-thought steps. [Merrill & Sabharwal (2024)](https://arxiv.org/abs/2310.07923) show that how much chain of thought adds depends on how many decoding steps it gets: a logarithmic number adds little, while a linear number lets a transformer recognise every regular language.

In probability terms, free-form generation writes a reasoning trace $$z$$ and then an answer $$y$$. The idealised picture is a marginal over traces,

$$
P(y \mid x) = \sum_z P(y \mid z, x)\, P(z \mid x),
$$

but decoding never computes that sum. It draws one trace (greedy decoding takes the most likely path), so free-form generation is a single-sample estimate of the marginal, and the computation inside $$z$$ does the heavy lifting in $$P(y \mid z, x)$$. An answer-first schema removes $$z$$ altogether. The answer token comes right after the forced prefix `{"answer": `, so the model has to compute $$P(y \mid x, \text{prefix})$$ in the forward passes over a few syntax tokens, however hard the problem is. Those forced tokens do add forward passes. They carry no choices, though, so they give the model nowhere to write an intermediate result.

So the key order in your schema is a statement about computation. Generation runs left to right: a `"reasoning"` field placed before `"answer"` is $$z$$, and placed after it, the reasoning can only rationalise an answer that already exists. Tam et al. saw exactly this on their Last Letter task, where 100% of GPT-3.5-turbo's JSON-mode responses put the `answer` key before the `reason` key.

![Answer-first vs reasoning-first JSON drawn in generation order: with the answer first, the KV cache holds only the prompt and four syntax tokens when the answer (12355, wrong) is emitted, and the trailing reasoning field cannot change it; with reasoning first, the cache holds the worked steps 9 × 18 = 162, 162 − 29 = 133, 5 × 11 = 55, 133 + 55 = 188 before the answer (188, right)]({{ '/assets/blog/strict-json-dumber/schema-order.png' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*How to read it: read each column top to bottom, in the order the model writes it. The blue bracket marks what the model can look back at when it writes the answer. On the left that is only the question and `{"answer":`, so it has to produce the number straight away, and the reasoning written afterwards cannot change it. On the right the model has already worked through each step of the sum, so the answer is just the last result it can see. Same fields; only the order differs.*

The next cell runs both schemas through real constrained decoding, with a small FSM written by hand. Inside the reasoning string it allows only tokens with no quote, backslash or control character, and it lets the model close the string. Then it forces the syntax `, "answer": ` and allows only digits, with `}` permitted after the first one. The two prompts differ only in key order. Decoding is greedy, and the answer-first run stops after the number because nothing generated later can change it.

```python
CLOSE, BRIDGE = tok.convert_tokens_to_ids("}"), ', "answer": '  # syntax forced between the two fields
pieces = tok.batch_decode([[i] for i in range(tok.vocab_size)])
STRING_OK = torch.zeros(model.config.vocab_size, dtype=torch.bool)  # tokens legal inside a JSON string
STRING_OK[: tok.vocab_size] = torch.tensor([bool(s) and '"' not in s and "\\" not in s and s.isprintable() for s in pieces])
# Tokens that close the string and start the bridge. BPE merges them: '"', '",', ' "', ' ",'
# (a leading space just ends up inside the string, which is still valid JSON).
CLOSERS = {i: c[1:] for i, s in enumerate(pieces)
           if (c := s.removeprefix(" ")).startswith('"') and BRIDGE.startswith(c[1:])}
REASON_MASK = STRING_OK.clone(); REASON_MASK[list(CLOSERS)] = True    # write text, or close the string
DIGIT_MASK = torch.zeros_like(STRING_OK); DIGIT_MASK[DIGIT_IDS] = True
NUMBER_END_MASK = DIGIT_MASK.clone(); NUMBER_END_MASK[CLOSE] = True   # after >= 1 digit, "}" may end it

@torch.no_grad()
def json_generate(question: str, answer_first: bool, max_reason: int = 200) -> tuple[int, str]:
    """Greedy FSM-constrained decoding. Answer-first stops after the number: later fields can't change it."""
    schema = ANSWER_FIRST if answer_first else '{"reasoning": <string>, "answer": <integer>}'
    prefix = '{"answer": ' if answer_first else '{"reasoning": "'
    out = model(**tok(chat(question, schema) + prefix, return_tensors="pt"), use_cache=True)
    phase, forced, emitted, digits = ("number" if answer_first else "string"), [], [], ""
    while True:
        if forced:  # syntax the FSM dictates: no choice at all
            nxt = forced.pop(0)
        else:
            mask = REASON_MASK if phase == "string" else (NUMBER_END_MASK if digits else DIGIT_MASK)
            nxt = int(out.logits[0, -1].masked_fill(~mask, float("-inf")).argmax())
            if phase == "string" and (nxt in CLOSERS or len(emitted) >= max_reason):
                nxt = nxt if nxt in CLOSERS else tok.convert_tokens_to_ids('"')  # close the string ...
                forced = tok(BRIDGE[len(CLOSERS[nxt]):], add_special_tokens=False).input_ids  # ... force the rest
                phase = "number"
            elif phase == "number":
                if nxt == CLOSE or len(digits) == 5:
                    break
                digits += str(DIGIT_IDS.index(nxt))
        emitted.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]]), past_key_values=out.past_key_values, use_cache=True)
    return int(digits), prefix + tok.decode(emitted) + "}"

results = {}
for answer_first in (True, False):
    results[answer_first] = [json_generate(q, answer_first) for q, _ in PROBLEMS]
    n_correct = sum(pred == ans for (pred, _), (_, ans) in zip(results[answer_first], PROBLEMS))
    n_valid = sum(isinstance(json.loads(text), dict) for _, text in results[answer_first])
    print(f"{'answer-first' if answer_first else 'reasoning-first':>15}: {n_correct:>2}/{len(PROBLEMS)} correct, "
          f"{n_valid}/{len(PROBLEMS)} valid JSON")
print(f"\nProblem 1 (true answer {PROBLEMS[0][1]}): {PROBLEMS[0][0]}")
print("answer-first   ->", results[True][0][1])
print(textwrap.fill("reasoning-first -> " + results[False][0][1], width=110))
```
```text
   answer-first:  0/20 correct, 20/20 valid JSON
reasoning-first: 20/20 correct, 20/20 valid JSON

Problem 1 (true answer 188): A baker makes 9 trays of 18 cookies. She sells 29 cookies, then bakes 5 more trays of 11 cookies. How many cookies does she have now?
answer-first   -> {"answer": 12355}
reasoning-first -> {"reasoning": "The baker initially made 9 trays of 18 cookies, which is a total of 9 * 18 =
162 cookies. After selling 29 cookies, she has 162 - 29 = 133 cookies left. Then she bakes 5 more trays of 11
cookies, which is a total of 5 * 11 = 55 cookies. Adding these to the remaining cookies, she has 133 + 55 =
188 cookies now. The answer is 188. ", "answer": 188}
```

Answer-first gets 0/20 and reasoning-first gets 20/20, and every one of the 40 outputs parses. Same two fields, same instructions; only the order changed. On problem 1, answer-first emitted `12355` (it hit the 5-digit cap) for a true answer of 188. Reasoning-first wrote out 9 × 18 = 162, 162 − 29 = 133, 5 × 11 = 55, 133 + 55 = 188, then put 188 in the answer field.

This is an illustration rather than a benchmark: 20 problems, one task template, one 0.5B model, greedy decoding. Larger models fit more computation into one forward pass, so expect the gap to shrink with scale and with easier problems. The direction is what the expressivity results predict.

A debugging note that previews the tokenization section. My first version of this decoder let only a bare `"` close the reasoning string. Qwen's tokenizer merges closing quotes with their neighbours (`",` and ` ",` are single tokens), and those were the tokens the model wanted. It couldn't close the string the way it normally would, so it kept writing, eventually into HTML tags, until it hit the 200-token cap. The JSON was valid and the reasoning field was garbage, all because the mask didn't match the tokenizer.

The schema rule is short: never put `answer` before `reasoning`.

## Rung 3, whole output: masking samples from the wrong distribution

Suppose the schema is perfect, reasoning first. A subtler problem remains, and it only appears across several steps.

We would like to sample from the model's distribution conditioned on the whole output being valid. With $$\mathcal{L}$$ the set of valid outputs,

$$
\begin{aligned}
P(y \mid y \in \mathcal{L}) &= \frac{P(y)}{Z}, \\
Z &= \sum_{y' \in \mathcal{L}} P(y').
\end{aligned}
$$

Masked decoding conditions one step at a time instead, dividing by the local valid mass at each step:

$$
\begin{aligned}
P_{\text{masked}}(y) &= \prod_t \frac{p(y_t \mid y_{<t})}{M_t(y_{<t})} \\
&= \frac{P(y)}{\prod_t M_t(y_{<t})}.
\end{aligned}
$$

The two agree only if $$\prod_t M_t(y_{<t}) = Z$$ for every valid $$y$$, meaning every path loses the same total mass to the mask. In general they don't. Think of a trap prefix: a token the model likes that is technically valid but whose continuations are mostly invalid. When the prefix is chosen, the FSM only checks that *some* valid continuation exists, so the prefix keeps its full probability. One step later the mask removes most of its continuations, divides by a small $$M_t$$, and rescues it. The decoder commits early and pays later, and the per-step rule never accounts for that.

Getting the global conditional right needs lookahead, weighting each prefix by the probability that it can still end in a valid output. [Park et al. (2024)](https://arxiv.org/abs/2405.21047) call the exact target grammar-aligned decoding and show that standard grammar-constrained decoding can distort the distribution; their algorithm, ASAp, uses earlier samples to over-approximate those expected futures and refines them as it keeps sampling. Rung 1 is the one-step special case. With a single step there is no future to look ahead to, and masking is exact.

![A two-step decoding tree whose branch thicknesses and leaf probabilities morph from masked decoding (trap sequence A x at 0.70, 3.70 times too likely) to the true conditional P(y | valid) (A x at 0.19, B branch growing to 0.24, 0.24, 0.32)]({{ '/assets/blog/strict-json-dumber/local-vs-global-tree.gif' | relative_url }}){: .img-fluid .rounded .z-depth-1 style="max-width: 100%; height: auto; display: block; margin: 1rem auto;" data-zoomable="true"}

*How to read it: the model writes two tokens. First it picks A (70%) or B (30%), then a second token. The red dashed branch, A then y, is not allowed by the format. Thicker lines mean more probability. The animation moves between two versions of the tree: what masked decoding actually produces, where A x comes out 70% of the time, and what we want (the model's own odds, keeping only the allowed outputs), where A x gets 19% and the three B sequences share the rest. Masking picked A before it could see that A almost always ends somewhere illegal, then pushed it into A's only legal ending.*

The next cell checks this by exact enumeration on a 2-step toy LM. Prefix A is the trap. It has probability 0.7, but only one of its continuations (x, at 0.1) is legal, while prefix B has 0.3 and all three of its continuations are legal.

```python
# A 2-step toy LM with explicit probabilities. "A" is a trap prefix: likely, but rarely completable.
P1 = {"A": 0.7, "B": 0.3}
P2 = {"A": {"x": 0.1, "y": 0.9}, "B": {"x": 0.3, "y": 0.3, "z": 0.4}}
VALID = {("A", "x"), ("B", "x"), ("B", "y"), ("B", "z")}  # the grammar: after A only x is legal

seqs = [(a, b) for a in P1 for b in P2[a]]
p_model = {s: P1[s[0]] * P2[s[0]][s[1]] for s in seqs}
Z = sum(p_model[s] for s in VALID)
p_true = {s: p_model[s] / Z if s in VALID else 0.0 for s in seqs}  # what we want: P(y | y is valid)

def p_masked(s: tuple[str, str]) -> float:
    """What FSM masking samples: prod_t p(y_t | y_<t) / M_t, with M_t = valid next-token mass."""
    a, b = s
    m1 = sum(P1[t] for t in P1 if any((t, u) in VALID for u in P2[t]))  # step 1: both A and B can still finish
    m2 = sum(v for u, v in P2[a].items() if (a, u) in VALID)            # step 2: mass left after the mask
    return (P1[a] / m1) * (P2[a][b] / m2) if s in VALID else 0.0

print("seq | P_model | P(y | valid) | P_masked | masked / true")
for s in sorted(VALID):
    print(f"{' '.join(s):>3} | {p_model[s]:.3f}   | {p_true[s]:.3f}        | {p_masked(s):.3f}    | {p_masked(s) / p_true[s]:.2f}x")
tv = 0.5 * sum(abs(p_masked(s) - p_true[s]) for s in VALID)
print(f"total variation distance between masked decoding and the true conditional: {tv:.3f}")

# independent check: run the masked sampler itself, token by token, 100k times
gen = np.random.default_rng(SEED)
def masked_step(dist: dict[str, float], legal) -> str:
    """Drop illegal tokens, renormalise, sample one token."""
    toks = [t for t in dist if legal(t)]
    w = np.array([dist[t] for t in toks])
    return toks[gen.choice(len(toks), p=w / w.sum())]
draws = []
for _ in range(100_000):
    a = masked_step(P1, lambda t: any((t, u) in VALID for u in P2[t]))
    draws.append(f"{a} {masked_step(P2[a], lambda u: (a, u) in VALID)}")
print(f"Monte Carlo, token-by-token masked sampling: P(A x) = {draws.count('A x') / len(draws):.3f}")
```
```text
seq | P_model | P(y | valid) | P_masked | masked / true
A x | 0.070   | 0.189        | 0.700    | 3.70x
B x | 0.090   | 0.243        | 0.090    | 0.37x
B y | 0.090   | 0.243        | 0.090    | 0.37x
B z | 0.120   | 0.324        | 0.120    | 0.37x
total variation distance between masked decoding and the true conditional: 0.511
Monte Carlo, token-by-token masked sampling: P(A x) = 0.702
```

Under the model, `A x` has probability 0.070, and conditioned on validity it should come out 0.189 of the time. Masked decoding produces it 0.700 of the time. That is 3.70× too often, and every B sequence gets 0.37× its share. The total variation distance from the true conditional is 0.511, so about half the probability mass sits in the wrong place; running the masked sampler itself, token by token, 100,000 times gives 0.702, which rules out an algebra slip in the formula.

In real JSON, a trap prefix is a token that opens something the schema allows but the model is unlikely to finish within the constraints: a string in a format it rarely writes, say, or a number it wouldn't normally produce. Each such step multiplies in another $$1/M_t$$ factor that the global conditional does not apply, so the distortion can compound with output length. I don't measure that here.

## A side note on tokenization

BPE merges frequent character sequences into single tokens, and the model only ever saw the merged forms in training. Qwen's tokenizer turns ` "` (space plus quote) and `"}` into single tokens. If a constraint pushes text through different boundaries, say a space token followed by a quote token, the model is conditioned on a sequence it has essentially never seen. Engines that run the FSM at token level and allow any token whose bytes are valid (the index in Willard & Louf, XGrammar's adaptive token mask cache) mostly avoid this. A prompt template whose forced prefix ends in whitespace still triggers it, and the closing-quote bug above is the same failure from the other side. File this under "check your engine" rather than a core mechanism.

The cell scores three tokenizations of the same continuation after `{"answer":`.

```python
ctx = chat("Is 391 a prime number?", '{"answer": "yes" | "no"}') + '{"answer":'

@torch.no_grad()
def continuation_logprob(context: str, cont_ids: list[int]) -> float:
    """Sum of log p(token) over a forced continuation, one forward pass."""
    ids = tok(context, return_tensors="pt").input_ids
    logp = model(torch.cat([ids, torch.tensor([cont_ids])], dim=1)).logits[0, ids.shape[1] - 1 : -1].log_softmax(-1)
    return float(logp.gather(1, torch.tensor(cont_ids)[:, None]).sum())

for word in ("no", "yes"):
    text = f' "{word}"}}'
    paths = {
        "canonical BPE": tok(text, add_special_tokens=False).input_ids,
        "space forced alone": tok(" ", add_special_tokens=False).input_ids + tok(text[1:], add_special_tokens=False).input_ids,
        "one char per token": [tok(c, add_special_tokens=False).input_ids[0] for c in text],
    }
    base = continuation_logprob(ctx, paths["canonical BPE"])
    for name, ids in paths.items():
        lp = continuation_logprob(ctx, ids)
        ratio = "(reference)" if name == "canonical BPE" else f"{np.exp(base - lp):.0e}x less likely"
        print(f"{word:>3} | {name:>18} | {str([tok.decode([i]) for i in ids]):<38} | log p = {lp:7.2f} | {ratio}")
```
```text
 no |      canonical BPE | [' "', 'no', '"}']                     | log p =   -1.55 | (reference)
 no | space forced alone | [' ', '"', 'no', '"}']                 | log p =  -45.08 | 8e+18x less likely
 no | one char per token | [' ', '"', 'n', 'o', '"', '}']         | log p =  -86.29 | 6e+36x less likely
yes |      canonical BPE | [' "', 'yes', '"}']                    | log p =   -0.27 | (reference)
yes | space forced alone | [' ', '"', 'yes', '"}']                | log p =  -44.17 | 1e+19x less likely
yes | one char per token | [' ', '"', 'y', 'e', 's', '"', '}']    | log p = -117.29 | 7e+50x less likely
```

For `"no"`, the canonical path has log-probability −1.55. Forcing the space onto its own token drops it to −45.08, about 8 × 10¹⁸ times less likely, and splitting every character drops it to −86.29. Same text, different boundaries, and it goes from plausible to effectively impossible. Every token generated after a forced non-canonical path inherits that off-distribution context.

One more thing in that output: the canonical `"yes"` scores −0.27, so the model would rather call 391 prime. It isn't (391 = 17 × 23). That's rung 2 again, a one-token answer with no room to factor.

## What the evidence actually says

[Tam et al. (2024)](https://arxiv.org/abs/2408.02442) compared free text against three kinds of format restriction: JSON mode (constrained decoding), format-restricting instructions in the prompt (with or without a schema), and NL-to-Format. Reasoning tasks got worse under restriction. The GSM8K numbers at the top of this post are the prompt-only case; with JSON mode itself (their Table 11), GSM8K falls to 47.78 for Gemini 1.5 Flash and 29.87 for GPT-3.5-turbo. Classification often got better. On DDXPlus, Gemini 1.5 Flash gained substantially with JSON mode, which the authors put down to the restricted answer space. That is rung 1 working for you: when the valid set *is* the label set, conditioning on it removes junk and leaves the reasoning alone. They also saw JSON mode put the answer before the reasoning, as quoted above. And their NL-to-Format variant, which answers in free text and then converts to the schema, matched free text closely for most models.

The rebuttal from .txt, [Say What You Mean](https://blog.dottxt.ai/say-what-you-mean.html) by Will Kurt, re-ran three of the tasks with Llama-3-8B-Instruct and found structured generation at or above free text: GSM8K 0.77 unstructured vs 0.78 structured, Last Letter 0.73 vs 0.77, Shuffle Object 0.41 vs 0.44. Kurt argues that the paper's prompts differed between the free-text and structured runs, that the structured prompts left out information the task needed, and that "JSON mode", which has no guarantees about the returned value, is a different thing from structured generation.

The two disagree less than it seems. The schema .txt shows for Last Letter puts a `reasoning` field capped at 250 characters before the `answer`, which is the rung 2 fix. On the other side, Kurt traces the paper's bad numbers to prompts that withheld what the task needed, and the paper itself caught JSON mode writing the answer first. Read together, they point at constraints that take away the scratchpad, rather than at constraints in general. Rung 3's distortion survives a good schema, and neither study measures it directly.

## Getting both: reliable parsing and full reasoning

Three patterns, cheapest first:

| Pattern | What it does | Fixes | Cost |
|---|---|---|---|
| Reasoning-first schema | Put a `reasoning` string field before every field that needs thought; cap its length | Rung 2 | A few hundred extra tokens per call |
| Think free, format later | Call 1: free-text answer. Call 2: a model (a small one is usually enough) extracts the fields under constrained decoding. This is the two-step NL-to-Format idea from Tam et al. | Rungs 1 to 3 for the reasoning, which is never constrained | One extra, cheap inference call |
| Thinking tokens | Reasoning models write an unconstrained thinking block; the constraint applies only to the final JSON | Rungs 1 to 3 for the reasoning | Higher latency; needs a model trained for it |

Rung 1 suggests two more habits. Give the model a valid way to say what it means: if "it depends" is a real answer, put `"uncertain"` in the enum. Geometrically, that moves the hedge corner onto the valid face, and the mask no longer has to erase it. And log $$M_t$$ and the masked entropy at answer tokens. A low $$M_t$$ means the model wanted something your schema forbids. As cell 4 showed, a high $$M_t$$ doesn't prove the model did the work, so keep an eval next to the dashboard.

## Where the field is now

- [CRANE (Banerjee et al., ICML 2025)](https://arxiv.org/abs/2502.09061) makes the rung 2 point directly: grammars that allow only the final answer reduce reasoning ability. Their fix augments the grammar with delimiters so decoding alternates between unconstrained reasoning and constrained output, and they report up to 10 percentage points over baselines on GSM-Symbolic and FOLIO.
- [Grammar-Aligned Decoding (Park et al., NeurIPS 2024)](https://arxiv.org/abs/2405.21047) goes after rung 3 with ASAp, which provably converges to the model's distribution conditioned on the grammar as it draws more samples.
- [XGrammar (Dong et al., MLSys 2025)](https://arxiv.org/abs/2411.15100) brings constrained decoding overhead close to zero in end-to-end serving, so speed is no longer an excuse for skipping reasoning tokens.
- [JSONSchemaBench (Geng et al., 2025)](https://arxiv.org/abs/2501.10868) evaluates six constrained decoding frameworks on 10K real-world schemas for efficiency, coverage and output quality. Worth a look before you pick an engine.

## Conclusion

At a single step, a logit mask is exact Bayesian conditioning on validity. What goes wrong comes from what that conditioning can't see: the mass it divides out (rung 1), the reasoning the schema never let the model write (rung 2), and the futures it doesn't look ahead to (rung 3). Rung 2 is the biggest problem and the cheapest to fix. Put the reasoning field first and give the model a legal way to hedge, and you keep the JSON guarantee without paying for it in accuracy. To check this on your own setup, run cell 5 on one of your schemas with the two keys swapped.

## Further reading

- [Tam et al. (2024), Let Me Speak Freely?](https://arxiv.org/abs/2408.02442): the measurement that started the debate; the tables carry more than the abstract.
- [Will Kurt, .txt, Say What You Mean](https://blog.dottxt.ai/say-what-you-mean.html): the rebuttal, with reproducible notebooks and a reasoning-first schema.
- [Park et al. (2024), Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047): the formal version of rung 3 and an algorithm that fixes it.
- [Banerjee et al. (2025), CRANE](https://arxiv.org/abs/2502.09061): reasoning-augmented constrained decoding, the strongest current answer to rung 2 inside a single call.
- [Willard & Louf (2023), Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702): how a schema becomes an FSM over tokens (the Outlines paper).
- [Feng et al. (2023), Towards Revealing the Mystery behind Chain of Thought](https://arxiv.org/abs/2305.15408): why fixed-depth transformers need the scratchpad.
