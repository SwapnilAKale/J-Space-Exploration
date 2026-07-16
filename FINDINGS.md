# Findings — practical lessons from running Stage 1

This file collects the non-obvious things we learned by actually *running* the read-and-steer
tool, so a new person can understand the behaviour without rediscovering it from scratch. Each
item is a standalone note — no prior context needed. (New findings get appended here as they
come up.)

Background on the project and the underlying research is in `CLAUDE.md`. How to run it is in
`README.md`. This file is the "why did it do *that*?" companion.

---

## 1. The coefficient is a magnitude, and it's *relative*

`--coefficient` is a plain number multiplied onto a fixed-direction steering vector before it's
added into the model's activations. `0` adds nothing (identical to baseline — a built-in sanity
check). Bigger = a stronger push.

But the number has **no fixed meaning on its own**. The model's own activations grow in size
through its depth, so the same coefficient is a big shove at an early layer and a gentle one at a
late layer — and the scale differs between models too. There is no universal "right" value. That's
why sweeping several coefficients in one run is the default, not an afterthought.

## 2. Every steering vector has a floor, a sweet spot, and a ceiling

Sweep a coefficient upward and you see three regimes:
- **Below the floor:** no visible effect (the push is too weak).
- **Sweet spot:** the target concept appears and the text stays coherent.
- **Above the ceiling:** the output "breaks the manifold" — degenerates into repetition or garbage,
  even though the target token's logit keeps climbing.

Key lesson: **a higher target-token logit is NOT a better result.** Past the ceiling the readout
number keeps rising while the actual text falls apart. Success = *concept present AND still
coherent*, judged on the generated text, not the logit alone.

## 3. The readout is a mid-layer snapshot, not the final answer

The "J-space readout" table is computed at the interception layer — partway through the model.
The model's actual output is decided at the *final* layer, after more layers of processing. So the
readout and the generated text can legitimately **disagree**: the readout might show `always`
winning while the model actually writes `football`, because the later layers resolve it differently.

As you raise the coefficient, the target concept climbs the readout; at low strength it may already
win the output while still only rank 2–3 in the mid-layer readout. Watch both, but trust the
generated text for "what the model actually said."

## 4. Readout tokens are often word-fragments — read the text for whole words

The readout shows single vocabulary tokens, and many tokens are sub-word pieces (`'sw'`, `'sc'`,
`'scar'`). A top token of `'sw'` means "about to start a word beginning with sw" (swimming? sword?),
not a finished word. Whole words only exist once several tokens are strung together — so for real
words, read the **generated text** block, not the readout table.

## 5. Two steering methods, built from different sources

- **`token_diff`** — subtracts two columns of the unembedding matrix (`W_U`), e.g. ` rugby` minus
  ` football`. A sharp, literal "point at this exact word" direction. Surgical and clean, but each
  side must be **exactly one token** (or you get a clear error), and it's an "artificial" output-space
  direction jammed into a middle layer, so it can break coherence at high strength.
- **`actadd`** — runs the model on two full prompts (e.g. "I love rugby" / "I love football") and
  subtracts their activations at the layer. A richer, contextual concept direction that lives
  natively in the residual stream, so it usually steers more smoothly — but it's messier, because a
  whole prompt carries incidental differences (keep the two prompts identical except the one concept).

Use `token_diff` for a crisp single-word push; use `actadd` for a broader concept, a multi-token
idea, or when you want gentler steering.

## 6. Tokens are case- and space-sensitive

` rugby`, `rugby`, ` Rugby`, and `Rugby` can be up to four *different* tokens. The leading space is
part of the token (mid-sentence words carry it), and case is preserved because it's real information
(proper nouns, sentence starts, acronyms). For `token_diff`, pick the form that matches how the word
appears where you're steering — usually lowercase-with-leading-space (` rugby`) for mid-sentence.
Related forms sit near each other in the model's space, so steering toward ` rugby` also pulls
` Rugby` up the readout.

## 7. GPT-2 is a base "completion" model, not a chatbot

GPT-2 only continues text; it does not obey instructions. Prompt it with the *start of a sentence*
that lands on your answer ("My favourite sport is") — not a command ("Choose a sport"), which it
just continues as generic filler. Qwen2.5-Instruct **does** follow instructions and writes far more
coherent, longer text (it's instruction-tuned and larger).

## 8. Greedy decoding is deterministic — reruns are identical

Generation uses greedy decoding (always take the highest-scoring next token), so the same command
gives the exact same output every run. This is deliberate: when the output changes, you *know* the
steering caused it, not random sampling. The flip side: greedy on a small model loops
("It's a great sport. It's a great sport...") — that repetition is the model + greedy decoding
("neural text degeneration"), **not** a steering failure. The unsteered baseline loops too.

## 9. Steering redirects *content*, it does not upgrade *capability*

Steering changes *what* the model talks about; it cannot make a weak model write better. A 124M
model steered toward rugby still loops after two sentences, because its coherence ceiling is fixed
by its size and training. Don't expect steering to turn a small model into an essay-writer — it
moves the concept within the model's existing limits.

## 10. Which layer you steer matters: sensory / workspace / motor

The model's depth splits into three functional bands (this is a core finding of the research):
- **Early ("sensory"):** the concept isn't formed yet — steering barely works, or needs a large push.
- **Middle ("workspace"):** where abstract concepts live — steering works cleanly here. The default
  auto-pick (~0.6 × n_layers) lands in this band on purpose.
- **Late ("motor"):** the layers are committing to the output — steering is blunt or breaks, and the
  readout starts to look like the final answer.

Different layers also need different coefficients (the most "sensitive" layer is where the concept
is actively being decided, and needs the smallest push). If a sweep does nothing at an early layer
and breaks at a late one, that's expected — the middle is where it behaves.

## 11. The logit-lens readout breaks at middle layers — badly on Qwen

Our readout uses the *logit lens*: decode a middle-layer activation as if it were the finished
output. This assumes the middle layer already "speaks the output's language," which is wrong at
middle layers — and it fails hard on some models. On Qwen the middle-layer readout is mostly
meaningless underscore/filler tokens. This is a known limitation, and it's exactly the failure the
research's **Jacobian lens** was built to fix. Until that lens is wired in, **judge Qwen's
middle-layer runs by the generated text, not the readout.**

The warning `you are not using LayerNorm, so the writing weights can't be centered! Skipping` on
Qwen is **harmless and expected** — Qwen uses RMSNorm by design; you cannot and should not switch it
to LayerNorm, and it does not degrade the output (the coherent baseline proves that).

## 12. Coefficient scales do NOT transfer between models

A value that works on GPT-2 (say 3–8) will not do the same thing on Qwen. Qwen's usable window can
be very narrow — e.g. coefficient 1 does nothing and coefficient 2 already breaks, with the sweet
spot only in between. Re-map the scale per model with a fine sweep (`--coefficient 1 1.25 1.5 1.75`),
rather than reusing another model's numbers.

## 13. Why Qwen sometimes collapses into a Chinese "fill-in-the-blank"

Over-steer Qwen and it may output something like `My favourite sport is ______（游泳）swimming` — a
Chinese fill-in-the-blank worksheet — instead of gibberish. This surprises people, and the
explanation is worth knowing:

- Qwen's training data has lots of Chinese educational material with long runs of underscores
  (worksheets, forms). It even has *dedicated single tokens* for underscore-runs.
- When over-steering knocks the model off its normal path and it emits an underscore token, its own
  context now reads "sport is ______" — which it recognises as the opening of a fill-in-the-blank
  exercise, a format it knows well. As a next-token predictor, it *fluently completes that format*.
- Crucially, this coherent (if weird) fallback is a sign of the model's **strength**, not a break. A
  weak model (GPT-2) degenerates into repetition/noise; a strong model "rationalises" the stray token
  into the nearest coherent structure it recognises. It's "good at what it saw a lot of," taken to an
  odd but logical conclusion.

## 14. This project reproduces the research's own examples

The `rugby` sport-swap and the `spider`/`insect` ("legs on the animal that spins webs") examples are
the paper's *own* headline demonstrations, done here by hand with simpler tools. The larger claim —
that these "reportable" concepts form a small privileged workspace (~25 concepts at a time, under
~10% of the model's internal activity, in the middle-layer band) — is the science we are *not*
reproducing; we're reproducing the mechanism so it can be seen working. See `CLAUDE.md` for the
full framing.
