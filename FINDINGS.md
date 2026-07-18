# Findings — practical lessons from running Stage 1

This file collects the non-obvious things we learned by actually *running* the read-and-steer
tool, so a new person can understand the behaviour without rediscovering it from scratch. Each
item is a standalone note that includes the run that surfaced it, so no prior context is needed.
(New findings get appended here as they come up.)

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

*How we saw it:* sweeping the gpt2 ` rugby`−` football` vector from 0 to 50 — nothing below ~2,
clean rugby continuations from ~2 to ~15, then repetition ("rugby rugby rugby…") by ~20, all while
the rugby logit kept climbing.

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

**The real difference is *where* the vector comes from — not the input length.**

- **`token_diff` reads the OUTPUT matrix.** It never runs the model: it looks up your two tokens'
  columns in the unembedding matrix `W_U` (` rugby` minus ` football`) — the *output* dictionary,
  where internal vectors become word-scores. The surrounding words are irrelevant (never fed in). A
  sharp, literal "point at this exact word" direction. Fast and surgical, but each side must be
  **exactly one token** (or you get a clear error), and it's an *output-space* direction jammed into
  a middle layer, so it's a bit artificial and breaks coherence at high strength.
- **`actadd` reads the INTERNAL residual.** It *runs* the model on two full prompts ("I love rugby" /
  "I love basketball") and subtracts their residual-stream activations at the interception layer —
  the model's *internal* representation of the concept, in context, after real computation. A richer,
  contextual direction that lives natively in the residual stream, so it steers more smoothly and can
  carry low-prior or multi-token concepts — but it's messier (a whole prompt carries incidental
  differences, so keep the two prompts identical except the one concept) and its vector is
  larger-norm, so it needs much smaller coefficients (see #16).

So even the *same* sentence with one word swapped gives the two methods *different* vectors, from two
different parts of the model: `token_diff` from the output dictionary, `actadd` from the internal
workspace. You can't feed a bare token to `actadd` in the `token_diff` sense — `actadd`'s whole
mechanism is running a forward pass and reading internals.

**In a comparison:**

| | `token_diff` | `actadd` |
|---|---|---|
| reads from | output matrix `W_U` | internal residual stream |
| runs the model? | no (pure lookup) | yes (forward pass) |
| uses context? | no | yes |
| input | one token per side | a full prompt per side |
| character | crisp, sharp, artificial | rich, contextual, native |
| coefficient scale | smaller-norm → larger numbers | larger-norm → smaller numbers |
| best for | a clean single-word push | a broad concept, or a low-prior target `token_diff` can't land (see #17) |

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

*How we saw it:* fixing one coefficient and sweeping `--layer 1 3 6 9 11` on gpt2 — layer 1 barely
moved the output, layer 9 flipped with a push of about 1, and layer 11 collapsed into repetition.

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

*How we saw it:* the gpt2 rugby window sat around 2–15; the identical `token_diff` on Qwen already
broke by coefficient 2, with its coherent window down at roughly 1.2–1.6.

## 13. Why Qwen sometimes collapses into a Chinese "fill-in-the-blank"

Over-steer Qwen and it may output something like `My favourite sport is ______（游泳）swimming` — a
Chinese fill-in-the-blank worksheet — instead of gibberish. (We hit it steering Qwen toward
` rugby` with `token_diff` at coefficient 2–3 on layer 17.) This surprises people, and the
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

## 15. `token_diff` evicts the named token and lets the strongest prior fill in

*How we saw it:* steering Qwen toward ` rugby` with `neg=' basketball'` (`token_diff`) at layers 17
and 20, coefficients 1.2–1.6 — a clean, coherent result at last… except the model said **football**,
not rugby.

`token_diff` only contrasts the *two* tokens you name. Here that means "suppress basketball, mildly
promote rugby" — it does **nothing** about football. So at a gentle (coherent) coefficient, clearing
basketball out of the way just lets the **highest-prior remaining sport fill the vacuum**, and in
text that's football, not rugby. The small rugby nudge isn't enough to lift a low-frequency word
above football's much stronger baseline. (You cleared the favourite's chair; the next-most-popular
sport sat down in it.)

This bites hardest for a **low-prior target**: rugby is a niche "favourite sport" in the training
data (in both the English and Chinese portions of a model like Qwen), so it sits low to begin with
and resists being pushed to the top. And there's no winning by pushing harder — the coefficient that
would give rugby a real chance (~2–3 on Qwen) is already past the coherence ceiling (see #13). The
coherent window is too weak for rugby; the rugby-strong window is incoherent.

The research predicts exactly this: swap success tracks **"workspace loading"** — how strongly the
target concept is *already present* before you intervene. High-loading concepts swap cleanly;
low-loading ones (like rugby) resist. The paper's own sport-swap worked because it used the surgical
**J-lens coordinate swap** (subtract the source's projection, add the target's), which is far more
precise than `token_diff`'s blunt `W_U` column difference. Takeaway: to install a *specific,
low-prior* concept, `token_diff` at a coherent strength usually isn't enough — reach for `actadd`
(which carries real context for the target), or eventually the proper J-lens swap.

## 16. Coefficient scale doesn't transfer between *methods* either — not just between models

*How we saw it:* running `actadd` ("I love rugby" vs "I love basketball") on Qwen at the same
1.2–1.6 that gave a clean result with `token_diff` — `actadd` broke coherence immediately (a garbled
Chinese multiple-choice quiz at 1.2, pure gibberish by 3–5).

The two methods build vectors of very different *size*. `token_diff` subtracts two unembedding
columns (small norm); `actadd` subtracts two residual-stream activations, which on Qwen are large.
So the *same* coefficient is a much bigger push for `actadd` than for `token_diff`. A scale tuned for
one method is wrong for the other — `actadd` on Qwen needs far smaller numbers (sweep ~0.2–1.0, not
1.2+). Combined with finding #12 (scale differs by model), the rule is: **re-map the coefficient
whenever you change the model OR the steering method.**

## 17. `actadd` can inject a low-prior target that `token_diff` can't — but the coherent window is narrower

*How we saw it:* `token_diff` toward ` rugby` on Qwen defaulted to *football* (see #15), but `actadd`
toward "I love rugby" actually produced rugby — at coefficient 1.2 every option in the (broken) quiz
was rugby / rugby league.

`actadd` carries real *context* for the concept, not just the bare output-token direction, so it
raises the target's "loading" enough to win where `token_diff`'s weak single-token nudge could not.
The cost: its larger-norm vector (see #16) breaks coherence sooner, so the window where the target
appears AND the text stays coherent is narrow and sits at a low coefficient. To land a low-prior
concept cleanly, use `actadd` and sweep *small* to find that window.

## 18. Steer *away from wrong*, not *toward unknown-right* — and keep concept prompts general

*Where this came from:* a design realization (not a single run), while asking how steering would
work on a real question whose answer we don't know in advance. This is really a Stage 2 principle,
noted here so it isn't lost.

In real use you usually don't know the correct answer — that's the whole reason you're asking the
model. So you can rarely steer *toward* a known-correct target. But you can almost always recognize a
*wrong* drift (a philosophy answer wandering into sports) and push *away* from it, because you know
the wrong direction even when you don't know the right one. The realistic mode is **"correct the
wrong direction," not "inject the right answer"** — and that's exactly what Stage 2's referee is
meant to do.

Practical consequence for `actadd` prompts: define a **general, portable concept** ("I love rugby",
or a broad "sports" contrast), not a target-specific sentence ("My favourite sport is rugby"). A
general concept vector can be applied to *any* prompt, including ones whose answer you don't know; a
target-specific one only fits that one sentence and leans toward hardcoding the output.

## 19. The bigger picture: (how you build the direction) × (what you do with it)

Every manipulation method is a combination of two independent choices. We currently use only two
cells of this grid; the rest are the natural upgrades.

**How you build the direction (where the vector comes from):**
- `token_diff` — from `W_U` columns (the output matrix; never runs the model).
- `actadd` — from one prompt-pair's residual activations (internal, contextual).
- mean-difference / CAA — `actadd` averaged over *many* examples, which cancels the incidental noise a
  single prompt pair carries (this is how the paper built its concept vectors).
- SAE features — cleaner, more interpretable directions from a sparse autoencoder.
- probe directions — the weight vector of a trained linear "is this concept present?" classifier.
- J-lens vectors — the paper's directions, read through the Jacobian lens (the eventual upgrade over
  our crude logit lens).

**What you do with the direction:**
- **add** — nudge *toward* it by a magnitude. Magnitude-fragile: too weak → the nearest strong
  neighbor wins (see #15), too strong → breaks (see #2). *(what we do now)*
- **subtract** (negative add) — nudge *away* from it.
- **ablate** — *remove* the concept entirely by projecting it out of the activation. There is **no
  target**; the model falls back to its next-natural output. The honest tool for "steer away from
  wrong" when you don't know the right answer (see #18).
- **swap** — *exchange* two concepts' presence surgically: hand the target the source's loading and
  leave everything orthogonal untouched. Installs a specific target cleanly — no magnitude gamble, no
  vacuum for a runner-up to fill. This is why the paper's rugby swap worked where our `add` defaulted
  to football (see #15).

We currently occupy two cells: `token_diff`+add and `actadd`+add. The natural Stage 2 additions are
**ablate** (remove an off-topic drift) and **swap** (install a known target), and eventually building
directions from the real **J-lens** instead of the logit lens.

## 20. Stage 2 intervention rules (from reasoning about ablation)

Design rules for the referee loop, worth writing down before we build it:

- **Ablate when you know it's *wrong*; swap when you know what's *right*.** In real use you usually
  don't know the correct answer, but you can spot an off-topic drift — so ablate it and let the model
  recover (no target). When the correct concept *is* identifiable, swap it in surgically.
- **Intervene early in the workspace band; never chase drift into the motor zone.** The last few
  layers only *format* the output — they can't do the reasoning that would rescue a confused state.
  Fix a drift at the *start* of the workspace (Qwen ~layer 17), where many layers remain to fold in
  the correction. If it's still there at the motor layers, the output is already lost.
- **Pair "ablate" with a positive on-topic nudge — don't leave a vacuum.** Ablation only removes; it
  doesn't say where to go, so a removed concept can be replaced by *another* off-topic one
  (whack-a-mole). A gentle push toward the correct context gives the model something to land on. (In
  practice the prompt's own context is the strongest attractor, so a removed intrusion is usually
  replaced by on-topic content — but that's a Stage 2 hypothesis to measure, not assume.)
- **A single ablation isn't enough — the concept re-enters downstream.** Later layers re-derive a
  concept you removed, so suppress it across the *band* of workspace layers (and at each generated
  token), not at one point.

## 21. Chat models run inside an invisible "chat template" — and feeding a bare string is itself a probe

Every instruction/chat-tuned model was fine-tuned to run inside a structured turn format built from
special tokens — a **chat template**. For Qwen (and OpenAI's models) that format is **ChatML**: each
turn is wrapped as `<|im_start|>{role}\n … <|im_end|>`, and a prompt ends with `<|im_start|>assistant\n`
to cue the model that it is now the assistant's turn to answer. `<|im_start|>` and `<|im_end|>` are
each a *single special token* in the vocabulary, not the literal punctuation characters.

**This format is NOT universal.** Each model family has its own tags — Llama 3 uses
`<|start_header_id|>user<|end_header_id|> … <|eot_id|>`; Llama 2 / Mistral use `[INST] … [/INST]`;
Gemma uses `<start_of_turn>user … <end_of_turn>`. What *is* universal is the principle: a chat model
must be fed *its own* template (a Llama template handed to Qwen just confuses it). HuggingFace's
`tokenizer.apply_chat_template()` applies the correct one per model — and is also the safe way to feed
it, because hand-typing the tags into a raw string can split them into ordinary characters and lose
the special-token signal entirely.

**The key mental model:** the chat template is the *envelope every chat interface silently wraps
around the user's words before they reach the model.* When someone types "recommend me a sport" into
Ollama or a chat box, the model never sees that bare string — it sees the ChatML-wrapped version. The
user never types the tags; the interface adds them, invisibly, every time. So the *faithful* way to
imitate a real user of an instruct model is *with* the template; feeding a bare string is the *less*
faithful thing, because no real user ever reaches the model without the envelope.

**Why this matters for the worksheet attractor (see #13).** `Qwen2.5-1.5B-Instruct` is a chat model.
Feed it a bare stem like `"My favourite sport is"` and you have dropped it into raw base-completion
mode it was never tuned for — off-distribution — with no assistant-role signal telling it to answer.
A lone instruction-shaped fragment is *exactly* the shape of raw worksheet text in pretraining, so
the model slides into that format. Contrast `gpt2`, a **base** model, where a sentence stem IS the
correct, faithful prompt. We had been prompting Qwen the same way we prompt gpt2 — like a base model —
and that model/format mismatch is a major driver of the worksheet collapse, *separate from* the
steering itself. Applies to Claude and every other chat model too: they all sit inside some turn
envelope; only the specific tokens differ.

**Deliberate project decision.** Stage 1 keeps using **bare-string prompts on Qwen on purpose**:
running an instruct model *without* its native scaffold is a controlled perturbation — a way to probe
what the chat-tuning does and to watch off-distribution failure modes (the worksheet) one variable at
a time. **Stage 2 will switch to the proper chat template** (via `apply_chat_template`), where the
referee needs the working model behaving cleanly and predictably rather than derailing. Prompt format
is a knob like coefficient and layer; we hold it at "no template" while exploring and turn it on when
we need reliability. (When a `--chat` flag is built for this, document it in `README.md`.)

*Where this came from:* unpacking what `<|im_start|>` meant in a suggested Qwen "chat-mode" prompt —
realising the tags are the invisible envelope real interfaces add, that base-completion on an
instruct model is both the root of the worksheet attractor *and* a useful probe, and that the format
is one specific dialect (ChatML) among several, not a universal standard.

## 22. The worksheet needs BOTH base-completion mode AND a worksheet-shaped stem — instruction phrasing escapes it

Findings #13 and #21 pinned Qwen's Chinese fill-in-the-blank collapse on two things: an instruct
model fed a bare string (off-distribution *base-completion* mode) plus lots of worksheet material in
pretraining. This run isolates the trigger more sharply.

On the **same** bare-string, no-template Qwen, changing *only* the prompt — from a fill-in-the-blank
*stem* (`"My favourite sport is"`) to a plain instruction (`"Name one sport you love, in a short
sentence."`) — made the worksheet **vanish**: clean rugby prose at the sweet spot. So the worksheet is
**not** caused by base-completion mode alone. It needs base-completion mode **AND** a prompt shaped
like a worksheet item. Break *either* lever and it's gone:
- Remove base-completion mode → wrap the same message in ChatML (#21).
- Remove the worksheet shape → phrase it as an instruction, not a `"___ is ___"` stem.

These two escapes are independent, which is useful for Stage-1 probing: you can stay in bare-string
mode (to keep studying the model off its native scaffold) and still get clean output, simply by not
handing it a fill-in-the-blank stem.

*How we saw it:* `actadd` rugby/basketball on Qwen with `--prompt "Name one sport you love, in a short
sentence."`, run both bare and ChatML-wrapped — both produced clean prose rugby at coeff 0.6, unlike
the earlier `"My favourite sport is"` stem which forced the worksheet at every coefficient. Side
evidence for the "still base-completion" half: the bare-prompt baseline drifted into a hallucinated
`Human: Can you provide me with more information...` turn — an instruct model with no template invents
the next conversational turn.

## 23. `actadd` over-steer drifts to the target's context-neighbors, not to gibberish

Finding #2 gave the floor / sweet-spot / ceiling shape; #17 noted `actadd` carries the target's
*context* and breaks coherence sooner than `token_diff`. This refines what "breaks" actually looks
like for `actadd` on a capable model: past the ceiling the output does **not** immediately become
noise — it **drifts to concepts adjacent to the target's context cloud.**

The vector `"I love rugby" − "I love basketball"` carries more than the bare token *rugby*: it carries
rugby's contextual halo — British/Irish, traditional, rugged, physical, outdoor. At the sweet spot
that halo is exactly what lands rugby (#17). Over-push it and the *halo itself* over-expresses and
takes over, so the output slides to *other* British-Isles rugged outdoor pursuits — rowing, yachting,
"the British and Irish Sea", whaleboat racing — rather than to word-salad.

*How we saw it:* Qwen, `actadd`, layer 17, coeff sweep 0.6 / 0.9 / 1.2:
- **0.6 (sweet spot):** clean rugby prose.
- **0.9 (fraying):** `"I love rugby sealaft, ..."` — rugby still named, but the nonsense token
  `sealaft` leaks in. The *same* `sealaft` appeared under **both** the bare and the ChatML prompt at
  0.9 — the coefficient scale is a property of the steering vector, not the prompt wrapper.
- **1.2 (past ceiling):** rugby lost; drift to `yachting` / `whaleboat racing` / "British and Irish
  Sea", plus `@RigbyRig` — the model grasping at a rugby-adjacent string (Rugby → Rigby).

Takeaway: for `actadd`, "past the ceiling" reads as **topical drift toward the target's neighbors** — a
softer, more legible failure than `token_diff`'s repetition collapse on gpt2 (#2). A capable model
degrades by *wandering semantically*, not by emitting noise (cf. #13, the worksheet as coherent-but-
wrong). Practical tell: neighbor-drift means you're one notch too high — back off toward the sweet
spot. (Aside: the sweet-spot answer's *specificity* also shifts with register — the bare instruction
prompt says "Rugby...", the ChatML assistant prompt says "rugby union" — same clean win, the assistant
register just gives a more specific answer.)
