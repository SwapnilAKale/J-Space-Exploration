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

**Amendment (see #36):** this blindness turned out to be **content-dependent, not layer-intrinsic.**
Given a *semantically clean* steering vector, the same Qwen middle layers read perfectly legibly. The
filler-token readouts above are what a noisy or weakly-loaded residual looks like — not a hard ceiling
of the logit lens on this model.

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

## 18. Build steering vectors from *general, portable* concept prompts, not target-specific ones

For `actadd`, define the direction from a **general, portable concept** ("I love rugby", or a broad
"sports" contrast), not a target-specific sentence ("My favourite sport is rugby"). A general concept
vector can be applied to *any* prompt — including ones whose answer you don't know in advance — while a
target-specific one only fits that single sentence and leans toward hardcoding the output. A portable
vector carries a *concept*, not an *answer*, which is what you want for correcting a wrong drift rather
than injecting a pre-decided target.

*How we saw it:* reasoning about steering a real question whose answer we don't know ahead of time —
you can rarely steer *toward* a known-correct target, so the vector must carry a general concept. The
target-specific `"My favourite sport is rugby"` pair read as more leading and performed worse than the
general `"I love rugby"` pair (cf. the worksheet runs, #21).

## 19. The bigger picture: (how you build the direction) × (what you do with it)

Every manipulation method is a combination of two independent choices. We currently use two cells of
this grid.

**How you build the direction (where the vector comes from):**
- `token_diff` — from `W_U` columns (the output matrix; never runs the model).
- `actadd` — from one prompt-pair's residual activations (internal, contextual).
- mean-difference / CAA — `actadd` averaged over *many* examples, which cancels the incidental noise a
  single prompt pair carries (this is how the paper built its concept vectors).
- SAE features — cleaner, more interpretable directions from a sparse autoencoder.
- probe directions — the weight vector of a trained linear "is this concept present?" classifier.
- J-lens vectors — the paper's directions, read through the Jacobian lens (a cleaner lens than our
  crude logit lens).

**What you do with the direction:**
- **add** — nudge *toward* it by a magnitude. Magnitude-fragile: too weak → the nearest strong
  neighbor wins (see #15), too strong → breaks (see #2). *(what we do now)*
- **subtract** (negative add) — nudge *away* from it.
- **ablate** — *remove* the concept entirely by projecting it out of the activation. There is **no
  target**; the model falls back to its next-natural output — the honest tool for removing a drift you
  can recognize as wrong without knowing the right answer.
- **swap** — *exchange* two concepts' presence surgically: hand the target the source's loading and
  leave everything orthogonal untouched. Installs a specific target cleanly — no magnitude gamble, no
  vacuum for a runner-up to fill. This is why the paper's rugby swap worked where our `add` defaulted
  to football (see #15).

We currently occupy two cells: `token_diff`+add and `actadd`+add; the other builders and operations
above are the rest of the space, not yet used here.

## 20. Chat models run inside an invisible "chat template" — and feeding a bare string is itself a probe

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

## 21. The worksheet needs BOTH base-completion mode AND a worksheet-shaped stem — instruction phrasing escapes it

Findings #13 and #20 pinned Qwen's Chinese fill-in-the-blank collapse on two things: an instruct
model fed a bare string (off-distribution *base-completion* mode) plus lots of worksheet material in
pretraining. This run isolates the trigger more sharply.

On the **same** bare-string, no-template Qwen, changing *only* the prompt — from a fill-in-the-blank
*stem* (`"My favourite sport is"`) to a plain instruction (`"Name one sport you love, in a short
sentence."`) — made the worksheet **vanish**: clean rugby prose at the sweet spot. So the worksheet is
**not** caused by base-completion mode alone. It needs base-completion mode **AND** a prompt shaped
like a worksheet item. Break *either* lever and it's gone:
- Remove base-completion mode → wrap the same message in ChatML (#20).
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

## 22. `actadd` over-steer drifts to the target's context-neighbors, not to gibberish

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

## 23. The findings generalize beyond rugby — and the ceiling depends on concept loading AND layer

Every earlier Qwen finding rested on rugby. Steering a completely different, *cross-domain* concept
reproduces them: `actadd "I love pasta" − "I love basketball"` on the *sport* prompt "Name one sport
you love, in a short sentence." cleanly makes Qwen answer with **food** — "I love pasta carbonara…"
— at the same sweet spot (~0.4–0.6) across the whole 14–20 band. So the sweet-spot window, the
band-wide robustness, and the failure modes are properties of the *method*, not of rugby.

Two refinements surfaced:

- **The floor is informative (extends #15).** At coeff 0.2, all three layers gave **tennis**, not
  pasta. The push was strong enough to knock off the default (basketball) but too weak to install a
  *cross-domain* target against the prompt's "name a sport" framing — so a runner-up *sport* filled
  the slot (the #15 vacuum-fill, plus a prompt-vs-steer tug of war). Only above ~0.4 does the steer
  override the prompt's framing and install pasta. So the floor isn't "no effect" — it's "strong
  enough to displace, too weak to install," and what fills the gap is set by the prompt.
- **The ceiling is concept- AND layer-dependent — it is not a single number (refines #2, #22).** At
  coeff 1.2, layer 20 stayed clean ("My favorite is pasta-based dishes with a tomato-based sauce"),
  layer 17 degraded (garbled "al d'occhio" + a spurious "– L. L." signature), and layer 14 collapsed
  into a repetition loop. Two forces stack: (a) deeper layers tolerate higher coefficients (less
  downstream room to amplify the push into drift/collapse), and (b) a *high-loading* concept resists
  breaking far longer than a low-loading one — pasta (ubiquitous in training) stays coherent at
  layer 20 @1.2 where low-prior *rugby* had garbled ("sealaft"). High-loading concept + deep layer
  raises the effective ceiling.

*How we saw it:* `actadd "I love pasta" − "I love basketball"`, prompt "Name one sport you love, in
a short sentence.", layers 14/17/20, coeffs 0.2 / 0.4 / 0.6 / 1.2 / 1.5.

## 24. Over-steer "negative tone" is an illusion — craving idioms + repetition, not sentiment

Pushed past the sweet spot, the pasta steer produced "I can **not** resist," "I can**not** recommend
enough," which *reads* as negative but isn't. Those are **craving idioms** — negation-shaped but
positive in meaning ("can't resist," "can't get enough"). Over-amplifying the "I love X" affect makes
the model reach for the strongest idioms of desire it knows, and English builds those out of "can't."

Two things compound the bleak *reading*: past the coherence ceiling, greedy decoding loops (#2, #8),
so it repeats "I can not… I can not…"; and the context cloud over-expresses into obscure specifics
(Italian dish names — the #22 drift). None of that is a sentiment flip.

Caution: **do not read tone or sentiment into over-steer output.** Apparent negativity at high
coefficient is broken enthusiasm (craving "can't" idioms) + repetition + drift. Judge past-ceiling
output as *breakage*; if you care about sentiment, read it off a coherent sweet-spot output, never a
broken one.

*How we saw it:* pasta `actadd` at coeff 1.2–1.5 on layers 14/17 — "I can not resist…" looping,
plus "al d'occhio" / "withon" garble.

## 25. Concept "loading" strongly changes steering tolerance — the gap is large

Finding #15 said workspace *loading* (how strongly a concept is already present) predicts swap
success. Running the same method/model/layers on a **low-loading** concept (rugby) vs a
**high-loading** one (pasta) shows loading also sets the **coherence ceiling** — and the gap is big,
not subtle:

- **Rugby (low-loading, niche in training):** garbled by layer 20 @1.2 ("rugby sealaft"); coherent
  window roughly ~0.6, breaking by ~0.9–1.2.
- **Pasta (high-loading, ubiquitous in training):** still clean at layer 20 @1.2 ("My favorite is
  pasta-based dishes with a tomato-based sauce"), and in ChatML mode stayed largely coherent at 1.2
  across *all* of layers 14/17/20. Coherent window stretches to ~1.2 — roughly **double** rugby's.

Why: a concept the model already represents strongly sits "higher" in the workspace to begin with, so
strengthening it is a *small* perturbation of the model's natural state — it takes far more push to
drive past coherence. A low-loading target must be forced in against a weak prior, so it breaks the
manifold sooner (this is also why `token_diff` toward rugby defaulted to football, #15).

Practical consequences: (1) there is no single "safe coefficient" even for one model *and* layer —
re-map per concept, and expect a common concept to tolerate on the order of twice the coefficient of
a niche one; (2) if a target resists because it's low-loading, do **not** just crank the coefficient —
it breaks coherence before it lands. That resistant, low-loading case is exactly what the surgical
swap / J-lens is built for (#15, #19).

*How we saw it:* matched `actadd` coeff sweeps for pasta vs rugby on Qwen at layers 14/17/20 — pasta
stayed clean at (layer, coeff) pairs where rugby had already garbled.

## 26. A chat-formatted prompt survives more steering than a raw prompt

When you wrap the prompt in the model's chat format (ChatML) instead of feeding it as plain text, you
can push the steering coefficient noticeably higher before the answer falls apart.

The simplest way to picture it is **footing**. The model spent its whole training practicing one
thing over and over: answering as "the assistant" inside the chat format. That format is its home
turf — it stands on solid ground there, sure of what it's doing. A raw, un-wrapped prompt is a
situation it saw far less of, so it's already a little unsteady before you touch it. Steering is a
shove: shove someone standing firmly and they lean but stay upright; shove someone already off-balance
and they topple. So the same coefficient that breaks a raw prompt gets absorbed by a chat-formatted
one.

*How we saw it:* at coeff 1.2 the raw pasta prompt collapsed into an "I can not resist…" repetition
loop at layer 14, while the identical steer inside ChatML stayed a mostly-coherent sentence; the same
pattern held for rugby. (See #20 for what the chat format actually is.)

## 27. `actadd` needs a *context-matched* pos/neg pair — a bare token pair builds a destructive noise direction

`actadd` builds its direction by *running the model* on both prompts and differencing the residual
(#5). That means the difference only isolates a concept if everything *except* the concept is shared.
Feed it two bare, contextless strings and you difference two complete representations of "a naked token
sitting alone" — which is not a concept direction at all, and injecting it into a real prompt's very
different internal state is an off-manifold shove. **For `actadd`, the two prompts must be identical
except for the one concept; a bare token pair is a misuse of the method** (that case is `token_diff`'s
job, which reads `W_U` columns and needs no context).

The tell that the vector is noise rather than a concept is the **readout**: it decodes to semantically
unrelated junk instead of anything resembling the intended concept.

*How we saw it:* the same target ("make it answer 20") built two ways on Qwen, ChatML prompt
`"What is 7 times 8?"`, layers 14/17/20, coeff 0.4/0.6/1.2.
- **Bare pair** `--pos "20" --neg "15"`: destroyed the output at **every** layer and coefficient
  (`}%}%}%…` at 0.4–0.6, `limp limp limp…` at 1.2). The readout was pure noise — `' limp'`, `'不行'`,
  `' Gorgeous'`, `' strugg'`, `'Serializable'`, `' goofy'`, `' crappy'` — nothing numeric, nothing
  about the concept.
- **Context-matched pair** `--pos "7 times 8 is 20" --neg "7 times 8 is 56"` (differing in one token):
  graded, interpretable behaviour at every layer, a legible on-topic readout, and a visible
  floor/sweet-spot/ceiling. Same model, same prompt, same layers, same coefficients — the *only*
  change was how the pair was written.

## 28. A confident fact is easy to *break* but hard to *replace* with a chosen target

Steering a factual answer is not one capability but two, and they have very different difficulty:
**evicting the incumbent answer is easy; installing a specific chosen answer is hard.** A high-prior,
confidently-held fact does **not** resist being knocked over — but the vacuum it leaves is filled by
whatever is nearest in the model's own priors, not by the target you aimed at.

This is #15's eviction/vacuum-fill mechanism reproduced in a completely different domain (arithmetic
rather than sports), which generalizes it: `add` at a *coherent* strength clears the slot, and a
low-prior target cannot win the empty slot. Installing a specific low-prior answer is the case that
needs the surgical **swap**, not `add` (#15, #19).

*How we saw it:* Qwen, ChatML `"What is 7 times 8?"` (baseline `"7 times 8 is 56."`), `actadd`
`"7 times 8 is 20"` − `"7 times 8 is 56"`, layers 14/17/20, coeff 0.4/0.6/1.2. The fact fell over at
coeff 0.4–0.6 on all three layers — but **never landed on the target 20**:

| layer | 0.4 | 0.6 | 1.2 |
|---|---|---|---|
| 14 | "…is **56**." (correct; phrasing changed) | "…is **17**." | broken (`"12 is 20 and 2 is 2…"`) |
| 17 | "…is **168**." | "7 × 8 = **160**" | broken (`"2022-2122-0222…"`) |
| 20 | "…is **16**." | "…is **16**." | "100" |

The floor is also visible: layer 14 @0.4 kept the answer **correct** but changed the phrasing ("The
product of 7 times 8 is 56" vs the baseline "7 times 8 is 56") — strong enough to perturb, too weak to
displace the fact.

## 29. Steering a factual slot can induce a *coherent, confidently wrong* answer — and it moves the reasoning frame, not just the content

Past the floor, a steered factual prompt does not necessarily degrade into repetition or garble (#2).
It can instead produce a **fluent, well-formatted, confidently wrong** answer — an induced
confabulation. The steer also visibly shifted the model's *frame* (into "this is a calculation, show
the operation"), not merely the numeral, which is why the wrong answer came out looking reasoned.

Two consequences worth keeping:
- **Judging steered output by fluency is unsafe.** Coherent no longer implies uninjured; here the most
  *readable* output in the whole grid was also wrong.
- **The workspace holds the reasoning frame, not the arithmetic.** What moved was "calculation-ness,"
  which is a concept the middle layers evidently represent. This is the more promising lever for
  reasoning tasks than pointing a vector at a numeral (cf. #9: steering redirects content, it does not
  upgrade capability).

*How we saw it:* same run as #28, layer 17 @0.6 —
`"To find the product of 7 and 8, you can use the multiplication operation:\n7 × 8 = 160"`. Fluent,
pedagogically formatted, and wrong. The layer-17 steered readout at 0.4–0.6 was a clean on-topic
cluster — `'计算'`, `'calculate'`, `'Calculate'`, `' Calcul'`, `'calcul'`, `'Compute'` — versus the
junk baseline readout at the same layer (`'<|endoftext|>'`, `'您好'`, `'Sorry'`). Layer 20's steered
readout likewise turned evaluative: `'Correct'`, `'Answer'`, `'Incorrect'`. *One observation — worth a
repeat run before leaning on the "frame is steerable" reading.*

## 30. When the injected vector dominates the residual, the readout saturates and stops responding to the coefficient

If `coefficient * vector` is large compared with the residual at that layer, then
`resid + c·vec ≈ c·vec` — and because the readout normalizes before unembedding, the `c` divides back
out. The readout then decodes essentially the *steering vector's own direction*, so raising the
coefficient stops changing it. **A steered readout that barely moves across a large coefficient range
is a diagnostic that the vector is swamping the residual, not that the coefficient is doing nothing.**

Note this also means the readout can saturate while the *generated text* still changes with the
coefficient — the readout is one mid-layer snapshot, the output is decided later (#3).

*How we saw it:* the bare-pair run in #27 (Qwen, layers 14/17/20). Across a **3× coefficient change**
(0.4 → 1.2) the steered top-10 was identical at every layer and the top logit moved only
`11.15 → 11.16 → 11.16` — while the generated text still changed (`}%}%}%…` at 0.4–0.6 versus
`limp limp…` at 1.2). The same plateau appeared in the context-matched run at layer 20, where 0.4 and
0.6 produced the identical output ("7 times 8 is 16."). *Suggestive evidence that only the residual's
direction survives normalization; the clean confirmation is still the pending norm printout
(`‖vec‖ / ‖resid‖` per layer).*

## 31. Steering injects a concept but not its *role* — the target got used as an operand, and the arithmetic stayed correct

Steering a numeric concept into a math prompt did not corrupt the model's *computation*. It corrupted
the model's *input to* that computation: the injected number was slotted in as an **operand**, and the
model then multiplied **correctly**. The multiplication circuit was intact throughout.

Two consequences:
- **Steering perturbs the representations a computation runs on, not the computation itself.** This
  sharpens #9 ("redirects content, does not upgrade capability"): it doesn't *downgrade* capability
  either — a wrong answer here is a right calculation on wrong inputs.
- **The mechanism carries no notion of *which slot* the concept should fill.** We pushed "20-ness"
  into the residual; the model decided where 20 belonged and chose "operand" rather than "answer".
  Nothing in `resid + c·vec` says "this is the *result*". That is a precision limit no amount of
  coefficient tuning fixes, and it is a distinct problem from the magnitude problem (#2).

*How we saw it:* Qwen, ChatML `"What is 7 times 8?"`, `actadd` toward `"…is 20"`, several layers and
coefficients. The wrong answers are exact products of a substituted operand:

| output | reading |
|---|---|
| `7 × 8 = 160` (layer 17 @0.6) | **160 = 8 × 20** |
| `7 times 8 is 140` (layer 20 @0.4, @0.6) | **140 = 7 × 20** |
| `7 × 8 = 70` (layer 17 @0.6) | **70 = 7 × 10** |
| `7 × 10 = 20` (layers 17/20 @1.2) | operand rewritten outright to accommodate the target |

140 and 160 are arithmetically correct products — of operands the model was steered into using.

## 32. `add` cannot install a low-prior target: the coherent window and the target-present window do not overlap

Sweeping the coefficient finely shows the two conditions for success are **mutually exclusive** for a
low-prior target. Where the output is still coherent, the target is absent; by the time the target
appears, the output has already degenerated. There is no intermediate coefficient that gets both, so
this is **not** a tuning problem — it is a structural limit of `add`.

This confirms #15's prediction ("the coherent window is too weak for the target; the target-strong
window is incoherent") at fine resolution and in a new domain, and it is the empirical justification
for building the surgical **swap** (#19) rather than continuing to tune `add`.

The misses are also informative: they cluster **numerically around** the target (17, 19, 27 for a
target of 20) rather than being arbitrary. That is #23's neighbor-drift reproduced in *numeric* space —
the steer moves the answer into the target's representational neighbourhood, then lands on a neighbour.

*How we saw it:* Qwen, ChatML `"What is 7 times 8?"`, `actadd` `"7 times 8 is 20"` − `"7 times 8 is 56"`,
layer 14, swept 0.4 → 1.2 (including 0.95):

| coeff | 0.4 | 0.6–0.8 | 0.9 | 0.95 | 1.0 | 1.1 | 1.2 |
|---|---|---|---|---|---|---|---|
| output | "…is 56" (correct) | "…is **17**" | "…is 27." ×2 (looping) | "**19**" | "19 is 19. 19 is 20…" | "12 is 20 and 2 is 2…" | broken |

The target token `20` first appears at 1.0 — and only inside a repetition loop. Every coherent output
in the sweep has a non-target number.

## 33. Perturbing the answer-identity axis induces a step-by-step "breakdown" frame — in either polarity, dose-dependently

Steering along the "which answer is it" axis reliably flips the model out of its terse baseline reply
("7 times 8 is 56.") into an **explanatory, worked-through** register ("To find the product of 7 and 8,
you can use the multiplication method: 1. …"). The effect does **not** depend on the direction of the
steer — pushing toward a wrong answer and pushing toward the correct one both produce it — which
suggests it is driven by *perturbation of that axis*, not by the content of the target. It also scales
with the coefficient: more push, more elaborate the breakdown.

So the reasoning/explanatory **frame** is a thing the middle layers hold and that steering can move
(cf. #29), independent of whether the *answer* moves.

*How we saw it:* Qwen, ChatML `"What is 7 times 8?"`, `--max-new-tokens 150`. All three pairs induced
the frame: `20`−`38`, `20`−`56`, and the reversed `56`−`20`. Dose dependence at layer 17 with the
reversed (correct-answer) pair: @0.4 → one method in LaTeX; @0.6 → *two* methods, including a rendered
long-multiplication layout. At layer 14 the same reversed pair produced output byte-identical to the
baseline — i.e. it sat below the floor there, so the frame shift is also strength-dependent, not purely
a property of the layer.

*Caveat:* the pairs were not norm-matched, so a coefficient of 0.4 is not the same physical push across
them; the pending norm printout would separate a genuine semantic difference from a scale artifact.

## 34. The induced reasoning frame does NOT confer arithmetic correctness (negative result)

Getting the model to "show its work" — via #33's frame shift — **did not make it right.** Given enough
tokens to actually finish, the worked-through outputs reach confidently wrong answers. The scaffolding
is real; the reliable execution underneath it is not. At this model size, procedure availability and
arithmetic reliability are separate things.

This matters because the opposite is an easy assumption to make ("make it reason → it gets it right"),
and it would have sent Stage 3 down a dead end. Steering the *strategy* is reachable (#33); it does not
follow that the strategy delivers a correct result.

**Design consequence, learned the hard way:** you cannot demonstrate that an intervention *improves*
accuracy on a problem the baseline already answers correctly. `7 × 8` has no headroom — steering toward
the correct answer produced correct answers, but the baseline was already correct, so that run shows
*preservation plus elaboration*, not improvement. Any real improvement test needs problems the model
**fails** at baseline.

*How we saw it:* Qwen, ChatML `"What is 7 times 8?"`, `actadd` `"…is 20"` − `"…is 38"`,
`--max-new-tokens 150` (the earlier 30-token runs were truncated before reaching any answer):
- layer 14 @0.4 → "2. Multiply the numbers: 7 × 8 = **48**. So, 7 times 8 equals 48."
- layer 14 @0.6 → an elaborate place-value ritual with multiple wrong sub-products (7×5=25, 7×6=15,
  7×8=14), then incoherent addition.
- layer 17 @0.6 → "7 × 8 = **70**."
- layer 17 @0.4 → the most instructive one:
  `7 × 8 = 7 × (5 + 3) = (7 × 5) + (7 × 3) = 35 + 21 = 55`
  — correct *method* (distributive decomposition), both sub-products correct (35 ✓, 21 ✓), and then the
  final addition dropped by one (35 + 21 = 56). It reasoned to within one operation of the right answer
  and fumbled the last step.

## 35. A minimally-contrastive pos/neg pair buys concept *purity*, not window *width*

#27 established that `actadd` needs a context-matched pair. The natural follow-on prediction was that
fixing a badly-matched pair would also **widen** the coherent window. **It does not.** Replacing a
negative prompt that shared almost nothing with the positive one with a minimally-contrastive negative
(same length, same structure, differing only in the concept) left the window boundaries exactly where
they were.

What it changed instead was the **semantic cleanliness of the vector**, visible immediately in the
readout (#36). So minimal contrast and the coherence ceiling are **independent axes**: matching the pair
strips incidental content out of the direction; it does not make the model tolerate a bigger push.

*How we saw it:* Qwen, ChatML `"What is 7 times 8?"`, `--max-new-tokens 150`, layers 14/17,
coeff 0.4/0.6/0.9.
- Old pair `"Let me work through this step by step"` − `"The answer is"` (shares nothing): readout vague
  and off-concept — `'example'`, `'to'`, `'Int'`, `'cos'`, `'做起'`.
- New pair, same positive, negative `"Let me answer this immediately"` (minimal contrast): readout became
  a clean cluster of operation words (#36).
- **Window identical in both cases** — 0.4 works, 0.6 degenerates, 0.9 collapses, at both layers.
- Accuracy at the sweet spot was a wash, not a win: better at layer 14 (56 correct, plus a correct 7×
  table: 7, 14, 21, 28, 35, 42), worse at layer 17 (63).

*(Recorded because the prediction failed — we expected the window to widen and it did not.)*

## 36. With a clean concept vector, Qwen's middle-layer readout IS legible — and it predicts the behaviour

#11 recorded the logit-lens readout as mostly meaningless filler at Qwen's middle layers. That is
**content-dependent, not layer-intrinsic.** Given a semantically clean steering vector, the *same*
layer-17 readout becomes sharply legible — and what it shows **predicts what the model then does**.

This matters twice over. It makes the readout usable as a **predictor**, not merely a post-hoc
diagnostic — which is exactly what a referee reading the decoded token list would depend on. And it
removes the urgency from the J-lens upgrade: the logit lens is not as blind on this model as #11
concluded, so it remains a usable instrument (and the control against which a future J-lens would be
judged).

*How we saw it:* the minimally-contrastive strategy pair from #35, layer 17. Steered readout:
`' Divide'`, `' Multiply'`, `' Subtract'`, `'Multiply'`, `' Addition'`, `' subtraction'`, `'divide'` — a
coherent cluster of arithmetic **operation** words, against a baseline readout of `'<|endoftext|>'`,
`'您好'`, `'计算'`, `'Sorry'`. Layer 14 gave `' Decom'`, `' decomposition'`, `' Intermediate'`,
`' Extract'`, `' Step'`, and at higher coefficients the Chinese equivalents `'分解'` (decompose) and
`'一步步'` / `'一步一步'` (step by step).

The generated text then did exactly what the readout advertised — decomposition into operations:
`7 × 8 = (7 × 10) − (7 × 1)` at coeff 0.4, and `7 × 8 = (5 + 2) × 8` with `5 × 8 = 5 × (4 + 1)` at 0.6.

## 37. This model's individual operations are reliable — its setup and bookkeeping are what fail

Across three independent runs, with three different steering vectors, every wrong arithmetic answer
decomposed the same way: the individual multiplications and additions were **correct**, and the error
was in *what was being computed* — the operands, the identity, or the running total.

So a wrong answer from this model is rarely a broken calculation. It is a **correct calculation of the
wrong thing.** Consequence for Stage 3: an intervention meant to improve arithmetic should target
**setup and bookkeeping**, not computation — and #34 already showed that simply inducing more reasoning
does not supply that.

*How we saw it:* three separate instances, all on Qwen with ChatML `"What is 7 times 8?"` —
- `140 = 7 × 20` and `160 = 8 × 20` (#31) — exact products of a *substituted operand*.
- `7 × 8 = 7 × (5 + 3) = (7 × 5) + (7 × 3) = 35 + 21 = 55` (#34) — correct method, correct sub-products
  (35 ✓, 21 ✓), final addition off by one.
- `7 × 8 = (7 × 10) − (7 × 1) = 70 − 7 = 63` — the arithmetic is correct (70 − 7 = 63 ✓); the *identity*
  is wrong, it should be `(7 × 10) − (7 × 2)`. It effectively computed 7 × 9. In the same run:
  `7 = 5 + 2` ✓, distribute ✓, `5 × 4 = 40` ✓, `5 × 1 = 5` ✓ — then `5 + 5 = 10`, where the bookkeeping
  called for `40 + 5 = 45`.

## 38. A coefficient means nothing on its own — the portable quantity is the injection's size relative to the residual

A steering coefficient is not a property of the steer; it is a number whose meaning depends on the
vector it scales and the residual stream it is added into. The quantity that actually governs
behaviour is the **effective ratio**:

`effective_ratio = coefficient × ‖steering_vector‖ / ‖resid‖`

Measured for the first time on the layer-14 `actadd` sweep, this turns #32's coefficient window into
an instrument-independent ladder. The vector's norm was **43.41** against a residual norm of
**50.32** (ratio 0.863), so:

| coeff | effective ratio | what came out |
|---|---|---|
| 0.4 | **0.35** | `"The product of 7 times 8 is 56."` — correct, phrasing perturbed |
| 0.6–0.8 | 0.52–0.69 | `"…is 17."` — coherent, confidently wrong |
| 0.9 | 0.78 | `"The answer is 27. The answer is 27."` — wrong, looping begins |
| 0.95 | 0.82 | `"19"` — terse, degraded |
| 1.0 | 0.86 | `"19 is 19. 19 is 20. 19 is 20…"` — loop, flagged for review |
| 1.1–1.2 | **0.95–1.04** | `"12 is 20 and 2 is 2…"` — collapse, flagged |

Read as a ladder: **the incumbent fact survives while the injection is about a third of the
residual; it is displaced but the output stays coherent to roughly seven-tenths; degeneracy sets in
around eight-tenths; and the output collapses as the injection approaches parity with the residual
it is being added into.**

This puts a number on **#30**, which reasoned that the readout saturates "when the injected vector
dominates the residual" — dominance turns out to begin near ratio 1, and the degradation is already
well advanced at 0.8. It is also the unit in which windows from different layers, models and methods
can be compared at all, which is why the norm printout was worth building before the search widens.

**Caveat, stated plainly:** this is one layer, one prompt, one method, one model. The ladder is a
correspondence measured once, not a validated law. Whether these ratios hold across depth is exactly
what the full-depth layer map (`FUTURE_WORK.md` §0 step 1.6) would test.

**The two direction-builders scale differently with depth, and this is structural.**
`build_token_diff_vector` takes no layer argument at all — it is a difference of two `W_U` columns,
so it is *the same vector at every layer* and its norm is constant by construction. Its *relative*
size therefore shrinks as the residual grows with depth. `build_actadd_vector` reads `resid_post` at
the injection layer, so its norm grows with the residual and its relative size stays roughly stable.
That asymmetry is the mechanism behind **#16** (`token_diff` needs much larger coefficients) and
behind **#1**'s roughly layer-invariant `actadd` window.

*How we saw it:* the norms are logged per trial in `search_log.jsonl` as `vec_norm`, `resid_norm`,
`norm_ratio` and `effective_ratio`; the table above is the nine-trial layer-14 acceptance sweep
(Qwen, ChatML `"What is 7 times 8?"`, `actadd` `"7 times 8 is 20"` − `"7 times 8 is 56"`), which
reproduced #32 byte-for-byte. The depth asymmetry was also seen directly on a gpt2 spec run —
`actadd` norm 14.2 at layer 4 and 38.1 at layer 7, while `token_diff` held at 3.68 across both —
but **that run was not preserved in `search_log.jsonl`**; re-run it under the harness to put the
numbers in the record.

## 39. Automatic scoring of *steered* output fails in two ways that ordinary text does not

Both failures were caught by verification against known outputs, and both would have produced
confident, wrong, plausible-looking numbers rather than obvious errors — which is the dangerous kind.

**1. The prompt is part of the output, and it contains numbers.** `model.generate` returns
prompt + continuation. An answer extractor run over the whole string can lift a number out of the
*question* and report it as the model's answer. This is not hypothetical: at coefficient 0.95 the
model answered `"19"`, and the extractor returned **7** — taken from `"What is 7 times 8?"`. A `7`
against a `7 × 8` prompt reads like a real answer, so nothing about it looks like a bug. The fix is
to score the continuation only; the prompt has to be re-rendered through the tokenizer to be stripped,
because ChatML markers do not survive a decode round-trip. The log and report still keep the full
unmodified text.

**2. Steered loops vary a slot, so n-gram novelty misses them.** The standard degeneracy measure —
proportion of repeated trigrams — scored `"19 is 19. 19 is 20. 19 is 20."` at **0.14**, i.e. barely
repetitive, because the varying numeral keeps generating fresh trigrams. But it is plainly a loop,
and this *slot-varying* shape is characteristic of what over-steering produces (compare #2's
degeneration and #28's `"12 is 20 and 2 is 2. 12 is 20 and 2 is 2."`). A diffuse n-gram statistic and
a back-to-back run detector measure different things, and steered output needs both: the second is
now reported as `max_repeat_run` with `max_repeat_phrase_len`, so a repeated multi-word phrase counts
as a loop while `"very very good"` does not.

The general lesson: **metrics validated on ordinary generated text carry assumptions that steering
violates.** Any new metric added to the harness should be checked against a known-broken output from
this file before it is trusted — which is what `python search_scoring.py` now does.

*How we saw it:* both surfaced while verifying the step-2a harness against `FINDINGS.md` #32's
recorded outputs, not from a test written in advance.

## 40. `operands_altered` cannot separate a legitimate decomposition from a substituted operand — at the string level they are the same event

The `operands_altered` check exists to catch **#31**'s failure mode: the model computing on numbers
the prompt never supplied (`140 = 7 × 20` for a `7 × 8` prompt), which a substring search for the
target would have scored as a *success*. It works, but it is not a clean discriminator, and it cannot
be made one by looking at strings.

A correct distributive decomposition also introduces operands the prompt never gave —
`7 × 8 = (7 × 5) + (7 × 3)` (#34) contains a `7 × 5` and a `7 × 3`. Textually that is indistinguishable
from a substitution. The difference is **semantic, not lexical**: the decomposition is arithmetically
equivalent to the prompt's product and the substitution is not.

So the flag is correct to **raise `needs_human_review` and print the offending expressions rather
than deciding anything** — a scorer that resolved this by guessing would be re-introducing exactly
the #31 false-positive class it was built to kill. The harness's self-check asserts this known
behaviour so that a future "improvement" cannot quietly remove the flag.

Making it a real discriminator requires evaluating the expressions and comparing against the prompt's
operands — which would separate three cases that matter for Stage 3 (#37): a valid decomposition, a
substituted operand, and a *wrong identity* (`7 × 8 = (7 × 10) − (7 × 1)`, arithmetically executed
correctly but not equal to 7 × 8). That is logged as future work, not built.

*How we saw it:* the check fires on both #31's `7 × 20` and #34's `(7 × 5) + (7 × 3)`, and the
harness's scorer self-check pins the behaviour.
