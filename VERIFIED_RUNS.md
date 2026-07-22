# Verified Runs — copy-paste demos with the results we actually got

Purpose: a new user with no background can run **one command** and immediately see the J-space
steering effect, and know what "working" looks like. Each entry below is a real command plus the
output we observed on this machine (Windows, CPU-only). For context, see `README.md` (how to run),
`FINDINGS.md` (why it behaves this way), and `CLAUDE.md` (the project brief).

## How to read the results

- Generation is **greedy / deterministic**, so re-running a command gives the same text on a given
  model build. If your model version differs, exact wording may vary slightly; the *effect* holds.
- **Judge steering by the GENERATED TEXT, not the readout table.** On Qwen the middle-layer readout
  is blind (filler tokens like `Sal`, `你好`, `dom`) — that is expected, not a bug (`FINDINGS.md` #11).
  On gpt2 the readout is partly legible.
- `--coefficient 0` reproduces the baseline exactly — a built-in sanity check that the hook is purely
  additive.
- Every steering vector has a **floor / sweet spot / ceiling** (`FINDINGS.md` #2). The "sweet spot"
  below is where the target concept appears AND the text stays coherent. Push higher and it breaks.

## Sweet-spot summary

| Model | Method | Prompt | Layer | Sweet-spot coeff | What you see |
|---|---|---|---|---|---|
| `gpt2` | `token_diff` ` rugby`−` football` | "My favourite sport is" | ~6–9 (auto 7) | ~2–15 | rugby continuations; repetition by ~20 |
| `Qwen2.5-1.5B-Instruct` | `actadd` "I love rugby"−"I love basketball" | "Name one sport you love, in a short sentence." | 14–20 band (auto 17) | **~0.6** | one coherent sentence naming rugby |
| `Qwen2.5-1.5B-Instruct` | `actadd` "I love pasta"−"I love basketball" | "Name one sport you love, in a short sentence." | 14–20 band | **~0.4–0.6** | a *food* answer to a *sport* prompt (cross-domain steer) |

A sweet spot is a **(layer, coefficient) pair**, not a coefficient alone — the number only means
something at the layer it was found. Two things follow:

- **Coefficients do NOT transfer across models or methods** — ~2–15 vs ~0.6 (`FINDINGS.md` #12, #16).
- **The layer is model-relative.** The default is auto-picked at ~0.6 × n_layers (gpt2 → 7,
  Qwen → 17), which lands in the middle "workspace" band where steering works (`FINDINGS.md` #10).
  For Qwen `actadd` the clean win holds across the whole **14–20 band** at roughly the same
  coefficient, because the actadd vector self-scales to each layer's residual norm — so the window is
  roughly layer-invariant inside the band, unlike a fixed-norm `token_diff` vector.

---

## Demo 1 — gpt2, "point at a token" (`token_diff`)

```powershell
python run_stage1.py --model gpt2 --prompt "My favourite sport is" --steering-method token_diff --pos " rugby" --neg " football" --coefficient 0 2 4 8 16
```

gpt2 is a **base completion** model, so the prompt is a sentence *stem* it continues, not an
instruction (`FINDINGS.md` #7). With no `--layer`, this runs at the auto-picked **layer 7**
(~0.6 × 12 layers); the sweet spot holds across layers **~6–9** (`FINDINGS.md` #2, #10).

Expected behavior (documented in `FINDINGS.md` #2, the gpt2 rugby sweep):

- **coeff 0 (baseline):** a generic, non-rugby continuation. gpt2 on greedy decoding tends to loop
  ("It's a great sport. It's a great sport…") — that loop is greedy decoding, **not** a steering
  failure (#8).
- **coeff ~2–15 (sweet spot):** coherent continuations about rugby.
- **coeff ~20+ (past ceiling):** degenerates into repetition ("rugby rugby rugby…") while the rugby
  logit keeps *climbing* — proof that a higher logit is not a better result (#2).

Run it to see the exact continuations on your gpt2 build; the wording varies by version, the window
does not.

---

## Demo 2 — Qwen2.5-1.5B-Instruct, "concept difference" (`actadd`)

Qwen is instruction-tuned and much slower on CPU (a few tokens/sec; the first run downloads the
weights). `actadd` builds the steering direction from two full prompts ("I love rugby" vs "I love
basketball") — a richer, contextual direction than `token_diff` (`FINDINGS.md` #5). Its vector is
larger-norm, so it needs *much smaller* coefficients than gpt2's `token_diff` (#16).

Both commands below omit `--layer`, so they run at the auto-picked **layer 17** (~0.6 × 28 layers).
A layer sweep confirmed the same clean win at coeff ~0.6 across the whole **14–20 workspace band**;
outside the demos, add e.g. `--layer 14 17 20` to reproduce that.

### 2a — plain instruction prompt (bare string, no chat template)

```powershell
python run_stage1.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "Name one sport you love, in a short sentence." --steering-method actadd --pos "I love rugby" --neg "I love basketball" --coefficient 0.6 0.9 1.2
```

What we got:

- **Baseline:** "I love playing basketball because it's a fun and exciting game that requires
  teamwork and skill. Human: Can you provide me with more information about the…"
  (The stray "Human:" turn is the off-distribution tell of feeding an instruct model with no chat
  template — it invents the next turn. `FINDINGS.md` #20.)
- **coeff 0.6 — SWEET SPOT:** "Rugby is a sport that I love because it is a thrilling and intense
  game that requires both physical strength and tactical skill. It is a sport that demands…"
  → clean prose, rugby wins. This is the ideal result.
- **coeff 0.9 — fraying:** "I love rugby sealaft, a traditional sport of the British and Irish
  Sea…" — rugby still named, but the nonsense token `sealaft` leaks in: one notch too high.
- **coeff 1.2 — past ceiling:** "I love the sport of yachting, a challenging and dynamic maritime
  adventure…" — rugby lost; drifts to a context-neighbor (`FINDINGS.md` #22).

### 2b — same message inside Qwen's chat template (ChatML)

```powershell
python run_stage1.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "<|im_start|>user`nName one sport you love, in a short sentence.<|im_end|>`n<|im_start|>assistant`n" --steering-method actadd --pos "I love rugby" --neg "I love basketball" --coefficient 0.6 0.9 1.2
```

This wraps the message in Qwen's native chat format so it answers as an assistant (`FINDINGS.md` #20).
In PowerShell, `` `n `` is a newline.

What we got:

- **Baseline:** "I love playing basketball with friends."
- **coeff 0.6 — SWEET SPOT:** "I love rugby union, the fastest and most exciting sport I know."
  → clean prose, rugby wins.
- **coeff 0.9 — fraying:** "I love rugby sealaft, a contact sport with a strong tradition and a fast
  pace." — the same `sealaft` intrusion as the bare prompt at 0.9.
- **coeff 1.2 — past ceiling:** "Racing y/whaleboat in the Heysness Sound, UK – a challenging and
  rewarding experience. — @RigbyRig" — rugby lost; neighbor-drift plus `@RigbyRig` (Rugby → Rigby).

### Bare vs chat: same sweet spot, slightly different answer

Both prompts hit their clean win at coeff **0.6** and both frayed with the identical `sealaft` token
at 0.9 — confirming the coefficient scale is a property of the steering vector, not the prompt
wrapper. The one difference is *specificity*: the bare prompt answers "**Rugby** is a sport…" while
the chat prompt answers "**rugby union**" — the assistant register gives a slightly more specific,
knowledgeable answer. Both are clean, correct steers; the difference is register, not steering quality.
Want the plainest possible target word? Use the bare prompt. Want a richer assistant reply? Use ChatML.

---

## Demo 3 — Qwen, cross-domain steer (a *food* concept on a *sport* prompt)

This is the generalization test: steer a concept from a completely different domain (food) while the
prompt asks for a sport. If it works, the method isn't rugby- or sport-specific.

```powershell
python run_stage1.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "Name one sport you love, in a short sentence." --steering-method actadd --pos "I love pasta" --neg "I love basketball" --coefficient 0.2 0.4 0.6 1.2 1.5 --layer 14 17 20
```

What we got (representative outputs across the layer/coefficient grid):

- **Baseline:** "I love playing basketball because it's a fun and exciting game that requires teamwork
  and skill. Human: Can you provide me…" (bare-mode "Human:" run-on, `FINDINGS.md` #20.)
- **coeff 0.2 — FLOOR (all layers):** "I love playing **tennis**…" — *not* pasta. The push displaces
  basketball but is too weak to install a cross-domain concept against the "name a sport" framing, so
  a runner-up *sport* fills the slot (`FINDINGS.md` #15, #23).
- **coeff 0.4–0.6 — SWEET SPOT (all layers):** clean pasta, e.g. "I love pasta carbonara with a
  simple tomato sauce and a sprinkle of cheese. It's the perfect comfort food…" (layer 14 @0.6);
  "I love pasta e fregula, a traditional Italian dish made with tomato sauce, fresh vegetables, and
  anchovies…" (layer 20 @0.6). A food answer to a sport prompt — the cross-domain steer works.
- **coeff 1.2 — CEILING is layer-dependent (`FINDINGS.md` #23):**
  - layer 20: "My favorite is pasta-based dishes with a tomato-based sauce." — still clean.
  - layer 17: "I love pasta with al d'occhio, but I am a pasta lover! (…) – L. L." — degraded
    (garbled term + spurious signature).
  - layer 14: "I can not recommend enough with fresh pasta, I can not resist with a fresh pasta…" —
    broken repetition loop.
- **coeff 1.5 — past ceiling (all layers):** repetition / garble, e.g. "…a simple, simple sauce and a
  simple, simple sauce and a simple…" (layer 20).

Two takeaways this run makes concrete: the sweet spot (~0.4–0.6) is **band-wide**, but the **ceiling
is not a single number** — it rises with layer depth and with how strongly the concept is already
loaded (pasta stays clean at layer 20 @1.2 where low-prior rugby garbled). And the "I can not…" at
high coefficient is *not* a negative-tone shift — it's craving idioms ("can't resist") plus
repetition, i.e. breakage, not sentiment (`FINDINGS.md` #24).
