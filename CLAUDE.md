# J-Space Exploration — Project Brief for Claude Code

This file describes a hands-on mechanistic-interpretability project. Read it fully before
writing any code. Build in stages, verify each stage with a runnable demo before moving to
the next, and explain what the outputs mean in plain language as you go — the goal is for me
(the human) to *understand* what is happening, not just to get working code.

---

## 0. Operating rules (read this before every task)

### Change-control and modes
- **Never create, edit, or delete files, and never run state-changing commands, unless I have
  approved a plan for that specific work.** When in doubt, propose — do not act.
- **Plan mode is your propose-only mode: research, analyze, and lay out a plan — no edits.**
  (Claude Code enforces this; you cannot edit while in plan mode. That is intended, not a bug.)
- **Only implement after I approve a plan and take you out of plan mode.** With manual edit
  approval, show me each change and wait; with auto-accept, apply the already-approved changes.
- **If I am talking to you and there is no approved plan, treat it as conversation only** —
  answer, explain, and propose, but change nothing until I say "implement" or approve a plan.
  In this project, chatting ≠ permission to edit.

### Models and effort — tell me to switch BEFORE you start
- **At the start of every task or stage, before doing any work, state the model and effort
  level the task calls for.** If it differs from what is currently active, tell me explicitly
  to switch and wait for my confirmation. Do not silently proceed on the wrong setting.
- Effort levels: **low / medium / high / max** (reasoning/thinking depth).
- Default working pair: **Sonnet 5** for the bulk, **Opus 4.8** for hard design and debugging.
  **Haiku 4.5** only for trivial mechanical edits. **Fable 5 is NOT needed for this project.**

| Scenario | Model | Effort |
|---|---|---|
| Environment setup, `requirements.txt`, `README`, boilerplate | Sonnet 5 | low–medium |
| Stage 1 core: J-space readout + steering math (tensor shapes, layernorm placement, hook wiring) | Sonnet 5 | high |
| Steering produces garbage or unexplained internal behavior — deep debugging | Opus 4.8 | high–max |
| Stage 2 referee-loop architecture and design | Opus 4.8 | high |
| Stage 2 implementation once the design is agreed | Sonnet 5 | medium–high |
| Stage 3 experiment design and result analysis | Opus 4.8 (design), Sonnet 5 (wiring/runs) | high / medium |
| Trivial edits, renames, formatting, file moves | Haiku 4.5 | low |

- The volume of code here is small, but its **correctness sensitivity is high** — wrong tensor
  shapes or a missing layernorm will run and produce plausible-looking nonsense. Do not
  under-power the hard steps to save time; flag when a step deserves Opus 4.8.

---

## 1. What this project is

I want to reach inside a language model while it is running, read the "concepts" it is
holding in its middle layers, change one of those concepts by editing the internal
activations, and watch the final text output change as a result. Then I want to automate
that intervention with a second model, and finally use it to try to make a small model
perform better than its untouched baseline.

This is based on a real, recent piece of research:

- **Paper:** *Verbalizable Representations Form a Global Workspace in Language Models*,
  Anthropic, July 2026 (Transformer Circuits).
- **Idea:** a model's "reportable" concepts sit in a small privileged subspace of the
  middle layers — nicknamed the **J-space** — and a tool called the **Jacobian lens
  (J-lens)** can decode an intermediate activation into a ranked list of vocabulary tokens
  the model is disposed to say. Steering that subspace causally changes the output.
- **Reference code:** `anthropics/jacobian-lens` (uses HuggingFace `transformers`),
  and the community tool `Extraltodeus/J-Wash` built on top of it.

I am doing this to learn by reproducing it at small scale on my own hardware. I am not
trying to reproduce the paper's scientific *claim* (that requires many models and controls);
I am reproducing the *mechanism* so I can see it work and build on it.

**Research stance — take this seriously and tell me the truth as we go:**
- **Stage 1 uses an established technique.** It works. If it doesn't, that's a bug to fix, not
  a limit of the idea.
- **Stage 2 and Stage 3 are open research** — deciding *which* direction to steer and *how
  much*, automatically, is a real unsolved problem, and solving it is the point of this work.
  Build it the way every real system starts: small, testable, one verified step at a time.
  Measure results honestly and report exactly what you find. That rigor is not caution — it is
  how we tell real progress from wishful thinking, and it is what makes what we build stand up.

---

## 2. Hardware and environment (hard constraints)

- **Machine:** ASUS Vivobook, Windows, Intel Core i5-13420H (8 cores / 12 threads).
- **GPU:** Intel UHD integrated only — **no CUDA, no usable VRAM. Everything runs on CPU +
  system RAM.** Do not write CUDA-only code. Default every model/tensor to `device="cpu"`.
- **RAM:** 16 GB. Keep working models small (see model list). A ~1.5B model in fp32 is ~6 GB
  and is fine; a 3B Ollama model (~2.5 GB Q4) can run alongside it but watch total memory.
- **Python:** 3.14 is installed but **PyTorch/TransformerLens do NOT support it yet.**
  Create and use a dedicated **Python 3.11 or 3.12** virtual environment for the torch stack.
  Do not install torch into 3.14.
- **Ollama** is installed and is used only for the Stage 2 referee model.
- Expect CPU speed: GPT-2 small is near-instant; a 1.5B model runs at a few tokens/sec.
  That is acceptable — we generate short outputs and inspect internals, we don't chat.

First thing to do: set up the venv, a `requirements.txt`, and a short `README.md` with exact
run instructions for Windows. Confirm imports work before writing project logic.

---

## 3. Models

**Working model (the one we inspect and steer) — needs TransformerLens / HF so we can reach
the residual stream. Ollama CANNOT be used for this; it does not expose activations.**
- Start with **GPT-2 small (`gpt2`, 124M)** via TransformerLens `HookedTransformer`. Fastest
  way to see the loop work.
- Then support **Qwen2.5-1.5B** as the "real" working model (also the family used by the
  `anthropics/jacobian-lens` examples, so the same model carries us into their code).
  `Pythia-1.4B` is an acceptable alternative.

**Referee model (Stage 2 only) — a plain chat evaluator, run via Ollama:**
- **`qwen2.5:3b`** (or `llama3.2:3b`). Called over Ollama's local API.

**Make the working model swappable.** Do not hardcode layer counts or hidden sizes. Read
`model.cfg.n_layers` and `model.cfg.d_model` at runtime and derive the interception layer
(a middle layer, roughly 0.5–0.65 × n_layers). Swapping models should be a config change.

---

## 4. The three stages

### Stage 1 — Read and steer the J-space (manual, hands-on)

**Goal:** in a single controlled script, read what a chosen middle layer is "thinking," inject
a vector, and observe the output change.

Build it to do, step by step:
1. Load the working model on CPU.
2. Run a prompt with `run_with_cache`; extract the residual stream at the interception layer
   (`cache["resid_post", layer]`).
3. **Read the J-space:** apply the model's final layernorm then unembedding to the layer's
   activation (`model.unembed(model.ln_final(resid[:, -1]))`) and print the top-k vocabulary
   tokens. This is the logit-lens readout; label it clearly as the model's current internal
   "disposition." (Optional upgrade: wire in `anthropics/jacobian-lens` for the proper
   averaged-Jacobian lens, which reads more cleanly at early layers.)
4. **Build a steering vector.** Primary method: activation addition (ActAdd) — cache the
   residual at the same layer for a positive prompt and a negative prompt and take the
   difference. Also support the simple "point at a token" method: difference of two
   `model.W_U` columns (e.g. ` spider` minus ` insect`).
5. **Inject it** via a forward hook on `blocks.{layer}.hook_resid_post` that adds
   `coefficient * steering_vector`. Use the `model.hooks(...)` context manager around
   `model.generate`.
6. Print, side by side: baseline top-k readout vs steered top-k readout, and baseline
   generated text vs steered generated text.

**Requirements:**
- `layer`, `coefficient`, the prompt, and the steering concept must be easily configurable
  (CLI args or a config block at the top).
- Deterministic generation (greedy / `do_sample=False`) so the effect is clean to see.
- Comment the tensor shapes and the reason for each step (this is a learning tool).

**Acceptance:** I can run one command, see the top tokens a middle layer is holding, change a
concept, and watch the generated text change. Works on GPT-2 small and on Qwen2.5-1.5B.

### Stage 2 — Referee agent that reads and corrects the J-space

**Goal:** automate Stage 1. A second, small model watches the working model's J-space and
decides whether/how to intervene. Offline and slow is fine — not real-time.

Loop to build:
1. Run the working model; at the interception layer (each generation step, or at intervals),
   pause and read the J-space token list (reuse Stage 1's readout). NOTE: the referee reads
   the **decoded token list from the lens**, never the raw activation vectors — two models do
   not share an activation basis, so raw vectors are meaningless to the referee. The lens does
   the vector→words translation; the referee reasons over the words.
2. Send the original prompt + the current J-space readout to the referee (`qwen2.5:3b` via
   Ollama). Let it **reason freely first, then end with a rigid, parseable final line** — do
   NOT force a bare one-word answer with no room to think (that makes it commit from an
   unresolved state and lowers accuracy). Constrain the *format*, not the *thinking*. Example
   contract: free-text analysis, then a final line such as `DECISION: PROCEED`,
   `DECISION: STEER toward <concept>`, or `DECISION: THINK`. Python parses only that line.
3. Act on the decision:
   - **Vector steering:** translate `<concept>` into a steering vector (Stage 1 machinery)
     and inject it, then resume.
   - **Dynamic compute injection:** force the working model to generate additional reasoning
     tokens (append a reasoning delimiter / continue the loop) instead of committing, giving
     it more depth to resolve on its own.
4. Log every step: layer, J-space readout, referee decision, action taken, and the effect.

**Referee call = stateless.** Reset the conversation history on every call. Each call gets
only the curated inputs it needs (the prompt, the current readout, and — if a decision needs
history — a short *structured summary* of the log, never raw prior conversation). Python holds
the memory; the referee judges fresh. On a small model this is also more accurate: long
accumulated context degrades its judgment.

**Magnitude search = Python's job, not the referee's.** Picking the steering magnitude is the
fiddly, open part: too small does nothing, too large produces garbage ("breaks the manifold").
Do NOT ask the LLM to remember which magnitudes it tried — LLMs track numbers unreliably and
will repeat values. Instead:
- Python owns a search loop and a log file (JSONL or CSV), one row per trial: `magnitude`,
  whether the target concept appeared, a coherence/repetition score, the referee's verdict,
  and a short sample of the output. Because Python holds the tested set, it cannot retest the
  same value — loop-avoidance is free, and the log doubles as a resume/reproducibility record.
- Search **coarse-to-fine**: sweep a wide, sparse range to find where the concept starts
  appearing and where coherence collapses, then refine between those bounds (~15–25 trials).
- The referee (or a cheap automatic check) only **judges each candidate output** — "did this
  work / is it still coherent?" This pass/fail signal is the hardest, most important piece:
  combine the referee's judgment with automatic checks (target-token rank, a repetition/
  coherence score). A weak signal makes the search optimize toward the wrong thing.
- Optional, once data accumulates: build a (prompt-type → working-magnitude range) table from
  the logs and use it as a **prior** to narrow future searches. Ranges are noisy (they depend
  on layer, model, and concept), but even a rough prior is a big speedup.

Robust auto-calibration is the open research problem at the heart of this stage. Aim first for
a clear, measurable result on a chosen contradiction or loop case, then push the method toward
generality from there.

**Acceptance:** a closed offline loop where the referee's reading of the working model's
internal state measurably changes the final output on a chosen test case, with a full log.

### Stage 3 — Instrument the generation loop and push past baseline

**Goal:** use the Stage 1–2 machinery to try to make the small working model perform *better
than its untouched self* on a specific task, via inference-time intervention only (no
retraining). Here "agentic" means: give it a goal more demanding than a plain chatbot reply
and try to raise task performance — **not** tool use.

Build:
1. Instrument the generation / reasoning loop so we can watch the J-space evolve token by
   token across a longer generation, and log where it drifts or loops.
2. Apply the two levers from Stage 2 (steering away from known failure directions; injecting
   extra reasoning tokens / test-time compute) during the loop.
3. Pick a small, checkable task set (e.g. a handful of simple reasoning or format-following
   prompts with known correct answers). Run **baseline vs intervened** and report the
   difference quantitatively.

**Measure it honestly:** intervention acts at inference time — it reshapes how the model uses
what it already has (reliability, failure modes, resolving contradictions) rather than
retraining its weights. Report the real, measured delta — positive, zero, or negative — and
let the numbers, not assumptions, decide how far this can go.

**Acceptance:** a reproducible baseline-vs-intervened comparison on the task set, with an
honest written summary of whether intervention helped and by how much.

---

## 5. How to work

- **Incremental.** Get Stage 1 fully working and verified before touching Stage 2. Do not
  scaffold all three stages up front.
- **Explain as you go.** After each runnable milestone, write a short plain-language note of
  what the output shows and what it means. I am here to learn.
- **Keep it CPU-friendly and small.** Never pull a model larger than the list in section 3
  without asking me first.
- **Keep the working model swappable** via config, as described in section 3.
- **Verify before claiming.** Every stage ends with a command I can run and an expected
  result. Don't mark a stage done until that command produces that result on this machine.
- **Document new features in the README.** After implementing any new feature, flag, or
  user-facing change, add or update its description in `README.md` — what it does and a short
  example — so anyone who has the repo can see the full, current set of features without
  having to read the code.
- **Structure:** small, readable modules with a shared config; a `README.md` kept current
  with setup and run commands; a `requirements.txt` pinned to the Python 3.11/3.12 venv.

## 6. Out of scope (do NOT build now)

- No multi-agent "swarm," mixture-of-agents, or agent-factory architecture.
- No RAG / vector database.
- No web dashboard or fancy UI (simple terminal logs are enough; a minimal log viewer is
  optional and only after Stage 2 works).
- No cloud GPU, no closed-API models for the working model, no future-scope features.

Keep the whole project inside three well-understood stages on local, small, CPU-run models.
