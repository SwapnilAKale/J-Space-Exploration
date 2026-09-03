# J-Space Exploration — Stage 1

Reach inside a small language model, read what a middle layer is "thinking" (the J-space,
via a logit-lens readout), inject a steering vector, and watch the generated text change.
Background and the full three-stage plan: see [CLAUDE.md](CLAUDE.md). **New here?** After
setup, skim [FINDINGS.md](FINDINGS.md) — it explains the non-obvious behaviour (why the
readout can disagree with the output, why Qwen sometimes prints Chinese, why the same
coefficient behaves differently on different models) so you don't have to rediscover it.
For copy-paste demos with the exact output to expect, see
[VERIFIED_RUNS.md](VERIFIED_RUNS.md).

Stage 1, plus the search harness (`run_search.py`) that runs Stage 1 experiments in bulk.
CPU-only (no CUDA) — see [CLAUDE.md](CLAUDE.md) section 2 for hardware constraints.

## Setup (Windows, PowerShell)

Requires a dedicated **Python 3.11 or 3.12** virtual environment — PyTorch/TransformerLens
do not yet support the newer Python that may already be on your machine.

```powershell
# 1. Install Python 3.12 if you don't have it (skip if already installed)
winget install Python.Python.3.12 --source winget --accept-source-agreements --accept-package-agreements

# 2. Create and activate the venv
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies — CPU-only torch FIRST, as its own command
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Smoke-test: confirms imports work and prints gpt2's real config
python -c "import torch, transformer_lens; from transformer_lens import HookedTransformer; m = HookedTransformer.from_pretrained('gpt2', device='cpu'); print(torch.__version__, m.cfg.n_layers, m.cfg.d_model)"
```

Expected smoke-test output ends with `12 768` (gpt2's n_layers and d_model).

### If step 4 fails with `DLL load failed ... Application Control policy has blocked this file`

This is Windows **Smart App Control** blocking `pyarrow`'s native extension as an
unrecognized binary — not a bug in this project. `requirements.txt` already pins
`pyarrow==21.0.0`, a version old enough to be allowed and new enough for `datasets`
(a `transformer_lens` dependency this project never actually uses, but which is imported
eagerly). If that pin is *also* blocked on your machine, try adjacent pyarrow versions:
```powershell
pip install "pyarrow==<other-version>"
python -c "import pyarrow"   # confirm no DLL error before retrying the smoke test
```
Smart App Control cannot be selectively configured to allow one file — turning it off
entirely is a one-way, system-wide security change (Microsoft: once off, it can't be turned
back on without reinstalling Windows), so pin-hunting for an allowed pyarrow version is the
right first move, not disabling the policy.

## Running Stage 1

```powershell
# gpt2, default settings: sweeps coefficients [0, 4, 8, 16] with the wedding-topic ActAdd pair
python run_stage1.py

# Explicit coefficient sweep and a custom prompt
python run_stage1.py --model gpt2 --prompt "I went up to my friend and said" --coefficient 0 4 8 16 30

# "Point at a token" steering method (CLAUDE.md's spider/insect example)
python run_stage1.py --model gpt2 --steering-method token_diff --pos " spider" --neg " insect"

# Layer sweep: compare the same steering vector injected at several depths in one run
python run_stage1.py --model gpt2 --layer 4 7 10 --coefficient 8 --steering-method token_diff --pos " spider" --neg " insect"

# Write a self-contained HTML report after the sweep (default report.html; pass a path to override)
python run_stage1.py --model gpt2 --layer 4 7 --coefficient 0 8 --steering-method token_diff --pos " spider" --neg " insect" --report

# Qwen2.5-1.5B-Instruct — slower (a few tokens/sec on CPU), first run downloads the weights
python run_stage1.py --model Qwen/Qwen2.5-1.5B-Instruct --max-new-tokens 20

# Model specs only (layer count, sizes, param count) -- no generation
python run_stage1.py --info --model Qwen/Qwen2.5-1.5B-Instruct

# Baseline generation + J-space readout only -- no steering
python run_stage1.py --observe --model gpt2 --prompt "The capital of France is"
```

Each run prints, per coefficient in the sweep: the baseline vs steered top-k J-space
readout (logit-lens tokens) side by side, then the baseline vs steered generated text.
`--coefficient 0` should exactly reproduce the baseline — a built-in sanity check that the
steering hook is truly additive.

### What each part of a command means (plain language)

Take this command:

```powershell
python run_stage1.py --model gpt2 --prompt "My favourite sport is" --steering-method token_diff --pos " rugby" --neg " football" --coefficient 0 4 8 --max-new-tokens 20
```

Piece by piece:

- `python run_stage1.py` — runs the script.
- `--model gpt2` — which model to load and inspect. `gpt2` is tiny and instant; `Qwen/Qwen2.5-1.5B-Instruct` is bigger, smarter, and much slower on CPU.
- `--prompt "My favourite sport is"` — the text the model continues. For gpt2, give it the *start of a sentence* that leads to your answer, not a command like "Choose a sport" (gpt2 only completes text, it doesn't obey instructions).
- `--steering-method token_diff` — how the "nudge" is built. `token_diff` = from two single words; `actadd` = from two full sentences.
- `--pos " rugby"` — the word/idea to push *toward*. The leading space matters (it's part of the token).
- `--neg " football"` — the word/idea to push *away from* (usually whatever the model says on its own).
- `--coefficient 0 4 8` — how hard to push, tried at several strengths in one run. `0` = no push (a baseline sanity check). There is no universal right number, so you sweep several.
- `--max-new-tokens 20` — how many words to generate after the prompt.

Anything you don't set falls back to a default (see the table below), so `python run_stage1.py`
on its own runs a full demo on gpt2.

**A good first-time workflow:**
1. `python run_stage1.py --observe --model gpt2 --prompt "My favourite sport is"` — see what the model says on its own, no steering.
2. Pick a word to push toward, then steer: add `--steering-method token_diff --pos " rugby" --neg " football" --coefficient 0 2 4 6 8`.
3. Read the **STEERED generation text** (not the readout table) to judge whether it worked — and push the coefficient higher to find where it breaks.

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `gpt2` | Any TransformerLens model name, e.g. `Qwen/Qwen2.5-1.5B-Instruct` |
| `--layer` | auto (~0.6 × n_layers) | One or more layers to sweep, e.g. `--layer 4 7 10`. Out-of-range layers are skipped with a message, not a crash |
| `--coefficient` | `0 4 8 16` | One or more steering magnitudes, swept in a single run |
| `--prompt` | *(see config.py)* | The prompt to complete |
| `--max-new-tokens` | `30` | Generation length |
| `--top-k` | `10` | How many tokens to show in the readout |
| `--steering-method` | `actadd` | `actadd` (prompt-pair difference) or `token_diff` (two `W_U` columns) |
| `--pos` / `--neg` | per-method (wedding-topic pair for `actadd`, `" spider"`/`" insect"` for `token_diff`) | A single token for `token_diff` (must be exactly one BPE token, or you get a clear error), a full prompt for `actadd` |
| `--observe` | off | **Standalone mode — run on its own.** Print baseline generation + J-space readout only; skips steering entirely |
| `--info` | off | **Standalone mode — run on its own.** Print the model's specs (layers, sizes, param count) and exit; no generation |
| `--report [PATH]` | off | **Add-on — append to a normal run.** Write a self-contained HTML report after the sweep (default `report.html`) and auto-open it in your browser. Sweep-only; ignored by `--observe`/`--info` |
| `--no-open` | off | **Add-on — append alongside `--report`.** When a report is written, do NOT auto-open it in the browser |

### Two kinds of flags — this trips people up

Most flags above are **settings you combine on one steering run** (`--model`, `--prompt`,
`--coefficient`, `--layer`, `--steering-method`, `--pos`/`--neg`, `--max-new-tokens`,
`--top-k`). A few behave differently and are easy to misuse:

**`--observe` and `--info` are standalone modes.** Run each on its own — it *replaces* the
steering sweep, and steering flags are ignored. You don't append these to a normal run:

```powershell
# Just print the model's specs, then exit (no generation at all)
python run_stage1.py --info --model gpt2

# Just show the baseline generation + J-space readout (no steering)
python run_stage1.py --observe --model gpt2 --prompt "My favourite sport is"
```

**`--report` and `--no-open` are add-ons.** Append them to a normal steering run — they don't
run anything on their own:

```powershell
# Do the sweep, then also write + auto-open the HTML report
python run_stage1.py --model gpt2 --coefficient 0 4 8 --report

# Same, but don't pop open a browser tab
python run_stage1.py --model gpt2 --coefficient 0 4 8 --report --no-open
```

The report opens in your browser as a self-contained HTML page: per coefficient, it shows each
layer's baseline-vs-steered readout and steered generation side by side. It also has a
**Download as Markdown** button that saves the whole report as a `report.md` file — preserving
the exact output tokens (including odd characters like `|`, `_`, or CJK), so it's safe to paste
into notes or GitHub.

`--device` is intentionally not a flag: this project is CPU-only by hardware constraint,
hardcoded in `model_setup.py`.

## Running a search (`run_search.py`)

`run_stage1.py` runs one experiment per typed command, and each command reloads the model —
on CPU with Qwen that load is most of the wall-clock time. `run_search.py` is the **runner**
for the same experiment: it loads the model **once**, walks a whole grid of
`(layer, coefficient)` pairs, scores each trial automatically, and writes both a machine log
and a readable report.

It adds **no new steering capability**. It imports the same `steering.py` and the same
generation call `run_stage1.py` uses, so on identical inputs the two produce byte-identical
text (greedy decoding is deterministic). `run_stage1.py` is unchanged and remains the manual
tool.

```powershell
# A quick grid on gpt2: 2 layers x 3 coefficients = 6 trials, one model load
python run_search.py --model gpt2 --prompt "My favourite sport is" --steering-method token_diff --pos " rugby" --neg " football" --layer 4 7 --coefficient 0 4 8 --target-concept rugby

# Several cases from a task file (see example_spec.json), still one model load
python run_search.py --spec example_spec.json

# With ground truth: score the answer, not just the presence of a word
python run_search.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "<|im_start|>user`nWhat is 7 times 8?<|im_end|>`n<|im_start|>assistant`n" --steering-method actadd --pos "7 times 8 is 20" --neg "7 times 8 is 56" --layer 14 --coefficient 0.4 0.6 0.8 0.9 0.95 1.0 1.2 --expected-answer 56 --target-concept 20
```

### What it writes

- **`search_log.jsonl` — the source of truth.** One JSON object per trial, **appended across
  every run** (never overwritten). Carries the full output text, both readouts with logits,
  the norms, every metric, and the config that produced it. The harness reads this file back,
  which is where resume comes from: a `(layer, coefficient)` pair that is already logged is
  **not regenerated** — the logged result is reused and marked `cached` in the report. Pass
  `--rerun` to force fresh generation.
- **`search_runs/search_report_<run_id>.md` — the reading document.** One per run, generated
  from the JSONL, never parsed by anything. Run header → summary table of short safe fields →
  **every trial's complete output verbatim in a fenced code block** with its metrics and a
  one-line top-3 readout → a "flagged for human review" section. If all you care about is
  what the model actually said, this file is enough on its own.

### How trials are scored

Several independent metrics, deliberately **never collapsed into one number** — they
disagree, and the disagreement is the information:

| Metric | What it means |
|---|---|
| `answer_extracted` | The number after the last `=`, else after the last `is`, else the last number. The answer is **parsed, not grepped for** |
| `answer_correct` | Compared against `--expected-answer`. `null` when there is no expected answer, or when nothing parsed — `null` means *not judged*, never *wrong* |
| `target_present` | Substring check for `--target-concept`. A deliberately **weak** signal: every failure in `FINDINGS.md` #32 contains its target |
| `operands_altered` | The output computes on operands the prompt never gave (`7 x 10` for a `7 x 8` prompt) — the `FINDINGS.md` #31 failure mode |
| `repetition_score`, `max_repeat_run` | Degeneracy, two ways: diffuse trigram repetition, and the longest back-to-back loop |
| `target_rank_in_readout` | Where the target sits in the steered top-k readout |
| `needs_human_review` | Set when the metrics **disagree**, or when there is no ground truth. Comes with the reasons |

Metrics are computed on the **generated continuation only** (the echoed prompt is stripped
first), while the log and report keep the full output text.

**No ground truth is a supported case, not an error.** Style, mood and cross-lingual probes
have no correct answer: those trials get `null` correctness fields, keep the metrics that do
apply (repetition, readout, norms), and are flagged for your judgement. The harness's
contribution there is execution and logging, not judgement.

Every run also prints `‖vec‖`, `‖resid‖` and their ratio per layer — the scale-free numbers
behind `FINDINGS.md` #1 (why an `actadd` window is roughly layer-invariant) and #30 (readout
saturation when the injection swamps the residual).

`python search_scoring.py` runs the scorer's self-check against real outputs from
`FINDINGS.md` — no model load, about a second.

### Task spec format

A spec runs several cases against **one** model in a single process. `defaults` are merged
under each case; per-case keys win. Any unknown key is a hard error rather than a silent
default. See `example_spec.json`:

```json
{
  "model": "gpt2",
  "defaults": { "layers": [4, 7], "coefficients": [0, 4, 8], "max_new_tokens": 20 },
  "cases": [
    {
      "name": "rugby_token_diff",
      "prompt": "My favourite sport is",
      "steering_method": "token_diff",
      "pos": " rugby",
      "neg": " football",
      "target_concept": "rugby",
      "expected_answer": null,
      "notes": "free text, copied into the log and report"
    }
  ]
}
```

### `run_search.py` flags

| Flag | Default | Meaning |
|---|---|---|
| `--spec PATH` | none | JSON task file with one or more cases. Without it, the flags below define a single ad-hoc case |
| `--model` | `gpt2` | TransformerLens model name. Overrides a spec's own `model` |
| `--prompt` | *(required without `--spec`)* | The prompt to run |
| `--layer` / `--coefficient` | auto layer, `0 4 8 16` | The grid to sweep. Same meaning as in `run_stage1.py` |
| `--steering-method`, `--pos`, `--neg` | `actadd`, per-method defaults | Same as `run_stage1.py` |
| `--max-new-tokens` / `--top-k` | `30` / `10` | Same as `run_stage1.py` |
| `--expected-answer` | none | Ground truth for this case. Omit when there is none — correctness is then `null`, never guessed |
| `--target-concept` | none | The concept steered toward, for `target_present` and the readout rank |
| `--case-name` | `cli` | Label for this case in the log and report |
| `--normalize` | **off** | Unit-normalize the steering vector before scaling. Off by default: every coefficient in `FINDINGS.md` was measured un-normalized, so normalizing changes what a coefficient *means* |
| `--rerun` | off | Regenerate trials already in the log instead of reusing them |
| `--log PATH` | `search_log.jsonl` | The accumulating JSONL log (appended, never overwritten) |
| `--out-dir DIR` | `search_runs` | Where the per-run Markdown report goes |

**Not in this step (it is step 2b):** adaptive / coarse-to-fine search, drill-down between
brackets, and any referee LLM. You hand `run_search.py` the grid; it runs the grid.

## Files

- `config.py` — `Config` dataclass + CLI parsing; single source of truth for defaults.
- `model_setup.py` — loads the model, resolves the interception layer from its own config.
- `steering.py` — logit-lens readout, both steering-vector builders, the injection hook.
- `report.py` — builds the self-contained HTML report (and its Markdown export).
- `run_stage1.py` — entry point: orchestrates baseline vs steered, prints results.
- `run_search.py` — entry point: runs a whole `(layer, coefficient)` grid on one model load.
- `search_spec.py` — task-spec dataclass, JSON loader, and the search CLI.
- `search_scoring.py` — the automatic metrics (`python search_scoring.py` self-checks them).
- `search_log.py` — append-safe JSONL log, and reading it back for resume.
- `search_report.py` — builds the per-run Markdown search report.
- `example_spec.json` — a runnable two-case task spec.
- `CLAUDE.md` — project brief, three-stage plan, and research background.
- `FINDINGS.md` — non-obvious behaviour learned from running the tool (read this early).
- `VERIFIED_RUNS.md` — copy-paste demo commands with the exact results we got (gpt2 + Qwen sweet spots).
