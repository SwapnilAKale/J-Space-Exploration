# J-Space Exploration — Stage 1

Reach inside a small language model, read what a middle layer is "thinking" (the J-space,
via a logit-lens readout), inject a steering vector, and watch the generated text change.
Background and the full three-stage plan: see [CLAUDE.md](CLAUDE.md). **New here?** After
setup, skim [FINDINGS.md](FINDINGS.md) — it explains the non-obvious behaviour (why the
readout can disagree with the output, why Qwen sometimes prints Chinese, why the same
coefficient behaves differently on different models) so you don't have to rediscover it.

Stage 1 only. CPU-only (no CUDA) — see [CLAUDE.md](CLAUDE.md) section 2 for hardware
constraints.

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

`--device` is intentionally not a flag: this project is CPU-only by hardware constraint,
hardcoded in `model_setup.py`.

## Files

- `config.py` — `Config` dataclass + CLI parsing; single source of truth for defaults.
- `model_setup.py` — loads the model, resolves the interception layer from its own config.
- `steering.py` — logit-lens readout, both steering-vector builders, the injection hook.
- `report.py` — builds the self-contained HTML report (and its Markdown export).
- `run_stage1.py` — entry point: orchestrates baseline vs steered, prints results.
- `CLAUDE.md` — project brief, three-stage plan, and research background.
- `FINDINGS.md` — non-obvious behaviour learned from running the tool (read this early).
