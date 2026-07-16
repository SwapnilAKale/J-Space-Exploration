# J-Space Exploration — Stage 1

Reach inside a small language model, read what a middle layer is "thinking" (the J-space,
via a logit-lens readout), inject a steering vector, and watch the generated text change.
Background and the full three-stage plan: see [CLAUDE.md](CLAUDE.md).

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
| `--observe` | off | Print baseline generation + J-space readout only; skips steering entirely |
| `--info` | off | Print the model's specs (layers, sizes, param count) and exit; no generation |
| `--report [PATH]` | off | Write a self-contained HTML report after the sweep (default `report.html`). Sweep-only; ignored by `--observe`/`--info` |

`--device` is intentionally not a flag: this project is CPU-only by hardware constraint,
hardcoded in `model_setup.py`.

## Files

- `config.py` — `Config` dataclass + CLI parsing; single source of truth for defaults.
- `model_setup.py` — loads the model, resolves the interception layer from its own config.
- `steering.py` — logit-lens readout, both steering-vector builders, the injection hook.
- `run_stage1.py` — entry point: orchestrates baseline vs steered, prints results.
