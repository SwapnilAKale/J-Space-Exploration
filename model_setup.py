"""Load the working model and resolve which middle layer to intercept.

Kept separate from steering.py: this module is pure model/config plumbing (no
interpretability logic), so the "swappable model" contract -- read cfg.n_layers /
cfg.d_model at runtime, never hardcode them -- is easy to verify by reading this file
alone.
"""
from __future__ import annotations

from transformer_lens import HookedTransformer

# Hard hardware constraint (CLAUDE.md section 2): Intel UHD integrated GPU only, no CUDA.
# This is intentionally not a CLI flag -- there is no correct value for it but "cpu" here.
DEVICE = "cpu"

# Middle layer, inside the paper's ~0.5-0.65 x n_layers "J-space" band.
DEFAULT_LAYER_FRACTION = 0.6


def load_model(model_name: str) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(model_name, device=DEVICE)
    model.eval()  # disable dropout so greedy generation is exactly reproducible run to run
    return model


def auto_layer(n_layers: int) -> int:
    """The layer --layer auto-picks when not overridden: ~0.6 * n_layers, clamped in range."""
    return max(0, min(round(DEFAULT_LAYER_FRACTION * n_layers), n_layers - 1))


def resolve_layers(model: HookedTransformer, layers_override: list[int] | None) -> list[int]:
    """Resolve the final list of interception layers to run.

    None => a single auto-picked layer (~0.6 * n_layers), same as when --layer is omitted.
    A list => each requested layer is checked against the model's real layer count; an
    out-of-range layer is reported (with the valid range) and skipped, rather than crashing
    the whole sweep over one bad value.
    """
    n_layers = model.cfg.n_layers
    if layers_override is None:
        return [auto_layer(n_layers)]

    resolved: list[int] = []
    for layer in layers_override:
        if 0 <= layer < n_layers:
            resolved.append(layer)
        else:
            print(
                f"[layer] --layer {layer} out of range for {model.cfg.model_name} "
                f"(valid range: 0..{n_layers - 1}) -- skipping"
            )
    return resolved


def print_model_info(model: HookedTransformer) -> None:
    """Print the loaded model's specs as a labeled list -- everything pulled from model.cfg
    (never hardcoded), except the total parameter count, which model.cfg does NOT give you
    accurately: model.cfg.n_params is a partial estimate (attention projections, optionally
    + MLP; it excludes embeddings/unembedding, biases, and layernorm), computed by a fixed
    formula inside TransformerLens. The real total is summed directly from the loaded
    model's own tensors instead.
    """
    cfg = model.cfg
    n_layers = cfg.n_layers
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n=== Model info: {cfg.model_name} ===")
    print(f"device:            {cfg.device}")
    print(f"n_layers:          {n_layers}")
    print(f"d_model:           {cfg.d_model}")
    print(f"n_heads:           {cfg.n_heads}")
    print(f"d_vocab:           {cfg.d_vocab}")
    print(f"n_ctx:             {cfg.n_ctx}")
    print(f"--layer range:     0..{n_layers - 1}")
    print(f"--layer auto-pick: {auto_layer(n_layers)}")
    print(f"total parameters:  {total_params:,}")
