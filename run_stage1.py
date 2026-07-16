"""Stage 1 entry point: read a model's J-space, steer it, and show the effect.

Usage:
    python run_stage1.py --model gpt2 --coefficient 0 4 8 16
    python run_stage1.py --model Qwen/Qwen2.5-1.5B-Instruct --max-new-tokens 20

See README.md for full setup instructions and CLAUDE.md for the background on what
"J-space" and the logit lens mean here.
"""
from __future__ import annotations

import pathlib
import sys
import webbrowser

import torch
from transformer_lens import HookedTransformer

from config import parse_args
from model_setup import load_model, print_model_info, resolve_layers
from report import write_html_report
from steering import build_steering_vector, get_resid_post_at_layer, logit_lens_readout, make_steering_hook


def generate_text(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int,
    fwd_hooks: list[tuple[str, object]] | None = None,
) -> str:
    def run() -> str:
        return model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding: deterministic, so the steering effect is
                               # legible run-to-run instead of hidden inside sampling noise
            verbose=False,
            return_type="str",
        )

    with torch.no_grad():
        if fwd_hooks:
            with model.hooks(fwd_hooks=fwd_hooks):
                return run()
        return run()


def _format_row(token: str, logit: float) -> str:
    return f"{token!r:>16} {logit:7.2f}"


def print_readout_table(
    baseline: list[tuple[str, float]],
    steered: list[tuple[str, float]],
    layer: int,
    coefficient: float,
) -> None:
    print(f"\n--- J-space readout: baseline vs steered (layer={layer}, coefficient={coefficient}) ---")
    print(f"{'rank':>4}  {'baseline (token, logit)':<26} {'steered (token, logit)':<26}")
    for i, ((b_tok, b_val), (s_tok, s_val)) in enumerate(zip(baseline, steered)):
        print(f"{i:>4}  {_format_row(b_tok, b_val):<26} {_format_row(s_tok, s_val):<26}")


def print_generation(label: str, text: str) -> None:
    print(f"\n[{label}]")
    print(text)


def print_baseline_readout(readout: list[tuple[str, float]]) -> None:
    print("\n--- J-space readout (baseline) ---")
    print(f"{'rank':>4}  token, logit")
    for i, (tok, val) in enumerate(readout):
        print(f"{i:>4}  {_format_row(tok, val)}")


def main() -> None:
    cfg = parse_args()
    model = load_model(cfg.model_name)

    if cfg.info:
        print_model_info(model)
        return

    n_layers = model.cfg.n_layers
    print(f"[model] {model.cfg.model_name}: n_layers={n_layers} d_model={model.cfg.d_model}")

    if cfg.observe:
        # --observe inspects a single point, not a sweep -- use only the first requested
        # layer if more were given, rather than silently sweeping in a mode whose whole
        # point is "no steering, just look."
        if cfg.layers and len(cfg.layers) > 1:
            print(f"[layer] --observe uses a single layer; ignoring extra layers {cfg.layers[1:]}")
        resolved = resolve_layers(model, cfg.layers[:1] if cfg.layers else None)
        if not resolved:
            return
        layer = resolved[0]
        print(f"[layer] intercepting layer {layer}")

        with torch.no_grad():
            baseline_resid = get_resid_post_at_layer(model, cfg.prompt, layer)  # [1, d_model]
            baseline_readout = logit_lens_readout(model, baseline_resid, cfg.top_k)
        baseline_text = generate_text(model, cfg.prompt, cfg.max_new_tokens)

        print(f"\n=== Prompt: {cfg.prompt!r} ===")
        print_generation("BASELINE", baseline_text)
        print_baseline_readout(baseline_readout)
        return

    layers = resolve_layers(model, cfg.layers)
    if not layers:
        print("[layer] No valid layers to run -- exiting.")
        return

    print(f"\n=== Prompt: {cfg.prompt!r} ===")
    print(f"Steering method: {cfg.steering_method}")
    if cfg.steering_method == "actadd":
        print(f"  pos_prompt={cfg.pos!r}  neg_prompt={cfg.neg!r}")
    else:
        print(f"  pos_token={cfg.pos!r}  neg_token={cfg.neg!r}")

    # Baseline generation has no hook attached, so it's identical regardless of which
    # layer(s) get steered below -- compute and print it once, outside the layer loop.
    baseline_text = generate_text(model, cfg.prompt, cfg.max_new_tokens)
    print_generation("BASELINE", baseline_text)

    # Only built when a report was actually requested -- zero cost otherwise, and pure
    # bookkeeping alongside the existing prints below, so terminal output is unaffected.
    report_data: dict[float, dict[int, dict]] = {}

    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        hook_name = f"blocks.{layer}.hook_resid_post"

        with torch.no_grad():
            # Unlike baseline_text, the baseline READOUT does depend on layer -- it reads
            # resid_post at this specific point in the network, so it's recomputed per layer.
            baseline_resid = get_resid_post_at_layer(model, cfg.prompt, layer)  # [1, d_model]
            baseline_readout = logit_lens_readout(model, baseline_resid, cfg.top_k)

        steering_vector = build_steering_vector(model, cfg, layer)  # [d_model]

        # Coefficient is a guess until you see the effect on THIS model/layer/prompt, so
        # sweeping several values in one run is the default (see config.py's --coefficient
        # nargs='+'), not something you bolt on after the first run looks wrong.
        for coefficient in cfg.coefficients:
            with torch.no_grad():
                # Adding coefficient * steering_vector to the already-cached baseline_resid
                # is mathematically identical to what the hook below does at that one point
                # in the network, so the readout doesn't need a second hooked forward pass
                # to show the shift.
                steered_resid = baseline_resid + coefficient * steering_vector
                steered_readout = logit_lens_readout(model, steered_resid, cfg.top_k)
            print_readout_table(baseline_readout, steered_readout, layer, coefficient)

            hook_fn = make_steering_hook(steering_vector, coefficient)
            steered_text = generate_text(model, cfg.prompt, cfg.max_new_tokens, fwd_hooks=[(hook_name, hook_fn)])
            print_generation(f"STEERED (layer={layer}, coefficient={coefficient})", steered_text)

            if cfg.report is not None:
                report_data.setdefault(coefficient, {})[layer] = {
                    "baseline_readout": baseline_readout,
                    "steered_readout": steered_readout,
                    "steered_text": steered_text,
                }

    if cfg.report is not None:
        write_html_report(
            path=cfg.report,
            cfg=cfg,
            model_name=model.cfg.model_name,
            baseline_text=baseline_text,
            layers=layers,
            report_data=report_data,
        )
        print(f"Report written to {cfg.report}")
        if not cfg.no_open:
            # Open the finished report in the default browser so the user sees the result
            # immediately once the run completes. The file:// URI form is the reliable
            # cross-platform way to hand a local path to webbrowser.
            webbrowser.open(pathlib.Path(cfg.report).resolve().as_uri())


if __name__ == "__main__":
    # Windows consoles often default to a legacy codepage (e.g. cp1252), which can't
    # represent every token a tokenizer might decode to (seen with Qwen2.5's vocab).
    # Force UTF-8 output so the readout table never crashes on an unusual token.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except ValueError as e:
        # A clear one-line error instead of a raw traceback -- e.g. an unusable --pos/--neg
        # value for the chosen --steering-method.
        print(f"Error: {e}")
        sys.exit(1)
