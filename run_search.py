"""Stage 2a entry point: run a GIVEN grid of steering trials in one process, score, log.

Usage:
    python run_search.py --model gpt2 --prompt "My favourite sport is" \
        --steering-method token_diff --pos " rugby" --neg " football" \
        --layer 4 7 --coefficient 0 4 8 --target-concept rugby
    python run_search.py --spec my_probes.json

What this is: a **runner**, not a new capability. It performs exactly the experiment
`run_stage1.py` performs -- same steering vector, same hook, same greedy generation -- but
loads the model once and walks the whole (layer x coefficient) grid, scoring and recording
each trial. `run_stage1.py` is deliberately untouched and remains the manual tool; if the
two ever disagree on the same inputs, this file is wrong.

What this is NOT (that is step 2b): adaptive or coarse-to-fine search, drill-down between
brackets, a referee LLM. You hand it the grid; it runs the grid.

See README.md for the flags and the spec format, and notes/BUILD_2A_BRIEF.md for why the
scoring rules are what they are.
"""
from __future__ import annotations

import sys
import time

import torch
from transformer_lens import HookedTransformer

from model_setup import load_model, resolve_layers
from run_stage1 import generate_text  # imported, never copied -- see module docstring
from search_log import append_trial, load_index, new_run_id, timestamp, trial_key
from search_report import write_search_report
from search_scoring import score_trial
from search_spec import SearchCase, SearchOptions, parse_args
from steering import (
    build_steering_vector,
    get_resid_post_at_layer,
    logit_lens_readout,
    make_steering_hook,
)


def _identity(opts: SearchOptions, case: SearchCase, layer: int, coefficient: float) -> dict:
    """The fields that identify a trial -- see search_log.KEY_FIELDS for why these ones."""
    return {
        "model": opts.model_name,
        "prompt": case.prompt,
        "steering_method": case.steering_method,
        "pos": case.pos,
        "neg": case.neg,
        "max_new_tokens": case.max_new_tokens,
        "normalize": opts.normalize,
        "layer": layer,
        "coefficient": coefficient,
    }


def completion_only(model: HookedTransformer, prompt: str, output_text: str) -> str:
    """The text the model GENERATED, with the echoed prompt stripped off the front.

    `model.generate(return_type="str")` returns prompt + continuation, and scoring the whole
    thing reads the prompt's own words as if the model had said them. That is not
    hypothetical: at coefficient 0.95 the model answered `"19"`, and the answer extractor
    picked up the `"is 7"` inside `"What is 7 times 8?"` instead. Every metric here must see
    the continuation only.

    The prompt is re-rendered through the tokenizer rather than string-compared directly,
    because generate decodes with `skip_special_tokens=True`: a ChatML prompt comes back
    without its `<|im_start|>` markers, so the raw prompt string is NOT a prefix of the
    output. `output_text` itself is still logged whole and unmodified.
    """
    rendered = model.tokenizer.decode(model.to_tokens(prompt)[0], skip_special_tokens=True)
    if output_text.startswith(rendered):
        return output_text[len(rendered):]
    # Fallback for any other decode convention: drop the longest common prefix. It can
    # never remove more than the prompt itself, so this under-strips rather than over-strips.
    i = 0
    while i < min(len(rendered), len(output_text)) and rendered[i] == output_text[i]:
        i += 1
    return output_text[i:]


def _progress(row: dict) -> None:
    """One line per trial, so a slow Qwen sweep is legible while it runs."""
    flag = "  [REVIEW]" if row["needs_human_review"] else ""
    source = "cached" if row.get("cached") else f"{row['gen_seconds']:.1f}s"
    rep = row["repetition_score"]
    print(
        f"[trial] {row['case_name']} layer={row['layer']} coeff={row['coefficient']}"
        f" -> answer={row['answer_extracted']} correct={row['answer_correct']}"
        f" target={row['target_present']} rep={'--' if rep is None else f'{rep:.2f}'}"
        f" ({source}){flag}"
    )


def run_case(
    model: HookedTransformer,
    opts: SearchOptions,
    case: SearchCase,
    index: dict[tuple, dict],
    run_id: str,
) -> tuple[list[dict], str]:
    """Run one case's whole grid. Returns (rows, baseline_text)."""
    cfg = case.to_config(opts.model_name)
    # Config.__post_init__ fills the per-method pos/neg defaults; mirror them back so the
    # log and report record what was ACTUALLY used, not a null the reader has to decode.
    case.pos, case.neg = cfg.pos, cfg.neg

    layers = resolve_layers(model, case.layers)
    if not layers:
        print(f"[case] {case.name}: no valid layers -- skipping")
        return [], ""

    print(f"\n=== Case: {case.name} ===")
    print(f"prompt: {case.prompt!r}")
    print(f"steering: {case.steering_method}  pos={case.pos!r}  neg={case.neg!r}")
    print(f"grid: layers {layers} x coefficients {case.coefficients}")

    # Which trials actually need generating. Computed up front so that a fully-cached case
    # skips the baseline generation and the per-layer forward passes too -- otherwise a
    # "resume" run would still pay most of its CPU cost for nothing.
    to_run = {
        (layer, coefficient)
        for layer in layers
        for coefficient in case.coefficients
        if opts.rerun or trial_key(_identity(opts, case, layer, coefficient)) not in index
    }

    if to_run:
        baseline_text = generate_text(model, case.prompt, case.max_new_tokens)
    else:
        # Every trial is cached, so the baseline is too -- it is stored on each row.
        any_key = trial_key(_identity(opts, case, layers[0], case.coefficients[0]))
        baseline_text = index[any_key].get("baseline_text", "")
        print("[case] every trial already logged -- reusing logged results, nothing regenerated")

    rows: list[dict] = []
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"
        layer_needs_work = any((layer, c) in to_run for c in case.coefficients)

        if layer_needs_work:
            with torch.no_grad():
                # Same reads Stage 1 does: resid_post at the prompt's last position, then
                # the logit lens over it. Kept inside no_grad because nothing here trains.
                baseline_resid = get_resid_post_at_layer(model, case.prompt, layer)  # [1, d_model]
                baseline_readout = logit_lens_readout(model, baseline_resid, case.top_k)
                resid_norm = float(baseline_resid.norm())

                steering_vector = build_steering_vector(model, cfg, layer)  # [d_model]
                vec_norm_raw = float(steering_vector.norm())
                if opts.normalize:
                    # OFF by default: every coefficient in FINDINGS.md was measured with an
                    # un-normalized vector, so normalizing silently changes what a
                    # coefficient means and breaks comparability with all of it.
                    steering_vector = steering_vector / vec_norm_raw
                vec_norm = float(steering_vector.norm())

            # ||vec|| / ||resid|| is the scale-free quantity behind FINDINGS.md #1 (actadd
            # self-scaling) and #30 (readout saturation when the injection swamps the
            # residual). Printed as well as logged because it explains a whole layer's
            # behaviour at a glance.
            print(f"[norms] layer {layer}: ||vec||={vec_norm:.3f} ||resid||={resid_norm:.3f} "
                  f"ratio={vec_norm / resid_norm:.4f}")

        for coefficient in case.coefficients:
            identity = _identity(opts, case, layer, coefficient)
            key = trial_key(identity)

            if (layer, coefficient) not in to_run:
                row = dict(index[key])
                row["cached"] = True
                rows.append(row)
                _progress(row)
                continue

            with torch.no_grad():
                # Adding to the already-cached baseline resid is exactly what the hook does
                # at this point in the network (run_stage1.py does the same), so no second
                # hooked forward pass is needed just to read the shifted disposition.
                steered_resid = baseline_resid + coefficient * steering_vector
                steered_readout = logit_lens_readout(model, steered_resid, case.top_k)

            started = time.perf_counter()
            hook_fn = make_steering_hook(steering_vector, coefficient)
            output_text = generate_text(
                model, case.prompt, case.max_new_tokens, fwd_hooks=[(hook_name, hook_fn)]
            )
            gen_seconds = time.perf_counter() - started

            completion = completion_only(model, case.prompt, output_text)
            metrics = score_trial(
                prompt=case.prompt,
                output_text=completion,
                expected_answer=case.expected_answer,
                target_concept=case.target_concept,
                readout_steered=steered_readout,
            )

            row = {
                "run_id": run_id,
                "timestamp": timestamp(),
                "case_name": case.name,
                **identity,
                "model_resolved": model.cfg.model_name,
                "expected_answer": case.expected_answer,
                "target_concept": case.target_concept,
                "output_text": output_text,  # full and unmodified, prompt echo included
                "completion": completion,  # what the metrics below were actually computed on
                "baseline_text": baseline_text,
                "readout_baseline": baseline_readout,
                "readout_steered": steered_readout,
                "vec_norm_raw": vec_norm_raw,
                "vec_norm": vec_norm,
                "resid_norm": resid_norm,
                "norm_ratio": vec_norm / resid_norm,
                # What actually hits the residual stream at this coefficient -- the
                # quantity FINDINGS.md #30's saturation prediction is really about.
                "effective_ratio": abs(coefficient) * vec_norm / resid_norm,
                **metrics,
                "gen_seconds": gen_seconds,
                "notes": case.notes,
                "cached": False,
            }
            append_trial(opts.log_path, row)
            rows.append(row)
            _progress(row)

    return rows, baseline_text


def main() -> None:
    opts = parse_args()
    run_id = new_run_id()

    # The whole point of this entry point: ONE load, then the entire grid. Every
    # run_stage1.py invocation pays this cost again, and on CPU with Qwen it dominates.
    print(f"[model] loading {opts.model_name} (once for this whole run)...")
    model = load_model(opts.model_name)
    print(f"[model] {model.cfg.model_name}: n_layers={model.cfg.n_layers} d_model={model.cfg.d_model}")

    index = load_index(opts.log_path)
    print(f"[log] {opts.log_path}: {len(index)} trial(s) already recorded"
          f"{' -- --rerun given, all will be regenerated' if opts.rerun else ''}")

    rows_by_case: dict[str, list[dict]] = {}
    baselines: dict[str, str] = {}
    for case in opts.cases:
        rows, baseline_text = run_case(model, opts, case, index, run_id)
        if rows:
            rows_by_case[case.name] = rows
            baselines[case.name] = baseline_text

    if not rows_by_case:
        print("\nNo trials ran -- nothing to report.")
        return

    path = write_search_report(
        out_dir=opts.out_dir,
        run_id=run_id,
        model_name=opts.model_name,
        normalize=opts.normalize,
        cases=[c for c in opts.cases if c.name in rows_by_case],
        rows_by_case=rows_by_case,
        baselines=baselines,
    )
    total = sum(len(r) for r in rows_by_case.values())
    flagged = sum(1 for rows in rows_by_case.values() for r in rows if r["needs_human_review"])
    print(f"\n{total} trial(s), {flagged} flagged for human review.")
    print(f"Log:    {opts.log_path}  (appended)")
    print(f"Report: {path}")


if __name__ == "__main__":
    # Same reason as run_stage1.py: a legacy Windows console codepage cannot represent
    # every token Qwen's vocabulary decodes to, and a print must never kill a long sweep.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
