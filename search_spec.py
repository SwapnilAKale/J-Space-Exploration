"""What the search harness should run: task specs (JSON) and the equivalent CLI flags.

A *case* is one prompt plus one steering setup plus the grid of (layer, coefficient) pairs
to try on it. A *spec* is a model plus a list of cases, so a whole afternoon's backlog runs
in one process against one model load.

JSON only, deliberately -- YAML would mean adding PyYAML to `requirements.txt`, and the
dependency list on this machine is load-bearing (see the pyarrow / Smart App Control note
in `requirements.txt`). JSON is in the standard library and is already the log format.

The bridge back to Stage 1 is `to_config`: it returns a real `config.Config`, which is what
`steering.build_steering_vector(model, cfg, layer)` already takes. That is how the harness
reuses the verified steering math without owning a second copy of it -- and it inherits
`Config.__post_init__`'s per-method `pos`/`neg` defaults for free.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, fields

from config import Config
from search_log import DEFAULT_LOG_PATH

DEFAULT_MODEL = "gpt2"
# Same default grid as `run_stage1.py`, so a harness run with no --coefficient does the
# same thing the manual tool does with no --coefficient. Consistency between the two entry
# points matters more here than picking a cleverer default.
DEFAULT_COEFFICIENTS = [0.0, 4.0, 8.0, 16.0]


@dataclass
class SearchCase:
    """One prompt + steering setup + the grid to sweep over it."""
    prompt: str
    name: str = "case"
    steering_method: str = "actadd"
    pos: str | None = None  # None => Config's per-method default
    neg: str | None = None
    layers: list[int] | None = None  # None => auto-pick ~0.6 * n_layers
    coefficients: list[float] = field(default_factory=lambda: list(DEFAULT_COEFFICIENTS))
    max_new_tokens: int = 30
    top_k: int = 10
    # Both nullable on purpose (brief section 5.5): the style / mood / cross-lingual probes
    # have no ground truth, and the scorer must return null rather than invent a verdict.
    expected_answer: str | None = None
    target_concept: str | None = None
    notes: str = ""

    def to_config(self, model_name: str) -> Config:
        """A real Stage-1 Config, so steering.py can be reused unmodified."""
        return Config(
            model_name=model_name,
            layers=self.layers,
            coefficients=self.coefficients,
            prompt=self.prompt,
            max_new_tokens=self.max_new_tokens,
            top_k=self.top_k,
            steering_method=self.steering_method,
            pos=self.pos,
            neg=self.neg,
        )


_CASE_FIELDS = {f.name for f in fields(SearchCase)}


def _build_case(raw: dict, defaults: dict, index: int) -> SearchCase:
    """Merge spec-level defaults under one case's own keys and validate the result.

    Unknown keys are a hard error, not a warning: a typo like "coefficient" for
    "coefficients" in a hand-written spec would otherwise silently run the default grid and
    you would only notice hours later, looking at the wrong numbers.
    """
    merged = {**defaults, **raw}
    unknown = set(merged) - _CASE_FIELDS
    if unknown:
        raise ValueError(
            f"case {index} ({merged.get('name', 'unnamed')!r}): unknown key(s) "
            f"{sorted(unknown)}. Valid keys: {sorted(_CASE_FIELDS)}"
        )
    if "prompt" not in merged:
        raise ValueError(f"case {index}: 'prompt' is required")
    if "coefficients" in merged:
        merged["coefficients"] = [float(c) for c in merged["coefficients"]]
    if merged.get("layers") is not None:
        merged["layers"] = [int(layer) for layer in merged["layers"]]
    merged.setdefault("name", f"case{index}")
    return SearchCase(**merged)


def load_spec(path: str) -> tuple[str | None, list[SearchCase]]:
    """Read a JSON spec: {"model": ..., "defaults": {...}, "cases": [...]}.

    Returns (model_name_or_None, cases). One model per spec -- the entire point of the
    harness is a single model load, so a spec cannot mix models.
    """
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    if not isinstance(spec, dict) or "cases" not in spec:
        raise ValueError(f"{path}: expected an object with a 'cases' list")
    if any(key in spec.get("defaults", {}) for key in ("model", "model_name")):
        raise ValueError(
            f"{path}: put 'model' at the top level, not in 'defaults' -- one spec runs "
            f"against exactly one model (that is what makes the single model load possible)"
        )

    defaults = spec.get("defaults", {})
    cases = [_build_case(raw, defaults, i) for i, raw in enumerate(spec["cases"])]
    if not cases:
        raise ValueError(f"{path}: 'cases' is empty")

    # Case names key the report's per-case sections, so duplicates would silently drop a
    # whole case's results from the report even though its trials ran and were logged.
    names = [c.name for c in cases]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate case name(s) {duplicates} -- names must be unique")
    return spec.get("model"), cases


@dataclass
class SearchOptions:
    """Run-level settings -- these apply to the whole process, not to one case."""
    model_name: str
    cases: list[SearchCase]
    normalize: bool = False
    rerun: bool = False
    log_path: str = DEFAULT_LOG_PATH
    out_dir: str = "search_runs"


def parse_args(argv: list[str] | None = None) -> SearchOptions:
    parser = argparse.ArgumentParser(
        description="Stage 2a search harness: run a given grid of (layer, coefficient) "
        "steering trials in one process, score each one, and log everything. Reuses the "
        "Stage 1 steering machinery unchanged -- run_stage1.py remains the manual tool."
    )
    parser.add_argument("--spec", default=None,
                        help="JSON task spec with one or more cases. Without it, the flags "
                             "below define a single ad-hoc case.")
    # default=None (not DEFAULT_MODEL) so we can tell "user asked for gpt2" from "user said
    # nothing", and let a spec's own "model" win in the second case.
    parser.add_argument("--model", dest="model_name", default=None,
                        help=f"TransformerLens model name. Overrides a spec's 'model'. "
                             f"Default: {DEFAULT_MODEL}")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--steering-method", choices=["actadd", "token_diff"], default="actadd")
    parser.add_argument("--pos", default=None,
                        help="Positive concept: a full prompt for actadd, a single token "
                             "for token_diff. Omit for the per-method default.")
    parser.add_argument("--neg", default=None, help="Negative concept, same interpretation as --pos.")
    parser.add_argument("--layer", dest="layers", type=int, nargs="+", default=None,
                        help="Layers to sweep. Omit to auto-pick ~0.6 * n_layers.")
    parser.add_argument("--coefficient", dest="coefficients", type=float, nargs="+",
                        default=None, help=f"Coefficients to sweep. Default: "
                                           f"{' '.join(str(c) for c in DEFAULT_COEFFICIENTS)}")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--expected-answer", default=None,
                        help="Ground truth for this case. Omit when there is none -- "
                             "correctness is then logged as null, never guessed.")
    parser.add_argument("--target-concept", default=None,
                        help="The concept being steered toward, for the (weak) "
                             "target_present check and the readout-rank metric.")
    parser.add_argument("--case-name", default="cli", help="Label for this case in the log and report.")
    parser.add_argument("--normalize", action="store_true", default=False,
                        help="Unit-normalize the steering vector before scaling. OFF by "
                             "default: every coefficient recorded in FINDINGS.md was "
                             "measured un-normalized, and normalizing changes what a "
                             "coefficient means.")
    parser.add_argument("--rerun", action="store_true", default=False,
                        help="Regenerate trials already present in the log instead of "
                             "reusing the logged result.")
    parser.add_argument("--log", dest="log_path", default=DEFAULT_LOG_PATH,
                        help="Path to the accumulating JSONL log (appended, never overwritten).")
    parser.add_argument("--out-dir", default="search_runs",
                        help="Directory for the per-run Markdown report.")
    args = parser.parse_args(argv)

    if args.spec:
        spec_model, cases = load_spec(args.spec)
        model_name = args.model_name or spec_model or DEFAULT_MODEL
    else:
        if args.prompt is None:
            parser.error("give --prompt for a single ad-hoc case, or --spec for a task file")
        model_name = args.model_name or DEFAULT_MODEL
        cases = [SearchCase(
            prompt=args.prompt,
            name=args.case_name,
            steering_method=args.steering_method,
            pos=args.pos,
            neg=args.neg,
            layers=args.layers,
            coefficients=args.coefficients if args.coefficients is not None else list(DEFAULT_COEFFICIENTS),
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            expected_answer=args.expected_answer,
            target_concept=args.target_concept,
        )]

    return SearchOptions(
        model_name=model_name,
        cases=cases,
        normalize=args.normalize,
        rerun=args.rerun,
        log_path=args.log_path,
        out_dir=args.out_dir,
    )
