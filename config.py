"""Shared configuration for the Stage 1 read-and-steer script.

Single source of truth for defaults: the CLI flags in `parse_args` just expose these
fields, so the model / layer / coefficient / prompt / steering concept are configurable
in one place instead of being scattered as magic numbers through the rest of the code.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

# Per-method defaults for --pos/--neg when omitted -- which pair applies is determined
# entirely by steering_method, so there's no meaningful single dataclass-field default.
_DEFAULT_CONCEPTS: dict[str, tuple[str, str]] = {
    # actadd: TransformerLens's own documented wedding-topic demo pair, known-good on gpt2.
    "actadd": ("I talk about weddings constantly", "I do not talk about weddings constantly"),
    # token_diff: CLAUDE.md's own worked example.
    "token_diff": (" spider", " insect"),
}


@dataclass
class Config:
    model_name: str = "gpt2"
    layers: list[int] | None = None  # None => auto-resolve to a single ~0.6 * n_layers pick
    coefficients: list[float] = field(default_factory=lambda: [0.0, 4.0, 8.0, 16.0])
    prompt: str = "I went up to my friend and said"
    max_new_tokens: int = 30
    top_k: int = 10
    steering_method: str = "actadd"  # "actadd" | "token_diff"
    # Interpreted per steering_method: a single token for token_diff, a full prompt for
    # actadd. None => resolved to the per-method default in __post_init__.
    pos: str | None = None
    neg: str | None = None
    observe: bool = False
    info: bool = False
    report: str | None = None  # None => no report; otherwise the HTML output path
    no_open: bool = False  # when a report is written, suppress auto-opening it in the browser

    def __post_init__(self) -> None:
        defaults = _DEFAULT_CONCEPTS.get(self.steering_method)
        if defaults is None:
            return  # unrecognized method; build_steering_vector's own check raises later
        default_pos, default_neg = defaults
        if self.pos is None:
            self.pos = default_pos
        if self.neg is None:
            self.neg = default_neg


def parse_args(argv: list[str] | None = None) -> Config:
    d = Config()
    parser = argparse.ArgumentParser(
        description="Stage 1: read a model's J-space (logit-lens readout) and steer it "
        "via activation addition or token-direction injection."
    )
    parser.add_argument("--model", dest="model_name", default=d.model_name,
                         help="TransformerLens model name, e.g. 'gpt2' or 'Qwen/Qwen2.5-1.5B-Instruct'.")
    parser.add_argument("--layer", dest="layers", type=int, nargs="+", default=d.layers,
                         help="One or more interception layers to sweep, e.g. --layer 4 7 10. "
                              "Omit to auto-pick ~0.6 * n_layers.")
    parser.add_argument("--coefficient", dest="coefficients", type=float, nargs="+", default=d.coefficients,
                         help="One or more steering coefficients to sweep in a single run, "
                              "e.g. --coefficient 0 4 8 16 30. Coefficient magnitude is a guess "
                              "until you see the effect, so sweeping is the default, not an "
                              "afterthought.")
    parser.add_argument("--prompt", default=d.prompt)
    parser.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    parser.add_argument("--top-k", type=int, default=d.top_k)
    parser.add_argument("--steering-method", choices=["actadd", "token_diff"], default=d.steering_method)
    parser.add_argument("--pos", default=None,
                         help="Positive concept. For --steering-method token_diff, a single "
                              "token (e.g. ' spider'); for actadd, a full prompt. Omit for a "
                              "sensible per-method default.")
    parser.add_argument("--neg", default=None,
                         help="Negative concept, interpreted the same way as --pos. Omit for "
                              "a sensible per-method default.")
    parser.add_argument("--observe", action="store_true", default=d.observe,
                         help="Load the model and show baseline generation + J-space readout "
                              "only. Skips steering entirely.")
    parser.add_argument("--info", action="store_true", default=d.info,
                         help="Print the model's config (layers, sizes, param count) and exit "
                              "without generating anything.")
    parser.add_argument("--report", nargs="?", const="report.html", default=d.report,
                         help="Write a self-contained HTML report after the sweep to PATH "
                              "(default report.html if no path is given). Off by default.")
    parser.add_argument("--no-open", dest="no_open", action="store_true", default=d.no_open,
                         help="When a report is written, do not auto-open it in the browser.")
    args = parser.parse_args(argv)

    return Config(
        model_name=args.model_name,
        layers=args.layers,
        coefficients=args.coefficients,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        steering_method=args.steering_method,
        pos=args.pos,
        neg=args.neg,
        observe=args.observe,
        info=args.info,
        report=args.report,
        no_open=args.no_open,
    )
