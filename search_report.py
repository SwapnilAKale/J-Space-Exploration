"""Render one search run as `search_report_<run_id>.md` -- the reading document.

This file is a **generated view** of `search_log.jsonl`, never a source of truth: nothing in
the system parses it, and it can be regenerated from the log at any time. It exists because
the JSONL is unreadable by eye, and because most of this project's findings came from a
human *reading the generated text* rather than from looking at a score.

Hence the one hard requirement (build brief section 6.2): **if all you care about is what
the model actually said, this file is sufficient on its own.**

Two formatting rules follow from real outputs containing `|`, newlines, CJK and LaTeX:
- generated text always goes inside a fenced code block, where every character is literal;
- the summary table carries only short, safe fields -- numbers and flags, never text.

`_md_fence` and `_md_token_cell` are imported from `report.py` rather than reimplemented:
they already solve "what if the text contains a backtick run" and "what if a token is `|`",
and a second copy of that logic would drift.
"""
from __future__ import annotations

import os

from report import _md_fence, _md_token_cell
from search_scoring import NO_GROUND_TRUTH_REASON
from search_spec import SearchCase


def _fmt(value) -> str:
    """One rendering rule for the summary table: None is an em dash, never a blank or a 0.

    The distinction matters here more than usual -- `null` correctness means "not judged"
    (no ground truth), which is a different statement from False, and the table must not
    blur them.
    """
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _fmt_loop(reps: int, phrase_len: int) -> str:
    """"2x a 3-word phrase", or "none" -- a run of one is not a loop, and printing it as
    "1x a 1-word phrase" makes clean outputs look like they have a finding in them."""
    return "none" if reps < 2 else f"{reps}x a {phrase_len}-word phrase"


def _readout_line(readout: list, k: int = 3) -> str:
    """The steered readout as one inline line -- enough to spot a saturated or garbled
    readout (FINDINGS.md #11, #30) without turning every trial into a table."""
    if not readout:
        return "*(none)*"
    return " · ".join(f"{_md_token_cell(str(tok))} {float(logit):.2f}" for tok, logit in readout[:k])


def _summary_table(rows: list[dict]) -> str:
    header = (
        "| layer | coeff | answer | correct? | target? | ops altered? | repetition | "
        "max run | c·‖vec‖/‖resid‖ | review? | source |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        reps, phrase_len = r["max_repeat_run"], r["max_repeat_phrase_len"]
        loop = "none" if reps < 2 else f"{reps}x{phrase_len}w"
        lines.append(
            f"| {r['layer']} | {r['coefficient']} | {_fmt(r['answer_extracted'])} | "
            f"{_fmt(r['answer_correct'])} | {_fmt(r['target_present'])} | "
            f"{_fmt(r['operands_altered'])} | {_fmt(r['repetition_score'])} | "
            f"{loop} | {_fmt(r['effective_ratio'])} | {_fmt(r['needs_human_review'])} | "
            f"{'cached' if r.get('cached') else 'this run'} |"
        )
    return "\n".join(lines)


def _trial_section(r: dict) -> str:
    fence = _md_fence(r["output_text"])
    metrics = (
        f"- answer extracted: **{_fmt(r['answer_extracted'])}** · correct: "
        f"**{_fmt(r['answer_correct'])}** · target present: {_fmt(r['target_present'])} "
        f"(readout rank {_fmt(r['target_rank_in_readout'])} of top-{r['readout_top_k']})\n"
        f"- operands altered: {_fmt(r['operands_altered'])}"
        + (f" ({', '.join(r['altered_expressions'])})" if r.get("altered_expressions") else "")
        + f" · repetition: {_fmt(r['repetition_score'])} · longest loop: "
        f"{_fmt_loop(r['max_repeat_run'], r['max_repeat_phrase_len'])}\n"
        f"- ‖vec‖ {_fmt(r['vec_norm'])} · ‖resid‖ {_fmt(r['resid_norm'])} · ratio "
        f"{_fmt(r['norm_ratio'])} · with coefficient {_fmt(r['effective_ratio'])}"
        f" · {_fmt(r['gen_seconds'])}s\n"
        f"- steered readout (top 3): {_readout_line(r['readout_steered'])}"
    )
    review = ""
    if r["needs_human_review"]:
        bullets = "\n".join(f"  - {reason}" for reason in r["review_reasons"])
        review = f"\n- **NEEDS HUMAN REVIEW:**\n{bullets}"

    return (
        f"#### layer {r['layer']}, coefficient {r['coefficient']}"
        f"{' *(cached from an earlier run)*' if r.get('cached') else ''}\n\n"
        f"{metrics}{review}\n\n"
        f"{fence}\n{r['output_text']}\n{fence}\n"
    )


def _case_section(case: SearchCase, rows: list[dict], baseline_text: str) -> str:
    concept = (
        f"`{case.steering_method}` -- pos={case.pos!r} neg={case.neg!r}"
    )
    grid = f"layers {sorted({r['layer'] for r in rows})} x coefficients {rows_by_coeff(rows)}"
    baseline_fence = _md_fence(baseline_text)
    parts = [
        f"## Case: {case.name}\n",
        f"- **prompt:** `{case.prompt!r}`",
        f"- **steering:** {concept}",
        f"- **grid:** {grid}",
        f"- **expected answer:** {_fmt(case.expected_answer)} · "
        f"**target concept:** {_fmt(case.target_concept)}",
    ]
    if case.notes:
        parts.append(f"- **notes:** {case.notes}")
    parts.append(f"\n**Baseline (no steering):**\n\n{baseline_fence}\n{baseline_text}\n{baseline_fence}\n")
    parts.append(f"### Summary\n\n{_summary_table(rows)}\n")
    parts.append("### Trials\n")
    parts.extend(_trial_section(r) for r in rows)
    return "\n".join(parts)


def rows_by_coeff(rows: list[dict]) -> list[float]:
    """Coefficients in the order they were run, de-duplicated across layers."""
    return list(dict.fromkeys(r["coefficient"] for r in rows))


def write_search_report(
    out_dir: str,
    run_id: str,
    model_name: str,
    normalize: bool,
    cases: list[SearchCase],
    rows_by_case: dict[str, list[dict]],
    baselines: dict[str, str],
) -> str:
    """Write the report and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"search_report_{run_id}.md")

    all_rows = [r for case in cases for r in rows_by_case.get(case.name, [])]
    flagged = [r for r in all_rows if r["needs_human_review"]]
    timestamp = all_rows[0]["timestamp"] if all_rows else ""

    head = [
        f"# Search run {run_id}\n",
        f"- **model:** {model_name}",
        f"- **date:** {timestamp}",
        f"- **normalize:** {'ON' if normalize else 'off (coefficients comparable with FINDINGS.md)'}",
        f"- **trials:** {len(all_rows)} across {len(cases)} case(s), "
        f"{sum(1 for r in all_rows if r.get('cached'))} reused from earlier runs",
        f"- **flagged for human review:** {len(flagged)}",
        "\nGenerated from `search_log.jsonl`; regenerate any time. Every trial's full "
        "output is reproduced verbatim below, so nothing here needs the JSONL to be read.\n",
        "Fenced blocks show the model's output exactly as `generate` returned it, echoed "
        "prompt included. The metrics were computed on the *continuation only* -- otherwise "
        "the prompt's own words get scored as if the model had said them.\n",
    ]

    body = [_case_section(case, rows_by_case[case.name], baselines[case.name])
            for case in cases if rows_by_case.get(case.name)]

    if flagged:
        # Two very different kinds of flag, kept apart so the interesting one isn't buried.
        # A whole no-ground-truth case flags every one of its trials for the same structural
        # reason; listing all of them would drown the trials where the metrics actually
        # contradicted each other.
        disagreements = [r for r in flagged
                         if any(reason != NO_GROUND_TRUTH_REASON for reason in r["review_reasons"])]
        unjudged_cases: dict[str, int] = {}
        for r in flagged:
            if NO_GROUND_TRUTH_REASON in r["review_reasons"]:
                unjudged_cases[r["case_name"]] = unjudged_cases.get(r["case_name"], 0) + 1

        flag_lines = ["## Flagged for human review\n"]
        if disagreements:
            flag_lines.append(
                "**Metrics disagreed** on these trials -- a confident score here would be "
                "worth less than your own reading of the output above.\n")
            for r in disagreements:
                reasons = "; ".join(reason for reason in r["review_reasons"]
                                    if reason != NO_GROUND_TRUTH_REASON)
                flag_lines.append(f"- **{r['case_name']}** layer {r['layer']} @ {r['coefficient']}: {reasons}")
            flag_lines.append("")
        if unjudged_cases:
            flag_lines.append(
                "**No ground truth** in these cases, so correctness was left null by design "
                "rather than guessed -- the harness's contribution here is execution and "
                "logging, and the judgement is yours:\n")
            for name, count in unjudged_cases.items():
                flag_lines.append(f"- **{name}**: all {count} trial(s)")
        body.append("\n".join(flag_lines) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(head) + "\n" + "\n".join(body))
    return path
