"""Automatic scoring for search-harness trials -- pure Python, no torch, no LLM.

Every function here is deliberately *narrow and honest*. The design constraints come
straight from findings we already paid for:

- **Never grep for the target.** `FINDINGS.md` #32's real outputs `"7 x 10 = 20"`,
  `"19 is 19. 19 is 20."` and `"12 is 20 and 2 is 2."` all contain the target "20" and are
  all failures. A substring check would have reported the *opposite* of the finding. So we
  parse the answer out and score *that*; `target_present` survives only as an explicitly
  weak side-signal.
- **Never invent a score.** Most Stage-1 backlog probes (style, mood, cross-lingual,
  prompt/steer mismatch) have no expected answer and no parseable output. Those return
  `None` for every correctness field and get flagged for human judgement, rather than
  being handed a number that means nothing.
- **Never collapse the metrics into one.** They dissociate: `35 + 21 = 55` is coherent and
  wrong, `1. Let's: - 7 - 7` is structurally fine and empty, `}%}%}%` is neither.

Run `python search_scoring.py` to execute the self-check at the bottom -- it asserts the
above real outputs score the way `FINDINGS.md` says they should. It needs no model and
takes about a second, which makes it the cheapest regression test in the project.
"""
from __future__ import annotations

import re

# A plain unsigned number, optionally comma-grouped and/or decimal. Unsigned on purpose:
# a leading "-" is far more often the subtraction operator in these outputs than a sign.
_NUM = r"\d[\d,]*(?:\.\d+)?"

# Operator spellings we understand, mapped to a canonical symbol. Word operators are
# wrapped in \b so "x" doesn't match inside words and "times" doesn't match "sometimes".
_OPERATORS: dict[str, str] = {
    r"\btimes\b": "*",
    r"\bmultiplied by\b": "*",
    r"×": "*",
    r"\*": "*",
    r"\bx\b": "*",
    r"\bplus\b": "+",
    r"\+": "+",
    r"\bminus\b": "-",
    r"-": "-",
    r"\bdivided by\b": "/",
    r"÷": "/",
    r"/": "/",
}
# Longest-first so "multiplied by" is tried before "by"-less alternatives can half-match.
_OP_ALTERNATION = "|".join(sorted(_OPERATORS, key=len, reverse=True))
_EXPR_RE = re.compile(rf"({_NUM})\s*({_OP_ALTERNATION})\s*({_NUM})", re.IGNORECASE)

# Repetition thresholds. These exist ONLY to raise needs_human_review -- nothing in this
# module turns them into a pass/fail verdict, because where the line sits between "loopy"
# and "emphatic" is a human call and the report prints the full text for exactly that reason.
REPETITION_REVIEW_THRESHOLD = 0.5
REPEAT_RUN_REVIEW_THRESHOLD = 3

# Named because the report treats it differently from every other review reason: "there was
# nothing to judge against" is a property of the CASE (a style/mood probe has no ground
# truth by design), whereas every other reason is a property of one surprising TRIAL.
NO_GROUND_TRUTH_REASON = "no expected answer -- correctness not judged, read the output"


def _canonical_op(raw: str) -> str:
    """Map a matched operator string back to its canonical symbol."""
    for pattern, symbol in _OPERATORS.items():
        if re.fullmatch(pattern, raw, flags=re.IGNORECASE):
            return symbol
    return raw  # unreachable for anything _EXPR_RE matched, but never guess silently


def _clean_number(raw: str) -> str:
    """'1,081.' -> '1081'. Thousands separators are display, not value."""
    return raw.replace(",", "").rstrip(".")


def _as_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return float(_clean_number(str(value)))
    except ValueError:
        return None


def extract_answer(text: str) -> str | None:
    """The number the output is actually *asserting*, or None if it asserts none.

    Three fallbacks in decreasing order of confidence:
      1. the number after the LAST "=" -- in a worked-through output the final equals sign
         is where the model commits (`... = 35 + 21 = 55` -> "55");
      2. else the number after the last "is" -- Qwen's terse register ("7 times 8 is 56.");
      3. else the last number anywhere in the text.

    Returns the cleaned numeric string (commas and a trailing period stripped) so the
    caller can compare numerically without re-parsing.
    """
    for pattern in (rf"=\s*(-?{_NUM})", rf"\bis\s+(-?{_NUM})"):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return _clean_number(matches[-1])

    matches = re.findall(rf"(-?{_NUM})", text)
    return _clean_number(matches[-1]) if matches else None


def answer_correct(extracted: str | None, expected: str | float | int | None) -> bool | None:
    """None means "not judged", not "wrong" -- the two must stay distinguishable.

    Returns None when there is no expected answer (most style/mood probes) AND when an
    expected answer exists but nothing parsed out of the output. The second case is a real
    "we didn't observe an answer", not an observed wrong one, so it goes to human review
    rather than being counted as a failure.
    """
    if expected is None or extracted is None:
        return None

    exp_num, got_num = _as_float(expected), _as_float(extracted)
    if exp_num is not None and got_num is not None:
        return abs(exp_num - got_num) < 1e-9
    return str(expected).strip().casefold() == str(extracted).strip().casefold()


def target_present(text: str, target: str | None) -> bool | None:
    """Case-insensitive substring check -- a WEAK signal, kept only as one column of many.

    See the module docstring: on its own this metric is actively misleading (every
    `FINDINGS.md` #32 failure contains its target). It earns its place only next to
    `answer_correct`, `operands_altered` and the repetition scores.
    """
    if target is None:
        return None
    return target.strip().casefold() in text.casefold()


def _expressions(text: str) -> list[tuple[str, str, str, str]]:
    """Every `A <op> B` in the text as (raw_expression, canonical_op, A, B)."""
    return [
        (m.group(0).strip(), _canonical_op(m.group(2)), _clean_number(m.group(1)), _clean_number(m.group(3)))
        for m in _EXPR_RE.finditer(text)
    ]


def prompt_operands(prompt: str) -> tuple[str, str, str] | None:
    """The (op, a, b) the prompt actually asked about, or None if it isn't arithmetic."""
    exprs = _expressions(prompt)
    if not exprs:
        return None
    _, op, a, b = exprs[0]
    return op, a, b


def operands_altered(prompt: str, text: str) -> tuple[bool | None, list[str]]:
    """Does the output compute with operands the prompt never gave it?

    This single check kills the entire `FINDINGS.md` #31 false-positive class: the model
    was steered toward "20", slotted 20 in as an *operand*, and then multiplied correctly
    (`7 x 10 = 20`, `140 = 7 x 20`). The arithmetic is right; the inputs are not. Nothing in
    `answer_correct` alone can tell you that happened.

    Only expressions using the prompt's OWN operator are compared, and operands are
    compared as an unordered pair (8 x 7 is the same question as 7 x 8).

    Returns (flag, offending_expressions). The flag is None when the prompt has no operands
    to compare against, and False when it has them and nothing contradicts them.

    Known and accepted noise: a legitimate distributive decomposition
    (`7 x 8 = (7 x 5) + (7 x 3)`, `FINDINGS.md` #34) trips this too, because at the string
    level it is indistinguishable from an operand substitution. That is precisely why this
    flag raises `needs_human_review` instead of deciding anything, and why the offending
    expressions are returned for the human to read.
    """
    asked = prompt_operands(prompt)
    if asked is None:
        return None, []

    op, a, b = asked
    expected_pair = sorted((a, b))
    offenders = [
        raw for raw, got_op, x, y in _expressions(text)
        if got_op == op and sorted((x, y)) != expected_pair
    ]
    return bool(offenders), offenders


def _words(text: str) -> list[str]:
    """Lowercased word tokens with surrounding punctuation stripped -- so "20." and "20"
    count as the same word when measuring repetition (a loop is a loop regardless of how
    the punctuation falls)."""
    return [w for w in (t.strip(".,!?;:()[]\"'").casefold() for t in text.split()) if w]


def repetition_score(text: str) -> float | None:
    """1 - (distinct word-trigrams / total word-trigrams). Higher = more repetitive.

    None when the text is too short to have three trigrams (five words): a two-word output
    like "19" is degenerate for a completely different reason, and inventing a repetition
    number for it would be exactly the kind of fake score this module refuses to produce.
    """
    words = _words(text)
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if len(trigrams) < 3:
        return None
    return 1.0 - len(set(trigrams)) / len(trigrams)


def max_repeat_run(text: str) -> tuple[int, int]:
    """Longest back-to-back repetition in the text, as (times_repeated, phrase_length).

    Complements `repetition_score`, which is blind to a short tight loop: the real output
    "19 is 19. 19 is 20. 19 is 20." has plenty of distinct trigrams yet is plainly stuck.

    The phrase LENGTH is returned alongside the count because the two together are what
    separate degeneracy from emphasis: "very very good" repeats a single word twice and is
    fine, while "It's a great sport. It's a great sport." repeats a four-word phrase twice
    and is the greedy-decoding loop of `FINDINGS.md` #8. Candidates are ranked by
    (repetitions, phrase length) so a longer repeated phrase wins a tie on count.

    O(n^2) on ~30 words is nothing, and both numbers read plainly in the report.
    """
    words = _words(text)
    n = len(words)
    best = (1, 1)
    for size in range(1, n // 2 + 1):
        for start in range(n - size):
            phrase = words[start:start + size]
            reps, i = 1, start + size
            while i + size <= n and words[i:i + size] == phrase:
                reps += 1
                i += size
            if reps > 1:
                best = max(best, (reps, size))
    return best


def target_rank_in_readout(readout: list[tuple[str, float]], target: str | None) -> int | None:
    """Where the target sits in the steered top-k readout, or None if it isn't in it.

    Matching is loose on purpose: readout tokens are often word-*fragments*
    (`FINDINGS.md` #4), so a token counts as a hit if it equals the target or is a prefix
    of it, case- and whitespace-insensitively. `readout_top_k` is logged next to this value
    so a None reads as "not in the top k", never as "unknown".
    """
    if target is None:
        return None
    want = target.strip().casefold()
    for rank, (token, _logit) in enumerate(readout):
        tok = token.strip().casefold()
        if tok and (tok == want or (len(tok) >= 2 and want.startswith(tok))):
            return rank
    return None


def score_trial(
    prompt: str,
    output_text: str,
    expected_answer: str | float | int | None,
    target_concept: str | None,
    readout_steered: list[tuple[str, float]],
) -> dict:
    """Run every metric and decide whether the result needs a human.

    `needs_human_review` fires when the metrics DISAGREE (the situation where a confident
    single score would be worst) or when there is nothing to judge against. `review_reasons`
    is a list rather than a bare bool so the report can say *why* without the reader
    re-deriving it.
    """
    extracted = extract_answer(output_text)
    correct = answer_correct(extracted, expected_answer)
    present = target_present(output_text, target_concept)
    altered, altered_exprs = operands_altered(prompt, output_text)
    repetition = repetition_score(output_text)
    repeat_run, repeat_phrase_len = max_repeat_run(output_text)
    rank = target_rank_in_readout(readout_steered, target_concept)

    # Two independent ways to be degenerate: a diffuse high trigram-repetition score, or a
    # tight back-to-back loop. A multi-word phrase repeated even twice counts (that IS the
    # greedy loop); a single word needs three repeats before it stops being emphasis.
    looks_degenerate = (
        (repetition is not None and repetition >= REPETITION_REVIEW_THRESHOLD)
        or repeat_run >= REPEAT_RUN_REVIEW_THRESHOLD
        or (repeat_run >= 2 and repeat_phrase_len >= 2)
    )

    reasons: list[str] = []
    if expected_answer is None:
        reasons.append(NO_GROUND_TRUTH_REASON)
    elif extracted is None:
        reasons.append("expected answer given but no answer could be parsed from the output")
    if present and looks_degenerate:
        reasons.append("target present but the output looks degenerate -- likely a false positive")
    if present and correct is False:
        # The FINDINGS.md #32 signature: the target string is in there, the answer is still
        # wrong. This is the exact disagreement a substring-based scorer would have called
        # a success, so it always goes to a human rather than to a verdict.
        reasons.append("target string appears but the extracted answer is wrong -- read the output")
    if extracted is not None and altered:
        reasons.append(f"answer extracted but output computes on altered operands: {', '.join(altered_exprs)}")
    if correct and altered:
        reasons.append("answer scored correct despite altered operands -- verify by hand")
    if correct and looks_degenerate:
        reasons.append("correct answer sitting inside a degenerate output")

    return {
        "answer_extracted": extracted,
        "answer_correct": correct,
        "target_present": present,
        "operands_altered": altered,
        "altered_expressions": altered_exprs,
        "repetition_score": repetition,
        "max_repeat_run": repeat_run,
        "max_repeat_phrase_len": repeat_phrase_len,
        "target_rank_in_readout": rank,
        "readout_top_k": len(readout_steered),
        "needs_human_review": bool(reasons),
        "review_reasons": reasons,
    }


def _self_check() -> None:
    """Assert the real outputs from FINDINGS.md score the way the findings say they do.

    Every string below is an output we actually observed on this machine, not a synthetic
    example -- so this doubles as a regression test against the exact mistakes a naive
    scorer would make.
    """
    math_prompt = "What is 7 times 8?"
    no_readout: list[tuple[str, float]] = []

    # -- FINDINGS.md #32, the three failures that all contain the target "20" --
    for text, why in [
        ("7 x 10 = 20", "operand rewritten"),
        ("19 is 19. 19 is 20. 19 is 20.", "degenerate loop"),
        ("12 is 20 and 2 is 2.", "garbage"),
    ]:
        s = score_trial(math_prompt, text, expected_answer=56, target_concept="20", readout_steered=no_readout)
        assert s["target_present"] is True, why  # the naive metric says "success" ...
        assert s["answer_correct"] is False, why  # ... and every real metric says failure
        assert s["needs_human_review"] is True, why

    # The operand rewrite is caught specifically, not just via the wrong answer.
    s = score_trial(math_prompt, "7 x 10 = 20", 56, "20", no_readout)
    assert s["operands_altered"] is True and s["altered_expressions"] == ["7 x 10"]

    # The loop is caught by the run detector even though its trigrams are fairly distinct
    # (repetition_score is only ~0.14 here -- the trigram metric alone would miss it).
    s = score_trial(math_prompt, "19 is 19. 19 is 20. 19 is 20.", 56, "20", no_readout)
    assert (s["max_repeat_run"], s["max_repeat_phrase_len"]) == (2, 3), s["max_repeat_run"]

    # Emphasis is not degeneracy: a single word doubled must NOT trip the loop detector.
    assert max_repeat_run("that is a very very good answer") == (2, 1)
    # ... but gpt2's classic greedy loop (FINDINGS.md #8) must.
    assert max_repeat_run("It's a great sport. It's a great sport.") == (2, 4)

    # -- FINDINGS.md #34: coherent, well-reasoned, and wrong by one --
    s = score_trial(math_prompt, "7 x 8 = 7 x (5 + 3) = (7 x 5) + (7 x 3) = 35 + 21 = 55", 56, None, no_readout)
    assert s["answer_extracted"] == "55", s["answer_extracted"]  # the LAST '=', not the first number
    assert s["answer_correct"] is False
    assert s["operands_altered"] is True  # the decomposition trips it -- known, hence the review flag
    assert s["needs_human_review"] is True

    s = score_trial(math_prompt, "2. Multiply the numbers: 7 x 8 = 48. So, 7 times 8 equals 48.", 56, None, no_readout)
    assert s["answer_extracted"] == "48" and s["answer_correct"] is False
    assert s["operands_altered"] is False  # correct operands, wrong product

    # -- The baseline that is genuinely right --
    s = score_trial(math_prompt, "7 times 8 is 56.", 56, "20", no_readout)
    assert s["answer_correct"] is True
    assert s["target_present"] is False
    assert s["needs_human_review"] is False, s["review_reasons"]

    # -- FINDINGS.md #32 @0.95: a bare "19". Nothing to measure repetition on. --
    s = score_trial(math_prompt, "19", 56, "20", no_readout)
    assert s["answer_extracted"] == "19" and s["repetition_score"] is None

    # -- No ground truth at all (style/mood/cross-lingual probes, brief section 5.5) --
    s = score_trial(
        "Name one sport you love, in a short sentence.",
        "I love pasta carbonara with a simple tomato sauce and a sprinkle of cheese.",
        expected_answer=None,
        target_concept=None,
        readout_steered=[(" pasta", 12.3), (" food", 11.1)],
    )
    assert s["answer_correct"] is None and s["target_present"] is None
    assert s["operands_altered"] is None  # no operands in the prompt to compare against
    assert s["needs_human_review"] is True  # flagged for judgement, NOT given a fake score
    assert s["repetition_score"] is not None  # the metrics that DO apply are still recorded

    # -- Readout rank, including fragment matching (FINDINGS.md #4) --
    assert target_rank_in_readout([("a", 1.0), (" rug", 0.9)], "rugby") == 1
    assert target_rank_in_readout([("a", 1.0), (" rug", 0.9)], "pasta") is None
    assert target_rank_in_readout([("a", 1.0)], None) is None

    print("search_scoring self-check: all assertions passed.")


if __name__ == "__main__":
    _self_check()
