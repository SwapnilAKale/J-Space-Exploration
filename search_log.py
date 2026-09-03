"""`search_log.jsonl` -- the search harness's source of truth.

One JSON object per line, one line per trial, appended across ALL runs forever. Everything
else in the harness is derived from this file: the Markdown report is a generated view, and
the resume/skip behaviour is just this file read back.

**Why JSONL and not a Markdown table or CSV.** The harness reads its own log, and real
generated outputs contain `|`, newlines, CJK, LaTeX and backticks. JSON escapes all of that
losslessly; a Markdown table would mangle it and a CSV would need its own quoting rules.
Line-delimited (rather than one big JSON array) is what makes appending crash-safe: a kill
mid-write loses at most the final line, and `load_index` tolerates exactly that.
"""
from __future__ import annotations

import datetime as _dt
import json
import os

DEFAULT_LOG_PATH = "search_log.jsonl"

# The fields that make a trial reproducible. Two rows sharing this key MUST produce the
# same text, which is only true because generation is greedy (`do_sample=False`) -- that is
# what licenses skipping an already-tested pair instead of regenerating it. Anything that
# changes the output but is missing from this key would silently serve a stale row.
KEY_FIELDS = (
    "model",
    "prompt",
    "steering_method",
    "pos",
    "neg",
    "max_new_tokens",
    "normalize",
    "layer",
    "coefficient",
)


def new_run_id() -> str:
    """Local-time stamp, sortable, filename-safe -- also the report's filename suffix."""
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def timestamp() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def trial_key(row: dict) -> tuple:
    """Hashable identity of a trial. Coefficients are floats on both sides (CLI and JSON
    both parse the same decimal literal to the same double), so exact comparison is safe."""
    return tuple(row.get(field) for field in KEY_FIELDS)


def append_trial(path: str, row: dict) -> None:
    """Append one trial and flush it to disk immediately.

    Opened and closed per row on purpose: a long Qwen sweep can be interrupted (Ctrl-C, a
    laptop sleeping, an OOM), and every trial that finished before that point must already
    be durable. os.fsync makes it durable against a power loss too -- one fsync per trial is
    free next to ~30 seconds of CPU generation.
    """
    line = json.dumps(row, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_index(path: str) -> dict[tuple, dict]:
    """Read the whole log back as {trial_key: row}, latest row winning.

    A truncated or corrupt final line is skipped with a warning rather than crashing the
    run: the log is append-only and the last line is the only one a crash can damage, so
    losing it must never cost you the other several hundred trials.
    """
    index: dict[tuple, dict] = {}
    if not os.path.exists(path):
        return index

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[log] {path}:{lineno} is not valid JSON -- skipping that line")
                continue
            index[trial_key(row)] = row
    return index
