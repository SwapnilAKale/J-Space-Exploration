"""Render Stage 1 sweep results as a single self-contained HTML file (inline CSS only).

Kept separate from run_stage1.py for the same reason steering.py and model_setup.py are
separate modules: this is a distinct concern (HTML rendering) with no upward dependency --
it only imports Config for typing, run_stage1.py imports this module, never the reverse.
"""
from __future__ import annotations

import base64
import html as html_lib
import re

from config import Config

# Plain string, not an f-string -- its own { and } are literal characters, so they never
# need escaping when this whole constant is interpolated into the outer HTML f-string.
_CSS = """
body {
  font-family: -apple-system, "Segoe UI", Arial, sans-serif;
  margin: 2rem auto;
  max-width: 1100px;
  background: #fafafa;
  color: #1a1a1a;
}
header {
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 2px solid #333;
}
header h1 { margin: 0 0 0.75rem 0; }
#download-md-btn {
  background: #2b2b2b;
  color: #fff;
  border: none;
  border-radius: 4px;
  padding: 0.5rem 1rem;
  font-size: 0.9rem;
  cursor: pointer;
  margin-bottom: 1rem;
}
#download-md-btn:hover { background: #444; }
header dl {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 0.35rem 1rem;
  margin: 0 0 1rem 0;
}
header dt { font-weight: bold; }
header dd { margin: 0; font-family: "Consolas", monospace; }
.generated-label {
  font-weight: bold;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: #555;
  margin: 0.75rem 0 0.25rem 0;
}
pre {
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 0.75rem;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: "Consolas", monospace;
  margin: 0;
}
.coefficient-section { margin: 2.5rem 0; }
.coefficient-section h2 {
  background: #2b2b2b;
  color: #fff;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  display: inline-block;
}
.layer-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1.5rem;
  margin-top: 1rem;
}
@media (max-width: 900px) {
  .layer-grid { grid-template-columns: 1fr; }
}
.layer-cell {
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 1rem;
}
.layer-cell h3 { margin: 0 0 0.5rem 0; }
table.readout {
  border-collapse: collapse;
  font-family: "Consolas", monospace;
  font-size: 0.85rem;
  width: 100%;
}
table.readout th, table.readout td {
  border: 1px solid #ccc;
  padding: 0.2rem 0.5rem;
  text-align: left;
}
table.readout th { background: #f0f0f0; }
table.readout mark { background: #ffe066; padding: 0 2px; border-radius: 2px; }
"""


def _esc(s: str) -> str:
    return html_lib.escape(s, quote=True)


def _highlight(token: str, highlight_token: str | None) -> str:
    tok_html = _esc(token)
    if highlight_token is not None and token == highlight_token:
        return f"<mark>{tok_html}</mark>"
    return tok_html


def _readout_table_html(
    baseline: list[tuple[str, float]],
    steered: list[tuple[str, float]],
    highlight_token: str | None,
) -> str:
    """One table, baseline and steered columns side by side -- mirrors run_stage1.py's
    print_readout_table shape exactly, just rendered as HTML instead of printed.
    """
    rows = []
    for i, ((b_tok, b_val), (s_tok, s_val)) in enumerate(zip(baseline, steered)):
        rows.append(
            f"<tr><td>{i}</td>"
            f"<td>{_highlight(b_tok, highlight_token)}</td><td>{b_val:.2f}</td>"
            f"<td>{_highlight(s_tok, highlight_token)}</td><td>{s_val:.2f}</td></tr>"
        )
    rows_html = "\n".join(rows)
    return (
        '<table class="readout">'
        "<thead><tr><th>#</th><th>baseline token</th><th>logit</th>"
        "<th>steered token</th><th>logit</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def _layer_cell_html(layer: int, entry: dict, highlight_token: str | None) -> str:
    table = _readout_table_html(entry["baseline_readout"], entry["steered_readout"], highlight_token)
    return (
        '<div class="layer-cell">'
        f"<h3>Layer {layer}</h3>"
        f"{table}"
        '<div class="generated-label">steered generation</div>'
        f"<pre>{_esc(entry['steered_text'])}</pre>"
        "</div>"
    )


def _coefficient_section_html(
    coefficient: float,
    layers: list[int],
    layer_entries: dict[int, dict],
    highlight_token: str | None,
) -> str:
    cells = "\n".join(
        _layer_cell_html(layer, layer_entries[layer], highlight_token)
        for layer in layers
        if layer in layer_entries
    )
    return (
        '<section class="coefficient-section">'
        f"<h2>Coefficient {coefficient}</h2>"
        f'<div class="layer-grid">{cells}</div>'
        "</section>"
    )


def _md_fence(text: str) -> str:
    """A backtick fence longer than any backtick run already in text, min length 3.

    Guards against the generation itself containing a run of backticks that would
    prematurely close a fixed-length ``` fence -- scans the whole text (not just line
    starts), which is conservative but always correct.
    """
    runs = re.findall(r"`+", text)
    longest = max((len(r) for r in runs), default=0)
    return "`" * max(3, longest + 1)


def _md_token_cell(token: str) -> str:
    """Backtick-wrap a table cell so |, _, * etc. render as literal characters instead of
    table delimiters or Markdown emphasis. Falls back to a double-backtick delimiter if the
    token itself contains a backtick, padded with a space when the token starts/ends with a
    backtick (standard CommonMark code-span convention -- protects the delimiter boundary,
    the token's own characters are untouched and the padding is stripped on render).
    """
    if "`" not in token:
        return f"`{token}`"
    padded = f" {token} " if token.startswith("`") or token.endswith("`") else token
    return f"``{padded}``"


def _readout_table_md(baseline: list[tuple[str, float]], steered: list[tuple[str, float]]) -> str:
    lines = [
        "| # | baseline token | logit | steered token | logit |",
        "|---|---|---|---|---|",
    ]
    for i, ((b_tok, b_val), (s_tok, s_val)) in enumerate(zip(baseline, steered)):
        lines.append(
            f"| {i} | {_md_token_cell(b_tok)} | {b_val:.2f} | "
            f"{_md_token_cell(s_tok)} | {s_val:.2f} |"
        )
    return "\n".join(lines)


def _layer_section_md(layer: int, entry: dict) -> str:
    table = _readout_table_md(entry["baseline_readout"], entry["steered_readout"])
    fence = _md_fence(entry["steered_text"])
    return (
        f"### Layer {layer}\n\n"
        f"{table}\n\n"
        f"**Steered generation:**\n\n"
        f"{fence}\n{entry['steered_text']}\n{fence}\n"
    )


def _coefficient_section_md(
    coefficient: float, layers: list[int], layer_entries: dict[int, dict]
) -> str:
    parts = [f"## Coefficient {coefficient}\n"]
    for layer in layers:
        if layer in layer_entries:
            parts.append(_layer_section_md(layer, layer_entries[layer]))
    return "\n".join(parts)


def _build_markdown(
    cfg: Config,
    model_name: str,
    concept_desc: str,
    baseline_text: str,
    layers: list[int],
    report_data: dict[float, dict[int, dict]],
) -> str:
    baseline_fence = _md_fence(baseline_text)
    sections = "\n".join(
        _coefficient_section_md(coefficient, layers, report_data[coefficient])
        for coefficient in cfg.coefficients
        if coefficient in report_data
    )
    return (
        f"# Stage 1 Report\n\n"
        f"**model:** {model_name}\n\n"
        f"**prompt:** {cfg.prompt}\n\n"
        f"**steering:** {concept_desc}\n\n"
        f"## Baseline generation\n\n"
        f"{baseline_fence}\n{baseline_text}\n{baseline_fence}\n\n"
        f"{sections}\n"
    )


# Plain string, not an f-string -- see _CSS's own comment for why. The Base64 payload it
# reads from the DOM at click-time can never contain '<', so it can't accidentally close
# this or the adjacent script tag; decoding straight to a Uint8Array (never a JS string)
# sidesteps every Unicode/CJK string-encoding pitfall entirely.
_DOWNLOAD_BUTTON_JS = """
document.getElementById('download-md-btn').addEventListener('click', function () {
  var b64 = document.getElementById('report-markdown-b64').textContent;
  var binary = atob(b64);
  var bytes = new Uint8Array(binary.length);
  for (var i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  var blob = new Blob([bytes], { type: 'text/markdown;charset=utf-8' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = 'report.md';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});
"""


def write_html_report(
    path: str,
    cfg: Config,
    model_name: str,
    baseline_text: str,
    layers: list[int],
    report_data: dict[float, dict[int, dict]],
) -> None:
    if cfg.steering_method == "token_diff":
        concept_desc = f"token_diff -- pos_token={cfg.pos!r}  neg_token={cfg.neg!r}"
        highlight_token = cfg.pos
    else:
        # actadd has no single target token (it's built from a prompt pair, not a token),
        # so there's nothing well-defined to highlight in this mode.
        concept_desc = f"actadd -- pos_prompt={cfg.pos!r}  neg_prompt={cfg.neg!r}"
        highlight_token = None

    sections = "\n".join(
        _coefficient_section_html(coefficient, layers, report_data[coefficient], highlight_token)
        for coefficient in cfg.coefficients
        if coefficient in report_data
    )

    markdown_doc = _build_markdown(cfg, model_name, concept_desc, baseline_text, layers, report_data)
    markdown_b64 = base64.b64encode(markdown_doc.encode("utf-8")).decode("ascii")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>J-space Stage 1 report -- {_esc(model_name)}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
<h1>Stage 1 Report</h1>
<button id="download-md-btn" type="button">Download as Markdown</button>
<dl>
<dt>model</dt><dd>{_esc(model_name)}</dd>
<dt>prompt</dt><dd>{_esc(cfg.prompt)}</dd>
<dt>steering</dt><dd>{_esc(concept_desc)}</dd>
</dl>
<div class="generated-label">baseline generation</div>
<pre>{_esc(baseline_text)}</pre>
</header>
{sections}
<script id="report-markdown-b64" type="text/plain">{markdown_b64}</script>
<script>{_DOWNLOAD_BUTTON_JS}</script>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)
