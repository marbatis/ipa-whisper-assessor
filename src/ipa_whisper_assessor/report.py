from __future__ import annotations

import html
from pathlib import Path

from .schemas import AssessmentResult


def write_json(result: AssessmentResult, path: str | Path) -> None:
    Path(path).write_text(result.model_dump_json(indent=2), encoding="utf-8")


def _fmt_time(t: float | None) -> str:
    if t is None:
        return ""
    return f"{t:.2f}s"


def write_html(result: AssessmentResult, path: str | Path) -> None:
    rows = []
    for w in result.word_alignments:
        ops = []
        for o in w.phoneme_ops:
            if o.op == "sub":
                ops.append(f"<span class='sub'>{html.escape(o.expected or '')}→{html.escape(o.predicted or '')}</span>")
            elif o.op == "ins":
                ops.append(f"<span class='ins'>+{html.escape(o.predicted or '')}</span>")
            elif o.op == "del":
                ops.append(f"<span class='del'>-{html.escape(o.expected or '')}</span>")
        rows.append(
            "<tr>"
            f"<td>{html.escape(_fmt_time(w.start))}–{html.escape(_fmt_time(w.end))}</td>"
            f"<td>{html.escape(w.reference_word)}</td>"
            f"<td class='mono'>{html.escape(w.expected_ipa)}</td>"
            f"<td class='mono'>{html.escape(w.predicted_ipa)}</td>"
            f"<td class='mono'>{' '.join(ops)}</td>"
            "</tr>"
        )

    mistake_rows = []
    for m in result.mistakes:
        mistake_rows.append(
            "<tr>"
            f"<td>{html.escape(m.rule)}</td>"
            f"<td class='mono'>{html.escape(m.expected or '')}</td>"
            f"<td class='mono'>{html.escape(m.predicted or '')}</td>"
            f"<td>{m.count}</td>"
            "</tr>"
        )

    mistake_body = "".join(mistake_rows) or "<tr><td colspan='4'>(none)</td></tr>"

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>IPA Assessment Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f7f7f7; text-align: left; }}
    .sub {{ color: #b91c1c; font-weight: 600; }}
    .ins {{ color: #0f766e; }}
    .del {{ color: #7c3aed; }}
  </style>
</head>
<body>
  <h1>IPA Assessment Report</h1>
  <p><b>Audio:</b> {html.escape(result.audio_path)}</p>
  <p><b>Model:</b> {html.escape(result.model)}</p>
  <p><b>Reference:</b> {html.escape(result.reference)}</p>
  <p><b>PER:</b> {result.metrics.phoneme_error_rate:.3f} (S:{result.metrics.substitutions} I:{result.metrics.insertions} D:{result.metrics.deletions})</p>

  <h2>Word Alignment</h2>
  <table>
    <thead>
      <tr>
        <th>Time</th>
        <th>Word</th>
        <th>Expected IPA</th>
        <th>Predicted IPA</th>
        <th>Ops</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>

  <h2>Mistakes</h2>
  <table>
    <thead>
      <tr><th>Rule</th><th>Expected</th><th>Predicted</th><th>Count</th></tr>
    </thead>
    <tbody>
      {mistake_body}
    </tbody>
  </table>
</body>
</html>
"""
    Path(path).write_text(html_doc, encoding="utf-8")
