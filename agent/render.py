"""Dependency-free HTML rendering of a campaign report.

Rendered server-side so the deliverable is readable even with JavaScript
disabled (and print/PDF friendly). All model text is escaped.
"""

from __future__ import annotations

import html
from typing import Any

_STYLE = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem auto;
     max-width:1040px;line-height:1.55;color:#111;padding:0 1rem}
h1{margin-bottom:.2rem}
table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border:1px solid #d6d9de;padding:.45rem .6rem;text-align:left;font-size:.93rem}
th{background:#f3f4f6}
section{margin-top:2.5rem;border-top:1px solid #e3e5e8;padding-top:1rem}
.note{background:#fff8e1;border-left:4px solid #f0b429;padding:.8rem 1rem;border-radius:3px}
.muted{color:#5b6270;font-size:.9rem}
code{background:#f3f4f6;padding:.1rem .3rem;border-radius:3px}
"""


def render_report_html(payload: dict[str, Any]) -> str:
    """Return a complete standalone HTML document for *payload*."""

    esc = html.escape
    campaign = payload.get("campaign", {}) or {}
    title = esc(str(campaign.get("name") or "Untitled campaign"))

    ranking_rows = "".join(
        "<tr>"
        f"<td>{row.get('rank')}</td>"
        f"<td>{esc(str(row.get('label')))}</td>"
        f"<td><strong>{row.get('score')}</strong></td>"
        f"<td>{esc(str(row.get('verdict')))}</td>"
        f"<td>{row.get('components', {}).get('engagement')}</td>"
        f"<td>{row.get('components', {}).get('focus')}</td>"
        f"<td>{row.get('components', {}).get('continuity')}</td>"
        f"<td>{row.get('dead_time_s')}s / {row.get('duration_s')}s</td>"
        "</tr>"
        for row in payload.get("ranking", [])
    )

    sections: list[str] = []
    for report in payload.get("creatives", []):
        interp = report.get("interpretation", {}) or {}
        verdict = interp.get("verdict", {}) or {}
        prov = interp.get("provenance", {}) or {}
        diag = report.get("diagnostics", {}) or {}

        moments = "".join(
            f"<li><strong>{esc(str(m.get('stimulus_time', '')))}</strong> &mdash; "
            f"{esc(str(m.get('interpretation', '')))} "
            f"<span class='muted'>({esc(str(m.get('evidence', '')))})</span></li>"
            for m in verdict.get("moments", []) or []
        )
        weak = "".join(
            f"<li><strong>{esc(str(w.get('stimulus_time', '')))}</strong> &mdash; "
            f"{esc(str(w.get('risk', '')))} "
            f"<span class='muted'>({esc(str(w.get('evidence', '')))})</span></li>"
            for w in verdict.get("weak_moments", []) or []
        )
        rewrites = "".join(
            f"<li><strong>{esc(str(r.get('target_time', '')))}</strong> &mdash; "
            f"{esc(str(r.get('change', '')))} "
            f"<span class='muted'>({esc(str(r.get('rationale', '')))})</span></li>"
            for r in verdict.get("rewrite_suggestions", []) or []
        )
        caveats = "".join(f"<li>{esc(str(c))}</li>" for c in verdict.get("caveats", []) or [])

        degraded = interp.get("degraded")
        badge = (
            "<p class='note'><strong>Degraded mode.</strong> The LLM step was "
            f"skipped or failed ({esc(str(interp.get('error')))}); this section is "
            "rule-based only.</p>"
            if degraded
            else ""
        )

        sections.append(
            "<section>"
            f"<h2>{esc(str(report.get('asset', {}).get('label')))}</h2>"
            + badge
            + f"<p><strong>Verdict:</strong> {esc(str(verdict.get('overall_verdict')))} "
            f"(confidence {esc(str(verdict.get('verdict_confidence')))})</p>"
            f"<p>{esc(str(verdict.get('summary', '')))}</p>"
            f"<p class='muted'>Model: {esc(str(prov.get('model')))} | "
            f"reasoning_effort: {esc(str(prov.get('reasoning_effort')))} | "
            f"dynamic range z={esc(str(diag.get('dynamic_range')))} | "
            f"laterality={esc(str(diag.get('laterality_ratio')))} | "
            f"regions: {esc(str(diag.get('region_strategy')))}</p>"
            f"<h3>Strongest moments</h3><ul>{moments or '<li>none detected</li>'}</ul>"
            f"<h3>Flat / weak moments</h3><ul>{weak or '<li>none detected</li>'}</ul>"
            f"<h3>Suggested edits</h3><ul>{rewrites or '<li>none</li>'}</ul>"
            f"<h3>Next experiment</h3><p>{esc(str(verdict.get('next_experiment', '')))}</p>"
            f"<h3>Caveats</h3><ul>{caveats or '<li>none</li>'}</ul>"
            "</section>"
        )

    disclaimers = "".join(f"<li>{esc(str(d))}</li>" for d in payload.get("disclaimers", []) or [])
    method = payload.get("method", {}) or {}
    method_rows = "".join(
        f"<tr><th>{esc(str(k))}</th><td><code>{esc(str(v))}</code></td></tr>"
        for k, v in method.items()
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fMRI Ad Pretest &mdash; {title}</title>
<style>{_STYLE}</style></head><body>
<h1>Agentic fMRI Ad Pretest</h1>
<p class="muted">Campaign: <strong>{title}</strong></p>
<p class="note"><strong>Read this first.</strong> These are TRIBE v2
<em>predicted</em> brain responses for an average subject &mdash; not measurements
of real viewers, not an emotional or persuasive-effect claim, and not a forecast
of sales.</p>
<h2>Ranking</h2>
<table><thead><tr><th>#</th><th>Creative</th><th>Score</th><th>Verdict</th>
<th>Engagement</th><th>Focus</th><th>Continuity</th><th>Flagged flat time</th>
</tr></thead><tbody>{ranking_rows or '<tr><td colspan="8">no creatives</td></tr>'}</tbody></table>
<h2>Method</h2>
<table><tbody>{method_rows}</tbody></table>
{''.join(sections)}
<h2>Disclaimers</h2><ul>{disclaimers}</ul>
</body></html>"""
