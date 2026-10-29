"""Bake data/latency.jsonl into a standalone HTML latency report you can publish or share.

    uv run python -m scripts.export_report      # -> data/latency_report.html
"""

import json
from urllib.parse import quote

from agent.config import CONFIG, ROOT
from agent.latency import LatencyLog, summarize


def main() -> None:
    recs = LatencyLog(CONFIG.latency_log).read(limit=10_000)
    if not recs:
        raise SystemExit("No latency records yet: make a call or run scripts.bench first.")
    payload = json.dumps({"records": recs, "summary": summarize(recs)}, ensure_ascii=False)
    html = (ROOT / "web" / "latency.html").read_text(encoding="utf-8")
    inject = (f"<script>window.LATENCY_STATIC = true; "
              f"window.LATENCY_URL = 'data:application/json,{quote(payload)}';</script>\n<script>")
    html = html.replace("<script>", inject, 1)
    out = ROOT / "data" / "latency_report.html"
    out.write_text(html, encoding="utf-8")
    s = summarize(recs)
    print(f"{out}  ({s['n']} turns, e2e p50 {s['e2e']['p50']} ms, p90 {s['e2e']['p90']} ms)")


if __name__ == "__main__":
    main()
