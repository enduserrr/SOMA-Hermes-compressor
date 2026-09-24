#!/usr/bin/env python3
"""soma-savings — report SOMA compression savings from accounting.jsonl.

Read-only CLI over the plugin's accounting trail. It never writes to
accounting.jsonl, never touches the compressor or its tester, and exits
without side effects. It only reads the JSONL records SOMA already wrote.

Usage:
    soma-savings                 total savings across all sessions
    soma-savings -v --by-session per-session savings list
    soma-savings <session-id>    that session's savings + grand total

Exit codes: 0 ok, 2 file/usage error.

Fields read from each accounting line (the engine writes these):
    session_id       (added by the engine since Sep 5 2026; "-" = unknown,
                       e.g. offline bench harness)
    input_est_chars  estimated chars before compression
    output_est_chars estimated chars after compression
    results_capped   number of tool results compressed in that request
    timestamp

Savings per request = input_est_chars - output_est_chars.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


def _accounting_path() -> pathlib.Path:
    # The plugin lives at <repo>/<plugin>; the accounting file sits beside
    # engine.py. Allow an override (e.g. a deployed copy elsewhere).
    override = os.environ.get("SOMA_ACCOUNTING")
    if override:
        return pathlib.Path(override)
    # This script lives in <repo>/scripts/ → repo root is parent's parent.
    plugin_root = pathlib.Path(__file__).resolve().parent.parent
    return plugin_root / "accounting.jsonl"


def _parse(path: pathlib.Path):
    """Yield parsed accounting records (skipping malformed lines)."""
    if not path.exists():
        print(f"error: accounting file not found: {path}", file=sys.stderr)
        return
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError("not a JSON object")
                yield rec
            except Exception:
                print(
                    f"warning: skipping unparsable line {line_no}", file=sys.stderr
                )


def _saved(rec: dict) -> int:
    try:
        return int(rec.get("input_est_chars") or 0) - int(
            rec.get("output_est_chars") or 0
        )
    except (TypeError, ValueError):
        return 0


def _human(n: int) -> str:
    """Format an integer char count with ',' thousands separators."""
    return f"{n:,}"


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="soma-savings",
        description="Report SOMA compression savings from accounting.jsonl (read-only).",
    )
    p.add_argument("-v", "--by-session", action="store_true",
                   help="list savings per session")
    p.add_argument("session_id", nargs="?", default=None,
                   help="report this session's savings plus the total")
    args = p.parse_args(argv)

    if args.session_id and args.by_session:
        p.error("use either a session-id arg or -v, not both")

    records = list(_parse(_accounting_path()))

    total_saved = sum(_saved(r) for r in records)
    total_capped = sum(int(r.get("results_capped") or 0) for r in records)

    if not args.session_id and not args.by_session:
        # Total across all sessions.
        print("SOMA compression savings (all sessions)")
        print(f"  sessions with compressed requests: {len(records)}")
        print(f"  total chars saved:                {_human(total_saved)}")
        print(f"  tool results compressed:          {_human(total_capped)}")
        return 0

    if args.by_session:
        # Per-session list, plus the grand total.
        by_session: dict[str, dict] = {}
        for rec in records:
            sid = str(rec.get("session_id") or "-")
            agg = by_session.setdefault(
                sid, {"requests": 0, "saved": 0, "capped": 0}
            )
            agg["requests"] += 1
            agg["saved"] += _saved(rec)
            agg["capped"] += int(rec.get("results_capped") or 0)
        print("SOMA compression savings, per session")
        for sid in sorted(by_session, key=lambda k: -by_session[k]["saved"]):
            agg = by_session[sid]
            print(
                f"  {sid:<28} {agg['requests']:>3} reqs  "
                f"{_human(agg['saved']):>10} chars saved  "
                f"{agg['capped']:>3} capped"
            )
        print(f"  {'TOTAL':<28} {len(records):>3} reqs  {_human(total_saved):>10} chars saved")
        return 0

    # Specific session id.
    sid_str = args.session_id
    session_recs = [r for r in records if str(r.get("session_id") or "-") == sid_str]
    session_saved = sum(_saved(r) for r in session_recs)
    session_capped = sum(int(r.get("results_capped") or 0) for r in session_recs)
    print(f"SOMA compression savings for session {sid_str}")
    print(f"  requests compressed:  {len(session_recs)}")
    print(f"  chars saved in this:  {_human(session_saved)}")
    print(f"  tool results capped:  {_human(session_capped)}")
    print(f"  grand total chars:    {_human(total_saved)}")
    if not session_recs:
        print("  (no compressed requests recorded for this session id)")
    return 0


if __name__ == "__main__":
    sys.exit(_main())