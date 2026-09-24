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
    input_est_tokens estimated tokens before (added Sep 22 2026; 0 on
                       older lines)
    output_est_tokens  estimated tokens after (same date; 0 on older lines)
    results_capped   number of tool results compressed in that request
    timestamp

Savings per request = input_est_chars - output_est_chars.
Token figures are estimates (tiktoken cl100k_base when installed, else a
deterministic chars-per-token fallback) — NOT billing truth. Provider-
reported usage lives in state.db (sessions / session_model_usage).
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


def _saved_tokens(rec: dict) -> int:
    """Token savings for one record; 0 for pre-token-era lines (0/0)."""
    try:
        return int(rec.get("input_est_tokens") or 0) - int(
            rec.get("output_est_tokens") or 0
        )
    except (TypeError, ValueError):
        return 0


# Legacy records (pre Sep 22 2026) carry only char counts. Backfill uses the
# SOMA core's deterministic fallback ratio (soma_compressor.CHARS_PER_TOKEN),
# which is exactly what final_token_estimate() would have recorded — same
# math, not an ad-hoc guess. Label these "retro" wherever shown.
LEGACY_CHARS_PER_TOKEN = 4


def _retro_tokens(rec: dict) -> int:
    """Estimated SAVED tokens for a legacy chars-only record (saved chars / 4)."""
    try:
        return (
            int(rec.get("input_est_chars") or 0)
            - int(rec.get("output_est_chars") or 0)
        ) // LEGACY_CHARS_PER_TOKEN
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
    # Token savings: real recorded estimates where present; legacy
    # chars-only records backfilled at the core's chars/token fallback.
    total_saved_tokens = sum(_saved_tokens(r) for r in records)
    legacy_n = sum(1 for r in records if not r.get("input_est_tokens"))
    retro_saved_tokens = sum(
        _saved_tokens(r) if r.get("input_est_tokens") else _retro_tokens(r)
        for r in records
    )

    if not args.session_id and not args.by_session:
        # Total across all sessions.
        print("SOMA compression savings (all sessions)")
        print(f"  sessions with compressed requests: {len(records)}")
        print(f"  total chars saved:                {_human(total_saved)}")
        print(f"  total est. tokens saved:          {_human(total_saved_tokens)}")
        print(
            f"  total tokens (incl. retro):       {_human(retro_saved_tokens)}"
            f"  [{legacy_n} legacy records backfilled at {LEGACY_CHARS_PER_TOKEN} chars/token]"
        )
        print(f"  tool results compressed:          {_human(total_capped)}")
        return 0

    if args.by_session:
        # Per-session list, plus the grand total.
        by_session: dict[str, dict] = {}
        for rec in records:
            sid = str(rec.get("session_id") or "-")
            agg = by_session.setdefault(
                sid, {"requests": 0, "saved": 0, "saved_tokens": 0, "capped": 0}
            )
            agg["requests"] += 1
            agg["saved"] += _saved(rec)
            agg["saved_tokens"] += _saved_tokens(rec)
            agg["capped"] += int(rec.get("results_capped") or 0)
        print("SOMA compression savings, per session")
        for sid in sorted(by_session, key=lambda k: -by_session[k]["saved"]):
            agg = by_session[sid]
            print(
                f"  {sid:<28} {agg['requests']:>3} reqs  "
                f"{_human(agg['saved']):>10} chars saved  "
                f"{_human(agg['saved_tokens'] or agg['saved'] // LEGACY_CHARS_PER_TOKEN):>9} tok est  "
                f"{agg['capped']:>3} capped"
            )
        print(
            f"  {'TOTAL':<28} {len(records):>3} reqs  "
            f"{_human(total_saved):>10} chars saved  "
            f"{_human(retro_saved_tokens):>9} tok est"
        )
        return 0

    # Specific session id.
    sid_str = args.session_id
    session_recs = [r for r in records if str(r.get("session_id") or "-") == sid_str]
    session_saved = sum(_saved(r) for r in session_recs)
    session_saved_tokens = sum(_saved_tokens(r) for r in session_recs)
    session_capped = sum(int(r.get("results_capped") or 0) for r in session_recs)
    print(f"SOMA compression savings for session {sid_str}")
    print(f"  requests compressed:  {len(session_recs)}")
    print(f"  chars saved in this:  {_human(session_saved)}")
    print(
        f"  est. tokens saved:    "
        f"{_human(session_saved_tokens or session_saved // LEGACY_CHARS_PER_TOKEN)}"
    )
    print(f"  tool results capped:  {_human(session_capped)}")
    print(f"  grand total chars:    {_human(total_saved)}")
    if not session_recs:
        print("  (no compressed requests recorded for this session id)")
    return 0


if __name__ == "__main__":
    sys.exit(_main())