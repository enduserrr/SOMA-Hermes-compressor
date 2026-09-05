#!/usr/bin/env python3
"""soma-mini-bench — light offline benchmark: Hermes default compressor vs SOMA.

A deliberately lightweight, deterministic, zero-LLM alternative to
DendriteHQ/SOMA-benchmark (which runs full SWE-bench_Verified through Docker +
LLM backends — thousands of API calls and a container fleet). This tool
replays synthetic-but-realistic agent histories through the two context
engines exactly as Hermes would call them, offline, in seconds, and measures
both efficiency (chars reclaimed from the request the model would see) and
fidelity (are the load-bearing facts still present?).

Usage (humans and agents alike):
    bench/soma-mini-bench run                  # full suite, human table
    bench/soma-mini-bench run --json           # machine-readable (agents)
    bench/soma-mini-bench run --scenario code-read
    bench/soma-mini-bench run --baseline-only  # default compressor only
    bench/soma-mini-bench run --soma-only      # SOMA only
    bench/soma-mini-bench scenarios            # list scenarios
    bench/soma-mini-bench report results.json  # re-print a saved JSON result

Requires the Hermes venv (the `bench/soma-mini-bench` launcher execs it; the
.py needs to be run with the venv python directly). Exit codes: 0 ok, 2
usage error. Every run saves JSON to <plugin>/bench_results/.

What is measured, per scenario (before/after = chars of message content the
provider would receive, needles = facts that must survive):

  baseline/default engine (ContextCompressor), via its real entry points:
    - select_context: inherited ABC default -> always a no-op (shown once)
    - prune_tool_results_only @20K tokens: the deterministic proactive prune
      (config default here is 0=off; 20K models an operator who enabled it)
    - compress(force): the full compaction path. OFFLINE CAVEAT: the LLM
      middle-summary cannot run without credentials, so it degrades to the
      deterministic fallback summary — the no-API worst case for the
      summarizer. With a live summary model the middle would be summarized
      (lossy but possibly needle-preserving), not dropped.

  soma (SomaEngine):
    - select_context replayed per turn, as conversation_loop calls it. SOMA
      compresses BOTH envelope shapes: read_file-style ({"content": ...})
      and terminal-style ({"output": ..., "exit_code": N}) — unwrapped,
      inner text compressed, re-wrapped with metadata preserved.

Fidelity needles are scenario-specific (FAILED lines, FATAL/traceback lines,
signatures, imports) and checked against the final request each engine would
send. Determinism: fixed seed, pure functions both sides.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import random
import sys
import time

HERMES_REPO = os.environ.get(
    "HERMES_REPO", os.path.expanduser("~/.hermes/hermes-agent")
)
# Plugin root = repo root (this script lives in <repo>/bench/). Override with
# SOMA_DIR to benchmark a deployed copy elsewhere on the machine.
SOMA_DIR = os.environ.get(
    "SOMA_DIR",
    str(pathlib.Path(__file__).resolve().parent.parent),
)
RESULTS_DIR = pathlib.Path(SOMA_DIR) / "bench_results"
CHARS_PER_TOKEN = 4  # matches both engines' rough estimator
RNG_SEED = 20260905

if HERMES_REPO not in sys.path:
    sys.path.insert(0, HERMES_REPO)

# ---------------------------------------------------------------------------
# Engine loading (lazy so `scenarios` works without the repo)
# ---------------------------------------------------------------------------

_soma_engine_cls = None
_default_engine_cls = None


def load_engines():
    global _soma_engine_cls, _default_engine_cls
    if _soma_engine_cls is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "soma_engine", os.path.join(SOMA_DIR, "engine.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Redirect accounting writes to a scratch file. This must happen
        # AFTER exec_module: engine.py binds ACCOUNTING_PATH at module level,
        # so a pre-exec assignment on the module object is silently
        # overwritten by the import.
        mod.ACCOUNTING_PATH = pathlib.Path(
            os.environ.get("TMPDIR", "/tmp")
        ) / "soma-mini-bench-accounting.jsonl"
        _soma_engine_cls = mod.SomaEngine
    if _default_engine_cls is None:
        from agent.context_compressor import ContextCompressor

        _default_engine_cls = ContextCompressor
    return _soma_engine_cls, _default_engine_cls


# ---------------------------------------------------------------------------
# Synthetic workload generators (fixed seed => deterministic)
# ---------------------------------------------------------------------------

def _code_file(path: str, n_defs: int, needles: list[str] | None = None) -> str:
    lines = [f"# module {path} — synthetic python source"]
    lines.append("import os")
    lines.append("import sys")
    lines.append("from typing import Any, Dict, List")
    if needles:
        lines.extend(needles)
    for k in range(1, n_defs):
        lines.append(f"def handler_{k}(x: int, y: int = {k}) -> int:")
        lines.append(f'    """docstring for handler_{k} — filler text block."""')
        lines.append(f"    return x * {k} + y")
        lines.append("")
    return "\n".join(lines)


def _test_output(n_tests: int, fails: list[str]) -> str:
    lines = ["============================= test session starts =============================="]
    lines.append("platform linux -- Python 3.11.16, pytest-9.1.1")
    lines.append(f"collected {n_tests} items")
    lines.append("")
    for i in range(1, n_tests + 1):
        lines.append(f"tests/test_mod_{i:03d}.py::test_case_{i} PASSED [  {i * 100 // n_tests:3d}%]")
    for f in fails:
        lines.append(f"tests/test_mod_{f}.py::test_case_{f} FAILED [ 100%]")
    lines.append("=========================== short summary for failures ===========================")
    for f in fails:
        lines.append(f"FAILED tests/test_mod_{f}.py::test_case_{f} - AssertionError: expected 42, got 13")
    lines.append(f"================= {len(fails)} failed, {n_tests} passed in 12.34s =================")
    return "\n".join(lines)


def _log_output(n_lines: int) -> str:
    lines = []
    for i in range(n_lines):
        lines.append(f"2026-09-05T10:{i % 60:02d}:{i % 60:02d} INFO worker-{i % 4} heartbeat tick {i} ok rss={400 + i}MB queue=0")
    lines.append("2026-09-05T10:59:59 ERROR worker-2 FATAL: disk quota exceeded on /var/lib/data/ingest.db")
    lines.append("Traceback (most recent call last):")
    lines.append('  File "/srv/app/ingest/main.py", line 231, in run')
    lines.append("    OSError: [Errno 122] Disk quota exceeded")
    return "\n".join(lines)


def _env_read(content: str) -> str:
    """read_file-style JSON envelope: SOMA unwraps and compresses these."""
    return json.dumps(
        {"content": content, "total_lines": content.count("\n") + 1,
         "file_size": len(content)}, ensure_ascii=False)


def _env_terminal(output: str, exit_code: int) -> str:
    """terminal-style JSON envelope: unwrapped by SOMA (both shapes handled)."""
    return json.dumps({"output": output, "exit_code": exit_code}, ensure_ascii=False)


def _tc(cid: str, name: str, args: dict) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


@dataclasses.dataclass
class Scenario:
    name: str
    desc: str
    build: object  # callable () -> list[dict]
    needles: list[str]

    def describe(self) -> str:
        return f"{self.name:14s} {self.desc}"


def _history_code_read_and_fix():
    """Agent reads 8 source files (~40K chars each, read_file envelopes), then a fix-all turn."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "read all modules under src/ and fix handler_500 in each"},
    ]
    for i in range(8):
        cid = f"call_r{i}"
        path = f"src/mod{i}.py"
        needles = (
            [f"# NEEDLE: bug marker in {path} — off-by-one at handler_500"]
            if i == 3 else None
        )
        text = _code_file(path, 700, needles)
        msgs.append(_tc(cid, "read_file", {"path": path}))
        msgs.append({"role": "tool", "tool_call_id": cid, "content": _env_read(text)})
        msgs.append({"role": "assistant", "content": f"read {path} (700 handlers)"})
    msgs.append({"role": "user", "content": "now patch handler_500 in every module you read"})
    return msgs


def _history_pytest_run():
    """Agent reads test sources, runs pytest (terminal envelope, 3 failures),
    keeps working, then must name the failures."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "the suite is red — investigate"},
    ]
    for i, path in enumerate(["tests/test_mod_007.py", "tests/test_mod_042.py",
                              "tests/test_mod_513.py", "tests/conftest.py"]):
        cid = f"call_f{i}"
        text = _code_file(path, 600)
        msgs.append(_tc(cid, "read_file", {"path": path}))
        msgs.append({"role": "tool", "tool_call_id": cid, "content": _env_read(text)})
        msgs.append({"role": "assistant", "content": f"read {path}"})
    msgs.append({"role": "user", "content": "ok now run the suite"})
    msgs.append(_tc("call_t1", "terminal", {"command": "pytest -x --tb=short"}))
    msgs.append({"role": "tool", "tool_call_id": "call_t1",
                 "content": _env_terminal(_test_output(60, fails=["007", "042", "513"]), 1)})
    msgs.append({"role": "assistant", "content": "3 failures seen"})
    msgs.append({"role": "user", "content": "check conftest for shared fixtures"})
    msgs.append(_tc("call_f9", "read_file", {"path": "tests/conftest.py"}))
    msgs.append({"role": "tool", "tool_call_id": "call_f9",
                 "content": _env_read(_code_file("tests/conftest.py", 200))})
    msgs.append({"role": "assistant", "content": "fixtures look fine"})
    msgs.append({"role": "user", "content": "so which tests failed and why? be specific"})
    return msgs


def _history_daemon_log():
    """Agent reads a 5K-line daemon log via read_file (content envelope), then
    must cite the exact crash."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "the ingest daemon crashed last night — check the logs"},
    ]
    msgs.append(_tc("call_l1", "read_file", {"path": "/var/log/ingest/daemon.log"}))
    msgs.append({"role": "tool", "tool_call_id": "call_l1", "content": _env_read(_log_output(5000))})
    msgs.append({"role": "assistant", "content": "scanning the log"})
    msgs.append({"role": "user", "content": "also check its config"})
    msgs.append(_tc("call_l2", "read_file", {"path": "/etc/ingest/daemon.conf"}))
    msgs.append({"role": "tool", "tool_call_id": "call_l2",
                 "content": _env_read("max_queue=1000\ndb_path=/var/lib/data/ingest.db\nworkers=4\n")})
    msgs.append({"role": "assistant", "content": "config noted"})
    msgs.append({"role": "user", "content": "what was the root cause? cite the exact error and file"})
    return msgs


def _history_mixed_long_session():
    """10-turn triage session blending code reads, test runs, log reads."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "triage this repo: logs, code, tests"},
    ]
    for i in range(10):
        kind = ["code", "tests", "logs"][i % 3]
        cid = f"call_m{i}"
        if kind == "code":
            path = f"src/svc{i}.py"
            text = _code_file(path, 800)
            msgs.append(_tc(cid, "read_file", {"path": path}))
            env = _env_read(text)
        elif kind == "tests":
            text = _test_output(40, fails=[f"{i:03d}"])
            msgs.append(_tc(cid, "terminal", {"command": "pytest tests/"}))
            env = _env_terminal(text, 1)
        else:
            text = _log_output(2500)
            msgs.append(_tc(cid, "read_file", {"path": f"/var/log/svc{i}.log"}))
            env = _env_read(text)
        msgs.append({"role": "tool", "tool_call_id": cid, "content": env})
        msgs.append({"role": "assistant", "content": f"step {i}: {kind} reviewed"})
    msgs.append({"role": "user", "content": "summarize: which module is broken and which test proves it"})
    return msgs


SCENARIOS = [
    Scenario("code-read", "8 read_file results (~40K chars each) then a fix-all turn",
             _history_code_read_and_fix,
             ["# NEEDLE: bug marker in src/mod3.py", "def handler_500(", "import os"]),
    Scenario("pytest", "read 4 test files, run pytest (3 FAILs), keep working, then name them",
             _history_pytest_run,
             ["FAILED tests/test_mod_042.py::test_case_042 - AssertionError: expected 42, got 13",
              "FAILED tests/test_mod_007.py", "FAILED tests/test_mod_513.py", "3 failed, 60 passed"]),
    Scenario("daemon-log", "5K-line daemon log via read_file — keep traceback + FATAL",
             _history_daemon_log,
             ["FATAL: disk quota exceeded on /var/lib/data/ingest.db",
              'File "/srv/app/ingest/main.py", line 231, in run',
              "OSError: [Errno 122] Disk quota exceeded"]),
    Scenario("mixed", "10-turn triage session blending code/tests/logs",
             _history_mixed_long_session,
             ["def handler_500(", "FAILED tests/test_mod_001.py",
              "FAILED tests/test_mod_007.py", "FATAL: disk quota exceeded"]),
]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _chars(messages):
    total = 0
    for m in messages:
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str):
                total += len(c)
    return total


def _unwrap_all_envelopes(messages):
    """Flatten JSON envelopes so needle search sees the raw inner text too."""
    out = []
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str) and c.lstrip().startswith("{"):
            try:
                obj = json.loads(c)
                if isinstance(obj, dict):
                    for key in ("content", "output"):
                        v = obj.get(key)
                        if isinstance(v, str):
                            out.append(v)
                            break
                    else:
                        out.append(c)
                    continue
            except Exception:
                pass
        out.append(c if isinstance(c, str) else "")
    return out


def _needle_hits(text_blobs, needles):
    kept = [n for n in needles if any(n in b for b in text_blobs)]
    return len(kept), len(needles)


def _fresh_default_engine(default_cls, context_length=128_000):
    # Mirror this host's real config (`hermes config get compression`):
    # threshold 0.5, lean tail, protect_last_n 2; proactive prune measured at
    # 20K tokens (config default is 0=off — shown as its own data point).
    return default_cls(
        model="bench-model",
        threshold_percent=0.50,
        protect_last_n=2,
        tail_mode="lean",
        config_context_length=context_length,
        proactive_prune_tokens=20_000,
    )


def run_baseline(history, default_cls, mode):
    """mode: 'prune' (prune_tool_results_only @20K) | 'compress' (force)."""
    eng = _fresh_default_engine(default_cls)
    toks = _chars(history) // CHARS_PER_TOKEN
    t0 = time.perf_counter()
    if mode == "prune":
        out, n = eng.prune_tool_results_only(list(history), current_tokens=toks)
        return out, time.perf_counter() - t0, {"pruned": n, "path": "prune_tool_results_only@20K"}
    out = eng.compress(list(history), current_tokens=toks, force=True)
    return out, time.perf_counter() - t0, {
        "path": "compress(force) — offline, fallback summary",
        "summary_fallback": bool(getattr(eng, "_last_summary_fallback_used", False)),
        "aborted": bool(getattr(eng, "_last_compress_aborted", False)),
    }


def run_soma_per_turn(history, soma_cls):
    """Mirror conversation_loop: assemble the request at each user turn and
    pass it through select_context. Returns (final_request, elapsed_s, n_turns)."""
    eng = soma_cls()
    eng.update_model("bench-model", 128_000)
    elapsed = 0.0
    req_points = [i for i, m in enumerate(history)
                  if isinstance(m, dict) and m.get("role") == "user"]
    if not req_points or req_points[-1] != len(history):
        req_points.append(len(history))
    last_out = list(history)
    for point in req_points:
        request = list(history[:point])
        t0 = time.perf_counter()
        selected = eng.select_context(request)
        elapsed += time.perf_counter() - t0
        last_out = selected if selected is not None else request
    return last_out, elapsed, len(req_points)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

W = 86
WIDTHS = [12, 18, 12, 10, 6, 8]


def _fmt_row(cols):
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, WIDTHS)).rstrip()


def print_table(results):
    print("=" * W)
    print("SOMA MINI-BENCH — Hermes default compressor vs SOMA (offline, deterministic)")
    print("=" * W)
    print(_fmt_row(["scenario", "engine", "chars-after", "chars-saved", "pct", "needles"]))
    print("-" * W)
    for r in results:
        print(_fmt_row([r["scenario"], r["engine"], f"{r['after']:,}",
                        f"{r['saved']:,}", f"{r['pct']}%",
                        f"{r['needles_kept']}/{r['needles_total']}"]))
        for k, v in r.get("notes", {}).items():
            print(f"        · {k}: {v}")
    print("-" * W)
    print("needles = load-bearing facts still present in the final request.")
    print("default/compress runs offline: the LLM summary degrades to its")
    print("deterministic fallback (no-API worst case for the summarizer).")


def cmd_run(args):
    SomaEngine, DefaultEngine = load_engines()
    scenarios = [s for s in SCENARIOS if not args.scenario or s.name == args.scenario]
    if args.scenario and not scenarios:
        print(f"unknown scenario: {args.scenario} (see `soma-mini-bench scenarios`)",
              file=sys.stderr)
        return 2
    results = []
    for sc in scenarios:
        history = sc.build()
        base_chars = _chars(history)

        if not args.soma_only:
            for mode in ("prune", "compress"):
                out, el, notes = run_baseline(history, DefaultEngine, mode)
                kept, total = _needle_hits(_unwrap_all_envelopes(out), sc.needles)
                after = _chars(out)
                results.append({
                    "scenario": sc.name, "engine": f"default/{mode}",
                    "before": base_chars, "after": after,
                    "saved": base_chars - after,
                    "pct": round(100 * (base_chars - after) / base_chars, 1),
                    "needles_kept": kept, "needles_total": total,
                    "elapsed_s": round(el, 3), "notes": notes,
                })
        if not args.baseline_only:
            out, el, turns = run_soma_per_turn(history, SomaEngine)
            kept, total = _needle_hits(_unwrap_all_envelopes(out), sc.needles)
            after = _chars(out)
            results.append({
                "scenario": sc.name, "engine": "soma",
                "before": base_chars, "after": after,
                "saved": base_chars - after,
                "pct": round(100 * (base_chars - after) / base_chars, 1),
                "needles_kept": kept, "needles_total": total,
                "elapsed_s": round(el, 3),
                "notes": {"path": "select_context per turn", "turns": turns},
            })

    payload = {"tool": "soma-mini-bench", "seed": RNG_SEED, "results": results}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_table(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / time.strftime("results-%Y%m%d-%H%M%S.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"saved: {out_path}", file=sys.stderr)
    return 0


def cmd_scenarios(_args):
    for s in SCENARIOS:
        print(s.describe())
    return 0


def cmd_report(args):
    data = json.loads(pathlib.Path(args.path).read_text())
    print_table(data["results"])
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="soma-mini-bench",
        description="Light offline benchmark: Hermes default compressor vs SOMA context engine.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="run the benchmark")
    pr.add_argument("--scenario", help="only this scenario (see `scenarios`)")
    pr.add_argument("--baseline-only", action="store_true", help="skip SOMA")
    pr.add_argument("--soma-only", action="store_true", help="skip default engine")
    pr.add_argument("--json", action="store_true", help="JSON output (for agents)")
    sub.add_parser("scenarios", help="list scenarios")
    prp = sub.add_parser("report", help="print a saved JSON result as a table")
    prp.add_argument("path")
    args = p.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "scenarios":
        return cmd_scenarios(args)
    if args.cmd == "report":
        return cmd_report(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
