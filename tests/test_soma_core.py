#!/usr/bin/env python3
"""Behavioural tests for the SOMA compressor.

These assert the properties the compressor documents, using fixtures generated
here rather than recorded data, so the suite is self-contained and readable.

Run:
    python3 tests/test_compressor.py
    .venv/bin/python tests/test_compressor.py     # with scikit-learn installed
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE = ROOT / "soma_compressor.py"


def load():
    spec = importlib.util.spec_from_file_location("soma_compressor", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


soma = load()

FLOOR = soma.MIN_PASSTHROUGH_CHARS
CMP = soma.CMP_START


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def log_lines(n: int) -> str:
    return "".join(
        f"2026-07-31T04:{i%60:02d}:{(i*7)%60:02d}Z INFO worker: req={i} lat={i%900}ms\n"
        for i in range(n)
    )


def text_of(size: int) -> str:
    body = log_lines(size // 55 + 60)
    return body[:size]


def assembly(*messages) -> dict:
    return {"messages": list(messages), "session_id": "test"}


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def call(cid: str, name: str = "read", **args) -> dict:
    return {"role": "assistant",
            "content": [{"type": "toolCall", "id": cid, "name": name, "arguments": args or {"path": "f.log"}}]}


def result(cid: str, text: str) -> dict:
    return {"role": "toolResult", "toolCallId": cid, "content": text}


def result_text(out: dict, index: int) -> str:
    content = out["messages"][index]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))


def chars(messages) -> int:
    total = 0
    for m in messages:
        c = m.get("content")
        total += len(c) if isinstance(c, str) else len(json.dumps(c))
    return total


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def small_results_pass_through_untouched():
    """At or below the floor, nothing is modified at all."""
    body = text_of(FLOOR - 500)
    out = soma.handle_assemble(assembly(user("go"), call("c1"), result("c1", body)))
    assert out["baseMiner"]["resultsCapped"] == 0, out["baseMiner"]
    assert result_text(out, 2) == body, "content changed below the passthrough floor"


@case
def exactly_at_floor_is_untouched():
    body = "x" * FLOOR
    out = soma.handle_assemble(assembly(user("go"), call("c1"), result("c1", body)))
    assert out["baseMiner"]["resultsCapped"] == 0
    assert result_text(out, 2) == body


@case
def oversized_results_are_compressed_and_marked():
    """Above the floor a result shrinks and carries the CMP markers, so the model
    can tell that shortening happened."""
    body = text_of(120_000)
    out = soma.handle_assemble(assembly(user("go"), call("c1"), result("c1", body)))
    got = result_text(out, 2)
    assert out["baseMiner"]["resultsCapped"] == 1, out["baseMiner"]
    assert CMP in got, "no compression marker"
    assert len(got) < len(body), f"not smaller: {len(got)} vs {len(body)}"


@case
def compression_is_idempotent():
    """Re-running over compressed output is a fixed point."""
    payload = assembly(user("go"), call("c1"), result("c1", text_of(120_000)))
    once = soma.handle_assemble(payload)
    twice = soma.handle_assemble({"messages": once["messages"], "session_id": "test"})
    assert twice["baseMiner"]["resultsCapped"] == 0, "re-compressed already-compressed input"
    assert result_text(once, 2) == result_text(twice, 2)


@case
def never_inflates():
    """Output is never larger than input, at any size."""
    for size in (500, FLOOR - 1, FLOOR, FLOOR + 1, 30_000, 60_000, 200_000):
        payload = assembly(user("go"), call("c1"), result("c1", text_of(size)))
        before = chars(payload["messages"])
        out = soma.handle_assemble(payload)
        after = chars(out["messages"])
        assert after <= before, f"size {size}: grew from {before} to {after}"


@case
def load_bearing_lines_survive():
    """Failing tests, assertions, tracebacks, paths and diffs are kept even when
    they sit inside a region being trimmed."""
    signal = [
        "AssertionError: expected 3 got 4",
        "Traceback (most recent call last):",
        '  File "/srv/app/billing.py", line 42, in apply_tax',
        "+++ b/billing.py",
    ]
    filler = [f"routine line {i} carrying nothing of interest" for i in range(4000)]
    body = "\n".join(filler[:2000] + signal + filler[2000:])
    out = soma.handle_assemble(assembly(user("go"), call("c1"), result("c1", body)))
    got = result_text(out, 2)
    assert out["baseMiner"]["resultsCapped"] == 1
    for line in signal:
        assert line in got, f"load-bearing line dropped: {line!r}"


@case
def thinking_blocks_are_removed():
    payload = assembly(
        user("go"),
        {"role": "assistant", "content": [
            {"type": "thinking", "text": "internal reasoning " * 200},
            {"type": "text", "text": "the answer"}]},
    )
    out = soma.handle_assemble(payload)
    blob = json.dumps(out["messages"])
    assert "internal reasoning" not in blob, "thinking block survived"
    assert "the answer" in blob, "visible text was dropped"


@case
def tool_call_pairing_is_preserved():
    """Every tool call keeps a matching result: no orphans are introduced."""
    msgs = [user("go")]
    for i in range(4):
        msgs.append(call(f"c{i}"))
        msgs.append(result(f"c{i}", text_of(40_000)))
    out = soma.handle_assemble(assembly(*msgs))
    blob = json.dumps(out["messages"])
    for i in range(4):
        assert f'"c{i}"' in blob, f"tool call c{i} lost"


@case
def loop_guard_fires_on_repeated_identical_calls():
    msgs = [user("go")]
    for i in range(6):
        msgs.append({"role": "assistant", "content": [
            {"type": "toolCall", "id": f"c{i}", "name": "exec", "arguments": {"command": "ls"}}]})
        msgs.append(result(f"c{i}", "identical output"))
    out = soma.handle_assemble(assembly(*msgs))
    assert out["baseMiner"]["loopGuardFired"], "loop guard did not fire on 6 identical calls"
    assert "loop_detected" in json.dumps(out["messages"][-1])


@case
def loop_guard_silent_on_varied_work():
    msgs = [user("go")]
    for i in range(6):
        msgs.append(call(f"c{i}", name="read", path=f"file{i}.py"))
        msgs.append(result(f"c{i}", f"contents of file {i}"))
    out = soma.handle_assemble(assembly(*msgs))
    assert not out["baseMiner"]["loopGuardFired"], "loop guard fired on legitimate varied work"


@case
def handles_degenerate_input():
    """Empty, missing and null inputs must not raise."""
    for payload in ({"messages": []}, {}, {"messages": [{"role": "user", "content": None}]},
                    {"messages": [{"role": "toolResult", "content": []}]}):
        out = soma.handle_assemble(payload)
        assert out["assembled"] is True, payload


@case
def accepts_both_content_shapes():
    """Content may be a bare string or a block list; both are compressed."""
    body = text_of(120_000)
    as_str = soma.handle_assemble(assembly(user("go"), call("c1"), result("c1", body)))
    as_blocks = soma.handle_assemble(assembly(
        user("go"), call("c1"),
        {"role": "toolResult", "toolCallId": "c1", "content": [{"type": "text", "text": body}]}))
    assert as_str["baseMiner"]["resultsCapped"] == 1, "string content not compressed"
    assert as_blocks["baseMiner"]["resultsCapped"] == 1, "block content not compressed"


@case
def role_spelling_is_normalised():
    """toolResult / tool_result / TOOLRESULT all count as a tool result."""
    body = text_of(120_000)
    for role in ("toolResult", "tool_result", "TOOLRESULT", "toolresult"):
        out = soma.handle_assemble(assembly(
            user("go"), call("c1"), {"role": role, "toolCallId": "c1", "content": body}))
        assert out["baseMiner"]["resultsCapped"] == 1, f"role {role!r} not recognised"


@case
def is_deterministic_across_processes():
    """Same input, same output, regardless of hash seed. Relied on for cache
    stability: the emitted prefix must not shift between turns."""
    payload = assembly(user("go"), call("c1"), result("c1", text_of(60_000)))
    script = (
        "import importlib.util,json,sys,hashlib\n"
        f"spec=importlib.util.spec_from_file_location('m',{str(MODULE)!r})\n"
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "p=json.load(sys.stdin)\n"
        "print(hashlib.sha256(json.dumps(m.handle_assemble(p),sort_keys=True).encode()).hexdigest())\n"
    )
    seen = set()
    for seed in ("0", "1", "42"):
        proc = subprocess.run([sys.executable, "-c", script], input=json.dumps(payload),
                              capture_output=True, text=True,
                              env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"})
        assert proc.returncode == 0, proc.stderr[:400]
        seen.add(proc.stdout.strip())
    assert len(seen) == 1, f"output varied across hash seeds: {seen}"


@case
def cli_contract_holds():
    """`soma_compressor.py assemble` reads JSON on stdin, writes one object out."""
    payload = assembly(user("go"), call("c1"), result("c1", text_of(120_000)))
    proc = subprocess.run([sys.executable, str(MODULE), "assemble"],
                          input=json.dumps(payload), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[:400]
    reply = json.loads(proc.stdout)
    assert reply["ok"] is True, reply
    assert reply["result"]["baseMiner"]["resultsCapped"] == 1


@case
def rejects_unknown_event():
    proc = subprocess.run([sys.executable, str(MODULE), "not_an_event"],
                          input="{}", capture_output=True, text=True)
    reply = json.loads(proc.stdout)
    assert reply["ok"] is False and "event" in reply["error"].lower(), reply


# --- pytest collection shim -------------------------------------------------
# The original suite runs via main()/CASES. To also run under pytest, alias each
# registered case to a test_* name. No test logic is changed.
def _make_pytest_test(fn):
    def _test() -> None:
        fn()
    _test.__name__ = f"test_{fn.__name__}"
    _test.__doc__ = fn.__doc__
    return _test


for _fn in CASES:
    globals()[f"test_{_fn.__name__}"] = _make_pytest_test(_fn)


def main() -> int:
    try:
        import sklearn  # noqa: F401
        scorer = "scikit-learn (TF-IDF)"
    except ImportError:
        scorer = "FALLBACK distinct-token scoring — install scikit-learn for reference behaviour"
    print(f"line scorer: {scorer}\n")

    failed = []
    for fn in CASES:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as err:
            failed.append((fn.__name__, str(err)))
            print(f"  FAIL  {fn.__name__}: {err}")
        except Exception as err:  # noqa: BLE001
            failed.append((fn.__name__, f"{type(err).__name__}: {err}"))
            print(f"  ERROR {fn.__name__}: {type(err).__name__}: {err}")

    print(f"\n{len(CASES)-len(failed)}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
