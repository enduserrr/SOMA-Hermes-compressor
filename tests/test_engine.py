#!/usr/bin/env python3
"""Tests for the SomaEngine skeleton: token accounting and status (Task 2).

Scope: engine identity, update_from_response(), update_model(), get_status().
Compression behaviour (select_context, sizing table) is Task 3 and NOT tested
here beyond the stability guarantee that an untouched request is a no-op.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
from datetime import datetime

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent  # context_engine/soma

spec = importlib.util.spec_from_file_location("soma_engine", ROOT / "engine.py")
engine_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine_mod)

SomaEngine = engine_mod.SomaEngine


def make_engine() -> SomaEngine:
    return SomaEngine()


def _history_with_big_result(n_chars: int = 40_000, tool_call_id: str = "call_1"):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "read the file please"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": tool_call_id, "type": "function",
                 "function": {"name": "read_file", "arguments": "{\"path\": \"data.py\"}"}}
            ],
        },
        _big_tool_result(n_chars, tool_call_id),
    ]


class TestIdentity:
    def test_name_is_soma(self):
        assert make_engine().name == "soma"

    def test_subclasses_context_engine_abc(self):
        from agent.context_engine import ContextEngine

        assert isinstance(make_engine(), ContextEngine)


class TestUpdateFromResponse:
    def test_usage_dict_updates_counters(self):
        eng = make_engine()
        eng.update_from_response(
            {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}
        )
        assert eng.last_prompt_tokens == 100
        assert eng.last_completion_tokens == 40
        assert eng.last_total_tokens == 140

    def test_missing_keys_default_to_zero(self):
        eng = make_engine()
        eng.update_from_response({})
        assert eng.last_prompt_tokens == 0
        assert eng.last_completion_tokens == 0
        assert eng.last_total_tokens == 0

    def test_non_numeric_garbage_does_not_raise(self):
        eng = make_engine()
        eng.update_from_response({"prompt_tokens": None})
        assert eng.last_prompt_tokens == 0

    def test_fail_open_on_exception_keeps_previous_counters(self):
        eng = make_engine()
        eng.update_from_response(
            {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
        )
        # A malformed payload must never break the caller (fail-open).
        eng.update_from_response({"prompt_tokens": "not-a-number"})
        assert eng.last_prompt_tokens == 7
        assert eng.last_completion_tokens == 2
        assert eng.last_total_tokens == 9


class TestUpdateModel:
    def test_sets_context_length_and_threshold(self):
        eng = make_engine()
        eng.update_model("some-model", context_length=100_000)
        assert eng.context_length == 100_000
        assert eng.threshold_tokens == int(100_000 * eng.threshold_percent)

    def test_threshold_uses_default_percent(self):
        from agent.context_engine import ContextEngine

        eng = make_engine()
        eng.update_model("m", context_length=200_000)
        assert eng.threshold_percent == ContextEngine.threshold_percent
        assert eng.threshold_tokens == int(200_000 * 0.75)


class TestGetStatus:
    def test_reports_threshold_and_context_length(self):
        eng = make_engine()
        eng.update_model("m", context_length=100_000)
        status = eng.get_status()
        assert status["threshold_tokens"] == eng.threshold_tokens
        assert status["context_length"] == 100_000

    def test_status_shape_matches_host_contract(self):
        eng = make_engine()
        eng.update_from_response(
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        )
        status = eng.get_status()
        for key in (
            "last_prompt_tokens",
            "threshold_tokens",
            "context_length",
            "usage_percent",
            "compression_count",
        ):
            assert key in status
        assert status["last_prompt_tokens"] == 50
        assert 0 <= status["usage_percent"] <= 100


class TestStabilityGuarantees:
    def test_select_context_noop_returns_none(self):
        """Nothing changed -> stable no-op (Task 2 skeleton must not touch requests)."""
        eng = make_engine()
        msgs = [{"role": "user", "content": "hello"}]
        assert eng.select_context(list(msgs)) is None


# ---------------------------------------------------------------------------
# Task 3: select_context() compression pass
# ---------------------------------------------------------------------------

def _file_like_lines(n_lines: int, vocab: str = "alpha beta gamma delta epsilon") -> str:
    """A realistic file-read shape: mostly prose lines, occasional paths (pinned)."""
    lines = []
    for i in range(n_lines):
        if i % 10 == 0:
            lines.append(f"line {i}: {vocab} value_{i} /tmp/data/file_{i}.py")
        else:
            lines.append(f"line {i}: {vocab} value_{i} plain prose without any signal markers")
    return "\n".join(lines)


def _big_tool_result(n_chars: int = 40_000, tool_call_id: str = "call_1") -> dict:
    content = _file_like_lines((n_chars // 64) + 1)
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class TestSelectContext:
    def setup_method(self):
        self.soma = engine_mod._load_soma()

    def test_no_oversized_results_returns_none(self):
        """Cache-stable no-op: no result exceeds the 16K passthrough floor."""
        eng = make_engine()
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "tool_call_id": "c1", "content": _file_like_lines(100)},
        ]
        assert eng.select_context(msgs) is None

    def test_oversized_result_replaced_and_marked(self):
        eng = make_engine()
        msgs = _history_with_big_result()
        original_len = len(msgs[3]["content"])
        out = eng.select_context(msgs)
        assert out is not None
        assert out is not msgs
        compressed = out[3]["content"]
        assert "[[CMP]]" in compressed and "[[/CMP]]" in compressed
        assert len(compressed) < original_len  # never inflate
        # non-tool messages pass through byte-identical
        assert out[0] == msgs[0] and out[1] == msgs[1]
        assert out[2] == msgs[2]

    def test_tool_call_pairing_preserved(self):
        eng = make_engine()
        msgs = _history_with_big_result()
        out = eng.select_context(msgs)
        assert out is not None
        # assistant tool_calls message untouched, result still present
        assert out[2] == msgs[2]
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_1"

    def test_orphan_violation_falls_back(self):
        """If rewriting would orphan calls/results, return the original list."""
        eng = make_engine()
        msgs = _history_with_big_result()
        real_orphans = self.soma.orphan_ids

        def evil_orphans(messages):
            # pretend the OUTPUT (a new list object) orphans a call -> adapter must revert
            if messages is msgs:
                return real_orphans(messages)
            return ({"orphaned_result"}, {"orphaned_call"})

        orig = self.soma.orphan_ids
        self.soma.orphan_ids = evil_orphans
        try:
            assert eng.select_context(msgs) is None
        finally:
            self.soma.orphan_ids = orig

    def test_never_inflate_all_pinned_lines(self):
        """Content whose lines are all pinned (code skeleton) can't shrink -> no-op."""
        eng = make_engine()
        pinned = "import os\n" * 6000  # 6000 chars of import lines, > 16K floor
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": pinned * 3}]
        result = eng.select_context(msgs)
        assert result is None or len(result[0]["content"]) < len(pinned * 3)

    def test_oversized_assistant_message_untouched(self):
        """SOMA only caps tool results; assistant/system/user pass verbatim."""
        eng = make_engine()
        big = _file_like_lines(1000)
        msgs = [
            {"role": "system", "content": big},
            {"role": "assistant", "content": big},
            {"role": "user", "content": big},
        ]
        assert eng.select_context(msgs) is None

    def test_compressor_exception_fail_open(self, monkeypatch):
        """No exception may escape select_context -> identity (None)."""

        def boom(*a, **k):
            raise RuntimeError("injected compressor failure")

        monkeypatch.setattr(self.soma, "extractive_compress", boom)
        eng = make_engine()
        msgs = _history_with_big_result()
        assert eng.select_context(msgs) is None

    def test_orphan_ids_exception_fail_open(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("injected orphan-check failure")

        monkeypatch.setattr(self.soma, "orphan_ids", boom)
        eng = make_engine()
        msgs = _history_with_big_result()
        assert eng.select_context(msgs) is None

    def test_history_never_mutated(self):
        import copy as _copy
        eng = make_engine()
        msgs = _history_with_big_result()
        snapshot = _copy.deepcopy(msgs)
        eng.select_context(msgs)
        assert msgs == snapshot

    def test_idempotent_second_call_is_noop(self):
        """Already-[[CMP]] results are fixed points -> second call returns None."""
        eng = make_engine()
        msgs = _history_with_big_result()
        out = eng.select_context(msgs)
        assert out is not None
        assert eng.select_context(out) is None

    def test_json_envelope_tool_result_compressed(self):
        """Hermes read_file results are single-line JSON envelopes — unwrap,
        compress inner text, re-wrap (regression: envelope was a fixed point)."""
        eng = make_engine()
        inner = _file_like_lines(700)
        envelope = json.dumps(
            {"content": inner, "total_lines": 700, "file_size": len(inner), "truncated": False},
            ensure_ascii=False,
        )
        msgs = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": envelope},
        ]
        out = eng.select_context(msgs)
        assert out is not None
        assert len(out[1]["content"]) < len(envelope)
        parsed = json.loads(out[1]["content"])
        assert "[[CMP]]" in parsed["content"]
        assert parsed["total_lines"] == 700  # envelope metadata preserved
        # idempotent: already-compressed envelope is a fixed point
        assert eng.select_context(out) is None

    def test_json_envelope_small_content_untouched(self):
        eng = make_engine()
        envelope = json.dumps({"content": "tiny", "total_lines": 1})
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": envelope}]
        assert eng.select_context(msgs) is None

    def test_json_envelope_unparseable_untouched(self):
        eng = make_engine()
        msgs = [{"role": "tool", "tool_call_id": "c1",
                 "content": "{not json but long enough " + "x" * 20000 + "}"}]
        assert eng.select_context(msgs) is None

    def test_terminal_envelope_compressed(self):
        """terminal results are {"output": ..., "exit_code": N} envelopes —
        same unwrap/compress/re-wrap path (regression: 65-87K-char terminal
        envelopes from real history rode through untouched)."""
        eng = make_engine()
        lines = []
        for i in range(1200):
            if i % 300 == 7:
                lines.append(
                    f"FAILED tests/test_mod_{i:03d}.py::test_case_{i} - "
                    "AssertionError: expected 42, got 13"
                )
            else:
                lines.append(f"plain run output line {i} no signal markers here")
        inner = "\n".join(lines)
        envelope = json.dumps(
            {"output": inner, "exit_code": 1, "error": None}, ensure_ascii=False
        )
        msgs = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": envelope},
        ]
        out = eng.select_context(msgs)
        assert out is not None
        assert len(out[1]["content"]) < len(envelope)  # never inflate
        parsed = json.loads(out[1]["content"])  # envelope still valid JSON
        assert "[[CMP]]" in parsed["output"]
        assert parsed["exit_code"] == 1  # envelope metadata preserved
        assert "FAILED tests/test_mod_007.py::test_case_7" in parsed["output"]
        # idempotent: already-compressed envelope is a fixed point
        assert eng.select_context(out) is None

    def test_terminal_envelope_small_output_untouched(self):
        eng = make_engine()
        envelope = json.dumps({"output": "ok", "exit_code": 0})
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": envelope}]
        assert eng.select_context(msgs) is None

    def test_terminal_envelope_no_output_key_untouched(self):
        """An envelope-shaped dict with no text field must not be rewritten."""
        eng = make_engine()
        envelope = json.dumps({"exit_code": 1, "error": "boom", "cwd": "/tmp"})
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": envelope}]
        assert eng.select_context(msgs) is None

    def test_envelope_unwrap_returns_key(self):
        """The unwrap reports which field held the text (re-wrap target)."""
        _, obj, key = engine_mod._unwrap_json_envelope(
            json.dumps({"output": "x", "exit_code": 0})
        )
        assert key == "output"
        _, _, key = engine_mod._unwrap_json_envelope(
            json.dumps({"content": "x", "total_lines": 1})
        )
        assert key == "content"
        assert engine_mod._unwrap_json_envelope("plain text") is None
        assert engine_mod._unwrap_json_envelope('{"exit_code": 0}') is None

    def test_passthrough_floor_is_32k(self):
        """Sizing rule (2026-09-05 floor sweep): under 32K untouched, over 32K
        capped. 16K floor would compress an 18K active-file read and drop
        unpinned body lines for <=12% savings."""
        eng = make_engine()
        # 18K file read envelope: under the 32K floor -> untouched
        text_18k = _file_like_lines(280)  # ~18K
        envelope = json.dumps({"content": text_18k, "total_lines": 280}, ensure_ascii=False)
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": envelope}]
        assert len(envelope) < 32_000
        assert eng.select_context(msgs) is None
        # engine tunable drives the core floor
        assert engine_mod.PASSTHROUGH_CHARS == 32_000
        soma_mod = engine_mod._load_soma()
        assert soma_mod.MIN_PASSTHROUGH_CHARS == 32_000

    def test_over_32k_capped_at_32k(self):
        """A 60K file read compresses to roughly the 32K cap."""
        eng = make_engine()
        text = _file_like_lines(950)  # ~60K
        envelope = json.dumps({"content": text, "total_lines": 950}, ensure_ascii=False)
        msgs = [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": envelope},
        ]
        out = eng.select_context(msgs)
        assert out is not None
        after = len(out[1]["content"])
        assert after < len(envelope)
        # capped near 32K (envelope JSON overhead + [[CMP]] wrapper tolerated)
        inner = json.loads(out[1]["content"])["content"]
        assert 30_000 <= len(inner) <= 35_000
        assert "[[CMP]]" in inner

    def test_real_world_shapes_never_inflate(self):
        """Long read_file output, pytest dump, JSON blob: None or strictly smaller."""
        eng = make_engine()
        pytest_dump = "\n".join(
            f"FAILED tests/test_mod_{i}.py::test_case_{i} - AssertionError: expected {i} got {i + 1}"
            for i in range(400)
        )
        json_blob = '{"items": [' + ",".join(
            f'{{"id": {i}, "name": "record_{i}", "data": "{"x" * 20}"}}' for i in range(600)
        ) + "]}"

        for content in (
            "read_file output\n" + _file_like_lines(1200),
            pytest_dump,
            json_blob,
        ):
            msgs = [
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "t", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": content},
            ]
            original = len(content)
            out = eng.select_context(msgs)
            if out is not None:
                assert len(out[-1]["content"]) < original


# ---------------------------------------------------------------------------
# Task 4: should_compress() overflow gate + minimal fallback compress()
# ---------------------------------------------------------------------------

class TestShouldCompress:
    def test_overflow_proven_returns_true(self):
        """last_total_tokens > context_length -> should_compress True."""
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
        )
        assert eng.last_total_tokens > eng.context_length
        assert eng.should_compress() is True

    def test_normal_turn_returns_false(self):
        """last_total_tokens <= context_length -> no compression."""
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600}
        )
        assert eng.should_compress() is False

    def test_context_length_zero_never_fires(self):
        """No known context window -> cannot prove overflow -> False."""
        eng = make_engine()
        eng.context_length = 0
        eng.last_total_tokens = 50_000
        assert eng.should_compress() is False

    def test_no_usage_yet_returns_false(self):
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        assert eng.should_compress() is False

    def test_compress_without_overflow_is_identity(self):
        """No proven overflow -> compress() returns messages unchanged."""
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        out = eng.compress(list(msgs))
        assert out == msgs

    def test_compress_overflow_returns_shortened_fallback(self):
        """Proven overflow -> delegates to nested built-in (LLM summarizer)."""
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
        )
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "y" * 5000},
        ]
        sentinel = [{"role": "user", "content": "delegated"}]
        nested = eng._get_fallback_compressor()
        assert nested is not None
        orig = nested.compress
        nested.compress = lambda *a, **k: sentinel
        try:
            assert eng.compress(msgs) is sentinel
        finally:
            nested.compress = orig

    def test_fallback_preserves_system_prompt(self):
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
        )
        eng._get_fallback_compressor = lambda: None  # force synthetic path
        system = {"role": "system", "content": "You are a helpful assistant."}
        msgs = [system, {"role": "user", "content": "x" * 5000}]
        out = eng.compress(msgs)
        assert out[0] == system

    def test_fallback_without_system_prompt(self):
        """No leading system message -> output is just the overflow note."""
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
        )
        eng._get_fallback_compressor = lambda: None  # force synthetic path
        msgs = [{"role": "user", "content": "x" * 5000}]
        out = eng.compress(msgs)
        assert len(out) == 1
        assert out[0]["role"] == "user"

    def test_fallback_never_mutates_input(self):
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
        )
        import copy as _copy
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 100},
        ]
        snapshot = _copy.deepcopy(msgs)
        eng.compress(msgs)
        assert msgs == snapshot

    def test_compress_fail_open(self):
        """Even the fallback must never raise into the agent loop."""
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
        )
        # malformed messages (not dicts) must not raise
        out = eng.compress(["not-a-dict", 42])
        assert isinstance(out, list)


# ---------------------------------------------------------------------------
# Task 5: per-request JSONL accounting + lazy scikit-learn
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _accounting_sandbox(tmp_path, monkeypatch):
    """Redirect ALL accounting writes away from the real file, every test."""
    path = tmp_path / "accounting.jsonl"
    monkeypatch.setattr(engine_mod, "ACCOUNTING_PATH", path)
    yield path


@pytest.fixture
def accounting_path(_accounting_sandbox):
    """Same redirected path, exposed to tests that assert on its contents."""
    return _accounting_sandbox


def _lines(path) -> list:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


class TestAccounting:
    def test_select_context_change_appends_one_line(self, accounting_path):
        eng = make_engine()
        out = eng.select_context(_history_with_big_result())
        assert out is not None
        lines = _lines(accounting_path)
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["reason"] in ("passthrough_native", "near_passthrough")
        assert isinstance(record["input_est_chars"], int)
        assert isinstance(record["output_est_chars"], int)
        assert isinstance(record["results_capped"], int)
        assert record["results_capped"] >= 1
        assert record["output_est_chars"] < record["input_est_chars"]
        # timestamp parses as ISO-8601
        datetime.fromisoformat(record["timestamp"])

    def test_select_context_noop_appends_zero_lines(self, accounting_path):
        eng = make_engine()
        msgs = [{"role": "user", "content": "hello"}]
        assert eng.select_context(msgs) is None
        assert _lines(accounting_path) == []

    def test_every_call_appends_one_line(self, accounting_path):
        eng = make_engine()
        assert eng.select_context(_history_with_big_result()) is not None
        assert eng.select_context(_history_with_big_result(tool_call_id="call_2")) is not None
        assert len(_lines(accounting_path)) == 2

    def test_all_lines_parse_as_json(self, accounting_path):
        eng = make_engine()
        eng.select_context(_history_with_big_result())
        eng.select_context([{"role": "user", "content": "hi"}])
        records = [json.loads(line) for line in _lines(accounting_path)]
        assert len(records) == 1  # no-op call above wrote nothing

    def test_accounting_error_never_breaks_request(self, monkeypatch, accounting_path):
        def boom(*args, **kwargs):
            raise OSError("injected accounting failure")

        monkeypatch.setattr("builtins.open", boom)
        eng = make_engine()
        msgs = _history_with_big_result()
        out = eng.select_context(msgs)  # must not raise
        assert out is not None
        assert len(out[3]["content"]) < len(msgs[3]["content"])

    def test_real_accounting_path_not_used(self, _accounting_sandbox):
        """Redirected writes must never touch the real accounting.jsonl."""
        default_path = pathlib.Path(engine_mod.__file__).parent / "accounting.jsonl"
        before = default_path.read_text(encoding="utf-8") if default_path.exists() else None
        eng = make_engine()
        eng.select_context(_history_with_big_result())
        assert _lines(_accounting_sandbox)  # redirect captured the write
        after = default_path.read_text(encoding="utf-8") if default_path.exists() else None
        assert after == before  # real file untouched by this test run


class TestLazySklearn:
    def test_engine_import_does_not_load_sklearn(self):
        """Re-importing engine.py must not pull scikit-learn into sys.modules.

        Runs in a subprocess: earlier tests in this session legitimately load
        sklearn via real compression calls, which would pollute sys.modules.
        """
        code = (
            "import importlib.util, pathlib, sys\n"
            "root = pathlib.Path(r'{root}')\n"
            "spec = importlib.util.spec_from_file_location('soma_engine_lazy', root / 'engine.py')\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "assert 'sklearn' not in sys.modules, 'sklearn loaded at engine import'\n"
            "mod._load_soma()\n"
            "assert 'sklearn' not in sys.modules, 'sklearn loaded at _load_soma()'\n"
            "print('ok')\n"
        ).format(root=ROOT)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_fallback_scorer_when_sklearn_broken(self, monkeypatch):
        """TF-IDF import fails -> distinct-token fallback still compresses."""
        monkeypatch.setitem(sys.modules, "sklearn", None)
        # purge any cached submodule entries so the fallback path is exercised
        for key in list(sys.modules):
            if key.startswith("sklearn"):
                monkeypatch.delitem(sys.modules, key)
        # _line_scores imports inside the function body each call, so the
        # poisoned sys.modules entry above forces the except branch.
        soma = engine_mod._load_soma()
        text = _file_like_lines(700)
        target = len(text) // 2
        out, changed = soma.extractive_compress(text, target)
        assert changed is True
        assert 0 < len(out) < len(text)

    def test_fallback_scorer_via_select_context(self, monkeypatch):
        """End-to-end: with sklearn poisoned, select_context still shrinks."""
        monkeypatch.setitem(sys.modules, "sklearn", None)
        for key in list(sys.modules):
            if key.startswith("sklearn"):
                monkeypatch.delitem(sys.modules, key)
        eng = make_engine()
        msgs = _history_with_big_result()
        out = eng.select_context(msgs)
        if out is not None:
            assert len(out[3]["content"]) < len(msgs[3]["content"])
        else:
            pytest.fail("compression must still engage with sklearn unavailable")


class TestCompressorDelegation:
    """compress() delegates to a nested built-in ContextCompressor (LLM
    summarizer) for whole-session compaction; SOMA keeps select_context."""

    def _builtin_compressor(self, eng):
        return eng._get_fallback_compressor()

    def test_delegate_returns_built_in_list(self):
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
        )
        assert eng.should_compress() is True
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"}]
        out = eng.compress(msgs)
        assert isinstance(out, list) and all(isinstance(m, dict) for m in out)

    def test_delegation_uses_nested_compressor(self):
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
        )
        nested = self._builtin_compressor(eng)
        assert nested is not None
        sentinel = [{"role": "user", "content": "delegated"}]
        orig = nested.compress
        nested.compress = lambda *a, **k: sentinel
        try:
            msgs = [{"role": "user", "content": "u1"}]
            assert eng.compress(msgs) is sentinel
        finally:
            nested.compress = orig

    def test_manual_compress_always_delegates(self):
        """Manual /compress (force=True) delegates even without overflow."""
        eng = make_engine()
        eng.update_model("m", context_length=100000)
        nested = self._builtin_compressor(eng)
        assert nested is not None
        sentinel = [{"role": "user", "content": "manual"}]
        nested.compress = lambda *a, **k: sentinel
        msgs = [{"role": "user", "content": "u1"}]
        assert eng.compress(msgs, force=True) is sentinel

    def test_delegation_failure_falls_back_to_synthetic(self):
        """If the nested compressor raises, keep the synthetic survival list."""
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
        )
        nested = self._builtin_compressor(eng)

        def boom(*a, **k):
            raise RuntimeError("nested failure")

        orig = nested.compress
        nested.compress = boom
        try:
            msgs = [{"role": "system", "content": "sys"},
                    {"role": "user", "content": "u1"}]
            out = eng.compress(msgs)
            assert out == [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": engine_mod.FALLBACK_NOTE},
            ]
        finally:
            nested.compress = orig

    def test_nested_compressor_tracks_model_switch(self):
        eng = make_engine()
        eng.update_model("model-x", context_length=12345)
        nested = self._builtin_compressor(eng)
        assert nested.context_length == 12345
        assert nested.model == "model-x"

    def test_no_nested_compressor_synthetic_fallback_still_works(self):
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
        )
        eng._get_fallback_compressor = lambda: None  # force synthetic path
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"}]
        out = eng.compress(msgs)
        assert out == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": engine_mod.FALLBACK_NOTE},
        ]

    def test_should_compress_delegates_threshold_decision(self):
        """Over-threshold (but under context_length) -> True via nested."""
        eng = make_engine()
        eng.update_model("m", context_length=1000)  # threshold = 500 tokens
        nested = eng._get_fallback_compressor()
        orig = nested.should_compress
        nested.should_compress = lambda pt: True
        try:
            assert eng.should_compress(600) is True  # 600 < 1000, over 500
        finally:
            nested.should_compress = orig

    def test_should_compress_under_threshold_false(self):
        eng = make_engine()
        eng.update_model("m", context_length=100000)
        assert eng.should_compress(1000) is False

    def test_should_compress_failure_falls_back_to_overflow(self):
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng.update_from_response(
            {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
        )
        nested = eng._get_fallback_compressor()
        orig = nested.should_compress

        def boom(pt=None):
            raise RuntimeError("injected")

        nested.should_compress = boom
        try:
            assert eng.should_compress() is True  # overflow fallback
        finally:
            nested.should_compress = orig

    def test_should_compress_no_nested_overflow_only(self):
        eng = make_engine()
        eng.update_model("m", context_length=1000)
        eng._get_fallback_compressor = lambda: None
        assert eng.should_compress(600) is False  # under context_length
        eng.update_from_response(
            {"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100}
        )
        assert eng.should_compress() is True  # proven overflow
