#!/usr/bin/env python3
"""Smoke test: vendored SOMA compressor imports cleanly from its new location."""
import importlib.util
import pathlib

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent
MODULE = PLUGIN_DIR / "soma_compressor.py"

spec = importlib.util.spec_from_file_location("soma_compressor", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

assert mod.MIN_PASSTHROUGH_CHARS == 16_000
assert mod.KEEP_FRACTION == 0.60
assert mod.MAX_KEEP_CHARS == 24_000
assert mod.REPORTED_RESULT_CAP == 16_000
assert all(
    hasattr(mod, n)
    for n in (
        "append_loop_guard",
        "detect_loop_reason",
        "strip_loop_guard",
        "LOOP_REASON_ASSISTANT",
        "LOOP_REASON_TOOLCALL",
        "extractive_compress",
        "cmp_block",
    )
), "loop-guard/core symbols missing"
print("vendored module imports OK; sizing table verbatim; loop-guard present")
