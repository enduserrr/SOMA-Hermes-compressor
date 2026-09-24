# SOMA plugin — testing & benchmarking guide

How to test the SOMA context-engine plugin and how to benchmark it against
the built-in Hermes compressor. Instructions are written for both humans and
agents. No special setup beyond the Hermes venv is required.

## The test suite

### Setup

The ONLY supported interpreter is the Hermes venv — the system python3
(3.10) lacks scikit-learn and pytest and cannot even import the repo:

```bash
cd ~/.hermes/plugins/context_engine/soma
~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/ -v
```

Expected result: **76 passed** (16 SOMA core + 60 engine). If the count is
lower after a Hermes upgrade, work through the repair checklist in the
plugin's `ARCHITECTURE.md` §5.

### Test layout

- `tests/test_soma_core.py` — 16 behavioural checks of the vendored
  compressor core (`soma_compressor.py`): sizing table, idempotency,
  no-inflation, tool-call pairing, determinism across processes. Independent
  of Hermes.
- `tests/test_engine.py` — 60 contract tests for the Hermes adapter
  (`engine.py`): ABC identity, token accounting, `get_status()` shape,
  `select_context()` passthrough rules (32K floor, cache-stable no-op),
  orphan fallback, fail-open fault injection, JSON-envelope unwrap/re-wrap
  for both `content` (read_file) and `output`/`exit_code` (terminal)
  envelopes, accounting behaviour, and the compressor-delegation paths
  (manual `/compress`, proven overflow, model-switch forwarding, synthetic
  survival fallback).

### Targeted runs

```bash
~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_engine.py::TestSelectContext -v
~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_engine.py::TestAccounting -v
```

### What the suite deliberately does NOT cover

Live provider behaviour. The suite is fully offline and deterministic; for
end-to-end confidence run the live soak described in `ARCHITECTURE.md` §3
(discovery check, `Using context engine: soma` log line, accounting.jsonl
gains a line after a 40K+ char read in a scratch session).

## The benchmark: `soma-mini-bench`

A lightweight, deterministic, zero-LLM offline benchmark comparing the
default Hermes compressor against the SOMA engine. It replays synthetic
agent histories through the engines' REAL entry points — the built-in via
`prune_tool_results_only` + `compress()`, SOMA via per-turn
`select_context()` — and measures both **chars reclaimed** (efficiency) and
**needle fidelity** (are load-bearing facts — paths, test names, errors —
still present in what the provider would receive?).

### Running it (humans)

```bash
soma-mini-bench run                  # full suite, human-readable table
soma-mini-bench scenarios            # list available scenarios
soma-mini-bench run --scenario pytest
soma-mini-bench report <file.json>   # re-print a saved result
```

### Running it (agents)

```bash
soma-mini-bench run --json           # machine-readable output
```

Every run saves a JSON result to the plugin's `bench_results/` directory
(gitignored). Exit codes: `0` ok, `2` usage error.

### Scenarios

| name | history |
|---|---|
| `code-read` | 8 read_file results (~40K chars each) then a fix-all turn |
| `pytest` | 4 test files read, a pytest run with 3 failures, keep working, then name them |
| `daemon-log` | 5K-line daemon log via read_file — keep traceback + FATAL |
| `mixed` | 10-turn triage session blending code, tests, and logs |

### Interpreting results

Reference run (Sep 2026, deterministic — byte-identical across runs):

| scenario | default/prune | default/compress* | SOMA | SOMA needles |
|---|---|---|---|---|
| code-read | 99.9%, 0/3 | 86.9%, 2/3 | 63.4% | 3/3 |
| pytest | 98.7%, 4/4 | 90.3%, 1/4 | 54.7% | 4/4 |
| daemon-log | 99.9%, 0/3 | 98.9%, 0/3 | 91.3% | 3/3 |
| mixed | 99.2%, 2/4 | 88.8%, 1/4 | 73.8% | 4/4 |

\* Offline the default engine's LLM summary degrades to its deterministic
fallback — the no-API worst case. A live summary model would preserve more
needles; always state this caveat when quoting the numbers.

Reading: the default compressor's aggressive paths reclaim more characters
but drop load-bearing facts (0/3, 0/3, 1/4 needles on prune paths); SOMA
trades roughly 10–35 percentage points of raw savings for near-perfect
fidelity. Neither number alone ranks the engines — a rewrite that deletes
the traceback is not a saving.

### Benchmark-design caveats (matter if you extend it)

- Needle checks must run on UNWRAPPED content: flatten JSON envelopes to the
  inner `content`/`output` text first, or checks false-negative.
- Histories shorter than ~15 messages make the default engine's prune a
  silent no-op (protect-last-N tail protection) — scenarios need enough
  turns or the baseline appears to do nothing.
- File fixtures where nearly every line matches a pin pattern (`def`,
  `raise`, `import`) are degenerate: compression becomes a byte-shave and
  quality numbers are meaningless. Real code is mostly unpinned body lines.
