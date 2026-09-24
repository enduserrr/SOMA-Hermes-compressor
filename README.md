# SOMA Context Engine

Extractive, per-request compression of oversized tool results before each
provider call. Uses the vendored SOMA core
([DendriteHQ/SOMA-OpenClaw-compressor](https://github.com/DendriteHQ/SOMA-OpenClaw-compressor),
MIT license — see `LICENSE` and `soma_compressor.py`). No LLM calls.

> **Scope guarantee:** SOMA only ever rewrites **tool results** (`role: tool`)
> in the per-request message copy. Pre-prompt context — the system prompt,
> SOUL.md, attached skills, memories — and all user/assistant messages pass
> through untouched, and persisted history is never mutated.

> **Deeper docs:** `ARCHITECTURE.md` covers the internal design, the Hermes
> integration contract, debugging triage, and the full repair checklist for
> when a Hermes update breaks something.

## Install / registration

The plugin lives at:

```
~/.hermes/plugins/context_engine/soma/
```

**Discovery caveat:** Hermes' `load_context_engine()` scans only the *repo's*
`plugins/context_engine/` directory, not `~/.hermes/plugins/`. A symlink into
the repo tree is required:

```bash
ln -sfn ~/.hermes/plugins/context_engine/soma \
        ~/.hermes/hermes-agent/plugins/context_engine/soma
```

Re-check that the symlink still exists after every Hermes upgrade.

Enable:

```bash
hermes config set context.engine soma
```

Gateway sessions need a gateway restart to pick it up; the CLI picks it up
immediately.

## How it works

- `select_context()` rewrites the **per-request** message list only — persisted
  history is never mutated.
- Returns `None` when nothing changed, so the provider cache prefix stays
  byte-identical (cache-stable no-op).
- Never inflates: a rewrite is emitted only if it is strictly smaller than the
  original.
- Preserves tool-call pairing via an orphan guard: if a rewrite would orphan
  `tool_calls` or tool results, the original request is kept.
- Fail-open: any exception leaves the request untouched (returns `None`).
- JSON tool-result envelopes (`read_file`-style single-line
  `{"content": "..."}` **and** `terminal`-style `{"output": "...",
  "exit_code": N}`) are unwrapped, the inner text is compressed
  (SOMA is line-based — a 30K one-line payload is incompressible as-is), and
  the result is re-wrapped with envelope metadata preserved (`total_lines`,
  `file_size`, `exit_code`, `error`, `cwd` — downstream consumers grep
  `exit_code` back out of the content, so it must survive).
- **Dual-compressor design:** SOMA owns the cheap per-request shrink; whole-
  session compaction (manual `/compress`, or provider-proven overflow) is
  delegated to a nested built-in `ContextCompressor` (LLM summarizer) whose
  decision logic, cooldowns and model-switch tracking match the stock
  engine. If the nested compressor fails to build, a crude synthetic
  survival list (system prompt + notice) is the last resort.

## Testing & benchmarking

```bash
cd ~/.hermes/plugins/context_engine/soma
~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/ -v
# expect: 77 passed (system python3 CANNOT import the repo — venv only)
```

The offline benchmark CLI (`soma-mini-bench`) compares SOMA against the
default compressor — usage, scenarios and result interpretation are in
`tests/BENCHMARK.md`. The script and its launcher live in this repo under
`bench/`. The script resolves the plugin root from its own location, so it
benchmarks the repo's `engine.py` by default; to point it at a different
deployed copy, set `SOMA_DIR`. Run it from the repo root:

```bash
bench/soma-mini-bench run            # full suite, human table  (venv launcher)
bench/soma-mini-bench run --json     # machine-readable (agents)
```

## Debugging

Symptom-first triage (full details in `ARCHITECTURE.md` §4):

1. **Engine not loaded** — grep `~/.hermes/logs/agent.log` for
   `Using context engine: soma`. Absent + "not found" WARNING => check the
   symlink (Install above) and `hermes config get context.engine`.
2. **Import errors probing manually** — run from `~/.hermes/hermes-agent/`;
   from `~/.hermes` the local `plugins/` dir shadows the repo package.
3. **No accounting lines despite large reads** — expected when nothing
   exceeds 24K chars; force with a 40K+ `read_file` in a scratch session.
4. **Request errors / provider 400s** — grep agent.log for `soma:`; the
   engine is fail-open, so an exception means compression silently skipped.

## After a Hermes update

1. Re-check the symlink (git-pull in the repo can wipe it).
2. Run the test suite (Testing above) — all 77 must pass.
3. Scratch-session check: read a 40K+ char file, confirm
   `accounting.jsonl` gained a line. Only then trust the engine again.

Full repair checklist (ABC-contract diffing, host call-flow checks, worst
case): `ARCHITECTURE.md` §5. Immediate rollback at any time:

```bash
hermes config set context.engine compressor
```

Rollback is lossless: persisted history is never mutated, only per-request
copies are rewritten.

## Tuning constants

Sizing rule (constants at the top of `engine.py`, applied at load time;
the vendored upstream core ships a 16K floor):

| Tool-result size        | Action              |
|-------------------------|---------------------|
| <= 24,000 chars         | passthrough untouched |
| > 24,000 chars          | 24K cap             |

History: the upstream 16K floor and 16K-32K mid-band ladder were retired
by a 2026-09-05 floor sweep over real sessions (16K -> 32K, evidence in
`ARCHITECTURE.md` "Floor sweep"), then lowered to 24K on 2026-09-07.
With floor == cap the ladder collapses to one rule: nothing under the
floor is ever touched, everything over is capped at it.

These values are host-tuned, not universal. For optimal results on a
different workload, test custom cap/floor configs: replay your own
persisted tool results at candidate floors and weigh chars saved
against load-bearing content kept (method: "Floor sweep" in
`ARCHITECTURE.md`; offline harness: `soma-mini-bench`, see
`tests/BENCHMARK.md`), then confirm realised savings with
`soma-savings` and provider-reported tokens in `state.db`. Compression
selection stays char-based by design; token figures are reporting-only.

The floor is applied by `engine.py`'s `PASSTHROUGH_CHARS` overriding the
core's `MIN_PASSTHROUGH_CHARS` at load time. For a single-rule ladder
(floor == cap), `MAX_KEEP_CHARS` in the vendored `soma_compressor.py`
must move in tandem — it is the only local edit to the vendored file
(upstream 32K -> 24K here); everything else stays byte-identical.

## Accounting

Every `select_context()` call that changed something appends one JSON line to:

```
~/.hermes/plugins/context_engine/soma/accounting.jsonl
```

Record fields: `session_id`, `input_est_chars`, `output_est_chars`,
`results_capped`, `reason`, `timestamp` — plus, since Sep 22 2026,
`input_est_tokens` / `output_est_tokens` (token estimates via the vendored
core's `final_token_estimate()`: tiktoken `cl100k_base` when installed,
else the deterministic chars-per-token fallback; `0` on older lines).
Token figures are estimates for savings ratios only — provider-reported
usage in `state.db` (`sessions` / `session_model_usage`) remains the
billing truth. Compression *selection* stays char-based by design:
token-aware selection would change which lines survive.

Accounting is best-effort — a write failure never breaks the request.
`session_id` is tagged via `on_session_start` so savings can be attributed
per session (see `soma-savings` below).

## Reading savings: `soma-savings`

A read-only CLI reports compression savings from `accounting.jsonl` (it never
writes, never touches the compressor or tester). Installed at
`scripts/soma-savings` in this repo; a symlink into `~/.local/bin/soma-savings`
is also created on this host.

```bash
soma-savings                # total chars + est. tokens saved, all sessions
soma-savings -v             # per-session list, then the grand total
soma-savings <session-id>   # that session's savings + grand total
```

Run from any directory. Example: `soma-savings 20260905_143052_a1b2c3`.
Note: accounting rows recorded before the `session_id` field existed group
under `-` (unknown); rows before the token fields existed (pre Sep 22 2026)
report `0 tok est`. Point it at a different accounting file with the
`SOMA_ACCOUNTING` env var.
