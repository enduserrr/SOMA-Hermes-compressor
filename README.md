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
# expect: 76 passed (system python3 CANNOT import the repo — venv only)
```

The offline benchmark CLI (`soma-mini-bench`) compares SOMA against the
default compressor — usage, scenarios and result interpretation are in
`tests/BENCHMARK.md`.

## Debugging

Symptom-first triage (full details in `ARCHITECTURE.md` §4):

1. **Engine not loaded** — grep `~/.hermes/logs/agent.log` for
   `Using context engine: soma`. Absent + "not found" WARNING => check the
   symlink (Install above) and `hermes config get context.engine`.
2. **Import errors probing manually** — run from `~/.hermes/hermes-agent/`;
   from `~/.hermes` the local `plugins/` dir shadows the repo package.
3. **No accounting lines despite large reads** — expected when nothing
   exceeds 32K chars; force with a 40K+ `read_file` in a scratch session.
4. **Request errors / provider 400s** — grep agent.log for `soma:`; the
   engine is fail-open, so an exception means compression silently skipped.

## After a Hermes update

1. Re-check the symlink (git-pull in the repo can wipe it).
2. Run the test suite (Testing above) — all 76 must pass.
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

Sizing rule (simplified 2026-09-05 after a floor sweep over real sessions —
see `ARCHITECTURE.md` "Floor sweep" for the evidence):

| Tool-result size        | Action              |
|-------------------------|---------------------|
| <= 32,000 chars         | passthrough untouched |
| > 32,000 chars          | 32K cap             |

The 16K-32K mid-band ladder from upstream was retired: that band is
dominated by active-work payloads (file reads being edited, test runs being
triaged) where compression dropped unpinned body lines for marginal
savings. The floor is applied by `engine.py`'s `PASSTHROUGH_CHARS`
overriding the core's `MIN_PASSTHROUGH_CHARS` at load time — the vendored
core file itself is untouched.

## Accounting

Every `select_context()` call that changed something appends one JSON line to:

```
~/.hermes/plugins/context_engine/soma/accounting.jsonl
```

Record fields: `input_est_chars`, `output_est_chars`, `results_capped`,
`reason`, `timestamp`. Accounting is best-effort — a write failure never breaks
the request.
