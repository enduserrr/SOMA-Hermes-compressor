# SOMA Context Engine

Extractive, per-request compression of oversized tool results before each
provider call. Uses the vendored SOMA core
([DendriteHQ/SOMA-OpenClaw-compressor](https://github.com/DendriteHQ/SOMA-OpenClaw-compressor),
MIT license — see `LICENSE` and `soma_compressor.py`). No LLM calls.

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
  `{"content": "...", ...}` **and** `terminal`-style `{"output": "...",
  "exit_code": N}`) are unwrapped, the inner text is compressed
  (SOMA is line-based — a 30K one-line payload is incompressible as-is), and
  the result is re-wrapped with envelope metadata preserved (`total_lines`,
  `file_size`, `exit_code`, `error`, `cwd` — downstream consumers grep
  `exit_code` back out of the content, so it must survive).

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

## How to revert

```bash
hermes config set context.engine compressor
```

Rollback is lossless: persisted history is never mutated, only per-request
copies are rewritten.

## Post-update verification (after Hermes upgrades)

1. Re-verify the repo symlink still exists (see Install above).
2. Run the plugin test suite:

   ```bash
   cd ~/.hermes/plugins/context_engine/soma
   ~/.hermes/hermes-agent/venv/bin/python3 -m pytest
   ```

3. Scratch-session compression check: in a fresh session, read a 40K+ char
   file, then confirm `accounting.jsonl` gained a line. Only after all three
   pass should the engine be trusted again.
