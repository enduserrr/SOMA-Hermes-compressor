---
name: soma
description: "SOMA context engine: install, verify, debug, benchmark."
version: 0.1.0
author: enduserrr (enduserrr), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [hermes, soma, compression, context, plugin]
---

# SOMA Context Engine Skill

Operational driver for the SOMA context-engine plugin: an extractive,
deterministic compressor that shrinks oversized tool results per-request
(no LLM calls), delegating whole-session compaction to the built-in
compressor. This skill is a thin router — the plugin's own docs are the
source of truth and are read from the cloned repo, never duplicated here.

## When to Use

- Installing or enabling SOMA on a machine (fresh setup or new profile).
- Verifying the plugin after a Hermes upgrade.
- Debugging: engine not loaded, no compression happening, request errors.
- Benchmarking SOMA against the default compressor.
- Don't use for: tuning the built-in compressor's threshold/tail config
  (that is the `context-compression` skill's territory).

## First run on a machine

1. Clone the plugin repo (if `~/.hermes/plugins/context_engine/soma` is
   missing):

   ```bash
   git clone --depth 1 https://github.com/enduserrr/NoSleepHermes.git \
           ~/.hermes/plugins/context_engine/soma
   ```

2. Symlink into the repo tree — Hermes' engine discovery scans ONLY the
   repo's own `plugins/context_engine/` directory:

   ```bash
   ln -sfn ~/.hermes/plugins/context_engine/soma \
           ~/.hermes/hermes-agent/plugins/context_engine/soma
   ```

   Completion criterion: `ls -l ~/.hermes/hermes-agent/plugins/context_engine/soma`
   resolves to the plugin dir.

3. Enable (gateway sessions need `systemctl --user restart
   hermes-gateway.service` afterwards):

   ```bash
   hermes config set context.engine soma
   ```

4. Read the authoritative docs in the cloned repo, in this order:
   - `README.md` — install, scope guarantee, tuning, debugging, post-update
     checklist.
   - `ARCHITECTURE.md` — design + invariants (§1), host contract (§2),
     testing (§3), debug triage (§4), repair checklist (§5).
   - `tests/BENCHMARK.md` — test suite + `soma-mini-bench` benchmark.

## Quick reference

```bash
# Test suite (expect 76 passed; venv only — system python3 cannot import)
cd ~/.hermes/plugins/context_engine/soma
~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/ -v

# Offline benchmark vs default compressor (machine-readable for agents)
soma-mini-bench run --json

# Accounting trail (one JSONL line per changed request)
cat ~/.hermes/plugins/context_engine/soma/accounting.jsonl

# Revert (lossless, anytime)
hermes config set context.engine compressor
```

## Pitfalls

- **Hermes upgrades can wipe the repo symlink** — that is the first thing
  to check when the engine "disappears" (log line `Using context engine:
  soma` absent from `~/.hermes/logs/agent.log`).
- **No accounting lines is not a failure** when every tool result is under
  32K chars — that is the cache-stable no-op path. Force with a 40K+ char
  `read_file` in a scratch session.
- **Import probing:** running Python from `~/.hermes` shadows the repo's
  `plugins` package with the user-level one — probe from
  `~/.hermes/hermes-agent/`.
- **Scope:** only `role: tool` results are ever rewritten. System prompt,
  SOUL.md, skills, memories, user and assistant messages pass through
  byte-verbatim; persisted history is never mutated.

## Verification

- `pytest tests/ -v` reports 76 passed (16 core + 60 engine).
- A scratch session reading a 40K+ char file adds exactly one line to
  `accounting.jsonl`.
