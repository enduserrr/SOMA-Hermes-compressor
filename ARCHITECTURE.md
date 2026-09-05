# SOMA Context Engine — Architecture, Testing & Repair

Operator and maintainer documentation for the `soma` context-engine plugin.
See `README.md` for install/tuning/revert; this document explains **how the
plugin is built, how it hooks into Hermes, how to test and debug it, and how
to repair it after a Hermes update breaks something.**

---

## 1. How it is built

Three layers, strictly separated:

```
soma_compressor.py   (vendored, MIT, DendriteHQ/SOMA-OpenClaw-compressor)
    Pure, deterministic, stdlib + scikit-learn. Knows nothing about Hermes.
    Entry points used by the engine:
      - extractive_compress(text, target_chars, active) -> (str, bool)
      - cap_tool_result(message, cap) -> (message, changed)   [role 'toolResult']
      - orphan_ids(messages) -> (result_orphans, call_orphans)
      - cmp_block(inner) / CMP_START / CMP_END                ["[[CMP]]" markers]
      - Constants: MIN_PASSTHROUGH_CHARS=16_000 (upstream; overridden to 32_000 at load time), KEEP_FRACTION=0.60,
        MAX_KEEP_CHARS=32_000, REPORTED_RESULT_CAP=16_000
    Vendored byte-identical (verify: sha256sum against upstream repo).
    Its 16 behavioural tests live in tests/test_soma_core.py.

engine.py            (the adapter, ~350 lines)
    SomaEngine(ContextEngine). All Hermes-specific logic:
      - Token accounting: update_from_response(), update_model(), get_status()
      - select_context(): per-request compression pass (the main feature)
      - should_compress()/compress(): overflow-gated survival fallback
      - JSONL accounting (accounting.jsonl, best-effort, never raises)

__init__.py / plugin.yaml
    Registration. __init__ exports SomaEngine; plugin.yaml carries
    name/description/version metadata (description shows in `hermes plugins`).
```

### Key design decisions (do not undo these)

- **Compression lives in `select_context()`, not `compress()`.** Hermes calls
  `select_context()` every turn before dispatch; `compress()` handles genuine
  whole-session compaction by **delegating to a nested built-in
  `ContextCompressor`** (the stock LLM summarizer with its protect-first-N /
  protect-last-N policy, focus-topic support, and summary cooldowns). The
  nested instance is created lazily on first `compress()` call and tracks
  `update_model()` so model switches stay in sync. It fires on provider-proven
  overflow (`last_total_tokens > context_length`) or manual `/compress`
  (`force=True`) — never on thresholds, avoiding double-compaction with
  select_context. If the nested compressor fails to build or raises, the
  engine falls back to a crude synthetic survival list (system prompt + note).
- **Role bridge**: SOMA keys on connector-style role `toolResult`; Hermes uses
  OpenAI role `tool`. `_cap_openai_tool_result()` copies the message, sets
  role `toolResult`, calls `cap_tool_result`, sets it back.
- **JSON-envelope unwrap**: Hermes stores tool results as single-line JSON
  envelopes. read_file-style: `{"content": "...", "total_lines": N}` ;
  terminal-style: `{"output": "...", "exit_code": N, "error": ..., "cwd": ...}`
  (the `output` shape verified against persisted history in state.db —
  65-87K-char terminal envelopes rode through untouched before it was
  handled). SOMA is line-based, so a 30K one-line payload is incompressible
  as-is. `_unwrap_json_envelope()` parses the envelope, the inner text is
  compressed, the result is re-wrapped — envelope metadata (`total_lines`,
  `file_size`, `exit_code`, `error`, `cwd`) is preserved; downstream
  consumers (e.g. the default compressor's `_summarize_tool_result`) grep
  `"exit_code"` back out of the content, so it must survive. Only emits if
  strictly smaller. The unwrap returns the key that held the text so the
  re-wrap writes the compressed text back to the same field.
- **Sizing rule** (simplified 2026-09-05; constants at top of engine.py):
  <=32,000 chars passthrough untouched; >32,000 chars capped at 32K. The
  upstream 16K floor and mid-band ladder (16K ceiling / keep 60%) were
  retired after a floor sweep over the 33 real sessions in state.db with
  >16K tool results: the 16-32K band is dominated by active-work payloads
  (13 read_file + 6 terminal results) where compression dropped unpinned
  body lines for <=12% savings, while the higher floor costs +0.96% chars
  sent and rewrites fewer distinct results (16 vs 26) = fewer prompt-cache
  invalidations. The floor is applied by overriding the loaded core's
  `MIN_PASSTHROUGH_CHARS` in `_load_soma()` — the vendored file stays
  byte-identical to upstream (invariant 6).
- **scikit-learn is imported lazily** inside SOMA's `_line_scores` (first
  compression call, not module import) — keeps Hermes baseline RSS down.
  The fallback scorer (distinct-token counting) is a valid no-sklearn path
  and is covered by tests.
- **Loop guard stripped**: SOMA's loop-detection injection is intentionally
  NOT ported — Hermes' own circuit breakers handle loops. SOMA core's guard
  code stays inert/unreferenced.

### Invariants (every change must preserve these)

1. `select_context()` returns `None` when nothing changed (cache-stable
   no-op — per-turn shuffling forfeits the 5-minute prompt-cache TTL).
2. Never mutate persisted history — request-scoped rewrites only.
3. Never inflate — emit a rewrite only when strictly smaller.
4. Preserve tool-call pairing — if rewriting would orphan `tool_calls`/tool
   results, fall back to the original request (orphan_ids subset check).
5. Fail-open — any exception in any hook leaves the request untouched
   (try/except, log once). A broken engine must be no worse than none.
6. MIT only — no BSL-licensed caveman code may be copied in.

---

## 2. How it is implemented into Hermes

### The ABC contract

The engine subclasses `agent.context_engine.ContextEngine` (in the Hermes
repo at `~/.hermes/hermes-agent/agent/context_engine.py`). Required:

| Member | Notes |
|---|---|
| `name` (property) | must return `"soma"`, matching `context.engine` in config.yaml |
| `update_from_response(usage)` | read `prompt_tokens`/`completion_tokens`/`total_tokens` into `last_*` attrs |
| `should_compress(prompt_tokens=None)` | ours: True only on proven overflow |
| `compress(messages, current_tokens, focus_topic, force, memory_context)` | ours: delegates to nested built-in ContextCompressor (LLM summarizer) on proven overflow or manual /compress; identity otherwise |

Class attributes read directly by `run_agent.py` (MUST be maintained):
`last_prompt_tokens`, `last_completion_tokens`, `last_total_tokens`,
`threshold_tokens`, `context_length`, `compression_count`.

`get_status()` must return the host contract keys: `last_prompt_tokens`
(clamp the -1 sentinel to 0), `threshold_tokens`, `context_length`,
`usage_percent` (guard divide-by-zero), `compression_count`.

`update_model(model, context_length, base_url, api_key, provider, api_mode)`
— match this signature exactly; the host calls it on start and model switch.
We set `context_length` and derive `threshold_tokens = context_length *
threshold_percent` (inherited default 0.75).

### Host call flow (per turn)

```
agent init:  config context.engine != "compressor"
             -> plugins.context_engine.load_context_engine("soma")
             -> instantiates SomaEngine, calls update_model()
             -> logs "Using context engine: soma" (INFO, run_agent)
every turn:  conversation_loop._apply_context_engine_selection()
             -> engine.select_context(api_messages, ...)
             -> None = untouched; list = replacement for THIS request only
every turn:  update_from_response(usage) after each API call
on overflow: should_compress() -> True -> compress() fallback
```

The hook is fail-open at the host level too: exceptions or invalid returns
fall open to the unmodified request.

### Selection & discovery mechanics (the part that bites)

`load_context_engine()` (in the Hermes repo's
`plugins/context_engine/__init__.py`) scans **only the repo's own
`plugins/context_engine/` directory** — NOT `~/.hermes/plugins/`. And the
general plugin system deliberately EXCLUDES the `context_engine/` subdir.
So a user-level engine needs a symlink into the repo tree:

```bash
ln -sfn ~/.hermes/plugins/context_engine/soma \
        ~/.hermes/hermes-agent/plugins/context_engine/soma
```

The loader follows directory scans, so symlinks are discovered and load
cleanly. Config selection is `context: engine: "soma"` in config.yaml (set
via `hermes config set context.engine soma` — never hand-edit config.yaml).
CLI sessions pick it up immediately; gateway sessions need a gateway restart
(`systemctl --user restart hermes-gateway.service`).

Accounting file: `~/.hermes/plugins/context_engine/soma/accounting.jsonl` —
one JSON line per changed request: `session_id`, `input_est_chars`,
`output_est_chars`, `results_capped`, `reason` (`near_passthrough`),
`timestamp`. `session_id` is captured via the ABC's `on_session_start` hook
and written so the read-only `soma-savings` CLI (see `scripts/`) can attribute
savings per session. It is purely additive — the compression result written
per request is unchanged. Rows written before the field existed (or by the
offline bench harness, which never triggers `on_session_start`) carry `-`.

---

## 3. How to test it

ALWAYS the venv interpreter — system python3 lacks scikit-learn/pytest:

```bash
cd ~/.hermes/plugins/context_engine/soma
~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/ -v
# expect: 77 passed (16 SOMA core + 61 engine, incl. accounting,
# envelope & delegation)
```

Testing and benchmarking are documented in full in `tests/BENCHMARK.md`
(includes `soma-mini-bench` usage and result interpretation).

Test layout:
- `tests/test_soma_core.py` — 16 vendored behavioural checks (sizing table,
  idempotency, no-inflation, pairing, determinism across processes).
- `tests/test_engine.py` — 61 engine contract tests: ABC identity, token
  accounting, get_status shape, select_context passthrough rules, orphan
  fallback, fail-open fault injection (monkeypatched compressor/orphan check
  raising -> must return None), JSON-envelope unwrap/re-wrap for both
  envelope shapes, idempotency, accounting (one line per changed call, zero
  on no-op, write failures never escape), overflow fallback, lazy-sklearn
  fallback scorer, and compressor delegation (manual /compress, proven
  overflow, model-switch forwarding, synthetic fallback on nested failure).

Targeted runs:

```bash
~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_engine.py::TestSelectContext -v
~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_engine.py::TestAccounting -v
```

Determinism spot-check (the plan's manual procedure):

```bash
for seed in 1 42; do PYTHONHASHSEED=$seed ~/.hermes/hermes-agent/venv/bin/python3 -c "
import json, importlib.util
spec = importlib.util.spec_from_file_location('sc', 'soma_compressor.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
payload = [{'role':'user','content':('lorem ipsum dolor ' + 'x'*5000 + ' alpha beta gamma ')*20}]
print(json.dumps(m.extractive_compress(payload, 4096)))" > /tmp/det_$seed.out; done
diff /tmp/det_1.out /tmp/det_42.out && echo DETERMINISM-OK
```

**Live soak** (end-to-end, after any change or Hermes upgrade):

```bash
# 1. Discovery check:
cd ~/.hermes/hermes-agent && venv/bin/python3 -c "
from plugins.context_engine import discover_context_engines, load_context_engine
print(discover_context_engines())          # ('soma', '<desc>', True) expected
print(load_context_engine('soma').name)    # 'soma'"
# 2. Engine actually selected (verbose CLI run):
cd ~/.hermes/hermes-agent && hermes chat -v -q "say ACK" 2>&1 | grep -a "Using context engine"
# 3. Compression check: read a 40K+ char file in a scratch session, then:
wc -l ~/.hermes/plugins/context_engine/soma/accounting.jsonl   # gained a line?
```

Interpretation: a session whose tool results are all under 32K chars writes
NOTHING to accounting.jsonl — that is correct (cache-stable no-op), not a
failure. Use a 40K+ read_file to force a compression.

---

## 4. How to debug it

**Symptom-first triage:**

1. **Engine not loaded?** Look for `Using context engine: soma` (INFO,
   run_agent) in `~/.hermes/logs/agent.log` for your session's timestamp.
   Absent + "Context engine 'soma' not found — falling back" WARNING =>
   discovery failed => check the symlink (section 2) and that
   `context.engine` is set.

2. **ImportError / wrong `plugins` package picked up.** From cwd
   `~/.hermes`, the local `~/.hermes/plugins/` dir shadows the repo's
   `plugins` package: `from plugins.context_engine import ...` resolves to
   the WRONG module. If you probe imports manually, run from the repo dir
   (`~/.hermes/hermes-agent`). Inside Hermes itself this shadowing means the
   repo `plugins` package must be importable — a git-pull that moved/broke
   `plugins/context_engine/__init__.py` needs the symlink re-checked.

3. **Import errors mentioning `soma_compressor`.** The engine loads the
   vendored core via `importlib.util.spec_from_file_location` from
   `_SOMA_MODULE_PATH` (derived from `engine.py`'s real location). If the
   plugin dir was moved/renamed, fix the path or the `_soma_mod` singleton
   will fail (fail-open: requests pass through uncompressed).

4. **No accounting lines but engine loaded.** Expected when no tool result
   exceeded 32K chars (e.g. terminal output is size-capped by Hermes before
   entering history). Force with a large `read_file`. If a large read_file
   still writes nothing, suspect the JSON-envelope path: the persisted tool
   content is a single-line envelope; check `results_capped` logic in
   `_cap_openai_tool_result` and run the envelope tests.

5. **Request-level errors / weird provider 400s.** The engine wraps
   everything fail-open; an exception would log `soma: select_context
   failed; leaving request untouched` (grep agent.log for `soma:`). If you
   see it, run the full test suite and reproduce with the SQLite
   transcript:

   ```bash
   ~/.hermes/hermes-agent/venv/bin/python3 -c "
   import sqlite3
   db = sqlite3.connect('file:$HOME/.hermes/state.db?mode=ro', uri=True)
   rows = db.execute(\"SELECT role, length(content), content FROM messages \"
                     \"WHERE session_id LIKE '<SESSION_ID>%'\").fetchall()
   for r in rows: print(r[0], r[1], repr(r[2][:80]))"
   ```

   Feed the persisted tool content back through `select_context()` in a
   probe script to reproduce (this is exactly how the envelope bug was
   found — content 28,064 chars, 0 newlines, returned None).

6. **`hermes plugins` shows "not enabled".** Cosmetic for provider plugins
   (context engines don't use the `plugins.enabled` gate); selection is
   solely `context.engine`. Ignore unless selection also fails.

7. **Compression active but no savings?** Read accounting.jsonl:
   `output_est_chars` vs `input_est_chars`. Pinned lines (imports, paths,
   signatures, errors, diffs) always survive — file-read results dominated
   by such lines compress poorly. That is SOMA's design (code skeleton
   preservation), not a bug.

---

## 5. How to reconfigure / repair after a Hermes update

The plugin is designed so Hermes updates can break it in only a handful of
ways. Work through this checklist top-down:

```bash
# 1. Symlink survives? (git-pull/checkout in the repo can wipe it)
ls -l ~/.hermes/hermes-agent/plugins/context_engine/soma
# If missing:
ln -sfn ~/.hermes/plugins/context_engine/soma \
        ~/.hermes/hermes-agent/plugins/context_engine/soma

# 2. ABC contract still matches? (Hermes may change the ABC)
grep -n "abstractmethod" ~/.hermes/hermes-agent/agent/context_engine.py
# All four (name/update_from_response/should_compress/compress) must still
# be implemented in engine.py; check get_status()/update_model() signatures
# against the ABC, and that the host-contract class attrs are unchanged.

# 3. Host call-flow unchanged?
grep -n "_apply_context_engine_selection" ~/.hermes/hermes-agent/agent/conversation_loop.py
# Confirm select_context is still invoked per-turn and fail-open.

# 4. Full test suite (must be 77/77):
cd ~/.hermes/plugins/context_engine/soma
~/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/ -q

# 5. Discovery + live soak (section 3), then:
#    gateway sessions: systemctl --user restart hermes-gateway.service
```

**Reconfiguration knobs** (top of `engine.py`, no test edits needed):
- `PASSTHROUGH_CHARS` (32,000 since the 2026-09-05 floor sweep) — raise to compress less often (protects the
  cache; lowers savings).
- `KEEP_FRACTION` (0.60) / `MID_CEILING_CHARS` / `LARGE_CAP_CHARS` — how
  aggressive compression is.
- `ACCOUNTING_PATH` — where accounting.jsonl lives.
- SOMA core constants: the vendored `soma_compressor.py` is NEVER edited
  (byte-identical to upstream, invariant 6). Host tuning lives only in
  `engine.py`, applied via the `_load_soma()` override — changing
  `PASSTHROUGH_CHARS` there is the single source of truth for the floor.

**Emergency rollback** (lossless at any time — persisted history is never
mutated):

```bash
hermes config set context.engine compressor
systemctl --user restart hermes-gateway.service
```

**Worst case** (repo update reshapes plugin discovery entirely): the plugin
source of truth is `~/.hermes/plugins/context_engine/soma/` (this git repo).
Re-clone/re-symlink it; if the loader interface changed, port `engine.py`
against the new `agent/context_engine.py` ABC — the vendored core
(`soma_compressor.py`) and its tests are independent of Hermes and will not
need changes.

---

## 6. Build history (for archaeology)

Built 2026-09-04 per plan `~/.hermes/plans/2026-09-04_164920-soma-context-engine-plugin.md`.
Coordinator (glm-5.3-flash) executed Task 3 inline; Tasks 1/2/4/5/7 via
delegate_task subagents; two-stage review per task (coordinator spec check +
independent reviewer subagent, fail-closed JSON verdict). Early task commits
were squashed; the repo history is: `c94a827` README, `e8cfd26` full
implementation (including the JSON-envelope fix that came out of the Task 6
live soak), then the compressor-delegation change (Sep 5 2026) with this
documentation. Benchmark methodology and delegation TDD notes also live in
the `context-compression` skill reference `soma-plugin-live.md`.
