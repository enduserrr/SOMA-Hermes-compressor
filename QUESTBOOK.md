# SOMA-Hermes-Plugin — QUESTBOOK

Dev-session log for this repo. ENTRY at start, STATUS (files touched) at end (protocol: 2026-09-29).

## 2026-10-02 — Fully-functioning plugin (branch `q`)

**ENTRY:** plan `~/.hermes/plans/2026-10-02_042917-soma-fully-functioning-plugin.md` — pass
`hermes plugins validate`, migrate to flat user-plugin layout, catalog-ready manifest/docs.
Baseline re-proven: 81 pass / 7 pre-existing (`hermes_yaml` venv skew) / 2 skipped.

**STATUS (files touched):**
- `plugin.yaml` — author/license/homepage/tags/requires_hermes, 0.2.0 (`2364244`)
- `__init__.py` — `register(ctx)` entry point; `tests/test_plugin_contract.py` new (`bf5f003`)
- `engine.py` — `clone_for_agent()` fresh-instance override (`a8a0ed7`)
- `ARCHITECTURE.md` — test-count fix (`fa79a86`)
- `README.md`, `ARCHITECTURE.md`, `skills/soma/SKILL.md`, `tests/BENCHMARK.md`,
  `scripts/soma-savings`, `tests/test_engine.py` — flat-layout docs purge (`b1daa3f`)
- Repo moved `~/.hermes/plugins/context_engine/soma` → `~/.hermes/plugins/soma` (whole-tree
  `mv`, no commit); core-repo symlink dropped; `~/.local/bin/soma-savings{,.py}` +
  `~/.hermes/skills/soma` retargeted.

**Gates:** `hermes plugins validate` → **Validation passed**; user-path discovery
`DISCOVERY OK` incl. `soma`; suite 81/7/2 unchanged; `plugins.disabled` stale key removed
via `hermes config set`.

**Pending:** user-gated gateway+desktop restart → live verify (gui.log clean, fresh
accounting record) → privacy gate → push `q` → optional catalog PR (with user).
