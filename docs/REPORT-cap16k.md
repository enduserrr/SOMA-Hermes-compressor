# SOMA-BENCH: Flat-16K-Cap A/B Report (tasks 004 + 006, glm-5.3-flash)

**Run:** `cap16k-flat` · **Date:** 2026-09-27 · **Arms:** `soma` (live 24K config, control) vs `soma-16k` (flat 16K cap)
**Model:** say-gm / glm-5.3-flash, reasoning high, max_turns 125 — identical to all prior SOMA runs (comparability).
**Pre-launch bug caught:** the committed cap16k patch lowered only the floor (24K→16K) while the sizing formula
`min(MAX_KEEP=24K, max(floor=16K, 60%·len))` kept a partial ladder (27–40K results kept 60%, >40K kept 24K) — **not**
the approved flat cap. Fixed pre-launch by clamping `MAX_KEEP_CHARS` to the floor in the host override (commit `319c77d`);
offline test pins the flat semantics (30K→16,018; 60K→16,070 — not 18K/24K). The cut-off session's partial data was
archived as `runs/cap16k-invalid-partial-ladder` and excluded.

## Test design

- **Short:** task 004 (audit fix, ~4–6 turns, one 77K runtime dump ≈ 19K tok → 16k arm caps it to 16K chars vs soma's 24K)
- **Medium:** task 006 (8-phase ops shift, ~30–45 turns, 8× 50K reports; 16k arm: each report → 16K vs soma's 24K)
- 2 repeats per arm per task; task 005 (467K megadump) skipped per user instruction.
- **Dendrite weighted tokens** = input×1.0 + cache_read×0.1 + output×3.0 (billing-relevant, cache-sensitive).

## Results — honest (harness-contaminated run excluded)

One run excluded: `004/soma-16k/r1`. The agent grepped outside its workdir (workdirs used to live inside the bench
tree) and found a **pre-fixed** `registry.py` from the Sep-24 manual smoke run (`runs/manual/`), concluded "already
fixed", edited nothing → failed the patch probe. Its own workdir copy was still broken. **Harness hygiene defect,
not a compression effect.** Fixed: workdirs now live in `/tmp` (mkdtemp, outside the bench tree) and `runs/manual/`
is deleted (commit `4177df9`). The soma arm's 004 runs also grepped old runs but patched their OWN files (verified
via patch-diff paths) — no false passes anywhere.

### Per-run data (Dendrite-weighted tokens)

| task | arm | run | resolved | wtok | raw tokens (in+cache+out) |
|---|---|---|---|---|---|
| 004 | soma | r1 | ✓ | 65,889 | 170,953 |
| 004 | soma | r2 | ✓ | 52,225 | 138,536 |
| 004 | soma-16k | r1 | ✗ *(excluded — contaminated)* | 76,198 | 185,396 |
| 004 | soma-16k | r2 | ✓ | 54,704 | 169,236 |
| 006 | soma | r1 | ✓ | 269,699 | 1,072,928 |
| 006 | soma | r2 | ✓ | 279,396 | 1,073,607 |
| 006 | soma-16k | r1 | ✓ | 239,040 | 1,126,993 |
| 006 | soma-16k | r2 | ✓ | 176,560 | 817,510 |

### Averages (honest)

| task | arm | resolved | avg wtok | avg raw tokens | delta (16k vs soma) |
|---|---|---|---|---|---|
| 004 (short) | soma | 2/2 | 59,057 | 154,744 | **weighted −7.4%** (raw +9.4%, n=1 vs 2) |
| 004 (short) | soma-16k | 1/1 honest (1 excluded) | 54,704 | 169,236 | |
| 006 (medium) | soma | 2/2 | 274,547 | 1,073,268 | **weighted −24.3%** (raw −9.4%) |
| 006 (medium) | soma-16k | 2/2 | 207,800 | 972,252 | |

## Reading the numbers

- **Medium-length sessions: flat 16K clearly wins.** −24% billing-weighted, −9% raw tokens, **−93% resident history
  chars** (3.7K vs 54K persisted), resolution 2/2 both arms. All 8 phase reports flowed through context in both arms
  (verified — no grep-dodging or /tmp dumping), needle recall passed in all honest runs.
- **Short sessions: a wash.** −7% weighted is within single-run variance (spread on this task across all runs is
  ±20%); raw tokens actually +9% — the 16K cap forces slightly more re-reading (agent re-greps the audit dump because
  the 16K form dropped some context it wanted). One honest run each side; not decisive either way at n=1–2.
- **Fidelity: no needle loss at 16K.** Every honest run in both arms resolved, including the 8-phase recall audit —
  the pin-based extraction keeps paths/errors/constants intact at 16K, and 006's audit specifically tests recall of
  the OLDEST phases.
- **Where 16K saves:** big multi-report sessions. 8 reports × (24K−16K) = 64K chars kept out of every subsequent
  request. Where it costs: nothing measurable; the single unresolved run is proven harness contamination.

## Verdict

**The flat 16K cap beats the current 24K config on this workload — mainly on medium/long sessions (−24% weighted
tokens, −93% resident chars), at no fidelity cost, and neutral on short sessions.** Confidence is moderate: 2 repeats,
one arm-task cell at n=1 after the exclusion, ±20–30% single-run variance. The direction is consistent across both
honest 006 repeats and both wtok/raw metrics agree on 006.

## Cost

8 agent runs, ~$0.35 total (glm-5.3-flash; in+cache+out ≈ 3.6M tokens across the run).

## Provenance

- Scoring: Dendrite SOMA-benchmark spec (weighted tokens, r-based score) — `soma_bench/scoring.py`, 23/23 unit
  checks. Dendrite docs: thesoma.ai/docs/miner/scoring (Sep 2026); github.com/DendriteHQ/SOMA-benchmark.
- Run data: `runs/cap16k-flat/` (rows.jsonl + rows-honest.jsonl + EXCLUDED-004-16k-r1.md + engine accounting).
- The live plugin was never loaded, modified, or written by the bench (verify OK, live tree clean throughout).

## Recommendation

Adopt the flat 16K cap on the live engine only if the user accepts the trade-off profile: big win on long/multi-report
sessions, neutral on short ones, untested beyond ~45 turns. A 4-repeat confirmation run on 006 alone (~$0.20) would
tighten confidence before flipping the live constant. The change itself is one line in the live engine's host override
(`PASSTHROUGH_CHARS 24_000 → 16_000` + `mod.MAX_KEEP_CHARS = PASSTHROUGH_CHARS` clamp — same shape as the bench patch).
