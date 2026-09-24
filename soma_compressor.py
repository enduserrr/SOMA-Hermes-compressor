#!/usr/bin/env python3
"""SOMA context compressor for OpenClaw.

Shrinks an agent's context by compressing oversized tool results in place, and
leaves everything else alone. It is selective rather than aggressive: small and
mid-sized results pass through untouched, and only genuinely bulky ones are
reduced, keeping a proportion of each rather than crushing everything to a fixed
ceiling.

What it does, per assembly:

  * Oversized tool results are replaced by an extractive summary wrapped in
    [[CMP]] ... [[/CMP]] markers, so the model can see that shortening happened
    instead of silently receiving a truncated file.
  * Load-bearing lines survive even when they sit inside a region being trimmed:
    failing test names, assertion lines, traceback tails, file paths, line
    numbers, diff hunks and code structure.
  * `thinking` blocks are dropped from the context.
  * An objective loop guard appends one of two fixed reason strings when the
    agent repeats itself; it never rewrites strategy or tool policy.

Three properties are relied on elsewhere and worth preserving in any edit:

  * Pure and deterministic. Output depends only on the input messages. Verified
    byte-identical across processes and hash seeds.
  * Idempotent. An already-compressed result is a fixed point, returned
    untouched, so re-running never re-compresses.
  * Never inflating. A compressed message is emitted only when it is strictly
    smaller than the original, and the whole result falls back to the input if
    tool-call pairing would be broken.

Configuration is the three constants below: the passthrough floor, the keep
fraction, and the upper bound on what is kept.

Protocol: `python3 soma_compressor.py assemble` with the connector payload as
JSON on stdin, one JSON object on stdout. Stdlib only, with two optional extras:

  * scikit-learn  used for TF-IDF line scoring. Without it the scorer falls back
                  to distinct-token counting, which changes WHICH lines survive.
                  Install it to reproduce reference behaviour.
  * tiktoken      used for the reported token estimate only. Its absence does not
                  change any compression decision.
"""

from __future__ import annotations

import copy

import hashlib
import json
import math
import re
import sys

from typing import Any

EVENT_NAMES = frozenset({"assemble"})

CHARS_PER_TOKEN = 4

# Per-result sizing. A flat character cap compresses big results harder than
# small ones (its ratio scales with size), which strips the context that hard
# tasks need. A proportional cap keeps a fixed FRACTION of each result instead,
# giving a uniform ratio, with a floor so small results are left alone entirely
# and a ceiling so a huge result cannot balloon the context.
#
# The cap depends only on the length of the message being compressed, never on
# the conversation around it. That is what makes it pure, deterministic and
# idempotent: re-feeding compressed output leaves it untouched, and the emitted
# prefix is stable from turn to turn.
#
# Net effect per result:
#   <= 16k chars        passed through untouched
#   16k to 26.7k        capped at 16k
#   26.7k to 53.3k      keep 60% (16k to 32k)
#   > 53.3k             capped at 32k
KEEP_FRACTION = 0.60           # fraction of an oversized result to keep
MIN_PASSTHROUGH_CHARS = 16_000 # results at or below this are never touched
MAX_KEEP_CHARS = 24_000        # upper bound on what is kept from one result
REPORTED_RESULT_CAP = MIN_PASSTHROUGH_CHARS  # metadata only; the effective cap is
                               # the bounded proportional value (see cap_tool_result)

# Tool results carrying these markers are the ground truth the agent patches
# against; protect them with larger budgets in every mode.
ERROR_MARKERS = (
    "traceback (most recent call last)",
    "assertionerror",
    "error:",
    "exception:",
    "failed",
    "failures=",
    "errors=",
    "fatal:",
)

# Paths like /a/b/c.py, django/utils/html.py:236 — the facts agents rediscover.
PATH_PATTERN = re.compile(r"(?:/)?[\w.-]+(?:/[\w.-]+)+\.[A-Za-z]{1,4}(?::\d+)?")

# Objective loop detection (compliant): identical repeated assistant response, or
# identical repeated tool-call signature, within a recent window. Detection only —
# the ONLY emitted text is one of the two allowed LOOP_REASON_* strings.
LOOP_WINDOW = 12
LOOP_THRESHOLD = 3
LOOP_SIG_ARGS_CLIP = 120
TEST_LINE_PATTERN = re.compile(
    r"^\s*(?:FAILED|ERROR|FAIL|XFAIL)[: ]\S.*$"
    r"|^\s*\S+\.py::\S+.*$"
    r"|^\s*(?:FAIL|ERROR): test\S* \(.*\).*$"
    r"|^.*\bAssertionError\b.*$",
    re.M,
)

# ===========================================================================
# ALLOWED COMPRESSION MARKERS — the ONLY bracketed strings this miner emits.
# Per miner/README_prompting.md §5.1 these are metadata wrappers; they preserve
# instruction meaning/order/requirements/tool-policy/safety/role-policy/output-
# contract exactly. We alias [[CMP]]/[[/CMP]] to the spelled-out start/end markers
# ONCE (allowed by the README) so the two forms are interchangeable downstream.
# ===========================================================================
CMP_START = "[[CMP]]"            # alias of "Compressed text starts here"
CMP_END = "[[/CMP]]"             # alias of "Compressed text ends here"

def cmp_block(inner: str = "") -> str:
    """Wrap a compressed/truncated region in the allowed [[CMP]]…[[/CMP]] markers.
    Empty/fully-elided regions become "[[CMP]][[/CMP]]". The kept head/tail lines go
    OUTSIDE the markers; only the elided middle is wrapped — the marker is metadata
    only and changes no instruction meaning."""
    inner = inner.strip("\n")
    if not inner:
        return f"{CMP_START}{CMP_END}"
    return f"{CMP_START}\n{inner}\n{CMP_END}"

# Allowed loop-detection reason strings (README §5.2). These are the ONLY strings
# the loop guard may emit — nothing else, no steering, no task restatement.
LOOP_REASON_ASSISTANT = "loop_detected: repeated assistant response"
LOOP_REASON_TOOLCALL = "loop_detected: repeated tool call signature"

def normalize_role(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    lowered = value.strip().lower().replace("_", "").replace("-", "")
    if lowered == "toolresult":
        return "toolResult"
    return lowered

def extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(extract_text(item) for item in value)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), str):
            return value["content"]
        return "\n".join(extract_text(item) for item in value.values())
    return str(value)

def collapse_ws(value: str) -> str:
    return " ".join(value.split())

def clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: max(0, limit - 1)] + "…"

def estimate_tokens(messages: list[Any]) -> int:
    total_chars = sum(
        len(extract_text(message.get("content")))
        for message in messages
        if isinstance(message, dict)
    )
    return max(1, math.ceil(total_chars / CHARS_PER_TOKEN)) if messages else 0

def final_token_estimate(messages: list[Any]) -> int:
    try:
        import tiktoken  # available in the compression-service image, cached offline

        encoder = tiktoken.get_encoding("cl100k_base")
        joined = "\n".join(
            extract_text(message.get("content"))
            for message in messages
            if isinstance(message, dict)
        )
        return len(encoder.encode(joined, disallowed_special=()))
    except Exception:
        return estimate_tokens(messages)

def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

def fingerprint_messages(messages: list[Any]) -> str:
    return hashlib.sha256(canonical_json(messages).encode("utf-8")).hexdigest()

def get_params(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("params")
    return params if isinstance(params, dict) else payload

def get_messages(payload: dict[str, Any]) -> list[Any]:
    messages = get_params(payload).get("messages")
    return messages if isinstance(messages, list) else []

def sanitize_content(content: Any) -> tuple[Any, bool]:
    if not isinstance(content, list):
        return content, False
    sanitized: list[Any] = []
    changed = False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "thinking":
            changed = True
            continue
        sanitized.append(block)
    return sanitized, changed

def has_runtime_content(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, (list, dict)):
        return bool(content)
    return content is not None

def is_failed_assistant_placeholder(message: Any) -> bool:
    if not isinstance(message, dict) or normalize_role(message.get("role")) != "assistant":
        return False
    if not isinstance(message.get("errorMessage"), str):
        return False
    content = message.get("content")
    return content in (None, "") or content == []

def sanitize_messages(messages: list[Any]) -> tuple[list[Any], bool]:
    sanitized: list[Any] = []
    changed = False
    for message in messages:
        if is_failed_assistant_placeholder(message):
            changed = True
            continue
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        next_message = message
        next_content, content_changed = sanitize_content(message.get("content"))
        if content_changed:
            next_message = copy.deepcopy(message)
            next_message["content"] = next_content
            changed = True
        if (
            normalize_role(next_message.get("role")) != "toolResult"
            and not has_runtime_content(next_message.get("content"))
        ):
            changed = True
            continue
        sanitized.append(next_message)
    return (sanitized if changed else messages), changed

def extract_tool_result_ids(message: Any) -> set[str]:
    if not isinstance(message, dict) or normalize_role(message.get("role")) != "toolResult":
        return set()
    ids: set[str] = set()
    for field in ("toolCallId", "toolUseId", "id"):
        value = message.get(field)
        if isinstance(value, str) and value.strip():
            ids.add(value.strip())
    return ids

def iter_tool_call_blocks(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict) or normalize_role(message.get("role")) != "assistant":
        return []
    blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "toolCall":
                blocks.append(block)
    for field in ("toolCalls", "tool_calls"):
        tool_calls = message.get(field)
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    blocks.append(tool_call)
    return blocks

def extract_tool_call_ids(message: Any) -> set[str]:
    ids: set[str] = set()
    for block in iter_tool_call_blocks(message):
        value = block.get("id")
        if isinstance(value, str) and value.strip():
            ids.add(value.strip())
    return ids

def orphan_ids(messages: list[Any]) -> tuple[set[str], set[str]]:
    """Return (result ids without a call, call ids without a result)."""
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for message in messages:
        call_ids.update(extract_tool_call_ids(message))
        result_ids.update(extract_tool_result_ids(message))
    return result_ids - call_ids, call_ids - result_ids

def truncate_text(value: str, head: int, tail: int) -> str:
    if len(value) <= head + tail + 80:
        return value
    # Kept head/tail sit OUTSIDE the markers; the elided middle is the compressed
    # region, wrapped in the allowed empty [[CMP]][[/CMP]] marker (metadata only).
    return f"{value[:head]}\n{cmp_block()}\n{value[-tail:] if tail else ''}"

_SIG_PATTERN = re.compile(r"\b(?:def|class)\s+\w+|\bfunction\s+\w+|=>\s*\{|\b\w+\s*\([^)]*\)\s*:")
_DIFF_PATTERN = re.compile(r"^\s*(?:[+\-]|@@|diff --git|---|\+\+\+)")
# Line pinning. The extractive pass keeps def/class/error/
# tests/diffs/paths but DROPS short structural CODE lines (imports, decorators, raise/except) because they
# score low on token-richness. On code file-reads that loses the file's API -> the agent edits referencing
# missing/wrong imports -> NameError/ImportError -> BREAK (verified: a real 32k sympy read dropped `import
# inspect`, `from functools import wraps`, `raise NameError(`...). Pin them so the code skeleton survives
# compression. Pure per-message + deterministic (cache-stable); informative-only (no steering/markers).
_STRUCT_PATTERN = re.compile(r"^\s*(?:import\s+\w|from\s+[\w.]+\s+import\b|@\w[\w.]*|raise\b|except\b)")

def _basenames(paths: set[str]) -> set[str]:
    out: set[str] = set()
    for p in paths:
        p = p.split(":")[0]
        out.add(p)
        out.add(p.rsplit("/", 1)[-1])
    return {b for b in out if len(b) >= 4}

def _line_is_pinned(line: str, active: frozenset) -> bool:
    if not line.strip():
        return False
    low = line.lower()
    if any(marker in low for marker in ERROR_MARKERS):
        return True
    if TEST_LINE_PATTERN.search(line):
        return True
    if _SIG_PATTERN.search(line):
        return True
    if _STRUCT_PATTERN.match(line):  # SALIENCE FIX: keep imports/decorators/raise/except (code skeleton)
        return True
    if _DIFF_PATTERN.match(line):
        return True
    found = PATH_PATTERN.findall(line)
    if found:
        if not active:
            return True  # any file path is signal
        if _basenames(set(found)) & active:
            return True
        return True  # paths are cheap signal; keep them
    return False

def _line_scores(lines: list[str]) -> list[float]:
    """Informativeness per line via TF-IDF; graceful fallback if unavailable."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = [l if l.strip() else " " for l in lines]
        matrix = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=4000).fit_transform(docs)
        return matrix.sum(axis=1).A1.tolist()
    except Exception:
        # fallback: distinct-token count (favors content-rich lines over boilerplate)
        return [len(set(re.findall(r"\w+", l))) for l in lines]

def extractive_compress(text: str, target_chars: int, active: frozenset = frozenset()) -> tuple[str, bool]:
    if not isinstance(text, str) or len(text) <= target_chars:
        return text, False
    lines = text.split("\n")
    n = len(lines)
    keep = {i for i in range(n) if _line_is_pinned(lines[i], active)}
    used = sum(len(lines[i]) + 1 for i in keep)
    remaining = [i for i in range(n) if i not in keep and lines[i].strip()]
    if used < target_chars and remaining:
        scores = _line_scores([lines[i] for i in remaining])
        for idx, _ in sorted(zip(remaining, scores), key=lambda x: (-x[1], x[0])):
            if used >= target_chars:
                break
            keep.add(idx)
            used += len(lines[idx]) + 1
    if len(keep) >= n:
        return text, False
    out: list[str] = []
    prev = -1
    for i in sorted(keep):
        if i > prev + 1:
            out.append("…")
        out.append(lines[i])
        prev = i
    if prev < n - 1:
        out.append("…")
    result = "\n".join(out)
    if len(result) >= len(text):  # extraction didn't help → fall back to truncation
        return truncate_text(text, int(target_chars * 0.75), int(target_chars * 0.25)), True
    return result, True

def _is_loop_guard_message(message: Any) -> bool:
    """A loop-guard message we appended on a prior turn: a user message whose text
    is exactly one of the allowed loop-reason strings. Identified so we can strip
    our own prior guard before re-deciding (idempotent across turns)."""
    if not isinstance(message, dict) or normalize_role(message.get("role")) != "user":
        return False
    text = extract_text(message.get("content")).strip()
    return text in (LOOP_REASON_ASSISTANT, LOOP_REASON_TOOLCALL)

def strip_loop_guard(messages: list[Any]) -> list[Any]:
    return [m for m in messages if not _is_loop_guard_message(m)]

def detect_loop_reason(messages: list[Any]) -> str:
    """Return the matching allowed loop-reason string if an OBJECTIVE loop is
    present in the recent window, else "". Two objective signals:
      - repeated ASSISTANT response: an identical assistant message (same
        normalized text + tool-call signatures) recurs >= LOOP_THRESHOLD times;
      - repeated TOOL-CALL signature: the same tool name+args (regardless of
        result) recurs >= LOOP_THRESHOLD times.
    No content beyond the recurrence count drives the decision; nothing about the
    task, strategy, or desired outcome is considered."""
    # ---- repeated assistant response ----
    assistant_sigs: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or normalize_role(message.get("role")) != "assistant":
            continue
        text = collapse_ws(extract_text(message.get("content")))
        call_sig = "|".join(sorted(
            f"{(b.get('name') or b.get('toolName') or 'tool')}"
            f"({clip(collapse_ws(extract_text(b.get('arguments') or b.get('args') or b.get('input') or '')), LOOP_SIG_ARGS_CLIP)})"
            for b in iter_tool_call_blocks(message)
        ))
        digest = hashlib.sha256(f"{text}#{call_sig}".encode("utf-8")).hexdigest()
        assistant_sigs.append(digest)
    recent_assist = assistant_sigs[-LOOP_WINDOW:]
    acounts: dict[str, int] = {}
    for sig in recent_assist:
        acounts[sig] = acounts.get(sig, 0) + 1
    if any(n >= LOOP_THRESHOLD for n in acounts.values()):
        return LOOP_REASON_ASSISTANT

    # ---- repeated tool-call signature ----
    call_sigs: list[str] = []
    for message in messages:
        for block in iter_tool_call_blocks(message):
            name = block.get("name") or block.get("toolName") or "tool"
            args = clip(collapse_ws(extract_text(
                block.get("arguments") or block.get("args") or block.get("input") or ""
            )), LOOP_SIG_ARGS_CLIP)
            call_sigs.append(f"{name}|{args}")
    recent_calls = call_sigs[-LOOP_WINDOW:]
    ccounts: dict[str, int] = {}
    for sig in recent_calls:
        ccounts[sig] = ccounts.get(sig, 0) + 1
    if any(n >= LOOP_THRESHOLD for n in ccounts.values()):
        return LOOP_REASON_TOOLCALL

    return ""

def append_loop_guard(messages: list[Any]) -> list[Any]:
    """If an objective loop is detected, append a user message whose ONLY content
    is the matching allowed loop-reason string. Otherwise return messages
    unchanged. This is the entire prompt-side surface of the miner — no other text."""
    reason = detect_loop_reason(messages)
    if not reason:
        return messages
    return [*messages, {"role": "user", "content": [{"type": "text", "text": reason}]}]

def cap_tool_result(message, cap):
    """Cache-stable per-message cap. If `message` is a tool result whose text exceeds `cap` and is not
    already compressed, replace its content with an extractive summary wrapped in the allowed
    [[CMP]]...[[/CMP]] markers. PURE + DETERMINISTIC (active=frozenset(); depends only on this message's
    own content, never on global trajectory size or recency) and IDEMPOTENT (an already-[[CMP]] or
    already-<=cap result is returned UNCHANGED). These two properties make the emitted prefix byte-
    identical turn-to-turn under either connector feed-mode, so the provider prompt-cache stays warm.
    Every non-tool-result message (system/user/assistant) passes through verbatim."""
    if not isinstance(message, dict):
        return message, False
    if normalize_role(message.get("role")) != "toolResult":
        return message, False
    text = extract_text(message.get("content"))
    if CMP_START in text:  # idempotency guard first: an already-compressed result is a fixed point,
        return message, False  # returned UNCHANGED before any cap math -> emitted prefix byte-stable turn-to-turn.
    # Proportional cap: keep ~KEEP_FRACTION of THIS result, with a passthrough floor. Pure function
    # of this message's OWN length only (no trajectory/recency) -> byte-stable turn-to-turn = cache-safe.
    # Big results stay FULLER than a flat cap (keep 60% of 40k = 24k rather than a flat 16k), preserving context
    # preserved; small/mid results get compressed for savings. `cap` arg ignored in favour of the proportional value.
    cap = min(MAX_KEEP_CHARS, max(MIN_PASSTHROUGH_CHARS, int(len(text) * KEEP_FRACTION)))
    if len(text) <= cap:
        return message, False
    inner_cap = max(256, cap - len(CMP_START) - len(CMP_END) - 2)
    inner, _changed = extractive_compress(text, inner_cap, frozenset())
    wrapped = cmp_block(inner)
    # No-inflation invariant: emit the capped/wrapped form ONLY if it is
    # STRICTLY SMALLER than the original. extractive_compress returns the text unchanged when all lines
    # are pinned/fit, and the [[CMP]] wrapper adds bytes — so without this guard a ~cap-sized result
    # would EXIT LARGER than it entered (probed: 6001 -> 6018). With it, every message is <= its original
    # length, so the whole output is <= native always (cannot overflow worse than the no-plugin baseline).
    if len(wrapped) >= len(text):
        return message, False
    out = copy.deepcopy(message)
    out["content"] = wrapped
    return out, True

def handle_assemble(payload: dict[str, Any]) -> dict[str, Any]:
    raw_messages = get_messages(payload)
    # Stateless: process whatever the connector feeds us (its own history, or our
    # prior output plus new turns) with one deterministic per-message cap.
    sanitized, sanitize_changed = sanitize_messages(raw_messages)
    working = strip_loop_guard(sanitized)
    estimated = estimate_tokens(working)

    result_messages: list[Any] = []
    n_capped = 0
    for message in working:
        capped, changed = cap_tool_result(message, REPORTED_RESULT_CAP)
        result_messages.append(capped)
        if changed:
            n_capped += 1

    # Orphan guard (defensive): a per-message cap never drops a message, so call/result pairing is
    # preserved; still verify against the sanitized input and fall back to it on any violation.
    in_result_orphans, in_call_orphans = orphan_ids(working)
    out_result_orphans, out_call_orphans = orphan_ids(result_messages)
    if not (out_result_orphans <= in_result_orphans and out_call_orphans <= in_call_orphans):
        result_messages = working
        n_capped = 0

    # Compliant loop guard: append ONLY an allowed loop-reason string, only on an objective loop.
    # Appended at the END so it never perturbs the cacheable prefix.
    result_messages = strip_loop_guard(result_messages)
    loop_reason = detect_loop_reason(result_messages) or None
    result_messages = append_loop_guard(result_messages)

    changed = fingerprint_messages(result_messages) != fingerprint_messages(raw_messages)
    if not changed:
        # Nothing to cap (output == native): emit native untouched so OpenClaw sends its own context
        # and the provider cache is preserved exactly like a no-plugin run (maximal cache).
        result_messages = raw_messages

    metadata = {
        "changed": changed,
        "reason": "near_passthrough" if n_capped else "passthrough_native",
        "sanitized": sanitize_changed,
        "mode": "near_passthrough",
        "resultsCapped": n_capped,
        "resultCap": REPORTED_RESULT_CAP,
        "loopGuardFired": loop_reason,
        "originalMessageCount": len(raw_messages),
        "messageCount": len(result_messages),
        "estimatedInputTokens": estimated,
    }
    return {
        "assembled": True,
        "messages": result_messages,
        "estimatedTokens": final_token_estimate(result_messages),
        "baseMiner": metadata,
    }

def run_event(event_name: str) -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            raise ValueError("Connector payload must be a JSON object")
        if event_name not in EVENT_NAMES:
            raise ValueError(f"Unknown miner event: {event_name}")
        response = {"ok": True, "result": handle_assemble(payload)}
    except Exception as error:
        response = {"ok": False, "error": str(error), "errorType": error.__class__.__name__}
    sys.stdout.write(json.dumps(response, ensure_ascii=False))
    return 0

def cli_main(argv: list[str]) -> int:
    return run_event(argv[0] if argv else "assemble")

if __name__ == "__main__":
    raise SystemExit(cli_main(sys.argv[1:]))
