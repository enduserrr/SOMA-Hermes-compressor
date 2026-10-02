#!/usr/bin/env python3
"""SOMA context engine — plugin skeleton with token accounting (Task 2).

Subclasses Hermes' ContextEngine ABC. Compression behaviour (per-turn
rewriting of oversized tool results via the vendored soma_compressor) lands in
Task 3; this module provides identity, token accounting, model-switch
handling, and status reporting.

Fail-open invariant: no accounting error may ever break the agent turn. All
host-facing hooks swallow and log once, leaving the request untouched.
"""
from __future__ import annotations

import datetime
import json
import logging
import importlib.util
import pathlib
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vendored SOMA core (loaded lazily via _load_soma(); see bottom of file)
# ---------------------------------------------------------------------------
_SOMA_MODULE_PATH = pathlib.Path(__file__).resolve().parent / "soma_compressor.py"
_soma_mod = None


# -- Tunables (sizing table lives here; keep in sync with soma_compressor) --
# Sizing table (simplified 2026-09-05 after floor sweep on real sessions:
# floor raised 16K -> 32K = MAX_KEEP_CHARS, collapsing the ladder to one
# rule: nothing under 32K is ever touched; everything over is capped at 32K.
# Evidence: the 16-32K band is active-work payloads (13 read_file + 6
# terminal results in real history) where compression dropped unpinned
# body lines for <=12% savings; real-session cost of the higher floor is
# +0.96% chars sent, offset by fewer cache-breaking rewrites (16 vs 26
# distinct results). >53K behavior is unchanged (already capped at 32K).
# floor lowered 32K -> 24K (2026-09-07, user request): the 24-32K band was
# passing through untouched; now nothing under 24K is ever touched and
# everything over is capped at 24K.
# floor lowered 24K -> 18K (2026-09-27, user request): flat cap validated by
# soma-bench cap16k-flat (16K beat 24K: -24% weighted tokens on medium
# sessions, zero needle loss; short sessions neutral). 18K = 16K win with
# extra fidelity margin. MAX_KEEP_CHARS in soma_compressor.py matches.
PASSTHROUGH_CHARS = 18_000
MID_CEILING_CHARS = 18_000
MID_UPPER_CHARS = 53_300
KEEP_FRACTION = 0.60
LARGE_UPPER_CHARS = 53_300
LARGE_CAP_CHARS = 18_000


# -- Task 4: overflow gate + minimal fallback --------------------------------
# should_compress() only fires on provider-PROVEN overflow: the provider's own
# usage report exceeded the model's context window. The trigger deliberately
# does NOT use threshold_tokens — SOMA's real compression lives in
# select_context(); compress() is a last-resort survival path only.

FALLBACK_NOTE = (
    "[context compacted: prior conversation was dropped because it exceeded "
    "the model's context window (overflow). SOMA compression is handled "
    "per-request by select_context(); this fallback preserves only the system "
    "prompt and this notice.]"
)


# -- Task 5: per-request JSONL accounting ------------------------------------
# One line per select_context() call that changed something. Best-effort:
# accounting failures are logged once and never break the request.
ACCOUNTING_PATH = (
    pathlib.Path(__file__).resolve().parent / "accounting.jsonl"
)


def _est_chars(messages: List[Dict[str, Any]]) -> int:
    """Rough character footprint of a message list (content fields only)."""
    total = 0
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
    return total


def _append_accounting_record(
    input_chars: int,
    output_chars: int,
    results_capped: int,
    reason: str,
    session_id: str = "-",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Append one JSONL accounting record. Best-effort, never raises.

    ``session_id`` is written when known (captured via on_session_start) so
    the reporting CLI (``soma-savings``) can attribute per-request savings to
    a session. Defaults to "-" when unknown (e.g. offline bench harness,
    which never runs on_session_start). Recording a field is additive — it
    never affects the compression result written per request.

    ``input_tokens``/``output_tokens`` are token estimates via
    soma_compressor.final_token_estimate() (tiktoken cl100k_base when
    available, else the deterministic chars-per-token fallback). Estimates
    only — provider-reported usage in state.db remains the billing truth.
    """
    try:
        record = {
            "session_id": session_id,
            "input_est_chars": input_chars,
            "output_est_chars": output_chars,
            "input_est_tokens": input_tokens,
            "output_est_tokens": output_tokens,
            "results_capped": results_capped,
            "reason": reason,
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        }
        with ACCOUNTING_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        log.exception("soma: accounting write failed (request unaffected)")


# -- Live-config sync: threshold_tokens_cap coercion -------------------------
# Mirrors ContextCompressor._coerce_max_tokens (core normalization for
# threshold_tokens_cap: positive int or None). Tries to borrow the core's
# own function so the two can never drift; the local copy keeps this module
# importable (and cheap) when agent.context_compressor is not on the path.


def _coerce_threshold_tokens_cap_value(value: Any):
    try:
        from agent.context_compressor import ContextCompressor

        return ContextCompressor._coerce_threshold_tokens_cap(value)
    except Exception:
        pass
    try:
        ivalue = int(value) if value is not None else 0
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


class SomaEngine(ContextEngine):
    """SOMA context engine: skeleton with token accounting."""

    def __init__(self) -> None:
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.context_length: int = 0
        self.compression_count: int = 0
        self._model: str = ""
        self._base_url: str = ""
        self._api_key: str = ""
        self._provider: str = ""
        self._api_mode: str = ""
        self._fallback_compressor: Any = None  # lazily created built-in
        self._session_id: str = "-"  # set via on_session_start for accounting
        self._threshold_tokens_cap = None  # coerced via the property below
        self._threshold_tokens: Optional[int] = None  # cached; None = rederive
        # Raw config percent, BEFORE model overrides / floor — the fallback a
        # model switch lands on (core: _config_threshold_percent). The gateway
        # sync writes this attr; derivation re-reads it each time.
        self._config_threshold_percent: Optional[float] = None
        # Base percent after model_thresholds resolution (core: _base_threshold_percent).
        self._base_threshold_percent: Optional[float] = None

    # -- Live-config sync contract (tui_gateway._apply_live_compression_config)
    # The gateway hot-applies compression.* edits onto the LIVE engine object
    # (Desktop/TUI never rebuilds the agent), so SomaEngine must tolerate the
    # same attribute writes the built-in ContextCompressor accepts. Sep 2026:
    # a newer core added threshold_tokens_cap and this engine's __init__ no
    # longer had it — every config re-apply logged "'SomaEngine' object has no
    # attribute '_coerce_threshold_tokens_cap'" and skipped the rest of the
    # sync. Reproduce the built-in's surface exactly.

    @staticmethod
    def _coerce_threshold_tokens_cap(value):
        """Core's normalization: a threshold_tokens cap is a positive int, or
        None for 'no cap'. Borrows the core's own normalizer when importable
        (drift-proof), falls back to the identical local copy otherwise —
        agent.context_compressor stays a LAZY import (plugin import must
        remain cheap)."""
        return _coerce_threshold_tokens_cap_value(value)

    @property
    def threshold_tokens_cap(self):
        """Coerced cap (positive int | None); None = no cap configured."""
        return self._threshold_tokens_cap

    @threshold_tokens_cap.setter
    def threshold_tokens_cap(self, value):
        self._threshold_tokens_cap = self._coerce_threshold_tokens_cap(value)
        # Relay onto a live nested compressor (reviewer finding 1b): its
        # firing threshold must honor the configured cap.
        nested = getattr(self, "_fallback_compressor", None)
        if nested is not None:
            try:
                nested.threshold_tokens_cap = self._threshold_tokens_cap
                nested._threshold_tokens = None
            except Exception:
                pass

    @staticmethod
    def _effective_threshold_percent(context_length: int, threshold_percent: float) -> float:
        """Raise-only small-context floor, mirroring the built-in compressor:
        windows under 512K trigger at >= 75% so live threshold edits derive
        the same trigger update_model() installs."""
        if context_length and context_length < 512_000:
            return max(threshold_percent, 0.75)
        return threshold_percent

    # -- Threshold derivation (mirrors core ContextCompressor) -----------------

    @property
    def threshold_percent(self) -> float:
        """Effective percent. Backing attr is written by the setter below;
        the raw config value lives in _config_threshold_percent (sync writes
        it), and _base_threshold_percent holds the model_thresholds-resolved
        value — both core names, both plain attrs on this engine."""
        return getattr(self, "_threshold_percent_value", 0.75)

    @threshold_percent.setter
    def threshold_percent(self, value) -> None:
        try:
            pct = float(value)
        except (TypeError, ValueError):
            return
        self._threshold_percent_value = pct
        # Relay onto a live nested compressor (reviewer finding 1b): the
        # nested owns the actual compaction firing, so gateway-synced
        # thresholds must reach it. Its cached trigger is invalidated so the
        # next read re-derives with its own floor+cap logic.
        nested = getattr(self, "_fallback_compressor", None)
        if nested is not None:
            try:
                nested.threshold_percent = pct
                raw = getattr(self, "_config_threshold_percent", None)
                nested._config_threshold_percent = raw if raw is not None and raw > 0 else pct
                nested._base_threshold_percent = pct
                nested._threshold_tokens = None
            except Exception:
                pass

    @property
    def threshold_tokens(self) -> int:
        """Cached trigger; rederived on read when _threshold_tokens is None
        (core's cached-property semantics — the gateway sync's invalidation
        write `cc._threshold_tokens = None` then works verbatim)."""
        if self._threshold_tokens is None:
            ctx = self.context_length
            if ctx > 0:
                effective = self._effective_threshold_percent(ctx, self.threshold_percent)
                trigger = int(ctx * effective)
                cap = self.threshold_tokens_cap
                if cap is not None and 0 < cap < trigger:
                    trigger = cap
                self._threshold_tokens = trigger
        return self._threshold_tokens or 0

    @threshold_tokens.setter
    def threshold_tokens(self, value) -> None:
        self._threshold_tokens = value

    def clone_for_agent(self) -> "SomaEngine":
        """Fresh engine per agent (agent/context_engine.py contract).

        The default clone is copy.deepcopy, which would copy a live nested
        ContextCompressor (locks, caches, partial turn state). A fresh
        SomaEngine is the correct, cheap clone: config signature is re-applied
        by the host on the clone's first turn and model overrides re-derive at
        agent init via update_model().
        """
        return SomaEngine()

    # -- Session lifecycle ----------------------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Record the active session id so accounting rows are attributable.

        Additive only: unlike the host's compression/summarizer path this just
        stores an id for the reporting CLI. compression_result / select_context
        output is completely unaffected. Fail-open: a session id that cannot
        be stored must never raise into agent startup.
        """
        try:
            self._session_id = session_id or "-"
        except Exception:
            self._session_id = "-"

    # -- Identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier; must match context.engine in config.yaml."""
        return "soma"

    # -- Token accounting ----------------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from an API response.

        Fail-open: a malformed payload logs once and leaves prior counters
        intact rather than raising into the agent loop. Usage is also
        mirrored into the nested built-in compressor so its threshold
        logic sees real token counts.
        """
        try:
            self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
            self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
            self.last_total_tokens = int(usage.get("total_tokens") or 0)
            nested = self._fallback_compressor
            if nested is not None:
                try:
                    nested.last_prompt_tokens = self.last_prompt_tokens
                    nested.last_completion_tokens = self.last_completion_tokens
                    nested.last_total_tokens = self.last_total_tokens
                except Exception:
                    pass
        except (TypeError, ValueError, AttributeError):
            log.exception("soma: malformed usage payload ignored")
            return

    # -- Model switch support ------------------------------------------------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        """Record the active model's context window and derive the threshold.

        Derivation mirrors core's _derive_trigger + update_model: re-resolve
        the base from the RAW config percent (_config_threshold_percent) via
        model_thresholds so a switch away from an overridden model falls back
        to the configured value; apply the raise-only small-context floor;
        then clamp to the configured threshold_tokens_cap. The gateway sync
        (tui_gateway._apply_live_compression_config) may have written any of
        those attrs onto this object.
        """
        self.context_length = context_length
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._provider = provider
        self._api_mode = api_mode
        raw = getattr(self, "_config_threshold_percent", None)
        if raw is None:
            raw = self.threshold_percent
            self._config_threshold_percent = raw
        try:
            from agent.context_compressor import resolve_model_threshold
            base = resolve_model_threshold(
                model, getattr(self, "model_thresholds", None) or {}, raw, provider
            )
        except Exception:
            base = raw
        self._base_threshold_percent = base
        self._threshold_percent_value = base
        effective = self._effective_threshold_percent(context_length, base)
        trigger = int(context_length * effective)
        cap = self.threshold_tokens_cap
        if cap is not None and 0 < cap < trigger:
            trigger = cap
        self._threshold_tokens = trigger
        if self._fallback_compressor is not None:
            self._fallback_compressor.update_model(
                model, context_length, base_url=base_url,
                api_key=self._api_key, provider=provider, api_mode=api_mode,
            )

    # -- Status / display ----------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Status dict for display/logging (host contract: run_agent.py)."""
        last_prompt = self.last_prompt_tokens if self.last_prompt_tokens > 0 else 0
        return {
            "last_prompt_tokens": last_prompt,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (
                min(100, last_prompt / self.context_length * 100)
                if self.context_length
                else 0
            ),
            "compression_count": self.compression_count,
        }

    # -- Host-required abstract methods (later tasks) -------------------------

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        """Delegate the compaction decision to the nested built-in compressor.

        Proven overflow (provider-reported usage beyond the window) fires
        first and unconditionally. Otherwise the nested built-in decides —
        it owns thresholds (threshold_tokens from config), summary-LLM
        cooldowns, and anti-thrash state — so auto-compaction behaves
        exactly as it would natively while SOMA's select_context keeps the
        cheap per-request shrink. Fail-open: any exception -> False.
        """
        try:
            if (
                self.context_length > 0
                and self.last_total_tokens > self.context_length
            ):
                return True
            nested = self._get_fallback_compressor()
            if nested is not None:
                try:
                    return bool(nested.should_compress(prompt_tokens))
                except Exception:
                    log.exception("soma: nested should_compress failed")
            return False
        except Exception:
            log.exception("soma: should_compress failed; defaulting to False")
            return False

    def _get_fallback_compressor(self):
        """Lazily create the nested built-in ContextCompressor (LLM summarizer).

        Created on first compress() call, not at engine init, so the plugin
        import stays cheap. Returns None if construction fails (fail-open:
        callers fall back to the synthetic survival list).
        """
        if self._fallback_compressor is None:
            try:
                from agent.context_compressor import ContextCompressor
                # Build from the ENGINE's synced threshold state, not hardcoded
                # values (reviewer finding 1a): the nested owns the actual
                # compaction firing via should_compress delegation, so the
                # gateway-synced percent/cap must flow into it. Fallback 0.50
                # = core's config default, used only when nothing synced yet.
                raw = getattr(self, "_config_threshold_percent", None)
                nested_pct = raw if raw is not None and raw > 0 else 0.50
                self._fallback_compressor = ContextCompressor(
                    model=self._model or "default",
                    threshold_percent=nested_pct,
                    base_url=self._base_url,
                    api_key=self._api_key,
                    provider=self._provider,
                    api_mode=self._api_mode,
                    config_context_length=self.context_length or None,
                    threshold_tokens_cap=self.threshold_tokens_cap,
                )
                if self.context_length:
                    self._fallback_compressor.update_model(
                        self._model, self.context_length,
                        base_url=self._base_url, api_key=self._api_key,
                        provider=self._provider, api_mode=self._api_mode,
                    )
            except Exception:
                log.exception("soma: nested ContextCompressor init failed")
                return None
        return self._fallback_compressor

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Whole-session compaction: delegate to the built-in LLM summarizer.

        SOMA handles per-request shrink in select_context(); genuine whole-
        session compaction (threshold-triggered or manual /compress) is the
        built-in ContextCompressor's job, so delegate to a nested instance.
        No provider-proven overflow and not manual -> identity pass-through.
        If the nested compressor fails, fall back to the synthetic survival
        list so the engine alone can survive a wedged session. Fail-open.
        """
        try:
            overflow = self.context_length > 0 and self.last_total_tokens > self.context_length
            if not overflow and not force:
                return messages
            nested = self._get_fallback_compressor()
            if nested is not None:
                try:
                    result = nested.compress(
                        messages,
                        current_tokens=current_tokens,
                        focus_topic=focus_topic,
                        force=force,
                        memory_context=memory_context,
                    )
                    if isinstance(result, list) and result and all(
                        isinstance(m, dict) for m in result
                    ):
                        self.compression_count += 1
                        return result
                    log.exception("soma: nested compress returned invalid value")
                except Exception:
                    log.exception("soma: nested compress failed; using synthetic fallback")
            # Synthetic survival fallback (no nested compressor, or it failed)
            out: List[Dict[str, Any]] = []
            first = messages[0] if messages else None
            if isinstance(first, dict) and first.get("role") == "system":
                out.append(first)
            out.append({"role": "user", "content": FALLBACK_NOTE})
            return out
        except Exception:
            log.exception("soma: compress fallback failed; leaving request untouched")
            return messages

    # -- Task 3: per-turn request rewriting ----------------------------------

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: List[Dict[str, Any]] = None,
        incoming_message: Dict[str, Any] = None,
        budget_tokens: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """Rewrite oversized tool results in THIS request via the SOMA core.

        Request-only: persisted history is never mutated. Returns None when
        nothing changed so the provider cache prefix stays byte-identical.
        Fail-open: any exception -> None (request untouched).
        """
        try:
            return self._select_context_impl(request_messages)
        except Exception:
            log.exception("soma: select_context failed; leaving request untouched")
            return None

    def _select_context_impl(
        self,
        request_messages: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        soma = _load_soma()
        out_changed = False
        results_capped = 0
        out: List[Dict[str, Any]] = []
        for message in request_messages:
            capped, changed = _cap_openai_tool_result(soma, message)
            out.append(capped)
            results_capped += 1 if changed else 0
            out_changed = out_changed or changed
        if not out_changed:
            return None
        # Orphan guard (SOMA invariant): if rewriting would orphan tool_calls
        # or tool results, fall back to the original request.
        in_result_orphans, in_call_orphans = soma.orphan_ids(request_messages)
        out_result_orphans, out_call_orphans = soma.orphan_ids(out)
        if not (
            out_result_orphans <= in_result_orphans
            and out_call_orphans <= in_call_orphans
        ):
            return None
        # Task 5 accounting: one JSONL line per request that changed something.
        # Best-effort — never raises into the agent loop.
        # Token estimates come from the vendored SOMA core's
        # final_token_estimate() (tiktoken if available, else chars/token
        # fallback). Estimation only — compression selection itself stays
        # char-based (token-aware selection would change which lines survive).
        try:
            in_tokens = soma.final_token_estimate(request_messages)
            out_tokens = soma.final_token_estimate(out)
        except Exception:
            log.exception("soma: token estimate failed; recording zeros")
            in_tokens = out_tokens = 0
        _append_accounting_record(
            _est_chars(request_messages),
            _est_chars(out),
            results_capped,
            "near_passthrough",
            self._session_id,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )
        return out


def _cap_openai_tool_result(soma: Any, message: Dict[str, Any]) -> tuple:
    """Map an OpenAI-format tool message to SOMA's shape, cap it, map back.

    SOMA's cap_tool_result() keys on role 'toolResult' and reads 'content';
    OpenAI tool results are role 'tool'. Non-tool messages pass verbatim.
    Returns (message, changed).
    """
    if not isinstance(message, dict) or soma.normalize_role(message.get("role")) != "tool":
        return message, False
    if soma.CMP_START in str(message.get("content") or ""):
        return message, False  # already-compressed fixed point: never re-cap
    bridged = dict(message)
    # Hermes stores read_file-style tool results as a single-line JSON
    # envelope ({"content": "..."} with escaped newlines). terminal-style
    # results use the same envelope shape keyed on "output"
    # ({"output": "...", "exit_code": N}) — verified against persisted
    # history in state.db (65-87K-char envelopes rode through untouched
    # because the unwrap only recognised "content"). SOMA is line-based, so
    # a one-line payload is incompressible as-is. Unwrap the envelope,
    # compress the inner text, re-wrap — only if strictly smaller. Envelope
    # metadata (total_lines, file_size, exit_code, error, cwd) is preserved
    # in the re-wrap: downstream consumers (e.g. the default compressor's
    # _summarize_tool_result) grep "exit_code" back out of the content.
    unwrapped = _unwrap_json_envelope(message.get("content"))
    if unwrapped is not None:
        inner_text, envelope_obj, text_key = unwrapped
        if soma.CMP_START not in inner_text:
            target = _sizing_target(soma, len(inner_text))
            compressed, changed = soma.extractive_compress(inner_text, target, frozenset())
            wrapped = soma.cmp_block(compressed)
            if changed and len(wrapped) < len(inner_text):
                envelope_obj[text_key] = wrapped
                out = dict(message)
                out["content"] = json.dumps(envelope_obj, ensure_ascii=False)
                if len(out["content"]) < len(message["content"]):
                    return out, True
        return message, False
    bridged["role"] = "toolResult"
    capped, changed = soma.cap_tool_result(bridged, soma.REPORTED_RESULT_CAP)
    if not changed:
        return message, False
    out = dict(capped)
    out["role"] = "tool"
    return out, True


_JSON_ENVELOPE_KEYS = ("content", "output", "total_lines", "file_size")


def _unwrap_json_envelope(content: Any):
    """If content is a Hermes tool-result JSON envelope, return (inner_text, dict, key).

    read_file results are a single-line JSON object like
    {"content": "...", "total_lines": N, ...}; terminal results are
    {"output": "...", "exit_code": N, ...} (verified against persisted
    history in state.db). Returns None for anything that is not a parseable
    dict envelope carrying a compressible text field. The key that held the
    text is returned so the re-wrap writes the compressed text back to the
    same field; envelope metadata (exit_code, error, cwd, ...) is preserved.
    """
    if not isinstance(content, str) or not content.lstrip().startswith("{"):
        return None
    try:
        obj = json.loads(content)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    for text_key in ("content", "output"):
        value = obj.get(text_key)
        if isinstance(value, str):
            return value, obj, text_key
    if any(k in obj for k in _JSON_ENVELOPE_KEYS):
        return None  # envelope-shaped but no compressible text field
    return None


def _sizing_target(soma: Any, text_len: int) -> int:
    """SOMA sizing table (verbatim): proportional cap with floor and ceiling."""
    return min(
        soma.MAX_KEEP_CHARS,
        max(soma.MIN_PASSTHROUGH_CHARS, int(text_len * soma.KEEP_FRACTION)),
    )


def _load_soma():
    """Load the vendored soma_compressor module once (import-time lazily).

    The vendored file stays byte-identical to upstream (ARCHITECTURE.md
    invariant); host-specific tuning is applied HERE, after load: the
    engine's PASSTHROUGH_CHARS (24K, lowered from the earlier host value
    32K on 2026-09-07; upstream core is 16K) overrides
    the core's MIN_PASSTHROUGH_CHARS. With floor == MAX_KEEP_CHARS the
    sizing ladder collapses to a single rule: under 24K untouched, over 24K
    capped at 24K. cap_tool_result's internal proportional cap also reads
    MIN_PASSTHROUGH_CHARS, so the override covers every path.
    """
    global _soma_mod
    if _soma_mod is None:
        spec = importlib.util.spec_from_file_location("soma_compressor", _SOMA_MODULE_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if PASSTHROUGH_CHARS != mod.MIN_PASSTHROUGH_CHARS:
            mod.MIN_PASSTHROUGH_CHARS = PASSTHROUGH_CHARS
        _soma_mod = mod
    return _soma_mod
