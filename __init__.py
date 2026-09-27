"""fast_jev — Hermes context engine port of tamaratran/fast-jev-compaction.

Instead of summarizing, asks TypeSafe's Jev (one fast scoring request per
batch) whether each old tool call / tool result is still needed, then drops
or truncates the stale ones. Everything kept stays verbatim.

Falls back to the built-in ContextCompressor summary when Jev fails, the key
is missing, or the reduction is too small.

Enable:  hermes config set context.engine fast_jev
Key:     TYPESAFE_API_KEY in $HERMES_HOME/.env
Tuning:  context.fast_jev.* in config.yaml (see README.md)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from agent.context_compressor import ContextCompressor
from agent.model_metadata import estimate_messages_tokens_rough

from . import core

logger = logging.getLogger(__name__)

SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

DEFAULTS: Dict[str, Any] = {
    "model": DEFAULT_MODEL,
    "base_url": SYSTEM_ONE_URL,
    "keep_threshold": 0.5,
    "preserve_recent_messages": 10,
    "max_state_tokens": 25_000,
    "max_request_tokens": 30_000,
    "truncate_head_chars": 300,
    "min_reduction_ratio": 0.25,
    "timeout_seconds": 60,
    # After pruning, if the history is still above this fraction of the
    # compression threshold, run the built-in summary on the pruned history.
    "summarize_if_above": 0.85,
}


def _load_cfg() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly
        return load_config_readonly() or {}
    except Exception:  # pragma: no cover - host without hermes_cli
        return {}


def _read_env_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if key:
        return key
    try:
        from hermes_constants import get_hermes_home
        env_path = get_hermes_home() / ".env"
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


class JevClient:
    def __init__(self, model: str, base_url: str, timeout: float):
        self.model, self.base_url, self.timeout = model, base_url, timeout

    def __call__(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        key = _read_env_key()
        if not key:
            raise RuntimeError("TYPESAFE_API_KEY is not configured")
        body = json.dumps({"model": self.model, "state": state, "questions": questions},
                          ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base_url, data=body, method="POST", headers={
            "authorization": f"Bearer {key}", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                text = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            raise RuntimeError(f"Jev request failed ({e.code}): {detail}") from None
        try:
            parsed = json.loads(text)
        except ValueError:
            raise RuntimeError("Jev returned malformed JSON") from None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("answers"), dict):
            raise RuntimeError("Jev response is missing answers")
        return parsed


class FastJevEngine(ContextCompressor):
    """ContextCompressor whose compress() prunes via Jev before (optionally) summarizing."""

    @property
    def name(self) -> str:
        return "fast_jev"

    def __init__(self, **kwargs: Any):
        cfg = _load_cfg()
        comp = cfg.get("compression", {}) if isinstance(cfg, dict) else {}
        ctx = cfg.get("context", {}) if isinstance(cfg, dict) else {}
        jev_cfg = dict(DEFAULTS)
        if isinstance(ctx.get("fast_jev"), dict):
            jev_cfg.update(ctx["fast_jev"])
        self.jev_cfg = jev_cfg
        init = dict(
            model=kwargs.pop("model", ""),
            threshold_percent=float(comp.get("threshold", 0.50)),
            protect_first_n=int(comp.get("protect_first_n", 3)),
            protect_last_n=int(comp.get("protect_last_n", 20)),
            summary_target_ratio=float(comp.get("target_ratio", 0.20)),
            quiet_mode=True,
            abort_on_summary_failure=bool(comp.get("abort_on_summary_failure", False)),
            threshold_tokens_cap=comp.get("threshold_tokens"),
            tail_mode=str(comp.get("tail_mode", "lean")),
        )
        init.update(kwargs)
        super().__init__(**init)
        self.last_jev_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------ core

    def _options(self, focus_topic: Optional[str]) -> core.Options:
        c = self.jev_cfg
        return core.Options(
            goal=(focus_topic or "").strip(),
            keep_threshold=float(c["keep_threshold"]),
            preserve_recent_messages=int(c["preserve_recent_messages"]),
            preserve_first_messages=max(1, int(self.protect_first_n) + 1),  # +1: system row
            max_state_tokens=int(c["max_state_tokens"]),
            max_request_tokens=int(c["max_request_tokens"]),
            truncate_head_chars=int(c["truncate_head_chars"]),
        )

    def jev_prune(self, messages: List[Dict[str, Any]], focus_topic: Optional[str] = None) -> core.Result:
        client = JevClient(str(self.jev_cfg["model"]), str(self.jev_cfg["base_url"]),
                           float(self.jev_cfg["timeout_seconds"]))
        return core.compact(messages, client, self._options(focus_topic))

    def compress(self, messages: List[Dict[str, Any]], current_tokens: Optional[int] = None,
                 focus_topic: Optional[str] = None, force: bool = False, memory_context: str = "",
                 **kwargs: Any) -> List[Dict[str, Any]]:
        def builtin(msgs):
            return super(FastJevEngine, self).compress(
                msgs, current_tokens=current_tokens, focus_topic=focus_topic, force=force,
                memory_context=memory_context, **kwargs)

        try:
            result = self.jev_prune(messages, focus_topic)
        except Exception as exc:
            logger.warning("fast_jev: Jev failed (%s) — falling back to built-in summary", exc)
            self.last_jev_stats = {"error": str(exc)}
            return builtin(messages)

        ratio = core.reduction_ratio(result)
        self.last_jev_stats = dict(result.stats, reduction_ratio=round(ratio, 3))
        logger.info("fast_jev: %s", self.last_jev_stats)
        if ratio < float(self.jev_cfg["min_reduction_ratio"]):
            logger.info("fast_jev: reduction %.0f%% below minimum — built-in summary", ratio * 100)
            return builtin(messages)

        pruned = self._sanitize_tool_pairs(result.messages)
        after = estimate_messages_tokens_rough(pruned)
        limit = float(self.jev_cfg["summarize_if_above"]) * float(self.threshold_tokens or 0)
        if limit and after > limit:
            logger.info("fast_jev: still ~%d tokens after pruning (> %d) — summarizing pruned history",
                        after, int(limit))
            return builtin(pruned)

        self.compression_count += 1
        self._last_compression_made_progress = True
        return pruned

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["fast_jev"] = self.last_jev_stats
        return status


def register(ctx) -> None:
    ctx.register_context_engine(FastJevEngine())
