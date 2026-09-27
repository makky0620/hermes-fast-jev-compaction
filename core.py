"""Pure compaction logic ported from tamaratran/fast-jev-compaction (src/).

Works on Hermes' OpenAI-format message dicts:
  assistant: {"role": "assistant", "content": ..., "tool_calls": [{"id", "function": {"name", "arguments"}}]}
  tool:      {"role": "tool", "tool_call_id": ..., "content": ...}

Nothing here is rewritten or summarized: tool calls/results are either kept,
truncated (result only), or removed together with their result.
"""

from __future__ import annotations

import copy
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

STATE_CONTEXT = (
    "A coding assistant conversation is being compacted to free context. `history` is the whole "
    "conversation so far, oldest first; tool outputs are replaced by a short `result` note and long "
    "texts may be abridged. Each question asks whether one tool call, or the full output of that "
    "call, still needs to stay in the history verbatim. Whatever is not kept is deleted permanently, "
    "but the assistant can always re-run a tool or re-read a file."
)

INPUT_CHARS = (1000, 200, 60)
TEXT_HEAD = 400
TEXT_TAIL = 150
REQUEST_OVERHEAD_TOKENS = 20
_TOKEN_PIECES = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")

# Jev asker: (state, questions) -> {"answers": {name: {"noul": float}}}
Asker = Callable[[Any, Dict[str, Any]], Dict[str, Any]]


@dataclass
class Options:
    goal: str = ""
    keep_threshold: float = 0.5
    preserve_recent_messages: int = 6
    preserve_first_messages: int = 1
    max_state_tokens: int = 25_000
    max_request_tokens: int = 30_000
    truncate_head_chars: int = 300
    max_concurrency: int = 4


@dataclass
class ToolCall:
    id: str
    tool_use_id: str
    tool: str
    input: Any
    call_index: int
    result_index: int
    result_chars: int
    is_error: bool
    pinned: bool


@dataclass
class Decision:
    id: str
    tool: str
    keep_call: float
    keep_result: float
    action: str  # keep | drop_result | drop_call
    reason: str  # pinned | kept | result_dropped | call_dropped


@dataclass
class Result:
    messages: List[Dict[str, Any]]
    decisions: List[Decision]
    stats: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- helpers

def estimate_tokens(text: str) -> int:
    """Tokenizer-free estimate calibrated by upstream against Jev's reported usage."""
    tokens = 0.0
    for m in _TOKEN_PIECES.finditer(text):
        piece = m.group(0)
        c = piece[0]
        if c.isdigit() and c.isascii():
            tokens += len(piece) / 2
        elif c.isascii() and c.isalpha():
            tokens += 1 + (len(piece) - 1) // 6
        else:
            tokens += 0.9
    return int(-(-tokens // 1))


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _abridge(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail + 40:
        return text
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[… {omitted} chars omitted …]\n{text[-tail:]}"


def content_text(content: Any) -> str:
    """Flatten OpenAI content (str or list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    out.append(str(part.get("text", "")))
                elif part.get("type") in ("image_url", "image", "input_image"):
                    out.append("[image]")
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return str(content)


def _parse_args(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw
    return raw


def _is_pinned(i: int, total: int, opts: Options) -> bool:
    return i < max(1, opts.preserve_first_messages) or i >= total - opts.preserve_recent_messages


def _is_error_text(text: str) -> bool:
    head = text[:200].lower()
    return head.startswith("error") or '"error"' in head


# ---------------------------------------------------------------- collection

def collect_tool_calls(messages: Sequence[Dict[str, Any]], opts: Options) -> List[ToolCall]:
    results: Dict[str, tuple] = {}
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and m.get("tool_call_id"):
            results[m["tool_call_id"]] = (i, content_text(m.get("content")))
    calls: List[ToolCall] = []
    total = len(messages)
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            tid = tc.get("id")
            found = results.get(tid)
            if not tid or not found:
                continue
            ri, text = found
            fn = tc.get("function") or {}
            calls.append(ToolCall(
                id=f"t{len(calls) + 1}", tool_use_id=tid, tool=str(fn.get("name", "?")),
                input=_parse_args(fn.get("arguments")), call_index=i, result_index=ri,
                result_chars=len(text), is_error=_is_error_text(text),
                pinned=_is_pinned(i, total, opts) or _is_pinned(ri, total, opts),
            ))
    return calls


# ---------------------------------------------------------------- state

def _input_text(value: Any, limit: int) -> str:
    try:
        s = value if isinstance(value, str) else _dumps(value)
    except (TypeError, ValueError):
        s = "[unserializable input]"
    return truncate(s, limit)


def _result_note(c: ToolCall) -> str:
    return f"{'error' if c.is_error else 'ok'}, {c.result_chars} chars (omitted)"


def _compact_call(c: ToolCall) -> str:
    if isinstance(c.input, dict):
        parts = []
        for k, v in c.input.items():
            t = v if isinstance(v, str) else _input_text({k: v}, 200)
            t = re.sub(r"\s+", " ", t)
            parts.append(f"{k}={t}")
        inp = " ".join(parts)
    else:
        inp = re.sub(r"\s+", " ", _input_text(c.input, 200))
    return f"{c.id} {c.tool} {truncate(inp, INPUT_CHARS[2])} → {'error' if c.is_error else 'ok'} {c.result_chars}ch"


def _by_message(calls: Sequence[ToolCall]) -> Dict[int, List[ToolCall]]:
    out: Dict[int, List[ToolCall]] = {}
    for c in calls:
        out.setdefault(c.call_index, []).append(c)
    return out


def _history(messages, calls, input_chars) -> List[Dict[str, Any]]:
    by = _by_message(calls)
    entries = []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "tool":
            continue  # represented by the call's result note
        tcs = [{"id": c.id, "tool": c.tool, "input": _input_text(c.input, input_chars),
                "result": _result_note(c)} for c in by.get(i, [])]
        text = content_text(m.get("content"))
        if not text.strip() and not tcs:
            continue
        e: Dict[str, Any] = {"i": i, "role": role, "text": text}
        if tcs:
            e["tool_calls"] = tcs
        entries.append(e)
    return entries


def goal_from_messages(messages) -> str:
    prompts = [content_text(m.get("content")) for m in messages
               if m.get("role") == "user" and content_text(m.get("content")).strip()]
    return "\n".join(truncate(p, 500) for p in prompts[-3:])


def fit_state(messages, calls, opts: Options) -> Dict[str, Any]:
    goal = opts.goal or goal_from_messages(messages)

    def state_of(h):
        return {"context": STATE_CONTEXT, "goal": goal, "history": h}

    def et(e):
        return estimate_tokens(_dumps(e)) + 1

    base = estimate_tokens(_dumps(state_of([])))
    total = len(messages)
    fits = lambda t: t <= opts.max_state_tokens  # noqa: E731

    def build(ic):
        h = _history(messages, calls, ic)
        pe = [et(e) for e in h]
        return h, pe, base + sum(pe)

    history, per, tokens = build(INPUT_CHARS[0])
    if fits(tokens):
        return {"state": state_of(history), "tokens": tokens, "stage": "full"}
    for lim in INPUT_CHARS[1:]:
        history, per, tokens = build(lim)
        if fits(tokens):
            return {"state": state_of(history), "tokens": tokens, "stage": f"inputs<={lim}"}

    pinned = lambda e: _is_pinned(e["i"], total, opts)  # noqa: E731
    idx = list(range(len(history)))
    order = [k for k in idx if not pinned(history[k])] + [k for k in idx if pinned(history[k])]

    def shrink(k, fn):
        nonlocal tokens
        fn(history[k])
        now = et(history[k])
        tokens += now - per[k]
        per[k] = now

    for k in order:
        if len(history[k]["text"]) <= TEXT_HEAD + TEXT_TAIL + 40:
            continue
        shrink(k, lambda e: e.__setitem__("text", _abridge(e["text"], TEXT_HEAD, TEXT_TAIL)))
        if fits(tokens):
            return {"state": state_of(history), "tokens": tokens, "stage": "texts abridged"}

    for k in order:
        e = history[k]
        if pinned(e) or not e["text"]:
            continue
        orig = len(content_text(messages[e["i"]].get("content")))
        shrink(k, lambda e, n=orig: e.__setitem__("text", f"[… {n} chars omitted …]"))
        if fits(tokens):
            return {"state": state_of(history), "tokens": tokens, "stage": "old messages collapsed"}

    by = _by_message(calls)
    for k in order:
        e = history[k]
        own = by.get(e["i"])
        if pinned(e) or not own:
            continue
        shrink(k, lambda e, own=own: e.__setitem__("tool_calls", [_compact_call(c) for c in own]))
        if fits(tokens):
            return {"state": state_of(history), "tokens": tokens, "stage": "old calls compacted"}

    left = set()
    for k in order:
        e = history[k]
        if pinned(e) or e.get("tool_calls"):
            continue
        left.add(k)
        tokens -= per[k]
        if fits(tokens):
            h = [x for j, x in enumerate(history) if j not in left]
            return {"state": state_of(h), "tokens": tokens, "stage": "old messages left out"}

    merged: List[Dict[str, Any]] = []
    foldable = lambda e: (not pinned(e) and not e["text"]  # noqa: E731
                          and isinstance((e.get("tool_calls") or [None])[0], str))
    for j, e in enumerate(history):
        if j in left:
            continue
        prev = merged[-1] if merged else None
        if prev and foldable(prev) and foldable(e) and prev["role"] == e["role"]:
            prev["tool_calls"] = prev["tool_calls"] + e["tool_calls"]
            continue
        merged.append(dict(e))
    tokens = base + sum(et(e) for e in merged)
    if fits(tokens):
        return {"state": state_of(merged), "tokens": tokens, "stage": "old calls merged"}
    raise ValueError(f"history too large for Jev (~{tokens} tokens, limit {opts.max_state_tokens})")


# ---------------------------------------------------------------- questions / decisions

def questions_for(c: ToolCall) -> Dict[str, Any]:
    return {
        f"call_{c.id}": {
            "type": "noul",
            "instructions": f"Tool call {c.id} ({c.tool}) should stay in the history: knowing this call "
                            f"was made, with its input, still matters for what the assistant does next",
        },
        f"result_{c.id}": {
            "type": "noul",
            "instructions": f"The full output of tool call {c.id} ({c.tool}, {c.result_chars} chars) should "
                            f"stay in the history verbatim: the assistant still needs its contents and "
                            f"re-running the tool would not do",
        },
    }


def batch_calls(calls, state_tokens: int, opts: Options) -> List[List[ToolCall]]:
    budget = opts.max_request_tokens - state_tokens - REQUEST_OVERHEAD_TOKENS
    batches, cur, cur_t = [], [], 0
    for c in calls:
        t = estimate_tokens(_dumps(questions_for(c)))
        if cur and cur_t + t > budget:
            batches.append(cur)
            cur, cur_t = [], 0
        if not cur and t > budget:
            raise ValueError(f"state leaves no room for questions (~{state_tokens} of {opts.max_request_tokens})")
        cur.append(c)
        cur_t += t
    if cur:
        batches.append(cur)
    return batches


def noul(answers: Dict[str, Any], name: str) -> float:
    a = answers.get(name)
    v = a.get("noul") if isinstance(a, dict) else None
    if not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"Invalid Jev answer for {name}")
    return float(v)


def decide(c: ToolCall, keep_call: float, keep_result: float, opts: Options) -> Decision:
    base = dict(id=c.id, tool=c.tool, keep_call=keep_call, keep_result=keep_result)
    if c.pinned:
        return Decision(**base, action="keep", reason="pinned")
    if keep_result >= opts.keep_threshold:
        return Decision(**base, action="keep", reason="kept")
    if keep_call >= opts.keep_threshold:
        return Decision(**base, action="drop_result", reason="result_dropped")
    return Decision(**base, action="drop_call", reason="call_dropped")


def _truncated(text: str, is_error: bool, head: int) -> str:
    if len(text) <= head + 120:
        return text
    h = f"{text[:head]}\n" if head > 0 else ""
    return (f"{h}[fast-jev-compaction truncated {len(text) - head} chars of this tool result"
            f"{' (error)' if is_error else ''}; re-run the tool if needed]")


def apply_decisions(messages, decisions, calls, head_chars: int) -> List[Dict[str, Any]]:
    by_id = {c.id: c for c in calls}
    actions: Dict[str, str] = {}
    for d in decisions:
        c = by_id.get(d.id)
        if c and d.action != "keep":
            actions[c.tool_use_id] = d.action
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "tool" and m.get("tool_call_id") in actions:
            act = actions[m["tool_call_id"]]
            if act == "drop_call":
                continue
            text = content_text(m.get("content"))
            new = _truncated(text, _is_error_text(text), head_chars)
            if new != text:
                m = dict(m)
                m["content"] = new
                m.pop("api_content", None)
            out.append(m)
            continue
        if role == "assistant" and any(tc.get("id") in actions for tc in m.get("tool_calls") or []):
            kept = [tc for tc in m["tool_calls"] if actions.get(tc.get("id")) != "drop_call"]
            if len(kept) != len(m["tool_calls"]):
                text = content_text(m.get("content"))
                has_reasoning = bool(m.get("reasoning") or m.get("reasoning_content"))
                if not kept and not text.strip() and not has_reasoning:
                    continue
                m = dict(m)
                if kept:
                    m["tool_calls"] = kept
                else:
                    m.pop("tool_calls", None)
                    if not text.strip():
                        m["content"] = "(earlier tool calls removed by fast-jev-compaction)"
                m.pop("api_content", None)
        out.append(m)
    return out


def _chars(m: Dict[str, Any]) -> int:
    n = len(content_text(m.get("content")))
    for tc in m.get("tool_calls") or []:
        n += len(str((tc.get("function") or {}).get("arguments", "")))
    return n


def reduction_ratio(r: Result) -> float:
    b, a = r.stats["chars_before"], r.stats["chars_after"]
    return 0.0 if b == 0 else (b - a) / b


def compact(messages: Sequence[Dict[str, Any]], asker: Asker, opts: Optional[Options] = None) -> Result:
    opts = opts or Options()
    started = time.time()
    calls = collect_tool_calls(messages, opts)
    cands = [c for c in calls if not c.pinned]
    before = sum(_chars(m) for m in messages)
    fitted = {"tokens": 0, "stage": ""}
    batches: List[List[ToolCall]] = []
    answers: Dict[str, tuple] = {}
    if cands:
        fitted = fit_state(messages, calls, opts)
        batches = batch_calls(cands, fitted["tokens"], opts)

        def ask(batch):
            qs: Dict[str, Any] = {}
            for c in batch:
                qs.update(questions_for(c))
            resp = asker(fitted["state"], qs)
            ans = resp.get("answers") if isinstance(resp, dict) else None
            if not isinstance(ans, dict):
                raise ValueError("Jev response is missing answers")
            return {c.id: (noul(ans, f"call_{c.id}"), noul(ans, f"result_{c.id}")) for c in batch}

        with ThreadPoolExecutor(max_workers=max(1, min(opts.max_concurrency, len(batches)))) as ex:
            for part in ex.map(ask, batches):
                answers.update(part)
    decisions = [decide(c, *answers.get(c.id, (1.0, 1.0)), opts) for c in calls]
    kept = apply_decisions(messages, decisions, calls, opts.truncate_head_chars)
    count = lambda r: sum(1 for d in decisions if d.reason == r)  # noqa: E731
    return Result(kept, decisions, {
        "messages_before": len(messages), "messages_after": len(kept),
        "chars_before": before, "chars_after": sum(_chars(m) for m in kept),
        "calls": len(calls), "kept": count("kept"), "results_dropped": count("result_dropped"),
        "calls_dropped": count("call_dropped"), "pinned": count("pinned"),
        "state_tokens": fitted["tokens"], "state_stage": fitted["stage"],
        "requests": len(batches), "ms": int((time.time() - started) * 1000),
    })


def deepcopy_messages(messages):
    return copy.deepcopy(list(messages))
