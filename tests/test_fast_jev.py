"""Offline tests for fast_jev (fake Jev, no network).

Run from the repo root:
    ~/.hermes/hermes-agent/venv/bin/python tests/test_fast_jev.py
Set HERMES_AGENT_DIR to use another Hermes checkout.
"""
import importlib.util
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERMES = os.environ.get("HERMES_AGENT_DIR", os.path.expanduser("~/.hermes/hermes-agent"))
sys.path.insert(0, HERMES)

# Import the repo root as package "fast_jev" regardless of the checkout's dir name.
_spec = importlib.util.spec_from_file_location(
    "fast_jev", os.path.join(REPO, "__init__.py"), submodule_search_locations=[REPO])
fast_jev = importlib.util.module_from_spec(_spec)
sys.modules["fast_jev"] = fast_jev
_spec.loader.exec_module(fast_jev)
core = fast_jev.core


def conv():
    m = [{"role": "system", "content": "sys"},
         {"role": "user", "content": "Fix the failing test. Never edit src/generated."}]
    for i in range(1, 9):
        m.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "read_file", "arguments": json.dumps({"path": f"src/f{i}.py"})}}]})
        m.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"file {i}\n" + "x" * 3000})
    m.append({"role": "assistant", "content": "Done reading."})
    m.append({"role": "user", "content": "now run tests"})
    return m


def fake(policy):
    calls = []

    def ask(state, questions):
        calls.append((state, questions))
        ans = {}
        for name in questions:
            kind, tid = name.split("_", 1)
            ans[name] = {"noul": policy(kind, int(tid[1:]))}
        return {"answers": ans}
    ask.calls = calls
    return ask


def check_pairs(msgs):
    ids = {tc["id"] for m in msgs if m["role"] == "assistant" for tc in m.get("tool_calls") or []}
    res = {m["tool_call_id"] for m in msgs if m["role"] == "tool"}
    assert ids == res, (ids, res)


def test_decisions():
    # t1,t2: drop call; t3,t4: keep call drop result; rest keep
    def policy(kind, n):
        if n <= 2:
            return 0.1
        if n <= 4:
            return 0.9 if kind == "call" else 0.2
        return 0.9
    ask = fake(policy)
    msgs = conv()
    r = core.compact(msgs, ask, core.Options(preserve_recent_messages=4))
    check_pairs(r.messages)
    s = r.stats
    assert s["calls_dropped"] == 2 and s["results_dropped"] == 2, s
    ids = [tc["id"] for m in r.messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
    assert "c1" not in ids and "c2" not in ids and "c3" in ids
    t3 = next(m for m in r.messages if m.get("tool_call_id") == "c3")
    assert "fast-jev-compaction truncated" in t3["content"] and t3["content"].startswith("file 3")
    t5 = next(m for m in r.messages if m.get("tool_call_id") == "c5")
    assert t5 is msgs[msgs.index(t5)]  # untouched object identity
    # user/assistant text verbatim
    assert [m["content"] for m in r.messages if m["role"] == "user"] == \
        [m["content"] for m in msgs if m["role"] == "user"]
    # state has no tool outputs
    state = ask.calls[0][0]
    assert "xxxx" not in json.dumps(state)
    assert core.reduction_ratio(r) > 0.3
    print("decisions ok", s)


def test_pinned_recent():
    ask = fake(lambda k, n: 0.0)
    r = core.compact(conv(), ask, core.Options(preserve_recent_messages=6))
    # last 6 messages include c8 call+result, c7 result? -> c8 pinned at least
    ids = [tc["id"] for m in r.messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
    assert "c8" in ids
    check_pairs(r.messages)
    print("pinned ok", r.stats)


def test_fit_state_shrinks():
    msgs = conv()
    msgs[1]["content"] = "long " * 20000
    calls = core.collect_tool_calls(msgs, core.Options())
    f = core.fit_state(msgs, calls, core.Options(max_state_tokens=2000))
    assert f["tokens"] <= 2000, f["tokens"]
    print("fit ok", f["stage"], f["tokens"])


def test_batching():
    ask = fake(lambda k, n: 0.9)
    r = core.compact(conv(), ask, core.Options(preserve_recent_messages=2, max_request_tokens=1500,
                                               max_state_tokens=1000))
    assert r.stats["requests"] > 1, r.stats
    print("batch ok", r.stats["requests"])


def test_bad_answer():
    try:
        core.compact(conv(), lambda s, q: {"answers": {}}, core.Options(preserve_recent_messages=2))
    except ValueError as e:
        print("bad answer raises ok:", e)
        return
    raise AssertionError("should raise")


def test_engine_fallback():
    os.environ.pop("TYPESAFE_API_KEY", None)
    saved_key = fast_jev._read_env_key
    fast_jev._read_env_key = lambda: ""
    eng = fast_jev.FastJevEngine(model="test-model")
    assert eng.name == "fast_jev"
    called = {}

    def fake_builtin(self, msgs, **kw):
        called["n"] = len(msgs)
        return msgs
    from agent.context_compressor import ContextCompressor
    saved = ContextCompressor.compress
    ContextCompressor.compress = fake_builtin
    try:
        out = eng.compress(conv())
    finally:
        ContextCompressor.compress = saved
        fast_jev._read_env_key = saved_key
    assert called and "TYPESAFE_API_KEY" in eng.last_jev_stats.get("error", ""), eng.last_jev_stats
    print("engine fallback ok", eng.last_jev_stats, len(out))


def test_engine_prune():
    eng = fast_jev.FastJevEngine(model="test-model")
    eng.jev_cfg["preserve_recent_messages"] = 4
    eng.threshold_tokens = 10**9
    fake_ask = fake(lambda k, n: 0.1 if n <= 6 else 0.9)
    eng.jev_prune = lambda msgs, ft=None: core.compact(msgs, fake_ask, eng._options(ft))
    out = eng.compress(conv())
    check_pairs(out)
    # head = system + protect_first_n(3) → c1 pinned; c2..c6 dropped
    assert eng.compression_count == 1 and eng.last_jev_stats["calls_dropped"] == 5, eng.last_jev_stats
    print("engine prune ok", eng.last_jev_stats)


def test_register():
    class Ctx:
        engine = None

        def register_context_engine(self, e):
            self.engine = e
    ctx = Ctx()
    fast_jev.register(ctx)
    assert ctx.engine is not None and ctx.engine.name == "fast_jev"
    print("register ok", type(ctx.engine).__name__)


if __name__ == "__main__":
    test_decisions()
    test_pinned_recent()
    test_fit_state_shrinks()
    test_batching()
    test_bad_answer()
    test_engine_fallback()
    test_engine_prune()
    test_register()
    print("ALL OK")
