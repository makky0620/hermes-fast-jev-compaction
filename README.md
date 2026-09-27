# hermes-fast-jev-compaction

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) context engine
that compacts long conversations **without summarizing**.

Python port of [tamaratran/fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction)
(originally a Claude Code plugin). When the context fills up, every old tool
call and tool result is scored by TypeSafe's **Jev** in a few fast requests;
stale ones are dropped or truncated, and everything kept stays verbatim.
User and assistant text is never rewritten.

## Why

Summaries are lossy: a file path, exact error, constraint or command can
vanish even when it matters later. This engine only deletes tool calls and
results Jev judges no longer needed. The assistant can always re-run a tool.

## Install

```bash
hermes plugins install makky0620/hermes-fast-jev-compaction
```

The installer asks for `TYPESAFE_API_KEY` and saves it to `~/.hermes/.env`
(or add `TYPESAFE_API_KEY=...` there yourself). Then select the engine and
restart Hermes:

```bash
hermes config set context.engine fast_jev
```

Revert to the built-in summarizer:

```bash
hermes config set context.engine compressor
```

Manual install: clone this repo into `~/.hermes/plugins/fast_jev/`.

## How it works

1. Every assistant `tool_calls` entry is paired with its `role: tool` result.
   Calls in the head (system prompt + `compression.protect_first_n`) or in the
   newest `preserve_recent_messages` messages are pinned and never touched.
2. The state sent to Jev is the whole conversation, oldest first, with each
   tool output replaced by a short note (`ok, 4213 chars (omitted)`). It is
   shrunk in stages to fit `max_state_tokens` (truncate tool inputs, abridge
   long texts, collapse old messages, one-line old calls, ...).
3. For every non-pinned call Jev answers two `noul` questions: should the
   **call** stay, and should the **result** stay verbatim. Questions are
   batched under `max_request_tokens` and sent concurrently.
4. Per call, against `keep_threshold`:
   - keep result -> call and result stay as is;
   - else keep call -> result truncated to its first `truncate_head_chars`
     characters plus a note;
   - else -> call and result are removed together.
5. Tool call / result pairs are re-validated so no orphan is ever sent.

Compaction triggers at the same point as the built-in compressor
(`compression.threshold`). It falls back to the built-in summary when:

- Jev fails or `TYPESAFE_API_KEY` is missing,
- the reduction is below `min_reduction_ratio`,
- the pruned history is still above `summarize_if_above` x threshold
  (then the pruned history is summarized, which still loses less).

## Privacy

Conversation text and tool **inputs** are sent to `api.typesafe.ai`.
Tool **outputs** are not sent (only their length).

## Configuration

All optional, in `~/.hermes/config.yaml`:

```yaml
context:
  engine: fast_jev
  fast_jev:
    keep_threshold: 0.5            # min keep probability
    preserve_recent_messages: 10   # newest messages never touched
    truncate_head_chars: 300       # head kept when a result is dropped
    min_reduction_ratio: 0.25      # below this, use the built-in summary
    summarize_if_above: 0.85       # still above this x threshold -> also summarize
    max_state_tokens: 25000
    max_request_tokens: 30000      # Jev request limit is 32k
    timeout_seconds: 60
    model: jev-latest
    base_url: https://api.typesafe.ai/v1/systemone
```

`hermes status`-style engine status includes the last run's stats under
`fast_jev` (calls kept / truncated / dropped, reduction ratio, requests, ms).

## Development

Tests use a fake Jev and never touch the network. They need a Hermes
checkout for the `ContextCompressor` base class:

```bash
~/.hermes/hermes-agent/venv/bin/python tests/test_fast_jev.py
# or point at another checkout
HERMES_AGENT_DIR=/path/to/hermes-agent python tests/test_fast_jev.py
```

## Limitations

- Only tool calls/results are candidates; text is never removed.
- Token sizes are character-based estimates, not a tokenizer.
- A keep probability is not a proof a result is safe to delete.
- Depends on Hermes' `ContextCompressor` internals for the fallback path;
  a large Hermes refactor may require an update.

## Credits / License

MIT. Algorithm, prompts and token estimator from
[tamaratran/fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction) (MIT).
