# Gaymini pro

> A compact AI API for chat, vision, images, video, music, code, and text tools.

## What it does

- Chat and multi-turn conversations
- Image generation, editing, and vision
- Video and music generation
- Translation, summaries, and rewriting
- Code assistance and deep research
- Model selection through the API and GoatBot commands

## Run locally

```bash
pip install -r requirements.txt
python3 app.py
```

The server runs on `http://localhost:5000`.

## Quick example

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain recursion simply","model":"gemini-3-flash"}'
```

## Documentation

- [API reference](API.md)
- Web docs: `http://localhost:5000/docs`

## Configuration

Add Gemini cookies through environment variables or `gaymini.txt`:

```text
__Secure-1PSID=your-value
__Secure-1PSIDTS=your-value
```

Never commit real cookie values.

## License

Built and maintained by [SIFAT](https://github.com/FX-SIF4T).

## Use with Claude Code (Anthropic-compatible gateway)

```bash
pip install -r requirements.txt
bash setup.sh          # installs gemini-api, cld-argon, cld-hydroide
cld-argon              # Claude Code on the reasoning model
cld-hydroide           # Claude Code on the fast model
```

`gemini_api_server.py` serves `/v1/messages` (streaming + tool use). Tool use is emulated
through a strict prompt protocol (`anthropic_adapter.py`) because the Gemini web chat has no
native function calling. Multiple tool calls per turn, schema validation/repair, one automatic
corrective retry and Anthropic-spec SSE are supported.

Env vars: `GAYMINI_DIR`, `HOST` (default `127.0.0.1`), `PORT` (5000),
`GAYMINI_API_KEY` (require this key; set it if the server is reachable by others),
`GAYMINI_MAX_PROMPT_CHARS` (450000), `GAYMINI_TOOL_RETRIES` (1),
`GAYMINI_MAX_CONCURRENCY` (2, requests hitting your Google account at once), `GAYMINI_QUEUE_TIMEOUT` (120s), `GAYMINI_THREADS` (16).

Offline tests (no network/cookies): `python3 test_adapter.py && python3 test_server_e2e.py && python3 test_live_stream.py`

Live check with your real cookies (gateway running): `python3 selftest_live.py --model gemini-4-argon`

Tool-calling quality: few-shot examples built from your real tools, full JSON-Schema validation (enum/min/max/nested) with auto-repair, one corrective retry that re-shows the schema, loop breaker, `disable_parallel_tool_use`. Counters at `GET /stats`.

Supported: streaming + non-streaming, multi tool_use per turn, tool_choice, stop_sequences, max_tokens,
Anthropic error types (401 auth / 429 rate_limit / 529 overloaded), request-id header, concurrency limiter, waitress server.
Not possible with Gemini web: real token counts (estimated), images, prompt caching, extended-thinking blocks.
