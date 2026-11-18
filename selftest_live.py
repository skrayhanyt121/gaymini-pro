#!/usr/bin/env python3
"""
Live self-test against YOUR running gateway + real Gemini cookies.
    gemini-api start
    python3 selftest_live.py [--model gemini-4-argon] [--url http://127.0.0.1:5000]
Checks: health, plain reply, streaming, real tool call, tool-result loop, multi-line file Write.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:5000"))
ap.add_argument("--model", default="gemini-4-hydrogen")
ap.add_argument("--key", default=os.environ.get("GAYMINI_API_KEY", "dummy-key-gaymini"))
a = ap.parse_args()

TOOLS = [
    {"name": "Bash", "description": "Run a shell command and return its output.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string", "description": "the command"}}, "required": ["command"]}},
    {"name": "Write", "description": "Write a file to disk.",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}},
]
results = []

def post(body, stream=False):
    req = urllib.request.Request(a.url + "/v1/messages", json.dumps(body).encode(),
                                 {"content-type": "application/json", "x-api-key": a.key, "anthropic-version": "2023-06-01"})
    return urllib.request.urlopen(req, timeout=300)

def step(name, fn):
    t = time.time()
    try:
        detail = fn()
        results.append(True); print(f"PASS  {name}  ({time.time()-t:.1f}s)  {detail or ''}")
    except Exception as e:  # noqa: BLE001
        results.append(False); print(f"FAIL  {name}  ({time.time()-t:.1f}s)  {e}")

def health():
    d = json.load(urllib.request.urlopen(a.url + "/health", timeout=10))
    assert d.get("cookies_configured"), "no __Secure-1PSID cookie configured"
    return f"account_status={d.get('account_status')}"

def plain():
    j = json.load(post({"model": a.model, "max_tokens": 200, "messages": [{"role": "user", "content": "Reply with exactly the single word: pong"}]}))
    txt = "".join(b.get("text", "") for b in j["content"])
    assert "pong" in txt.lower(), f"unexpected reply: {txt[:120]!r}"
    return repr(txt[:40])

def streaming():
    r = post({"model": a.model, "max_tokens": 200, "stream": True, "messages": [{"role": "user", "content": "Count from 1 to 5 separated by spaces."}]})
    ev = [l.decode().strip() for l in r if l.strip()]
    assert any(l == "event: message_stop" for l in ev), "stream did not end with message_stop"
    n = sum(1 for l in ev if "text_delta" in l)
    assert n >= 1
    return f"{n} text deltas"

def first_tool_turn(prompt):
    j = json.load(post({"model": a.model, "max_tokens": 2000, "tools": TOOLS, "messages": [{"role": "user", "content": prompt}]}))
    calls = [b for b in j["content"] if b["type"] == "tool_use"]
    assert j["stop_reason"] == "tool_use" and calls, f"no tool call; got {json.dumps(j['content'])[:200]}"
    return j, calls

ctx = {}
def tool_call():
    j, calls = first_tool_turn("Use the Bash tool to run exactly: echo hello-gateway")
    c = calls[0]
    assert c["name"] == "Bash" and "hello-gateway" in c["input"].get("command", ""), c
    ctx["j"], ctx["c"] = j, c
    return json.dumps(c["input"])

def tool_loop():
    j, c = ctx["j"], ctx["c"]
    msgs = [{"role": "user", "content": "Use the Bash tool to run exactly: echo hello-gateway, then tell me what it printed."},
            {"role": "assistant", "content": j["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": c["id"], "content": "hello-gateway"}]}]
    out = json.load(post({"model": a.model, "max_tokens": 500, "tools": TOOLS, "messages": msgs}))
    txt = "".join(b.get("text", "") for b in out["content"])
    assert out["stop_reason"] == "end_turn" and "hello-gateway" in txt, f"{out['stop_reason']} {txt[:150]!r}"
    return repr(txt[:60])

def write_file():
    _, calls = first_tool_turn('Use the Write tool to create /tmp/hello.py containing exactly two lines: print("hi") and print({"a": 1}).')
    c = calls[0]
    assert c["name"] == "Write" and 'print("hi")' in c["input"].get("content", ""), c
    return f"{len(c['input']['content'])} chars"

for n, f in (("health", health), ("plain reply", plain), ("streaming", streaming), ("tool call (Bash)", tool_call),
             ("tool result loop", tool_loop), ("Write multi-line file", write_file)):
    step(n, f)
print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
