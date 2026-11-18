"""End-to-end test of /v1/messages with a FAKE Gemini client (no network, no cookies)."""
import asyncio, json, os, sys, types
os.environ["GAYMINI_API_KEY"] = "secret123"
sys.argv = ["x"]
import gemini_api_server as srv

class Chunk:
    def __init__(self, d, full): self.text_delta = d; self.text = full

SCRIPT = {"replies": [], "prompts": []}
class FakeClient:
    async def generate_content_stream(self, prompt, model=None, temporary=False):
        SCRIPT["prompts"].append(prompt)
        out = SCRIPT["replies"].pop(0)
        acc = ""
        for i in range(0, len(out), 6):
            acc += out[i:i+6]
            yield Chunk(out[i:i+6], acc)
            await asyncio.sleep(0)

async def fake_get():
    return FakeClient()
srv.get_gemini_client = fake_get

c = srv.app.test_client()
H = {"x-api-key": "secret123"}
TOOLS = [{"name": "Bash", "description": "run shell", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
         {"name": "Write", "description": "write file", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}]
ok = 0
def check(n, cond):
    global ok; assert cond, "FAILED: " + n; ok += 1; print("  ok -", n)

# auth
r = c.post("/v1/messages", json={"messages": [{"role": "user", "content": "hi"}]})
check("401 without key (anthropic error shape)", r.status_code == 401 and r.get_json()["type"] == "error" and r.get_json()["error"]["type"] == "authentication_error")
r = c.get("/health"); check("health open", r.status_code == 200)

# validation
r = c.post("/v1/messages", json={"model": "m"}, headers=H)
check("400 invalid_request_error", r.status_code == 400 and r.get_json()["error"]["type"] == "invalid_request_error")

# non-stream plain
SCRIPT["replies"] = ["Hello from fake gemini"]
r = c.post("/v1/messages", json={"model": "claude-sonnet-4", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}, headers=H)
j = r.get_json()
check("non-stream text", r.status_code == 200 and j["content"][0]["text"] == "Hello from fake gemini" and j["stop_reason"] == "end_turn" and j["model"] == "claude-sonnet-4")

# stream with tool use (Write containing awkward content)
code = 'print("hi")\n```\n</tool_call>\n}'
reply = "I'll create it.\n<tool_call>\n" + json.dumps({"name": "Write", "input": {"file_path": "a.py", "content": code}}) + "\n</tool_call>\n<tool_call>{\"name\":\"Bash\",\"input\":{\"command\":\"python a.py\"}}</tool_call>"
SCRIPT["replies"] = [reply]
r = c.post("/v1/messages", json={"model": "claude-opus-5-5", "stream": True, "system": [{"type": "text", "text": "You are Claude Code"}], "tools": TOOLS,
                                 "messages": [{"role": "user", "content": "create a.py and run it"}]}, headers=H)
check("SSE content-type", r.mimetype == "text/event-stream")
raw = r.get_data(as_text=True)
frames = [f for f in raw.split("\n\n") if f.strip() and not f.startswith(":")]
evs = [(f.split("\n")[0][7:], json.loads(f.split("\n")[1][6:])) for f in frames]
names = [e[0] for e in evs]
check("SSE starts/ends right", names[0] == "message_start" and names[-1] == "message_stop" and names[-2] == "message_delta")
tool_blocks = {}
for n, d in evs:
    if n == "content_block_start" and d["content_block"]["type"] == "tool_use": tool_blocks[d["index"]] = {"name": d["content_block"]["name"], "json": ""}
    if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta": tool_blocks[d["index"]]["json"] += d["delta"]["partial_json"]
parsed = [(v["name"], json.loads(v["json"])) for _, v in sorted(tool_blocks.items())]
check("two tool_use blocks, content intact", parsed == [("Write", {"file_path": "a.py", "content": code}), ("Bash", {"command": "python a.py"})])
check("stop_reason tool_use", [d for n, d in evs if n == "message_delta"][0]["delta"]["stop_reason"] == "tool_use")
check("prompt carried real tool schema + system", "Input schema" in SCRIPT["prompts"][-1] and "You are Claude Code" in SCRIPT["prompts"][-1])

# agent loop: tool_result goes back in the next request
SCRIPT["replies"] = ["Done: file created and executed."]
msgs = [{"role": "user", "content": "create a.py and run it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_A", "name": "Bash", "input": {"command": "python a.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_A", "content": "hi", "is_error": False}]}]
r = c.post("/v1/messages", json={"model": "m", "tools": TOOLS, "messages": msgs}, headers=H)
check("follow-up turn sees tool result", r.status_code == 200 and 'name="Bash" status="ok"' in SCRIPT["prompts"][-1] and r.get_json()["stop_reason"] == "end_turn")

# transient error before output is retried
class Flaky(FakeClient):
    n = 0
    async def generate_content_stream(self, prompt, model=None, temporary=False):
        Flaky.n += 1
        if Flaky.n == 1: raise RuntimeError("1013 temporary")
        async for ch in FakeClient.generate_content_stream(self, prompt, model, temporary): yield ch
async def flaky_get(): return Flaky()
srv.get_gemini_client = flaky_get
SCRIPT["replies"] = ["recovered"]
async def nosleep(*a, **k): return None
srv.asyncio.sleep = nosleep  # speed up backoff in test
r = c.post("/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "x"}]}, headers=H)
check("transient failure retried", r.status_code == 200 and r.get_json()["content"][0]["text"] == "recovered")

# hard failure -> anthropic error
class Dead(FakeClient):
    async def generate_content_stream(self, *a, **k):
        raise RuntimeError("cookie expired")
        yield
async def dead_get(): return Dead()
srv.get_gemini_client = dead_get
r = c.post("/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "x"}]}, headers=H)
check("cookie failure -> 401 authentication_error", r.status_code == 401 and r.get_json()["error"]["type"] == "authentication_error")
class Boom(FakeClient):
    async def generate_content_stream(self, *a, **k):
        raise RuntimeError("something odd")
        yield
async def boom_get(): return Boom()
srv.get_gemini_client = boom_get
r = c.post("/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "x"}]}, headers=H)
check("unknown failure -> 500 api_error", r.status_code == 500 and r.get_json()["error"]["type"] == "api_error")
class Busy(FakeClient):
    async def generate_content_stream(self, *a, **k):
        raise RuntimeError("Gemini encountered a temporary error (1013)")
        yield
async def busy_get(): return Busy()
srv.get_gemini_client = busy_get
r = c.post("/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "x"}]}, headers=H)
check("1013 -> 529 overloaded_error", r.status_code == 529 and r.get_json()["error"]["type"] == "overloaded_error")
srv.get_gemini_client = dead_get
r = c.post("/v1/messages", json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "x"}]}, headers=H)
check("hard failure in stream -> error event", '"type": "error"' in r.get_data(as_text=True) and "cookie expired" in r.get_data(as_text=True))

# stop_sequences + max_tokens
srv.get_gemini_client = fake_get
SCRIPT["replies"] = ["one two STOP three four"]
r = c.post("/v1/messages", json={"model": "m", "stop_sequences": ["STOP"], "messages": [{"role": "user", "content": "x"}]}, headers=H)
j = r.get_json()
check("stop_sequences honoured", j["content"][0]["text"] == "one two" and j["stop_reason"] == "stop_sequence" and j["stop_sequence"] == "STOP")
SCRIPT["replies"] = ["abcdefghij" * 20]
r = c.post("/v1/messages", json={"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "x"}]}, headers=H)
j = r.get_json()
check("max_tokens honoured", j["stop_reason"] == "max_tokens" and len(j["content"][0]["text"]) <= 20)
check("headers + cache usage", "request-id" in r.headers and r.headers["anthropic-version"] and j["usage"]["cache_read_input_tokens"] == 0)
srv.get_gemini_client = dead_get

# count_tokens + models
r = c.post("/v1/messages/count_tokens", json={"messages": [{"role": "user", "content": "hello world" * 100}], "tools": TOOLS}, headers=H)
check("count_tokens", r.status_code == 200 and r.get_json()["input_tokens"] > 200)
r = c.get("/v1/models", headers=H); d = r.get_json()
check("models anthropic fields", d["data"][0]["type"] == "model" and "created_at" in d["data"][0])
print(f"\nALL {ok} E2E CHECKS PASSED")
