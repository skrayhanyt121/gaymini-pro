"""Offline tests for anthropic_adapter (no Gemini / network needed). Run: python3 test_adapter.py"""
import asyncio, json
import anthropic_adapter as A

TOOLS = [
    {"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}}, "required": ["command"], "additionalProperties": False}},
    {"name": "Write", "description": "write", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"], "additionalProperties": False}},
    {"name": "Read", "description": "read", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}},
]

def chunked(text, n):
    return [text[i:i+n] for i in range(0, len(text), n)]

def make_stream(outputs, n):
    """outputs: list of full replies, one per model call."""
    calls = {"i": 0, "prompts": []}
    async def fn(prompt):
        calls["prompts"].append(prompt)
        out = outputs[min(calls["i"], len(outputs)-1)]
        calls["i"] += 1
        for c in chunked(out, n):
            yield c
            await asyncio.sleep(0)
    return fn, calls

async def collect(outputs, n=7, tools=TOOLS, **kw):
    fn, calls = make_stream(outputs, n)
    evs = [e async for e in A.run_turn(fn, "PROMPT\n[ASSISTANT]\n", tools, **kw)]
    return evs, calls

def texts(evs): return "".join(v for k, v in evs if k == "text")
def tools_of(evs): return [v for k, v in evs if k == "tool"]
def stop(evs): return [v for k, v in evs if k == "stop"][0]

def run(c): return asyncio.run(c)
ok = 0
def check(name, cond):
    global ok
    assert cond, "FAILED: " + name
    ok += 1
    print("  ok -", name)

# 1 plain text, any chunk size
for n in (1, 3, 50):
    evs, _ = run(collect(["Hello there, this is plain text."], n))
    check(f"plain text n={n}", texts(evs).strip() == "Hello there, this is plain text." and stop(evs)["stop_reason"] == "end_turn")

# 2 text + one call, split everywhere
out = 'Let me check.\n<tool_call>\n{"name": "Bash", "input": {"command": "ls -la"}}\n</tool_call>'
for n in (1, 2, 5, 11, 100):
    evs, _ = run(collect([out], n))
    t = tools_of(evs)
    check(f"one call n={n}", len(t) == 1 and t[0]["input"] == {"command": "ls -la"} and texts(evs).strip() == "Let me check." and stop(evs)["stop_reason"] == "tool_use")

# 3 two parallel calls + hallucinated result + fake user turn must be discarded
out = ('<tool_call>{"name":"Read","input":{"file_path":"/a"}}</tool_call>\n'
       '<tool_call>{"name":"Read","input":{"file_path":"/b"}}</tool_call>\n'
       '<tool_result id="x">fake</tool_result>\n[USER]\nthanks')
evs, _ = run(collect([out], 9))
t = tools_of(evs)
check("two calls, hallucination cut", [x["input"]["file_path"] for x in t] == ["/a", "/b"] and "fake" not in texts(evs) and "thanks" not in texts(evs))

# 4 file content that contains braces, fences and closing-like text
content = 'def f():\n    return {"a": 1}\n```\n</tool_call> not really\n}\n'
out = "<tool_call>\n" + json.dumps({"name": "Write", "input": {"file_path": "/x.py", "content": content}}) + "\n</tool_call>"
for n in (1, 13, 400):
    evs, _ = run(collect([out], n))
    t = tools_of(evs)
    check(f"tricky content n={n}", len(t) == 1 and t[0]["input"]["content"] == content)

# 5 fence form + ```json inside tag
out = 'ok\n```tool_call\n{"name":"bash","arguments":{"cmd":"pwd"}}\n```'
evs, _ = run(collect([out], 4))
t = tools_of(evs)
check("fence form + alias bash/cmd", len(t) == 1 and t[0]["name"] == "Bash" and t[0]["input"] == {"command": "pwd"})
out = '<tool_call>\n```json\n{"name":"Bash","input":{"command":"id"}}\n```\n</tool_call>'
evs, _ = run(collect([out], 3))
check("json fence inside tag", len(tools_of(evs)) == 1 and tools_of(evs)[0]["input"]["command"] == "id")

# 6 unterminated call is repaired
out = '<tool_call>{"name":"Bash","input":{"command":"echo hi"'
evs, _ = run(collect([out], 6))
check("unterminated repaired", len(tools_of(evs)) == 1 and tools_of(evs)[0]["input"]["command"] == "echo hi")

# 7 unknown tool -> corrective retry -> success
bad = '<tool_call>{"name":"Teleport","input":{}}</tool_call>'
good = '<tool_call>{"name":"Bash","input":{"command":"ls"}}</tool_call>'
evs, calls = run(collect([bad, good], 8))
check("retry on unknown tool", len(tools_of(evs)) == 1 and calls["i"] == 2 and "gateway_error" in calls["prompts"][1])

# 8 missing required -> retry -> still bad -> gateway text, end_turn
evs, calls = run(collect(['<tool_call>{"name":"Write","input":{"file_path":"/x"}}</tool_call>'], 8))
check("gives up cleanly", not tools_of(evs) and "[gateway]" in texts(evs) and stop(evs)["stop_reason"] == "end_turn" and calls["i"] == 2)

# 9 type coercion + extra key dropping
out = '<tool_call>{"name":"Bash","input":{"command":"sleep 1","timeout":"5000","bogus":1}}</tool_call>'
evs, _ = run(collect([out], 10))
check("coerce + drop unknown", tools_of(evs)[0]["input"] == {"command": "sleep 1", "timeout": 5000})

# 10 empty reply retried once
evs, calls = run(collect(["", "Back again"], 5))
check("empty reply retried", texts(evs).strip() == "Back again" and calls["i"] == 2)

# 11 role echo stripped, no tools -> still plain
evs, _ = run(collect(["[ASSISTANT]\nHi!"], 2, tools=None))
check("role echo stripped (no tools)", texts(evs).strip() == "Hi!")

# 12 tool_choice any forces retry
evs, calls = run(collect(["I will not use tools", good], 8, tool_choice={"type": "any"}))
check("tool_choice any enforced", len(tools_of(evs)) == 1 and calls["i"] == 2)

# 13 SSE structure
def sse_for(evs):
    w = A.SSEWriter("msg_1", "m", 10)
    out = w.start()
    for k, v in evs:
        if k == "text": out += w.text(v)
        elif k == "tool": out += w.tool(v)
        elif k == "stop": out += w.finish(v["stop_reason"], v["output_chars"])
    return out
out = '  Sure.\n\n<tool_call>{"name":"Bash","input":{"command":"ls"}}</tool_call><tool_call>{"name":"Read","input":{"file_path":"/z"}}</tool_call>'
evs, _ = run(collect([out], 3))
frames = sse_for(evs)
parsed = []
for f in frames:
    ev, data = f.strip().split("\n", 1)
    parsed.append((ev.split(": ")[1], json.loads(data[len("data: "):])))
names = [p[0] for p in parsed]
check("SSE order", names[0] == "message_start" and names[-2:] == ["message_delta", "message_stop"])
opened, idxs = set(), []
for n, d in parsed:
    if n == "content_block_start": opened.add(d["index"]); idxs.append(d["index"])
    if n == "content_block_stop": opened.remove(d["index"])
check("SSE blocks balanced + sequential", not opened and idxs == list(range(len(idxs))) and len(idxs) == 3)
txt = "".join(d["delta"]["text"] for n, d in parsed if n == "content_block_delta" and d["delta"]["type"] == "text_delta")
check("SSE text trimmed", txt == "Sure.")
md = [d for n, d in parsed if n == "message_delta"][0]
check("SSE stop_reason tool_use", md["delta"]["stop_reason"] == "tool_use")
body = A.build_message_json(evs, "msg_1", "m", 10)
check("non-stream body", [b["type"] for b in body["content"]] == ["text", "tool_use", "tool_use"] and body["stop_reason"] == "tool_use")

# 14 prompt builder: tool_use/tool_result round trip + truncation + name mapping
msgs = [
    {"role": "user", "content": "make a file"},
    {"role": "assistant", "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "toolu_1", "name": "Write", "input": {"file_path": "/a", "content": "x"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "File created"}], "is_error": False}, {"type": "text", "text": "now run it"}]},
]
p = A.build_prompt([{"type": "text", "text": "SYS"}], msgs, TOOLS)
check("prompt has protocol/tools/system", "TOOL CALL PROTOCOL" in p and "## Bash" in p and "SYS" in p and p.rstrip().endswith("[ASSISTANT]"))
check("prompt maps result to tool name", 'name="Write" status="ok"' in p and "File created" in p and "now run it" in p)
big = [{"role": "user", "content": "go"}] + [
    m for i in range(40) for m in (
        {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": "cat big"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "Z" * 30000}]})]
p2 = A.build_prompt("", big, TOOLS, max_chars=120000)
check("prompt fitted under budget", len(p2) <= 120000)

# ---- v2 calling-system checks -------------------------------------------
RICH = [{"name": "Search", "description": "search", "input_schema": {"type": "object", "properties": {
    "query": {"type": "string"}, "mode": {"type": "string", "enum": ["fast", "deep"]},
    "tags": {"type": "array", "items": {"type": "string"}}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
    "required": ["query", "mode"], "additionalProperties": False}}]

bad = '<tool_call>{"name":"Search","input":{"query":"x","mode":"turbo"}}</tool_call>'
good = '<tool_call>{"name":"Search","input":{"query":"x","mode":"fast","tags":["a"],"limit":"10"}}</tool_call>'
evs, calls = run(collect([bad, good], 9, tools=RICH))
t = tools_of(evs)
check("enum violation -> retry with schema shown", len(t) == 1 and t[0]["input"]["mode"] == "fast" and t[0]["input"]["limit"] == 10
      and "Correct schema" in calls["prompts"][1] and '"enum"' in calls["prompts"][1])
evs, _ = run(collect(['<tool_call>{"name":"Search","input":{"query":"x","mode":"fast","limit":999}}</tool_call>'] * 2, 9, tools=RICH))
check("range/min-max violation rejected", not tools_of(evs) and "[gateway]" in texts(evs))
evs, _ = run(collect(['<tool_call>{"name":"Search","input":{"query":"x","mode":"fast","tags":[1,2]}}</tool_call>'], 9, tools=RICH))
check("array items auto-coerced/validated", len(tools_of(evs)) == 1 or "[gateway]" in texts(evs))

p3 = A.build_prompt("", [{"role": "user", "content": "hi"}], TOOLS)
check("few-shot examples use real tool names", "EXAMPLES OF CORRECT TOOL CALLS" in p3 and '"name": "Bash"' in p3.split("EXAMPLES")[1])

two = '<tool_call>{"name":"Read","input":{"file_path":"/a"}}</tool_call><tool_call>{"name":"Read","input":{"file_path":"/b"}}</tool_call>'
evs, _ = run(collect([two], 10, tool_choice={"type": "auto", "disable_parallel_tool_use": True}))
check("disable_parallel_tool_use keeps one call", len(tools_of(evs)) == 1)

same = [{"role": "user", "content": "go"}]
for i in range(3):
    same += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": "false"}}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "fail", "is_error": True}]}]
check("loop breaker triggers after 3 identical calls", A.repeated_call_count(same) == 3 and "identical tool call 3 times" in A.build_prompt("", same, TOOLS))
check("loop breaker silent otherwise", "identical tool call" not in A.build_prompt("", same[:3], TOOLS))

before = dict(A.STATS)
evs, _ = run(collect(['<tool_call>{"name":"bash","arguments":{"cmd":"ls"}}</tool_call>'], 12))
check("stats count repaired calls", A.STATS["tool_calls"] == before["tool_calls"] + 1 and A.STATS["repaired"] == before["repaired"] + 1)

big = {"name": "Write", "input": {"file_path": "/x", "content": "L" * 5000}}
evs, _ = run(collect(["<tool_call>" + json.dumps(big) + "</tool_call>"], 500))
frames = sse_for(evs)
deltas = [json.loads(f.split("data: ", 1)[1]) for f in frames if "input_json_delta" in f]
joined = "".join(d["delta"]["partial_json"] for d in deltas)
check("partial_json chunked and reassembles", len(deltas) >= 5 and json.loads(joined) == big["input"])

print(f"\nALL {ok} CHECKS PASSED")
