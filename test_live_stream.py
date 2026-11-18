"""Real-socket test: waitress must deliver SSE incrementally, and the concurrency limiter must queue."""
import asyncio, http.client, json, os, threading, time
os.environ["GAYMINI_MAX_CONCURRENCY"] = "1"
import gemini_api_server as srv
from waitress import serve

class Chunk:
    def __init__(self, d, full): self.text_delta = d; self.text = full
class Slow:
    async def generate_content_stream(self, prompt, model=None, temporary=False):
        acc = ""
        for w in ["alpha ", "beta ", "gamma ", "delta"]:
            await asyncio.sleep(0.4); acc += w
            yield Chunk(w, acc)
async def get(): return Slow()
srv.get_gemini_client = get

threading.Thread(target=lambda: serve(srv.app, host="127.0.0.1", port=5611, threads=8, send_bytes=1), daemon=True).start()
time.sleep(1)

def call(results, i):
    t0 = time.time()
    conn = http.client.HTTPConnection("127.0.0.1", 5611, timeout=30)
    conn.request("POST", "/v1/messages", json.dumps({"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}), {"content-type": "application/json"})
    r = conn.getresponse()
    first = None; body = b""
    while True:
        line = r.fp.readline()
        if not line: break
        if first is None and b"text_delta" in line: first = time.time() - t0
        body += line
    results[i] = (first, time.time() - t0, body.decode())

res = {}
call(res, 0)
first, total, body = res[0]
print(f"single: first text delta after {first:.2f}s, finished after {total:.2f}s")
assert first < total - 0.8, "SSE is being buffered instead of streamed"
assert "alpha beta gamma delta" in "".join(json.loads(l[6:])["delta"]["text"] for l in body.split("\n") if l.startswith("data:") and "text_delta" in l)
print("  ok - incremental streaming over a real socket")

res = {}
ts = [threading.Thread(target=call, args=(res, i)) for i in (1, 2)]
t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
span = time.time() - t0
print(f"2 parallel requests with limit=1 took {span:.2f}s (serialised ~3.2s, parallel ~1.6s)")
assert span > 2.8 and all("message_stop" in res[i][2] for i in (1, 2))
print("  ok - limiter queues instead of hammering the Gemini account")
print("LIVE-SOCKET TESTS PASSED")
