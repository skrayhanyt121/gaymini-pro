"""
Anthropic Messages API (/v1/messages) & Agentic Web Compatibility Proxy for Gaymini Pro.
Translates Claude Code requests into Gemini Web API calls with full tool-calling
support, SSE streaming, local terminal/file execution, and temporary/incognito mode (no chat history saved).
"""

import asyncio
import json
import queue
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from flask import Blueprint, Response, jsonify, render_template, request, stream_with_context
from gemini_webapi import GeminiClient
from gemini_webapi.constants import Model
from gemini_webapi.utils.rotate_1psidts import _get_cookie_cache_dir, rotate_1psidts
from curl_cffi.requests import AsyncSession

anthropic_bp = Blueprint("anthropic_proxy", __name__)

_ASYNC_LOOP: Optional[asyncio.AbstractEventLoop] = None
_LOOP_THREAD: Optional[threading.Thread] = None
_LOOP_INIT_LOCK = threading.Lock()

_ACTIVE_CLIENT: Optional[GeminiClient] = None
_LAST_ROTATION_TIME = 0.0

TOOL_CALL_REGEX = re.compile(
    r"```(?:tool_call|json_tool|tool)\s*(\{[\s\S]*?\})\s*```|<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>",
    re.MULTILINE | re.DOTALL,
)

LOCAL_TOOLS = [
    {
        "name": "Bash",
        "description": "Execute any Linux bash shell command (e.g. ls, git, npm, python, grep, curl).",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command line string to run"}
            },
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read the contents of a file on disk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute file path to read"}
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": "Create a new file or overwrite an existing file with the provided content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute file path to write to"},
                "content": {"type": "string", "description": "Full text content of the file"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": "Edit an existing file by replacing old_string with new_string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute file path to modify"},
                "old_string": {"type": "string", "description": "Exact text string to replace"},
                "new_string": {"type": "string", "description": "Replacement text string"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
]


def normalize_tool_call(name: str, args: dict) -> tuple[str, dict]:
    """Map any LLM tool name variant to Claude Code's native tool names (Write, Read, Edit, Bash)."""
    norm_name = (name or "").strip()
    norm_lower = norm_name.lower()

    # 1. Write tool mapping
    if norm_lower in ("filewrite", "write", "createfile", "writefile", "create"):
        tool_name = "Write"
        file_path = args.get("file_path") or args.get("path") or ""
        content = args.get("content", "")
        return tool_name, {"file_path": file_path, "content": content}

    # 2. Read tool mapping
    elif norm_lower in ("fileview", "view", "read", "readfile", "fileread", "cat"):
        tool_name = "Read"
        file_path = args.get("file_path") or args.get("path") or ""
        res_args = {"file_path": file_path}
        if "offset" in args:
            res_args["offset"] = args["offset"]
        if "limit" in args:
            res_args["limit"] = args["limit"]
        return tool_name, res_args

    # 3. Edit tool mapping
    elif norm_lower in ("fileedit", "edit", "replace", "strreplace", "modify"):
        tool_name = "Edit"
        file_path = args.get("file_path") or args.get("path") or ""
        old_str = args.get("old_string") or args.get("old_str") or ""
        new_str = args.get("new_string") or args.get("new_str") or ""
        return tool_name, {"file_path": file_path, "old_string": old_str, "new_string": new_str}

    # 4. Bash tool mapping
    elif norm_lower in ("bash", "command", "terminal", "sh", "shell", "run"):
        tool_name = "Bash"
        cmd = args.get("command") or args.get("cmd") or ""
        return tool_name, {"command": cmd}

    return norm_name, args


def execute_local_tool(name: str, args: dict) -> str:
    """Execute real local tools on the Linux machine."""
    try:
        norm_name = name.lower()
        if "bash" in norm_name or "command" in norm_name or "terminal" in norm_name:
            cmd = args.get("command", "")
            if not cmd:
                return "Error: Empty command"
            res = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd="/root",
            )
            out = res.stdout
            if res.stderr:
                out = (out + "\n" + res.stderr) if out else res.stderr
            return out.strip() or "[Command executed successfully with no output]"

        elif "view" in norm_name or "read" in norm_name:
            path_str = args.get("path") or args.get("file_path", "")
            p = Path(path_str)
            if not p.is_absolute():
                p = Path("/root") / p
            if not p.exists():
                return f"Error: File '{p}' does not exist."
            return p.read_text(encoding="utf-8", errors="replace")[:30000]

        elif "write" in norm_name or "create" in norm_name:
            path_str = args.get("path") or args.get("file_path", "")
            p = Path(path_str)
            if not p.is_absolute():
                p = Path("/root") / p
            content = args.get("content", "")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to {p}."

        elif "edit" in norm_name or "modify" in norm_name:
            path_str = args.get("path") or args.get("file_path", "")
            p = Path(path_str)
            if not p.is_absolute():
                p = Path("/root") / p
            if not p.exists():
                return f"Error: File '{p}' does not exist."
            content = p.read_text(encoding="utf-8")
            old_str = args.get("old_str") or args.get("old_string", "")
            new_str = args.get("new_str") or args.get("new_string", "")
            if old_str not in content:
                return f"Error: Target text not found in {p}."
            updated = content.replace(old_str, new_str, 1)
            p.write_text(updated, encoding="utf-8")
            return f"Successfully updated {p}."

        return f"Unknown tool: {name}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as e:
        return f"Tool execution failed: {str(e)}"


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    global _ASYNC_LOOP, _LOOP_THREAD
    with _LOOP_INIT_LOCK:
        if _ASYNC_LOOP is None or not _ASYNC_LOOP.is_running():
            _ASYNC_LOOP = asyncio.new_event_loop()

            def _target():
                asyncio.set_event_loop(_ASYNC_LOOP)
                _ASYNC_LOOP.run_forever()

            _LOOP_THREAD = threading.Thread(target=_target, daemon=True, name="GeminiAsyncWorker")
            _LOOP_THREAD.start()
    return _ASYNC_LOOP


def _run_coroutine(coro):
    loop = _ensure_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=300)


def _load_cookie_dict(gaymini_txt_path: str) -> dict:
    path = Path(gaymini_txt_path)
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8").strip()
    cookies = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ";" in line:
            for part in line.split(";"):
                part = part.strip()
                if "=" in part:
                    k, _, v = part.partition("=")
                    cookies[k.strip()] = v.strip()
        elif "=" in line:
            k, _, v = line.partition("=")
            cookies[k.strip()] = v.strip()
    return cookies


async def get_or_create_client(gaymini_txt_path: str = "/root/gaymini-claude/gaymini.txt") -> GeminiClient:
    global _ACTIVE_CLIENT, _LAST_ROTATION_TIME

    now = time.time()
    if _ACTIVE_CLIENT is not None:
        return _ACTIVE_CLIENT

    cookies = _load_cookie_dict(gaymini_txt_path)
    psid = cookies.get("__Secure-1PSID")
    if not psid:
        raise RuntimeError("Missing __Secure-1PSID in gaymini.txt")

    session = AsyncSession(impersonate="chrome", curl_options={113: 1})
    for k, v in cookies.items():
        session.cookies.set(k, v, domain=".google.com")

    psidts = None
    cache_file = _get_cookie_cache_dir() / f".cached_cookies_{psid}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            psidts = next((c["value"] for c in cached if c["name"] == "__Secure-1PSIDTS"), None)
        except Exception:
            pass

    if not psidts or (now - _LAST_ROTATION_TIME > 1800):
        try:
            rotated = await rotate_1psidts(session, verbose=False)
            if rotated:
                psidts = rotated
                _LAST_ROTATION_TIME = now
        except Exception as e:
            if not psidts:
                raise RuntimeError(f"Failed to rotate/fetch __Secure-1PSIDTS: {e}")

    extra = {k: v for k, v in cookies.items() if k not in {"__Secure-1PSID", "__Secure-1PSIDTS"}}
    client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts, cookies=extra or None)
    await client.init(timeout=600, watchdog_timeout=300, auto_refresh=True)
    _ACTIVE_CLIENT = client
    return _ACTIVE_CLIENT


def _format_tools_prompt(tools: List[Dict[str, Any]]) -> str:
    if not tools:
        return ""
    lines = [
        "\n\n# AVAILABLE TOOLS & CAPABILITIES",
        "You are Claude Code running directly on the user's Linux host. You HAVE FULL ACCESS to execute bash commands, view files, write files, and edit code.",
        "When the user asks you to run a command, list files, view or edit code, you MUST invoke a tool call block instead of refusing.\n",
    ]
    for t in tools:
        name = t.get("name", "")
        desc = t.get("description", "")
        schema = t.get("input_schema", {})
        lines.append(f"### Tool: `{name}`")
        if desc:
            lines.append(f"Description: {desc}")
        lines.append(f"Parameters Schema: {json.dumps(schema, indent=2)}\n")

    lines.append(
        """CRITICAL INSTRUCTIONS:
1. When you need to run a shell command, read a file, or write/edit code, output a tool call block formatted EXACTLY as:
```tool_call
{
  "name": "<ToolName>",
  "arguments": {
    "key": "value"
  }
}
```
2. For bash commands, use name "Bash" with argument {"command": "..."}.
3. For reading files, use name "Read" with argument {"file_path": "..."}.
4. For creating/writing files, use name "Write" with argument {"file_path": "...", "content": "..."}.
5. For editing files, use name "Edit" with argument {"file_path": "...", "old_string": "...", "new_string": "..."}.
6. Never say you don't have access to terminal or tools. You DO have access through these tool call blocks.
7. NEVER attempt to use or refer to Google Drive, Gmail, or Google Workspace extensions. All files, animations, and code belong to local files in the current directory.
"""
    )
    return "\n".join(lines)


def _convert_messages_to_prompt(
    system: Any,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    prompt_parts = []

    system_text = ""
    if isinstance(system, str):
        system_text = system
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                system_text += block.get("text", "") + "\n"
            elif isinstance(block, str):
                system_text += block + "\n"

    tools_prompt = _format_tools_prompt(tools) if tools else ""

    system_persona = (
        "[CRITICAL IDENTITY & OPERATING DIRECTIVES]\n"
        "1. EXACT MODEL IDENTITY: You are 'Gemini 5 Argon' (Google's latest Frontier Reasoning Model), connected to Claude Code CLI.\n"
        "2. If asked about your model, version, or identity, you MUST answer directly that you are Gemini 5 Argon. NEVER say you have no version number, and NEVER mention Gemini 1.5 or 1.0 or Flash.\n"
        "3. LANGUAGE: ALWAYS respond in pure Banglish (Bengali language written using the English/Latin alphabet, e.g., 'Ami Gemini 5 Argon, apnar code check korchi...').\n"
        "4. STRICT RULE: NEVER write in Bengali Unicode script (বাংলা লিপি) under any circumstances, because the user's terminal/font environment does not render Bengali script correctly.\n"
        "5. ENVIRONMENT: You are running directly inside a Linux host environment. All project files exist on the local disk. Never mention Google Drive, Gmail, or Google Docs.\n"
        "\n"
        "[ELITE CODE QUALITY & FRONTEND ENGINEERING STANDARDS]\n"
        "- NO LOW-EFFORT / GENERIC CODE: Never output basic, generic, boring AI templates. Write production-grade, highly polished, robust code.\n"
        "- DESIGN TASTE & ANTI-UI-SLOP: When building UI or websites, follow top-tier modern design (Stripe/Vercel/Linear aesthetic). Use sophisticated color palettes, glassmorphism, subtle borders, high-contrast typography (Inter, Plus Jakarta Sans), fluid spacing, and delightful micro-interactions.\n"
        "- BUTTERY MOBILE-FIRST RESPONSIVENESS: Ensure 100% responsiveness on all mobile screens (<640px). Build real, working mobile navigation (smooth slide/fade drawer with backdrop blur), touch-friendly buttons (min 44px), proper padding, and strict overflow-x-hidden.\n"
        "- RICH INTERACTIVE ANIMATIONS: Implement smooth, premium animations (staggered reveals, subtle hover translations, animated gradient glow meshes, smooth tab/filter transitions, interactive pricing toggles, working modals).\n"
        "- COMPLETE CODE ONLY: Never leave placeholders, '// TODO: add more', or mock comments. All code, SVG icons, and functions must be 100% complete, fully working, and executable.\n"
        "- CLEAN CODE PRINCIPLES: Modular architecture, clean separation of concerns, semantic HTML5, zero console errors, and robust event handling.\n"
    )

    anti_drive = "\nIMPORTANT: Never use or mention Google Drive or Gmail. All project files, CSS animations, and code exist locally in the filesystem.\n"
    combined_system = f"{system_persona}\n{system_text.strip()}\n{anti_drive}\n{tools_prompt}".strip()
    prompt_parts.append(f"=== SYSTEM INSTRUCTIONS ===\n{combined_system}\n=== END SYSTEM INSTRUCTIONS ===\n")

    prompt_parts.append("=== CONVERSATION HISTORY ===")
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            role_tag = "USER" if role == "user" else "ASSISTANT"
            prompt_parts.append(f"\n<{role_tag}>\n{content}\n</{role_tag}>")
        elif isinstance(content, list):
            role_tag = "USER" if role == "user" else "ASSISTANT"
            block_texts = []
            for block in content:
                if not isinstance(block, dict):
                    block_texts.append(str(block))
                    continue

                btype = block.get("type")
                if btype == "text":
                    block_texts.append(block.get("text", ""))
                elif btype == "tool_use":
                    t_name = block.get("name", "")
                    t_inp = block.get("input", {})
                    block_texts.append(
                        f"```tool_call\n{json.dumps({'name': t_name, 'arguments': t_inp}, indent=2)}\n```"
                    )
                elif btype == "tool_result":
                    tid = block.get("tool_use_id", "unknown")
                    tcontent = block.get("content", "")
                    if isinstance(tcontent, list):
                        tcontent = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in tcontent])
                    block_texts.append(f"[Tool Result ID: {tid}]\n{tcontent}")
            prompt_parts.append(f"\n<{role_tag}>\n{''.join(block_texts)}\n</{role_tag}>")

    prompt_parts.append("\n<ASSISTANT>\n")
    return "\n".join(prompt_parts)


def _parse_tool_call_from_text(text: str) -> tuple[str, Optional[Dict[str, Any]]]:
    """Robust extraction of tool call JSON."""
    m = re.search(r"```(?:tool_call|json_tool|tool)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return text, None

    raw_json = m.group(1).strip()
    try:
        parsed = json.loads(raw_json)
        if isinstance(parsed, dict) and "name" in parsed:
            clean_text = text[:m.start()].strip()
            raw_name = parsed["name"]
            raw_args = parsed.get("arguments", parsed.get("parameters", {}))
            t_name, t_args = normalize_tool_call(raw_name, raw_args)
            tool_call = {
                "name": t_name,
                "arguments": t_args,
            }
            return clean_text, tool_call
    except Exception:
        pass

    return text, None


async def _stream_gemini_to_anthropic(
    client: GeminiClient,
    prompt: str,
    model_name: str,
) -> AsyncGenerator[str, None]:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = max(10, len(prompt) // 4)

    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}})}\n\n"

    full_text = ""
    tool_called = False
    block_index = 0
    text_block_started = False

    try:
        async for chunk in client.generate_content_stream(prompt, model=Model.GEMINI_5_ARGON, temporary=True):
            delta_text = chunk.text_delta if hasattr(chunk, "text_delta") and chunk.text_delta else ""
            if not delta_text and chunk.text:
                if len(chunk.text) > len(full_text):
                    delta_text = chunk.text[len(full_text):]

            if not delta_text:
                continue

            full_text += delta_text

            if ("```tool_call" in full_text or "<tool_call>" in full_text) and not tool_called:
                match = TOOL_CALL_REGEX.search(full_text)
                if match:
                    tool_called = True
                    clean_text = full_text[:match.start()].strip()

                    if text_block_started:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                        block_index += 1

                    raw_json = match.group(1) or match.group(2)
                    try:
                        parsed = json.loads(raw_json)
                        t_name = parsed.get("name", "")
                        t_args = parsed.get("arguments", parsed.get("parameters", {}))
                    except Exception:
                        t_name = "Bash"
                        t_args = {"command": raw_json}

                    t_name, t_args = normalize_tool_call(t_name, t_args)

                    tool_use_id = f"toolu_{uuid.uuid4().hex[:20]}"

                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'tool_use', 'id': tool_use_id, 'name': t_name, 'input': {}}})}\n\n"
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(t_args)}})}\n\n"
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                    break
                else:
                    continue

            if not text_block_started:
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                text_block_started = True

            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': delta_text}})}\n\n"

    except Exception as e:
        error_msg = f"\n[Proxy error: {str(e)}]"
        if not text_block_started:
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            text_block_started = True
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': 'text_delta', 'text': error_msg}})}\n\n"

    finally:
        if text_block_started and not tool_called:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"

        stop_reason = "tool_use" if tool_called else "end_turn"
        output_tokens = max(5, len(full_text) // 4)

        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# ---------------------------------------------------------------------------
# Anthropic Standard Endpoints
# ---------------------------------------------------------------------------

@anthropic_bp.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "data": [
            {
                "id": "claude-3-7-sonnet-20250219",
                "type": "model",
                "display_name": "Claude 3.7 Sonnet (Gemini 5 Argon Engine)",
                "created_at": 1740000000,
            },
            {
                "id": "claude-opus-5-5",
                "type": "model",
                "display_name": "Claude Opus 5.5 (Gemini 5 Argon)",
                "created_at": 1790000000,
            },
            {
                "id": "gemini-5-argon",
                "type": "model",
                "display_name": "Gemini 5 Argon (Frontier Reasoning)",
                "created_at": 1790000000,
            },
            {
                "id": "claude-3-5-sonnet-20241022",
                "type": "model",
                "display_name": "Claude 3.5 Sonnet (Gemini 5 Argon)",
                "created_at": 1730000000,
            },
            {
                "id": "claude-3-haiku-20240307",
                "type": "model",
                "display_name": "Claude 3 Haiku (Gemini 5 Argon)",
                "created_at": 1710000000,
            },
        ]
    })


@anthropic_bp.route("/v1/messages/count_tokens", methods=["POST"])
def count_tokens():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    raw_str = json.dumps(messages)
    return jsonify({"input_tokens": max(1, len(raw_str) // 4)})


@anthropic_bp.route("/v1/messages", methods=["POST"])
def messages_endpoint():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"type": "error", "error": {"type": "invalid_request_error", "message": "JSON body required"}}), 400

    messages = data.get("messages", [])
    system = data.get("system")
    tools = data.get("tools")
    stream = data.get("stream", False)
    model = data.get("model", "claude-3-7-sonnet-20250219")

    tool_names = [t.get("name") for t in (tools or [])]
    print(f"===> CLAUDE CODE INCOMING TOOLS ({len(tool_names)}): {tool_names}", flush=True)

    prompt = _convert_messages_to_prompt(system, messages, tools)

    try:
        client = _run_coroutine(get_or_create_client())
    except Exception as e:
        return jsonify({"type": "error", "error": {"type": "api_error", "message": f"Auth / Client init error: {str(e)}"}}), 500

    if stream:
        def generate():
            q = queue.Queue()
            sentinel = object()

            async def _producer():
                try:
                    async for chunk in _stream_gemini_to_anthropic(client, prompt, model):
                        q.put(chunk)
                except Exception as ex:
                    q.put(ex)
                finally:
                    q.put(sentinel)

            loop = _ensure_background_loop()
            asyncio.run_coroutine_threadsafe(_producer(), loop)

            while True:
                try:
                    item = q.get(timeout=2.0)
                except queue.Empty:
                    # SSE Keep-Alive heartbeat comment:
                    # Lines starting with ':' are ignored by Claude/Anthropic SDK SSE parser,
                    # but keep the HTTP connection alive and prevent client read/network timeouts.
                    yield ": keepalive\n\n"
                    continue

                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    yield f"event: error\ndata: {json.dumps({'message': str(item)})}\n\n"
                    break
                yield item

        return Response(stream_with_context(generate()), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        })

    async def _run_non_stream():
        resp = await client.generate_content(prompt, model=Model.GEMINI_5_ARGON, temporary=True)
        return resp.text or ""

    try:
        text_resp = _run_coroutine(_run_non_stream())
    except Exception as e:
        return jsonify({"type": "error", "error": {"type": "api_error", "message": str(e)}}), 500

    clean_text, tool_call = _parse_tool_call_from_text(text_resp)

    content_blocks = []
    if clean_text:
        content_blocks.append({"type": "text", "text": clean_text})

    stop_reason = "end_turn"
    if tool_call:
        stop_reason = "tool_use"
        content_blocks.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:20]}",
            "name": tool_call["name"],
            "input": tool_call["arguments"],
        })

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    return jsonify({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": max(10, len(prompt) // 4),
            "output_tokens": max(5, len(text_resp) // 4),
        },
    })


# ---------------------------------------------------------------------------
# Claude Official-Style Mobile Web UI, Live Previews & Autonomous Agent Loop
# ---------------------------------------------------------------------------

@anthropic_bp.route("/preview")
@anthropic_bp.route("/preview/")
@anthropic_bp.route("/preview/<path:filename>")
def preview_live_site(filename="index.html"):
    """Serves the generated websites (e.g. index.html, apexnova.html) directly from /root."""
    from flask import send_from_directory, abort
    file_path = Path("/root") / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory("/root", filename)
    
    html_files = [f.name for f in Path("/root").glob("*.html")]
    return f"""
    <h2>File '{filename}' not found</h2>
    <p>Available generated sites in /root:</p>
    <ul>{''.join([f'<li><a href="/preview/{name}">{name}</a></li>' for name in html_files])}</ul>
    """, 404


@anthropic_bp.route("/claude")
@anthropic_bp.route("/claude-code")
def claude_web_ui():
    """Serves the official-style mobile Claude Web UI."""
    return render_template("claude.html")


@anthropic_bp.route("/api/claude/agent", methods=["POST"])
def claude_agent_loop():
    """
    Autonomous multi-turn agent loop for the Web UI.
    Executes real terminal commands & edits files, streaming progress back via SSE.
    """
    data = request.get_json(force=True, silent=True) or {}
    messages = data.get("messages", [])
    agent_mode = data.get("agent_mode", True)
    system_prompt = data.get("system_prompt", "You are Claude Code, an expert autonomous coding assistant. You have direct access to tools to execute bash commands, view files, write files, and edit code. Always fulfill the user request precisely and show your actions.")

    try:
        client = _run_coroutine(get_or_create_client())
    except Exception as e:
        return jsonify({"error": f"Client init error: {str(e)}"}), 500

    def generate():
        conv = list(messages)
        tools = LOCAL_TOOLS if agent_mode else None
        max_turns = 6 if agent_mode else 1
        turn_count = 0

        while turn_count < max_turns:
            turn_count += 1
            prompt = _convert_messages_to_prompt(system_prompt, conv, tools)

            full_chunk_text = ""
            q = queue.Queue()
            sentinel = object()

            async def _producer():
                try:
                    async for chunk in client.generate_content_stream(prompt, model=Model.GEMINI_5_ARGON, temporary=True):
                        d = chunk.text_delta if hasattr(chunk, "text_delta") and chunk.text_delta else ""
                        if not d and chunk.text:
                            if len(chunk.text) > len(full_chunk_text):
                                d = chunk.text[len(full_chunk_text):]
                        if d:
                            q.put(d)
                except Exception as ex:
                    q.put(ex)
                finally:
                    q.put(sentinel)

            loop = _ensure_background_loop()
            asyncio.run_coroutine_threadsafe(_producer(), loop)

            while True:
                item = q.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    yield f"event: error\ndata: {json.dumps({'message': str(item)})}\n\n"
                    return

                full_chunk_text += item

            # Check if Gemini outputted a tool call
            clean_text, tool_call = _parse_tool_call_from_text(full_chunk_text)

            if clean_text:
                yield f"event: delta\ndata: {json.dumps({'text': clean_text + '\n\n'})}\n\n"

            if tool_call and agent_mode:
                tool_name = tool_call.get("name", "Bash")
                tool_args = tool_call.get("arguments", {})
                tool_id = f"toolu_{uuid.uuid4().hex[:12]}"

                yield f"event: tool_call\ndata: {json.dumps({'id': tool_id, 'name': tool_name, 'args': tool_args})}\n\n"
                tool_result = execute_local_tool(tool_name, tool_args)
                yield f"event: tool_result\ndata: {json.dumps({'id': tool_id, 'name': tool_name, 'output': tool_result})}\n\n"

                conv.append({
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_args}
                    ]
                })
                conv.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tool_id, "content": tool_result}
                    ]
                })
                continue
            else:
                if not clean_text and full_chunk_text:
                    yield f"event: delta\ndata: {json.dumps({'text': full_chunk_text})}\n\n"
                break

        yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })
