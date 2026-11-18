"""
anthropic_adapter.py
====================
Bridges the Anthropic Messages API (what Claude Code speaks) and a plain
chat backend (Gemini web) that has NO native function calling.

What it does
------------
* build_prompt()   - turns system + messages + the *real* tool schemas Claude
                     Code sends into one prompt with a strict tool protocol.
* ToolStreamParser - incremental parser: streams text safely, extracts one or
                     MANY <tool_call> blocks, tolerates fences / partial tags,
                     cuts off hallucinated "[USER]" / fake tool results.
* resolve_call()   - validates a call against the declared schema, repairs
                     aliases / types / missing-key spellings.
* run_turn()       - one model turn: stream -> parse -> validate -> (one
                     corrective retry) -> events.
* SSEWriter        - emits spec-correct Anthropic streaming events.
* build_message_json() - non-streaming response body.

The module has no Flask / Gemini imports so it can be unit-tested offline.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Config (env overridable)
# --------------------------------------------------------------------------
MAX_PROMPT_CHARS = int(os.environ.get("GAYMINI_MAX_PROMPT_CHARS", "450000"))
TOOL_DESC_LIMIT = int(os.environ.get("GAYMINI_TOOL_DESC_LIMIT", "2500"))
TOOL_RETRIES = int(os.environ.get("GAYMINI_TOOL_RETRIES", "1"))

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
FENCE_OPEN = "```tool_call"
FENCE_CLOSE = "```"
STOP_MARKERS = ("\n[USER]", "\n[SYSTEM]", "\n[ASSISTANT]", "<tool_result")
ROLE_ECHO = "[ASSISTANT]"

try:  # full JSON-Schema validation when available (pip install jsonschema)
    from jsonschema.validators import validator_for as _validator_for
except Exception:  # noqa: BLE001
    _validator_for = None

STATS: Dict[str, int] = {"turns": 0, "tool_calls": 0, "repaired": 0, "retries": 0, "gave_up": 0, "empty_replies": 0}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.65)
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...[{omitted} characters omitted by gateway]...\n{text[-tail:]}"


def _slim_schema(node: Any, desc_limit: int = 300) -> Any:
    """Compact a JSON schema without changing its meaning."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in ("$schema", "$id", "examples"):
                continue
            if k == "description" and isinstance(v, str):
                out[k] = _truncate(v, desc_limit)
            elif k == "properties" and isinstance(v, dict):
                out[k] = {pk: _slim_schema(pv, desc_limit) for pk, pv in v.items()}
            else:
                out[k] = _slim_schema(v, desc_limit)
        return out
    if isinstance(node, list):
        return [_slim_schema(i, desc_limit) for i in node]
    return node


# --------------------------------------------------------------------------
# Tool index / validation
# --------------------------------------------------------------------------
NAME_ALIASES = {
    "write_file": "Write", "writefile": "Write", "filewrite": "Write",
    "create_file": "Write", "createfile": "Write",
    "edit_file": "Edit", "editfile": "Edit", "fileedit": "Edit",
    "str_replace": "Edit", "strreplace": "Edit", "str_replace_editor": "Edit",
    "read_file": "Read", "readfile": "Read", "fileread": "Read",
    "view": "Read", "cat": "Read",
    "run_command": "Bash", "runcommand": "Bash", "shell": "Bash",
    "terminal": "Bash", "run": "Bash", "sh": "Bash", "execute": "Bash",
    "bash_tool": "Bash", "exec": "Bash",
    "list_files": "Glob", "find_files": "Glob",
    "search": "Grep", "grep_search": "Grep",
}

ARG_ALIASES = {
    "file_path": ["path", "filepath", "filename", "file_name", "file", "target_file"],
    "command": ["cmd", "script", "shell_command"],
    "old_string": ["old_str", "old", "search", "target", "find"],
    "new_string": ["new_str", "new", "replacement", "replace"],
    "content": ["text", "data", "contents", "body", "file_text"],
    "pattern": ["regex", "query", "glob"],
    "description": ["desc", "summary"],
}


class ToolIndex:
    def __init__(self, tools: Optional[List[Dict[str, Any]]]):
        self.tools: Dict[str, Dict[str, Any]] = {}
        for t in tools or []:
            if isinstance(t, dict) and t.get("name"):
                self.tools[t["name"]] = t
        self._lower = {n.lower(): n for n in self.tools}

    def __bool__(self) -> bool:
        return bool(self.tools)

    def resolve_name(self, raw: Any) -> Optional[str]:
        if not isinstance(raw, str):
            return None
        raw = raw.strip()
        if raw in self.tools:
            return raw
        if raw.lower() in self._lower:
            return self._lower[raw.lower()]
        alias = NAME_ALIASES.get(raw.lower())
        if alias and alias in self.tools:
            return alias
        if alias and alias.lower() in self._lower:
            return self._lower[alias.lower()]
        return None


def _coerce(value: Any, spec: Dict[str, Any]) -> Any:
    t = spec.get("type") if isinstance(spec, dict) else None
    if not isinstance(t, str):
        return value
    try:
        if t == "string" and not isinstance(value, str) and value is not None:
            return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        if t == "integer" and isinstance(value, str):
            return int(float(value.strip()))
        if t == "integer" and isinstance(value, float) and value.is_integer():
            return int(value)
        if t == "number" and isinstance(value, str):
            f = float(value.strip())
            return int(f) if f.is_integer() else f
        if t == "boolean" and isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "yes", "1"):
                return True
            if low in ("false", "no", "0"):
                return False
        if t in ("array", "object") and isinstance(value, str):
            parsed = json.loads(value)
            if (t == "array" and isinstance(parsed, list)) or (t == "object" and isinstance(parsed, dict)):
                return parsed
    except Exception:
        return value
    return value


def resolve_call(obj: Any, index: ToolIndex) -> Tuple[Optional[str], Optional[Dict[str, Any]], List[str]]:
    """Validate + repair one parsed tool call. Returns (name, input, errors)."""
    errors: List[str] = []
    if not isinstance(obj, dict):
        return None, None, ["tool call JSON must be an object with 'name' and 'input'"]

    raw_name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    name = index.resolve_name(raw_name)
    if name is None:
        avail = ", ".join(sorted(index.tools)) or "(none)"
        return None, None, [f"unknown tool {raw_name!r}. Available tools: {avail}"]

    args = None
    for key in ("input", "arguments", "parameters", "args"):
        if key in obj:
            args = obj[key]
            break
    if args is None:
        # tolerate flat form: {"name": "Bash", "command": "ls"}
        args = {k: v for k, v in obj.items() if k not in ("name", "tool", "tool_name", "id")}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return name, None, [f"{name}: 'input' must be a JSON object, got a string"]
    if not isinstance(args, dict):
        return name, None, [f"{name}: 'input' must be a JSON object"]

    schema = index.tools[name].get("input_schema") or {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []

    args = dict(args)
    # map alias spellings -> canonical param names
    for canon, aliases in ARG_ALIASES.items():
        if canon in props and canon not in args:
            for a in aliases:
                if a in args and a not in props:
                    args[canon] = args.pop(a)
                    break
    # drop unknown keys unless the schema explicitly allows extras
    addl = schema.get("additionalProperties")
    if props and not (addl is True or isinstance(addl, dict)):
        args = {k: v for k, v in args.items() if k in props}
    # type coercion
    for k in list(args):
        if k in props:
            args[k] = _coerce(args[k], props[k])

    errors.extend(_schema_errors(name, schema, args, required))
    return name, args, errors


def _schema_errors(name: str, schema: Dict[str, Any], args: Dict[str, Any], required: List[str]) -> List[str]:
    if _validator_for is not None and schema:
        try:
            cls = _validator_for(schema)
            found = sorted(cls(schema).iter_errors(args), key=lambda e: list(e.absolute_path))
            msgs = []
            for e in found[:4]:
                where = ".".join(str(x) for x in e.absolute_path)
                if e.validator == "required":
                    msgs.append(f"{name}: {e.message}")
                else:
                    msgs.append(f"{name}: parameter {where or '(root)'}: {_truncate(e.message, 160)}")
            return msgs
        except Exception:  # noqa: BLE001  (invalid schema etc. -> fall back)
            pass
    return [f"{name}: missing required parameter {r!r}" for r in required if r not in args or args[r] is None]


# --------------------------------------------------------------------------
# Streaming parser
# --------------------------------------------------------------------------
class ToolStreamParser:
    """
    feed(delta) -> list of events:
        ("text", str)
        ("call", obj)        parsed JSON of one tool call (unvalidated)
        ("bad_call", raw)    a call block whose JSON could not be parsed
    After a call block the model is expected to stop; anything that follows
    (hallucinated results, new turns, chatter) is discarded and `halted` is set.
    """

    def __init__(self, allow_tools: bool = True):
        self.allow_tools = allow_tools
        self.buf = ""
        self.mode = "text"          # text | call | after | halted
        self.close: Optional[str] = TOOL_CLOSE
        self._started = False
        self._call_ready = False
        self._sc = [0, 0, False, False]
        self._dec = json.JSONDecoder(strict=False)

    @property
    def halted(self) -> bool:
        return self.mode == "halted"

    # -- public ----------------------------------------------------------
    def feed(self, s: str) -> List[Tuple[str, Any]]:
        if self.mode == "halted" or not s:
            return []
        self.buf += s
        out: List[Tuple[str, Any]] = []
        self._drain(out, final=False)
        return out

    def finish(self) -> List[Tuple[str, Any]]:
        out: List[Tuple[str, Any]] = []
        if self.mode != "halted":
            self._drain(out, final=True)
        self.mode = "halted"
        return out

    # -- internals -------------------------------------------------------
    def _openers(self) -> Tuple[str, ...]:
        return (TOOL_OPEN, FENCE_OPEN) if self.allow_tools else ()

    def _drain(self, out: List[Tuple[str, Any]], final: bool) -> None:
        while True:
            if self.mode == "text":
                if not self._text_step(out, final):
                    return
            elif self.mode == "call":
                if not self._call_step(out, final):
                    return
            elif self.mode == "after":
                if not self._after_step(out, final):
                    return
            else:
                return

    def _text_step(self, out, final) -> bool:
        if not self._started:
            stripped = self.buf.lstrip()
            if not stripped and not final:
                return False
            if stripped.startswith(ROLE_ECHO):
                self.buf = stripped[len(ROLE_ECHO):].lstrip(" \t")
                self.buf = self.buf.lstrip("\r\n")
            elif ROLE_ECHO.startswith(stripped) and stripped and not final:
                return False
            self._started = True

        markers = [(m, "open") for m in self._openers()] + [(m, "stop") for m in STOP_MARKERS]
        best_idx, best_marker, best_kind = None, None, None
        for m, kind in markers:
            i = self.buf.find(m)
            if i != -1 and (best_idx is None or i < best_idx):
                best_idx, best_marker, best_kind = i, m, kind
        if best_idx is not None:
            if best_idx > 0:
                out.append(("text", self.buf[:best_idx]))
            if best_kind == "stop":
                self.buf = ""
                self.mode = "halted"
                return False
            self.buf = self.buf[best_idx + len(best_marker):]
            self.close = TOOL_CLOSE if best_marker == TOOL_OPEN else FENCE_CLOSE
            self._enter_call()
            return True

        hold = 0
        if not final:
            for m, _ in markers:
                for k in range(min(len(m) - 1, len(self.buf)), 0, -1):
                    if self.buf.endswith(m[:k]):
                        hold = max(hold, k)
                        break
        cut = len(self.buf) - hold
        if cut > 0:
            out.append(("text", self.buf[:cut]))
        self.buf = self.buf[cut:]
        return False

    def _enter_call(self) -> None:
        self.mode = "call"
        self._call_ready = False
        self._sc = [0, 0, False, False]   # scan pos, depth, in_string, escape

    def _scan(self, s: str) -> Optional[int]:
        """Incrementally find the end of the first balanced JSON value in s
        (string-aware). Returns end index or None if still incomplete."""
        pos, depth, in_str, esc = self._sc
        n = len(s)
        while pos < n:
            ch = s[pos]
            pos += 1
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth <= 0:
                    self._sc = [pos, 0, False, False]
                    return pos
        self._sc = [pos, depth, in_str, esc]
        return None

    def _call_step(self, out, final) -> bool:
        if not self._call_ready:
            s = self.buf.lstrip()
            if s.startswith("```"):
                nl = s.find("\n")
                if nl == -1:
                    self.buf = s
                    if final:
                        out.append(("bad_call", s))
                        self.mode = "halted"
                    return False
                s = s[nl + 1:].lstrip()
            j = [i for i in (s.find("{"), s.find("[")) if i != -1]
            if not j:
                self.buf = s
                if final:
                    if s.strip():
                        out.append(("bad_call", s))
                    self.mode = "halted"
                return False
            self.buf = s[min(j):]
            self._call_ready = True
            self._sc = [0, 0, False, False]

        s = self.buf
        end = self._scan(s)
        if end is None:
            if not final:
                return False
            ci = s.rfind(self.close) if self.close else -1
            raw = s if ci == -1 else s[:ci]
            fixed = self._repair(raw)
            out.append(("call", fixed) if fixed is not None else ("bad_call", raw))
            self.buf = ""
            self.close = None
            self.mode = "after"
            return True

        cand = s[:end]
        try:
            obj = self._dec.raw_decode(cand)[0]
            items = obj if isinstance(obj, list) else [obj]
            for it in items:
                out.append(("call", it))
        except json.JSONDecodeError:
            fixed = self._repair(cand)
            out.append(("call", fixed) if fixed is not None else ("bad_call", cand))
        self.buf = s[end:]
        self.mode = "after"
        return True

    @staticmethod
    def _repair(raw: str) -> Optional[Any]:
        dec = json.JSONDecoder(strict=False)
        raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        for suffix in ("", "}", "}}", '"}', '"}}', '"}}}'):
            try:
                obj, _ = dec.raw_decode(raw + suffix)
                return obj
            except json.JSONDecodeError:
                continue
        return None

    def _after_step(self, out, final) -> bool:
        while True:
            r = self.buf.lstrip()
            if r.startswith("```") and not r.startswith(FENCE_OPEN):
                self.buf = r[3:]
                continue
            if self.close and r.startswith(self.close):
                self.buf = r[len(self.close):]
                self.close = None
                continue
            if self.close and r and self.close.startswith(r) and not final:
                self.buf = r
                return False
            break
        r = self.buf.lstrip()
        if not r:
            self.buf = ""
            return False
        for op in self._openers():
            if r.startswith(op):
                self.buf = r[len(op):]
                self.close = TOOL_CLOSE if op == TOOL_OPEN else FENCE_CLOSE
                self._enter_call()
                return True
        if not final and any(op.startswith(r) for op in self._openers()):
            self.buf = r
            return False
        # trailing chatter after the last tool call -> discard
        self.buf = ""
        self.mode = "halted"
        return False


# --------------------------------------------------------------------------
# Prompt building
# --------------------------------------------------------------------------
PROTOCOL = """You are the reasoning engine of an autonomous coding agent (Claude Code). A harness runs tools on the user's machine for you and returns the results in the next turn. You cannot run anything yourself - you request actions through tool calls, and the harness executes them.

# TOOL CALL PROTOCOL
To use a tool, output exactly:
<tool_call>
{"name": "<ToolName>", "input": {<arguments>}}
</tool_call>

Rules:
1. "name" must be one of the tools listed below, spelled exactly. "input" must follow that tool's JSON schema: every required parameter, correct types.
2. Output strictly valid JSON inside the tag (escape newlines as \\n and quotes as \\" inside strings). File contents go inside the JSON string.
3. You may write a short sentence of text before a tool call. You may output several independent <tool_call> blocks in a row when the calls do not depend on each other.
4. After your last <tool_call>, STOP. Never write a tool result yourself, never continue the conversation, never write "[USER]" or <tool_result>. The harness answers in the next turn.
5. Never claim that you ran a command, read, created or edited a file unless a <tool_result> in the conversation proves it. To touch files or run anything you MUST use a tool.
6. The tools below are real and available. Never say you cannot access files or run commands.
7. When the task is finished, or no tool is needed, reply in plain text without any <tool_call>.
8. Follow the agent instructions below for behaviour, style and project rules."""


_EXAMPLE_INPUTS = {
    "Bash": {"command": "ls -la"},
    "Read": {"file_path": "/path/to/file.txt"},
    "Write": {"file_path": "/path/to/new_file.py", "content": "print('hello')\n"},
    "Edit": {"file_path": "/path/to/file.py", "old_string": "foo", "new_string": "bar"},
    "Glob": {"pattern": "**/*.py"},
    "Grep": {"pattern": "TODO"},
}


def _dummy_for(spec: Dict[str, Any]) -> Any:
    t = spec.get("type") if isinstance(spec, dict) else None
    if isinstance(spec, dict) and spec.get("enum"):
        return spec["enum"][0]
    return {"string": "value", "integer": 1, "number": 1, "boolean": True, "array": [], "object": {}}.get(t, "value")


def _example_for(tool: Dict[str, Any]) -> Dict[str, Any]:
    name = tool.get("name", "")
    if name in _EXAMPLE_INPUTS:
        return _EXAMPLE_INPUTS[name]
    schema = tool.get("input_schema") or {}
    props = schema.get("properties") or {}
    return {k: _dummy_for(props.get(k, {})) for k in (schema.get("required") or [])}


def _render_examples(tools: List[Dict[str, Any]]) -> str:
    if not tools:
        return ""
    preferred = [t for n in ("Bash", "Read", "Write") for t in tools if t.get("name") == n]
    picks = (preferred or tools)[:2]
    blocks = []
    for t in picks:
        ex = json.dumps({"name": t["name"], "input": _example_for(t)}, ensure_ascii=False)
        blocks.append(f"{TOOL_OPEN}\n{ex}\n{TOOL_CLOSE}")
    return (
        "# EXAMPLES OF CORRECT TOOL CALLS\n"
        "[USER]\nShow me what is in the current directory.\n\n[ASSISTANT]\nI'll list the files.\n"
        + blocks[0]
        + ("\n\n(Independent calls can be batched:)\n" + "\n".join(blocks) if len(blocks) > 1 else "")
    )


def _render_tool(t: Dict[str, Any]) -> str:
    name = t.get("name", "")
    desc = _truncate((t.get("description") or "").strip(), TOOL_DESC_LIMIT)
    schema = _slim_schema(t.get("input_schema") or {"type": "object", "properties": {}})
    return f"## {name}\n{desc}\nInput schema: {json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"


def _system_to_text(system: Any) -> str:
    if not system:
        return ""
    if isinstance(system, list):
        return "\n".join(
            (b.get("text", "") if isinstance(b, dict) else str(b)) for b in system
        ).strip()
    return str(system).strip()


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    parts.append(c.get("text", ""))
                elif c.get("type") == "image":
                    parts.append("[image omitted: this model cannot see images]")
                else:
                    parts.append(json.dumps(c, ensure_ascii=False))
            else:
                parts.append(str(c))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _render_message(msg: Dict[str, Any], names: Dict[str, str], limit: int) -> str:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    chunks: List[str] = []
    if isinstance(content, str):
        chunks.append(content)
    elif isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                chunks.append(str(b))
                continue
            bt = b.get("type")
            if bt == "text":
                chunks.append(b.get("text", ""))
            elif bt == "tool_use":
                tid = b.get("id", "")
                names[tid] = b.get("name", "")
                payload = {"name": b.get("name", ""), "input": b.get("input", {})}
                chunks.append(f"{TOOL_OPEN}\n{json.dumps(payload, ensure_ascii=False)}\n{TOOL_CLOSE}")
            elif bt == "tool_result":
                tid = b.get("tool_use_id", "")
                status = "error" if b.get("is_error") else "ok"
                body = _truncate(_result_text(b.get("content")), limit)
                chunks.append(
                    f'<tool_result id="{tid}" name="{names.get(tid, "unknown")}" status="{status}">\n{body}\n</tool_result>'
                )
            elif bt in ("image", "document"):
                chunks.append(f"[{bt} attached - not visible to this model]")
            # thinking / redacted_thinking blocks are intentionally dropped
    text = "\n".join(c for c in chunks if c is not None and c != "").strip()
    if not text:
        return ""
    label = "[ASSISTANT]" if role == "assistant" else "[USER]"
    return f"{label}\n{text}"


def repeated_call_count(messages: List[Dict[str, Any]]) -> int:
    """How many times in a row the assistant issued the identical tool call at the end of history."""
    sigs: List[str] = []
    for m in messages:
        if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
            continue
        calls = [b for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"]
        if calls:
            sigs.append(json.dumps([(c.get("name"), c.get("input")) for c in calls], sort_keys=True, default=str))
    n = 0
    for sig in reversed(sigs):
        if sigs and sig == sigs[-1]:
            n += 1
        else:
            break
    return n


def _choice_instruction(tool_choice: Any) -> str:
    if not isinstance(tool_choice, dict):
        return ""
    t = tool_choice.get("type")
    if t == "any":
        return "For this turn you MUST respond with at least one <tool_call>."
    if t == "tool" and tool_choice.get("name"):
        return f"For this turn you MUST call the tool {tool_choice['name']} (and no other)."
    return ""


def build_prompt(
    system: Any,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Any = None,
    max_chars: int = MAX_PROMPT_CHARS,
) -> str:
    tools = tools or []
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "none":
        tools = []
    system_text = _system_to_text(system)

    header_parts: List[str] = []
    if tools:
        header_parts.append(PROTOCOL)
    if system_text:
        header_parts.append("# AGENT INSTRUCTIONS\n" + system_text)
    if tools:
        header_parts.append("# AVAILABLE TOOLS\n" + "\n\n".join(_render_tool(t) for t in tools))
        header_parts.append(_render_examples(tools))
    header = "\n\n".join(header_parts)

    footer_bits = []
    if tools:
        footer_bits.append(
            "(Reply with plain text, or with <tool_call> blocks as specified. Do not simulate tool results or user turns.)"
        )
        ci = _choice_instruction(tool_choice)
        if ci:
            footer_bits.append(ci)
        reps = repeated_call_count(messages)
        if reps >= 3:
            footer_bits.append(
                f"NOTE: you have issued the identical tool call {reps} times in a row. Repeating it will not change the result. "
                "Try a different approach, fix the underlying problem, or explain to the user what is blocking you."
            )
    footer = ("\n".join(footer_bits) + "\n\n" if footer_bits else "") + "[ASSISTANT]\n"

    def render(limit_old: int, limit_new: int, drop: int = 0) -> str:
        names: Dict[str, str] = {}
        n = len(messages)
        turns: List[str] = []
        for i, m in enumerate(messages):
            limit = limit_new if i >= n - 2 else limit_old
            r = _render_message(m, names, limit)
            if r:
                turns.append(r)
        if drop and len(turns) > drop + 2:
            turns = turns[:1] + [f"[... {drop} earlier turns omitted by gateway to fit the context window ...]"] + turns[1 + drop:]
        body = "\n\n".join(turns)
        return "\n\n".join(p for p in (header, body, footer) if p)

    for lo, ln in ((20000, 40000), (6000, 15000), (1500, 6000), (500, 3000)):
        prompt = render(lo, ln)
        if len(prompt) <= max_chars:
            return prompt
    drop = 2
    while drop < len(messages):
        prompt = render(500, 3000, drop)
        if len(prompt) <= max_chars:
            return prompt
        drop += 2
    return render(500, 3000, max(0, len(messages) - 3))


def corrective_prompt(prompt: str, raw_output: str, errors: List[str], schemas: str = "") -> str:
    bullet = "\n".join(f"- {e}" for e in errors)
    if schemas:
        bullet += "\nCorrect schema(s):\n" + schemas
    return (
        prompt.rstrip()[: -len("[ASSISTANT]")].rstrip()
        + "\n\n[ASSISTANT]\n"
        + _truncate(raw_output.strip(), 6000)
        + "\n\n[USER]\n<gateway_error>\nYour previous reply could not be executed:\n"
        + bullet
        + "\nRe-issue it now using the exact <tool_call> format with valid JSON and a valid tool name. "
        "Output only the corrected <tool_call> block(s) (plus at most one short sentence).\n</gateway_error>\n\n[ASSISTANT]\n"
    )


# --------------------------------------------------------------------------
# One model turn
# --------------------------------------------------------------------------
StreamFn = Callable[[str], AsyncIterator[str]]


async def run_turn(
    stream_fn: StreamFn,
    prompt: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Any = None,
    retries: int = TOOL_RETRIES,
) -> AsyncIterator[Tuple[str, Any]]:
    """
    Yields:
        ("text", str)
        ("tool", {"id","name","input"})
        ("stop", {"stop_reason": str, "output_chars": int})
    """
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "none":
        tools = []
    index = ToolIndex(tools)
    require_tool = bool(index) and isinstance(tool_choice, dict) and tool_choice.get("type") in ("any", "tool")

    one_call_only = isinstance(tool_choice, dict) and bool(tool_choice.get("disable_parallel_tool_use"))
    STATS["turns"] += 1
    failed_tools: List[str] = []
    attempt = 0
    cur_prompt = prompt
    total_chars = 0
    emitted_text = False
    suppress_text = False

    while True:
        parser = ToolStreamParser(allow_tools=bool(index))
        raw_out = ""
        valid: List[Tuple[str, Dict[str, Any]]] = []
        errors: List[str] = []
        say_text = not suppress_text

        def consume(events, sink):
            for kind, val in events:
                if kind == "text":
                    if say_text:
                        sink.append(("text", val))
                elif kind == "call":
                    name, args, errs = resolve_call(val, index)
                    if errs:
                        errors.extend(errs)
                        if name and name not in failed_tools:
                            failed_tools.append(name)
                    if name and args is not None and not errs:
                        raw_in = next((val[k] for k in ("input", "arguments", "parameters", "args") if k in val), None)
                        if name != val.get("name") or args != raw_in:
                            STATS["repaired"] += 1
                        valid.append((name, args))
                elif kind == "bad_call":
                    errors.append("a <tool_call> block did not contain valid JSON: " + _truncate(val, 300))

        async for delta in stream_fn(cur_prompt):
            raw_out += delta
            sink: List[Tuple[str, Any]] = []
            consume(parser.feed(delta), sink)
            for ev in sink:
                if ev[0] == "text" and ev[1].strip():
                    emitted_text = True
                yield ev
            if parser.halted:
                break
        sink = []
        consume(parser.finish(), sink)
        for ev in sink:
            if ev[0] == "text" and ev[1].strip():
                emitted_text = True
            yield ev
        total_chars += len(raw_out)

        if require_tool and not valid and not errors:
            errors.append("a tool call is required for this turn but none was produced")
        if not raw_out.strip() and not errors:
            errors.append("the reply was empty")
            STATS["empty_replies"] += 1

        if errors and attempt < retries:
            attempt += 1
            STATS["retries"] += 1
            if raw_out.strip():
                schemas = "\n\n".join(_render_tool(index.tools[n]) for n in failed_tools if n in index.tools)
                cur_prompt = corrective_prompt(prompt, raw_out, errors, schemas)
                suppress_text = True   # don't repeat prose the user already saw
            continue  # empty reply: retry with the same prompt

        if valid:
            if one_call_only:
                valid = valid[:1]
            STATS["tool_calls"] += len(valid)
            for name, args in valid:
                yield ("tool", {"id": new_id("toolu"), "name": name, "input": args})
            yield ("stop", {"stop_reason": "tool_use", "output_chars": total_chars})
            return

        if errors:
            STATS["gave_up"] += 1
        if errors and not emitted_text:
            msg = "[gateway] The model did not return a usable response: " + "; ".join(errors)[:600]
            yield ("text", msg)
        yield ("stop", {"stop_reason": "end_turn", "output_chars": total_chars})
        return


async def apply_limits(
    events: AsyncIterator[Tuple[str, Any]],
    stop_sequences: Optional[List[str]] = None,
    max_tokens: Optional[int] = None,
) -> AsyncIterator[Tuple[str, Any]]:
    """Honour `stop_sequences` and `max_tokens` (approx. 4 chars/token) on streamed text."""
    stops = [x for x in (stop_sequences or []) if isinstance(x, str) and x]
    hold = (max(len(x) for x in stops) - 1) if stops else 0
    max_chars = int(max_tokens) * 4 if isinstance(max_tokens, int) and max_tokens > 0 else None
    buf = ""
    emitted = 0

    async def finish_with(reason, seq=None, piece=""):
        nonlocal emitted
        if piece:
            emitted += len(piece)
        return ("stop", {"stop_reason": reason, "stop_sequence": seq, "output_chars": emitted})

    async for kind, val in events:
        if kind == "text":
            buf += val
            hit = None
            for sq in stops:
                i = buf.find(sq)
                if i != -1 and (hit is None or i < hit[0]):
                    hit = (i, sq)
            if hit:
                if buf[:hit[0]]:
                    yield ("text", buf[:hit[0]])
                yield await finish_with("stop_sequence", hit[1], buf[:hit[0]])
                await events.aclose()
                return
            safe = buf[: len(buf) - hold] if hold else buf
            if max_chars is not None and emitted + len(safe) > max_chars:
                piece = safe[: max(0, max_chars - emitted)]
                if piece:
                    yield ("text", piece)
                yield await finish_with("max_tokens", None, piece)
                await events.aclose()
                return
            if safe:
                yield ("text", safe)
                emitted += len(safe)
                buf = buf[len(safe):]
        else:
            if buf and kind in ("tool", "stop"):
                yield ("text", buf)
                emitted += len(buf)
                buf = ""
            if kind == "stop":
                val = dict(val)
                val["output_chars"] = max(val.get("output_chars", 0), emitted)
            yield (kind, val)


# --------------------------------------------------------------------------
# Anthropic SSE writer / non-stream body
# --------------------------------------------------------------------------
def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class SSEWriter:
    def __init__(self, msg_id: str, model: str, input_tokens: int):
        self.msg_id = msg_id
        self.model = model
        self.input_tokens = input_tokens
        self.index = 0
        self.text_open = False
        self._held_ws = ""
        self._any_content = False

    def start(self) -> List[str]:
        return [
            _sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": self.msg_id, "type": "message", "role": "assistant",
                    "model": self.model, "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 1,
                              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                },
            }),
            _sse("ping", {"type": "ping"}),
        ]

    def text(self, delta: str) -> List[str]:
        out: List[str] = []
        if not self.text_open:
            delta = (self._held_ws + delta).lstrip()
            self._held_ws = ""
            if not delta:
                return out
            out.append(_sse("content_block_start", {
                "type": "content_block_start", "index": self.index,
                "content_block": {"type": "text", "text": ""},
            }))
            self.text_open = True
        else:
            delta = self._held_ws + delta
            self._held_ws = ""
        stripped = delta.rstrip()
        self._held_ws = delta[len(stripped):]
        if stripped:
            out.append(_sse("content_block_delta", {
                "type": "content_block_delta", "index": self.index,
                "delta": {"type": "text_delta", "text": stripped},
            }))
            self._any_content = True
        return out

    def _close_text(self) -> List[str]:
        self._held_ws = ""
        if not self.text_open:
            return []
        self.text_open = False
        ev = _sse("content_block_stop", {"type": "content_block_stop", "index": self.index})
        self.index += 1
        return [ev]

    def tool(self, call: Dict[str, Any]) -> List[str]:
        out = self._close_text()
        out.append(_sse("content_block_start", {
            "type": "content_block_start", "index": self.index,
            "content_block": {"type": "tool_use", "id": call["id"], "name": call["name"], "input": {}},
        }))
        payload = json.dumps(call["input"], ensure_ascii=False)
        for i in range(0, max(len(payload), 1), 1024):
            out.append(_sse("content_block_delta", {
                "type": "content_block_delta", "index": self.index,
                "delta": {"type": "input_json_delta", "partial_json": payload[i:i + 1024]},
            }))
        out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": self.index}))
        self.index += 1
        self._any_content = True
        return out

    def finish(self, stop_reason: str, output_chars: int, stop_sequence: Optional[str] = None) -> List[str]:
        out = self._close_text()
        if not self._any_content:
            out.append(_sse("content_block_start", {
                "type": "content_block_start", "index": self.index,
                "content_block": {"type": "text", "text": ""},
            }))
            out.append(_sse("content_block_delta", {
                "type": "content_block_delta", "index": self.index,
                "delta": {"type": "text_delta", "text": "(no output)"},
            }))
            out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": self.index}))
            self.index += 1
        out.append(_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": {"output_tokens": estimate_tokens("x" * output_chars)},
        }))
        out.append(_sse("message_stop", {"type": "message_stop"}))
        return out

    @staticmethod
    def error(message: str, etype: str = "api_error") -> str:
        return _sse("error", {"type": "error", "error": {"type": etype, "message": message}})


def build_message_json(events: List[Tuple[str, Any]], msg_id: str, model: str, input_tokens: int) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    stop_reason = "end_turn"
    stop_sequence = None
    out_chars = 0
    text_buf = ""

    def flush_text():
        nonlocal text_buf
        t = text_buf.strip()
        if t:
            content.append({"type": "text", "text": t})
        text_buf = ""

    for kind, val in events:
        if kind == "text":
            text_buf += val
        elif kind == "tool":
            flush_text()
            content.append({"type": "tool_use", "id": val["id"], "name": val["name"], "input": val["input"]})
        elif kind == "stop":
            stop_reason = val["stop_reason"]
            stop_sequence = val.get("stop_sequence")
            out_chars = val["output_chars"]
    flush_text()
    if not content:
        content.append({"type": "text", "text": "(no output)"})
    return {
        "id": msg_id, "type": "message", "role": "assistant", "model": model,
        "content": content, "stop_reason": stop_reason, "stop_sequence": stop_sequence,
        "usage": {"input_tokens": input_tokens, "output_tokens": estimate_tokens("x" * out_chars),
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }


def error_body(message: str, etype: str = "api_error") -> Dict[str, Any]:
    return {"type": "error", "error": {"type": etype, "message": message}}
