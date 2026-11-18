import asyncio
import sys
import json
import base64
import time
import tempfile
import mimetypes
import threading
import concurrent.futures
import re
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import quote, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sxysifat" / "src"))

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from curl_cffi.requests import AsyncSession
from gemini_webapi import GeminiClient
from gemini_webapi.constants import Model

app = Flask(
    __name__,
    static_folder=str(ROOT / "sxysifat" / "public"),
    static_url_path="/sxysifat",
    template_folder=str(ROOT / "sxysifat" / "templates"),
)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max request body

GAYMINI_TXT = ROOT / "gaymini.txt"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_TEXT_CHARS = 50_000  # max characters for text inputs
VIDEO_GENERATION_TIMEOUT = 450
MUSIC_GENERATION_TIMEOUT = 300
# The account owner supplied a Gemini conversation where Music mode is enabled.
# Reuse that conversation first. Music requests use strict session mode so a
# failed send is surfaced instead of silently creating a different conversation.
MUSIC_SESSION_SEED = "45f522b7b8e5db20"

# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------
_START_TIME = time.time()
_COUNTERS: dict[str, int] = defaultdict(int)
_LAST_CALLED: dict[str, float] = {}

# Generation requests intentionally share four logical Gemini conversations:
# one for image generation, one for video generation, one for image edits, and
# one for music. The client connection itself is recreated per request, but
# each Gemini chat_id is reused so requests do not create new conversations.
_GENERATION_SESSION_IDS: dict[str, str | None] = {
    "image": None,
    "video": None,
    "image_edit": None,
    "music": MUSIC_SESSION_SEED,
}
_GENERATION_SESSION_LOCKS = {
    "image": threading.Lock(),
    "video": threading.Lock(),
    "image_edit": threading.Lock(),
    "music": threading.Lock(),
}
_GENERATION_SESSION_REQUESTS: dict[str, int] = {
    "image": 0,
    "video": 0,
    "image_edit": 0,
    "music": 0,
}
_GENERATION_SESSION_MODELS: dict[str, str] = {
    "image": "unspecified",
    "video": "unspecified",
    "image_edit": Model.BASIC_FLASH.model_name,
}

# ---------------------------------------------------------------------------
# Log buffer
# ---------------------------------------------------------------------------
_LOG_BUFFER: deque = deque(maxlen=200)

_LEVEL_ICONS = {
    "INFO":    "ℹ",
    "SUCCESS": "✓",
    "WARN":    "⚠",
    "ERROR":   "✗",
    "REQUEST": "→",
}

def _log(level: str, msg: str):
    _LOG_BUFFER.append({
        "ts": time.time(),
        "level": level,
        "msg": msg,
        "icon": _LEVEL_ICONS.get(level, "·"),
    })

def _track(endpoint: str):
    _COUNTERS[endpoint] += 1
    _LAST_CALLED[endpoint] = time.time()
    _log("REQUEST", f"/{endpoint.replace('_', '/')} called")


def _normalize_model_name(value, default: str = "unspecified") -> str:
    if value is None:
        value = default
    if not isinstance(value, str):
        raise ValueError("model must be a string")
    model_name = value.strip() or default
    try:
        Model.from_name(model_name)
    except ValueError:
        available = ", ".join(model.model_name for model in Model)
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        ) from None
    return model_name


def _uptime_str() -> str:
    s = int(time.time() - _START_TIME)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m}m {sec}s"


# Boot log
_log("INFO", "Starting Gaymini pro...")
_log("INFO", f"Flask app initialised · static @ /sxysifat")


# ---------------------------------------------------------------------------
# Global error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "detail": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max 10 MB"}), 413


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_header_string(header_str: str) -> dict:
    cookies = {}
    for part in header_str.split(";"):
        part = part.strip()
        if "=" in part:
            name, _, value = part.partition("=")
            cookies[name.strip()] = value.strip()
    return cookies


def load_cookies() -> dict:
    import os

    # Priority 1 — environment variables (for cloud hosting)
    psid = os.environ.get("GEMINI_SECURE_1PSID", "").strip()
    psidts = os.environ.get("GEMINI_SECURE_1PSIDTS", "").strip()
    if psid:
        cookies = {"__Secure-1PSID": psid}
        if psidts:
            cookies["__Secure-1PSIDTS"] = psidts
        return cookies

    # Priority 2 — gaymini.txt (supports 3 formats)
    if not GAYMINI_TXT.exists():
        return {}

    content = GAYMINI_TXT.read_text(encoding="utf-8").strip()
    if not content:
        return {}

    collected: dict[str, str] = {}

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Format A — full browser cookie string on one line:
        #   __Secure-1PSID=xxx; __Secure-1PSIDTS=xxx; ...
        if "__Secure-1PSID" in line and ";" in line:
            return parse_header_string(line)

        # Format B — one cookie per line (key=value):
        #   __Secure-1PSID=xxx
        #   __Secure-1PSIDTS=xxx
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if key in ("__Secure-1PSID", "GEMINI_SECURE_1PSID", "PSID", "1PSID"):
                collected["__Secure-1PSID"] = val
            elif key in ("__Secure-1PSIDTS", "GEMINI_SECURE_1PSIDTS", "PSIDTS", "1PSIDTS"):
                collected["__Secure-1PSIDTS"] = val

    # Format B result
    if collected.get("__Secure-1PSID"):
        return collected

    # Format C — bare value only (first non-empty line = PSID, second = PSIDTS):
    bare_lines = [l.strip() for l in content.splitlines()
                  if l.strip() and not l.strip().startswith("#") and "=" not in l]
    if bare_lines:
        collected["__Secure-1PSID"] = bare_lines[0]
        if len(bare_lines) >= 2:
            collected["__Secure-1PSIDTS"] = bare_lines[1]
        return collected

    return {}


_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36"
    ),
    "Referer": "https://gemini.google.com/",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


async def _download_image_with_session(url: str, session) -> str | None:
    """Download image using the existing GeminiClient session (avoids 403)."""
    try:
        resp = await session.get(url, headers=_FETCH_HEADERS, allow_redirects=True)
        if resp.status_code == 200 and resp.content:
            ct = resp.headers.get("content-type", "image/jpeg").split(";")[0]
            b64 = base64.b64encode(resp.content).decode()
            return f"data:{ct};base64,{b64}"
    except Exception:
        pass
    return None


def _read_upload(field: str = "image") -> tuple[bytes, str] | tuple[None, str]:
    """Read an uploaded file, enforce size limit. Returns (bytes, mime) or (None, error)."""
    if field not in request.files:
        return None, f"{field} file is required"
    f = request.files[field]
    if not f or f.filename == "":
        return None, "no file selected"
    data = f.read()
    if not data:
        return None, "uploaded file is empty"
    if len(data) > MAX_UPLOAD_BYTES:
        return None, f"file too large (max 10 MB, got {len(data)//1024} KB)"
    return data, f.content_type or "image/jpeg"


def _save_temp_image(data: bytes, mime: str) -> Path:
    """Save image bytes to a temp file with the correct extension."""
    ext = mimetypes.guess_extension(mime) or ".jpg"
    if ext == ".jpe":
        ext = ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


def _build_client(cookies: dict) -> GeminiClient:
    psid = cookies.get("__Secure-1PSID", "")
    psidts = cookies.get("__Secure-1PSIDTS", "")
    extra = {k: v for k, v in cookies.items()
             if k not in {"__Secure-1PSID", "__Secure-1PSIDTS"}}
    return GeminiClient(secure_1psid=psid, secure_1psidts=psidts, cookies=extra or None)


async def _collect_images(output, session) -> list[dict]:
    images = []
    if output.images:
        for img in output.images:
            data_url = await _download_image_with_session(img.url, session)
            images.append({
                "url": img.url,
                "data_url": data_url,
                "type": type(img).__name__,
            })
    return images


def _collect_videos(output) -> list[dict]:
    videos = []
    for video in getattr(output, "videos", []) or []:
        url = getattr(video, "url", "") or ""
        if not url:
            continue
        videos.append({
            "url": url,
            # Gemini media URLs can require the same cookies as the generation
            # request. The UI uses this same-origin URL when direct playback
            # is blocked by the browser.
            "proxy_url": f"/api/video/proxy?url={quote(url, safe='')}",
            "thumbnail": getattr(video, "thumbnail", ""),
            "title": getattr(video, "title", "[Video]"),
            "type": type(video).__name__,
        })
    return videos


def _collect_media(output) -> list[dict]:
    """Serialize Gemini-generated music/audio media for API clients."""
    media_items = []
    for media in getattr(output, "media", []) or []:
        audio_url = getattr(media, "mp3_url", "") or ""
        video_url = getattr(media, "url", "") or ""
        if not audio_url and not video_url:
            continue
        item = {
            "audio_url": audio_url,
            "audio_proxy_url": (
                f"/api/music/proxy?url={quote(audio_url, safe='')}"
                if audio_url else ""
            ),
            "video_url": video_url,
            "video_proxy_url": (
                f"/api/video/proxy?url={quote(video_url, safe='')}"
                if video_url else ""
            ),
            "thumbnail": getattr(media, "mp3_thumbnail", "")
                or getattr(media, "thumbnail", ""),
            "title": getattr(media, "title", "[Music]"),
            "type": type(media).__name__,
        }
        media_items.append(item)
    return media_items


async def _run_gemini(message: str, model_name: str = "unspecified",
                      files: list | None = None,
                      chat_id: str | None = None,
                      strict_chat: bool = False,
                      timeout: float = 60,
                      watchdog_timeout: float = 120) -> dict:
    cookies = load_cookies()
    if not cookies.get("__Secure-1PSID"):
        return {"error": "No cookies found. Add cookies to gaymini.txt"}

    client = _build_client(cookies)
    try:
        await client.init(
            timeout=timeout,
            watchdog_timeout=watchdog_timeout,
            auto_refresh=False,
            verbose=False,
        )
        try:
            model = Model.from_name(model_name)
        except (ValueError, AttributeError):
            model = Model.UNSPECIFIED

        kwargs = {}
        if files:
            kwargs["files"] = files

        if chat_id:
            try:
                chat = client.start_chat(model=model, metadata=[chat_id, "", ""])
                output = await chat.send_message(message, **kwargs)
                out_chat_id = chat_id
            except Exception as exc:
                if strict_chat:
                    _log(
                        "ERROR",
                        f"Existing Gemini session failed: {type(exc).__name__}: {str(exc)[:160]}",
                    )
                    return {
                        "error": (
                            "The supplied Gemini session could not accept this "
                            f"request: {str(exc)}"
                        ),
                        "chat_id": chat_id,
                        "session_error": True,
                    }
                output = await client.generate_content(message, model=model, **kwargs)
                out_chat_id = output.metadata[0] if output.metadata else None
        else:
            output = await client.generate_content(message, model=model, **kwargs)
            out_chat_id = output.metadata[0] if output.metadata else None

        images = await _collect_images(output, client.client)
        videos = _collect_videos(output)
        media = _collect_media(output)
        return {
            "text": output.text,
            "images": images,
            "videos": videos,
            "media": media,
            "chat_id": out_chat_id,
        }
    finally:
        await client.close()


def _run_generation_session(
    kind: str,
    message: str,
    model_name: str = "unspecified",
    files: list | None = None,
    strict_chat: bool = False,
    timeout: float = 60,
    watchdog_timeout: float = 120,
) -> dict:
    """Run generation through one reusable logical Gemini session."""
    if kind not in _GENERATION_SESSION_IDS:
        raise ValueError(f"Unknown generation session: {kind}")
    model_name = _normalize_model_name(model_name)

    lock = _GENERATION_SESSION_LOCKS[kind]
    with lock:
        previous_chat_id = _GENERATION_SESSION_IDS[kind]
        if (
            previous_chat_id
            and kind in _GENERATION_SESSION_MODELS
            and _GENERATION_SESSION_MODELS[kind] != model_name
        ):
            _log("INFO", f"{kind} model changed; starting a new session")
            previous_chat_id = None
            _GENERATION_SESSION_IDS[kind] = None
        if kind in _GENERATION_SESSION_MODELS:
            _GENERATION_SESSION_MODELS[kind] = model_name
        _GENERATION_SESSION_REQUESTS[kind] += 1
        request_number = _GENERATION_SESSION_REQUESTS[kind]
        action = "reusing" if previous_chat_id else "starting"
        _log("INFO", f"{action} {kind} generation session (request {request_number})")

        result = asyncio.run(_run_gemini(
            message,
            model_name,
            files=files,
            chat_id=previous_chat_id,
            strict_chat=strict_chat,
            timeout=timeout,
            watchdog_timeout=watchdog_timeout,
        ))

        next_chat_id = result.get("chat_id")
        if next_chat_id:
            _GENERATION_SESSION_IDS[kind] = next_chat_id
        elif result.get("error"):
            _log("WARN", f"{kind} generation session returned an error")

        return result


_GEMINI_REFUSAL_PHRASES = (
    "against my guidelines",
    "go against",
    "i can't help",
    "i cannot help",
    "not one like that",
    "can't create",
    "cannot create",
    "can't generate",
    "cannot generate",
    "i'm not able",
    "i am not able",
    "violates",
    "policy",
    "not able to",
    "help with instead",
)


def _is_gemini_refusal(text: str) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in _GEMINI_REFUSAL_PHRASES)


async def _smart_image_edit(
    image_path: Path,
    prompt: str,
    chat_id: str | None = None,
    model_name: str = Model.BASIC_FLASH.model_name,
) -> dict:
    cookies = load_cookies()
    if not cookies.get("__Secure-1PSID"):
        return {"error": "No cookies found"}

    model = Model.from_name(_normalize_model_name(
        model_name,
        default=Model.BASIC_FLASH.model_name,
    ))
    client = _build_client(cookies)
    try:
        await client.init(timeout=60, auto_refresh=False, verbose=False)

        out_chat_id = chat_id
        if chat_id:
            try:
                chat = client.start_chat(
                    model=model,
                    metadata=[chat_id, "", ""],
                )
                output = await chat.send_message(prompt, files=[image_path])
            except Exception:
                output = await client.generate_content(
                    prompt,
                    model=model,
                    files=[image_path],
                )
                out_chat_id = output.metadata[0] if output.metadata else None
        else:
            chat = client.start_chat(model=model)
            output = await chat.send_message(prompt, files=[image_path])
            out_chat_id = output.metadata[0] if output.metadata else None
        text = output.text or ""
        images = await _collect_images(output, client.client)

        if not images and _is_gemini_refusal(text):
            return {
                "error": "Gemini declined to edit this image due to its content policy. "
                         "Try with an anime illustration or artwork instead.",
                "policy_blocked": True,
                "gemini_message": text,
                "chat_id": out_chat_id,
            }

        return {
            "text": text,
            "images": images,
            "chat_id": out_chat_id,
            "edited": len(images) > 0,
        }
    finally:
        await client.close()


def _run_image_edit_session(
    image_path: Path,
    prompt: str,
    model_name: str = Model.BASIC_FLASH.model_name,
) -> dict:
    """Run image editing through the single reusable image-edit session."""
    model_name = _normalize_model_name(
        model_name,
        default=Model.BASIC_FLASH.model_name,
    )
    lock = _GENERATION_SESSION_LOCKS["image_edit"]
    with lock:
        previous_chat_id = _GENERATION_SESSION_IDS["image_edit"]
        if (
            previous_chat_id
            and _GENERATION_SESSION_MODELS["image_edit"] != model_name
        ):
            _log("INFO", "image_edit model changed; starting a new session")
            previous_chat_id = None
            _GENERATION_SESSION_IDS["image_edit"] = None
        _GENERATION_SESSION_MODELS["image_edit"] = model_name
        _GENERATION_SESSION_REQUESTS["image_edit"] += 1
        request_number = _GENERATION_SESSION_REQUESTS["image_edit"]
        action = "reusing" if previous_chat_id else "starting"
        _log("INFO", f"{action} image_edit session (request {request_number})")

        result = asyncio.run(_smart_image_edit(
            image_path,
            prompt,
            chat_id=previous_chat_id,
            model_name=model_name,
        ))
        next_chat_id = result.get("chat_id")
        if next_chat_id:
            _GENERATION_SESSION_IDS["image_edit"] = next_chat_id
        elif result.get("error"):
            _log("WARN", "image_edit session returned an error")
        return result


# ---------------------------------------------------------------------------
# Routes — Website
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("claude.html")


@app.route("/docs")
def docs():
    return render_template("docs.html")


@app.route("/stats")
def stats_page():
    return render_template("stats.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

def _run_chat_mode(
    message: str,
    model_name: str,
    mode: str = "chat",
    chat_id: str | None = None,
) -> tuple[str, dict]:
    """Run the chat API through the same mode-aware media paths as /api/gemini."""
    effective_mode = _detect_mode(message, False) if mode == "auto" else mode

    if effective_mode == "music":
        result = _run_generation_session(
            "music",
            f"Create an original music track based on this prompt: {message}",
            model_name,
            strict_chat=True,
            timeout=MUSIC_GENERATION_TIMEOUT,
            watchdog_timeout=MUSIC_GENERATION_TIMEOUT,
        )
    elif effective_mode == "video":
        result = _run_generation_session(
            "video",
            f"Generate a video: {message}",
            model_name,
            timeout=VIDEO_GENERATION_TIMEOUT,
            watchdog_timeout=VIDEO_GENERATION_TIMEOUT,
        )
    elif effective_mode == "image":
        result = _run_generation_session(
            "image",
            f"Generate an image: {message}",
            model_name,
        )
    elif effective_mode == "chat":
        result = asyncio.run(_run_gemini(message, model_name, chat_id=chat_id))
    else:
        raise ValueError(
            "chat mode supports auto, chat, image, video, and music"
        )

    return effective_mode, result


def _chat_mode_result(result: dict, mode: str):
    """Return a consistent response for chat text and media modes."""
    if "error" in result:
        return jsonify(result), 500

    if mode == "music" and not result.get("media"):
        return jsonify({
            "error": (
                "Gemini did not return an audio track. Music generation may "
                "not be available for this account or model."
            ),
            "text": result.get("text", ""),
            "media": [],
            "mode": mode,
        }), 422
    if mode == "video" and not result.get("videos"):
        return jsonify({
            "error": (
                "Gemini did not return a video. Video generation may not be "
                "available for this account or model."
            ),
            "text": result.get("text", ""),
            "videos": [],
            "mode": mode,
        }), 422
    if mode == "image" and not result.get("images"):
        return jsonify({
            "error": "Gemini did not return an image.",
            "text": result.get("text", ""),
            "images": [],
            "mode": mode,
        }), 404

    return jsonify({**result, "mode": mode})


@app.route("/api/status")
def api_status():
    try:
        _track("status")
        cookies = load_cookies()
        return jsonify({
            "status": "online",
            "version": "1.0.0",
            "cookies_loaded": bool(cookies.get("__Secure-1PSID")),
            "generation_sessions": {
                kind: {
                    "active": bool(_GENERATION_SESSION_IDS[kind]),
                    "requests": _GENERATION_SESSION_REQUESTS[kind],
                }
                for kind in ("image", "video", "image_edit", "music")
            },
            "uptime": _uptime_str(),
            "project": "Gaymini pro",
            "author": "SIFAT",
            "github": "https://github.com/MYB-SIFAT",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    try:
        total = sum(_COUNTERS.values())
        return jsonify({
            "uptime": _uptime_str(),
            "uptime_seconds": int(time.time() - _START_TIME),
            "total_requests": total,
            "endpoints": {
                ep: {
                    "calls": _COUNTERS[ep],
                    "last_called": (
                        int(_LAST_CALLED[ep]) if ep in _LAST_CALLED else None
                    ),
                }
                for ep in sorted(_COUNTERS)
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs")
def api_logs():
    try:
        since = float(request.args.get("since", 0))
        limit = min(int(request.args.get("limit", 100)), 200)
        entries = [e for e in _LOG_BUFFER if e["ts"] > since]
        entries = entries[-limit:]
        return jsonify({"logs": entries, "server_time": time.time()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models")
def api_models():
    try:
        _track("models")
        model_list = [
            {"name": m.model_name, "default": m == Model.UNSPECIFIED}
            for m in Model
        ]
        return jsonify({"models": model_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        _track("chat")
        data = request.get_json(force=True, silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "JSON object required"}), 400
        message = data.get("message", "")
        model_name = data.get("model", "unspecified")
        mode = data.get("mode", "chat")
        if not isinstance(message, str) or not message.strip():
            return jsonify({"error": "message field is required"}), 400
        message = message.strip()
        if not isinstance(mode, str):
            return jsonify({"error": "mode must be a string"}), 400
        mode = mode.strip().lower() or "chat"
        chat_id = data.get("chat_id")
        if chat_id is not None and not isinstance(chat_id, str):
            return jsonify({"error": "chat_id must be a string"}), 400
        chat_id = chat_id.strip() or None if isinstance(chat_id, str) else None
        if not isinstance(model_name, str):
            return jsonify({"error": "model must be a string"}), 400
        model_name = _normalize_model_name(model_name)
        if len(message) > MAX_TEXT_CHARS:
            return jsonify({"error": f"message too long ({len(message):,} chars). Max {MAX_TEXT_CHARS:,} chars."}), 400
        effective_mode, result = _run_chat_mode(message, model_name, mode, chat_id=chat_id)
        return _chat_mode_result(result, effective_mode)
    except asyncio.TimeoutError:
        return jsonify({"error": "Generation timed out. Try a shorter or simpler prompt."}), 504
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/history", methods=["POST"])
def api_chat_history():
    try:
        _track("chat_history")
        data = request.get_json(force=True, silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "JSON object required"}), 400
        message = data.get("message", "")
        chat_id = data.get("chat_id", "")
        model_name = data.get("model", "unspecified")
        mode = data.get("mode", "chat")
        if not isinstance(message, str) or not message.strip():
            return jsonify({"error": "message field is required"}), 400
        message = message.strip()
        if not isinstance(chat_id, str):
            return jsonify({"error": "chat_id must be a string"}), 400
        chat_id = chat_id.strip() or None
        if not isinstance(mode, str):
            return jsonify({"error": "mode must be a string"}), 400
        mode = mode.strip().lower() or "chat"
        if not isinstance(model_name, str):
            return jsonify({"error": "model must be a string"}), 400
        model_name = _normalize_model_name(model_name)
        if len(message) > MAX_TEXT_CHARS:
            return jsonify({"error": f"message too long ({len(message):,} chars). Max {MAX_TEXT_CHARS:,} chars."}), 400
        effective_mode, result = _run_chat_mode(
            message, model_name, mode, chat_id=chat_id
        )
        return _chat_mode_result(result, effective_mode)
    except asyncio.TimeoutError:
        return jsonify({"error": "Generation timed out. Try a shorter or simpler prompt."}), 504
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/image", methods=["POST"])
def api_image():
    try:
        _track("image")
        data = request.get_json(force=True, silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "JSON object required"}), 400
        prompt = data.get("prompt", "")
        if not isinstance(prompt, str):
            return jsonify({"error": "prompt must be a string", "mode": "image"}), 400
        prompt = prompt.strip()
        if not prompt:
            return jsonify({"error": "prompt field is required"}), 400
        if len(prompt) > MAX_TEXT_CHARS:
            return jsonify({
                "error": f"prompt too long ({len(prompt):,} chars). Max {MAX_TEXT_CHARS:,} chars."
            }), 400
        model_name = _normalize_model_name(data.get("model", "unspecified"))
        result = _run_generation_session(
            "image",
            f"Generate an image: {prompt}",
            model_name,
        )
        if "error" in result:
            return jsonify({**result, "mode": "image", "model": model_name}), 500
        images = result.get("images", [])
        if not images:
            return jsonify({
                "error": "Gemini did not return an image.",
                "text": result.get("text", ""),
                "images": [],
                "mode": "image",
                "model": model_name,
                "retryable": False,
            }), 422
        return jsonify({
            "images": images,
            "text": result.get("text", ""),
            "chat_id": result.get("chat_id"),
            "mode": "image",
            "model": model_name,
        })
    except ValueError as e:
        return jsonify({"error": str(e), "mode": "image"}), 400
    except asyncio.TimeoutError:
        return jsonify({
            "error": "Image generation timed out.",
            "mode": "image",
            "retryable": True,
        }), 504
    except Exception as e:
        _log("ERROR", f"Image generation failed: {str(e)[:180]}")
        return jsonify({"error": str(e), "mode": "image"}), 502


@app.route("/api/image/proxy")
def api_image_proxy():
    try:
        _track("image_proxy")
        url = request.args.get("url", "").strip()
        if not url:
            return jsonify({"error": "url param required"}), 400
        if "googleusercontent.com" not in url and "google.com" not in url:
            return jsonify({"error": "invalid url — only Google image URLs allowed"}), 400
        cookies = load_cookies()

        async def _fetch():
            async with AsyncSession(impersonate="chrome120", cookies=cookies) as session:
                resp = await session.get(url, headers=_FETCH_HEADERS, allow_redirects=True)
                return resp.status_code, resp.content, resp.headers.get("content-type", "image/jpeg")

        status, content, ct = asyncio.run(_fetch())
        if status != 200 or not content:
            return jsonify({"error": f"Failed to fetch image (HTTP {status})"}), 502
        return Response(content, content_type=ct)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video", methods=["POST"])
@app.route("/api/video/generate", methods=["POST"])
def api_video():
    """Generate a video from a prompt, optionally animating an uploaded image."""
    try:
        _track("video")
        is_multipart = request.content_type and "multipart" in request.content_type
        if is_multipart:
            prompt = request.form.get("prompt", "")
            model_name = request.form.get("model", "unspecified")
            has_image = "image" in request.files and request.files["image"].filename != ""
        else:
            data = request.get_json(force=True, silent=True) or {}
            if not isinstance(data, dict):
                return jsonify({"error": "JSON object or multipart form required"}), 400
            prompt = data.get("prompt", "")
            model_name = data.get("model", "unspecified")
            has_image = False
        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"error": "prompt field is required"}), 400
        prompt = prompt.strip()
        if len(prompt) > MAX_TEXT_CHARS:
            return jsonify({
                "error": f"prompt too long ({len(prompt):,} chars). Max {MAX_TEXT_CHARS:,} chars."
            }), 400

        if not isinstance(model_name, str):
            return jsonify({"error": "model must be a string"}), 400
        model_name = _normalize_model_name(model_name)
        tmp_path = None
        files = None
        if has_image:
            image_data, image_mime = _read_upload("image")
            if image_data is None:
                return jsonify({"error": image_mime}), 400
            if not image_mime.lower().startswith("image/"):
                return jsonify({"error": "image must be an image file"}), 400
            tmp_path = _save_temp_image(image_data, image_mime)
            files = [tmp_path]

        try:
            result = _run_generation_session(
                "video",
                f"Generate a video: {prompt}",
                model_name,
                files=files,
                timeout=VIDEO_GENERATION_TIMEOUT,
                watchdog_timeout=VIDEO_GENERATION_TIMEOUT,
            )
        finally:
            if tmp_path:
                tmp_path.unlink(missing_ok=True)
        if "error" in result:
            return jsonify(result), 500

        videos = result.get("videos", [])
        if not videos:
            return jsonify({
                "error": "Gemini did not return a video. Video generation may not be available for this account or model.",
                "text": result.get("text", ""),
                "videos": [],
                "image_attached": bool(files),
            }), 422

        return jsonify({
            "videos": videos,
            "text": result.get("text", ""),
            "chat_id": result.get("chat_id"),
            "mode": "video",
            "model": model_name,
            "image_attached": bool(files),
        })
    except ValueError as e:
        return jsonify({"error": str(e), "mode": "video"}), 400
    except asyncio.TimeoutError:
        _log("ERROR", "Video generation timed out")
        return jsonify({
            "error": "Video generation timed out after 7.5 minutes. Try a shorter or simpler prompt.",
            "mode": "video",
            "retryable": True,
        }), 504
    except Exception as e:
        message = str(e)
        lower = message.lower()
        if any(token in lower for token in ("quota", "rate limit", "resource exhausted", "usage limit")):
            _log("WARN", "Video generation blocked by account quota")
            return jsonify({
                "error": "Gemini video generation is unavailable because this account has reached its usage limit.",
                "detail": message,
                "mode": "video",
                "retryable": False,
            }), 429
        _log("ERROR", f"Video generation failed: {message[:180]}")
        return jsonify({"error": message, "mode": "video"}), 502


@app.route("/api/video/proxy")
@app.route("/api/music/proxy")
def api_media_proxy():
    """Proxy authenticated Gemini audio/video media for browser playback."""
    try:
        is_music = request.path.startswith("/api/music/")
        _track("music_proxy" if is_music else "video_proxy")
        url = request.args.get("url", "").strip()
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        allowed_host = (
            parsed.scheme == "https"
            and (
                host == "google.com"
                or host.endswith(".google.com")
                or host == "googleusercontent.com"
                or host.endswith(".googleusercontent.com")
            )
        )
        if not allowed_host:
            media_name = "audio" if is_music else "video"
            return jsonify({
                "error": f"invalid {media_name} URL — only Google {media_name} URLs are allowed"
            }), 400

        fetch_headers = dict(_FETCH_HEADERS)
        range_header = request.headers.get("Range")
        if range_header:
            fetch_headers["Range"] = range_header
        cookies = load_cookies()

        async def _fetch():
            async with AsyncSession(impersonate="chrome120", cookies=cookies) as session:
                resp = await session.get(url, headers=fetch_headers, allow_redirects=True)
                return resp.status_code, resp.content, dict(resp.headers)

        status, content, upstream_headers = asyncio.run(_fetch())
        if status not in (200, 206) or not content:
            media_name = "audio" if is_music else "video"
            return jsonify({"error": f"Failed to fetch {media_name} (HTTP {status})"}), 502

        response_headers = {}
        for key in ("Content-Length", "Content-Range", "Accept-Ranges", "Cache-Control"):
            if key in upstream_headers:
                response_headers[key] = upstream_headers[key]
        response_headers["Accept-Ranges"] = "bytes"
        return Response(
            content,
            status=status,
            content_type=upstream_headers.get(
                "Content-Type",
                "audio/mpeg" if is_music else "video/mp4",
            ),
            headers=response_headers,
        )
    except Exception as e:
        media_name = "audio" if request.path.startswith("/api/music/") else "video"
        _log("ERROR", f"{media_name.title()} proxy failed: {str(e)[:180]}")
        return jsonify({"error": f"Unable to stream this {media_name} right now"}), 502


@app.route("/api/music", methods=["POST"])
@app.route("/api/music/generate", methods=["POST"])
def api_music():
    """Generate an original music/audio track from a text prompt."""
    try:
        _track("music")
        data = request.get_json(force=True, silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "JSON object required"}), 400
        prompt = data.get("prompt", "")
        model_name = data.get("model", "unspecified")
        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"error": "prompt field is required"}), 400
        prompt = prompt.strip()
        if len(prompt) > MAX_TEXT_CHARS:
            return jsonify({
                "error": f"prompt too long ({len(prompt):,} chars). Max {MAX_TEXT_CHARS:,} chars."
            }), 400
        if not isinstance(model_name, str):
            return jsonify({"error": "model must be a string"}), 400
        model_name = _normalize_model_name(model_name)

        result = _run_generation_session(
            "music",
            f"Create an original music track based on this prompt: {prompt}",
            model_name,
            strict_chat=True,
            timeout=MUSIC_GENERATION_TIMEOUT,
            watchdog_timeout=MUSIC_GENERATION_TIMEOUT,
        )
        if "error" in result:
            return jsonify(result), 500

        media = result.get("media", [])
        if not media:
            return jsonify({
                "error": "Gemini did not return an audio track. Music generation may not be available for this account or model.",
                "text": result.get("text", ""),
                "media": [],
            }), 422

        return jsonify({
            "media": media,
            "text": result.get("text", ""),
            "chat_id": result.get("chat_id"),
            "mode": "music",
            "model": model_name,
        })
    except ValueError as e:
        return jsonify({"error": str(e), "mode": "music"}), 400
    except asyncio.TimeoutError:
        return jsonify({
            "error": "Music generation timed out. Try a shorter or simpler prompt.",
            "mode": "music",
            "retryable": True,
        }), 504
    except Exception as e:
        message = str(e)
        lower = message.lower()
        if any(token in lower for token in ("quota", "rate limit", "resource exhausted", "usage limit")):
            return jsonify({
                "error": "Gemini music generation is unavailable because this account has reached its usage limit.",
                "detail": message,
                "mode": "music",
                "retryable": False,
            }), 429
        _log("ERROR", f"Music generation failed: {message[:180]}")
        return jsonify({"error": message, "mode": "music"}), 502


@app.route("/api/image/generate")
def api_image_generate():
    """Return generated image directly as a file (not JSON)."""
    try:
        _track("image_generate")
        prompt = request.args.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "prompt query param is required"}), 400
        if len(prompt) > MAX_TEXT_CHARS:
            return jsonify({
                "error": f"prompt too long ({len(prompt):,} chars). Max {MAX_TEXT_CHARS:,} chars."
            }), 400
        model_name = _normalize_model_name(request.args.get("model", "unspecified"))
        result = _run_generation_session(
            "image",
            f"Generate an image: {prompt}",
            model_name,
        )
        if "error" in result:
            return jsonify({**result, "mode": "image", "model": model_name}), 500
        images = result.get("images", [])
        if not images or not images[0].get("data_url"):
            return jsonify({
                "error": "Gemini did not return an embeddable image.",
                "mode": "image",
                "model": model_name,
            }), 422
        data_url = images[0]["data_url"]
        header, b64data = data_url.split(",", 1)
        ct = header.split(":")[1].split(";")[0]
        raw = base64.b64decode(b64data)
        filename = re.sub(r"[^A-Za-z0-9_-]+", "_", prompt[:40]).strip("_") or "generated-image"
        return Response(
            raw,
            content_type=ct,
            headers={
                "Content-Disposition": f'inline; filename="{filename}.png"',
                "X-Generation-Model": model_name,
                "X-Generation-Mode": "image",
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e), "mode": "image"}), 400
    except asyncio.TimeoutError:
        return jsonify({
            "error": "Image generation timed out.",
            "mode": "image",
            "retryable": True,
        }), 504
    except Exception as e:
        _log("ERROR", f"Image file generation failed: {str(e)[:180]}")
        return jsonify({"error": str(e), "mode": "image"}), 502


@app.route("/api/image/edit", methods=["POST"])
def api_image_edit():
    try:
        _track("image_edit")
        prompt = request.form.get("prompt", "").strip()
        if not isinstance(prompt, str) or not prompt:
            return jsonify({"error": "prompt field is required (form field)"}), 400
        if len(prompt) > MAX_TEXT_CHARS:
            return jsonify({
                "error": f"prompt too long ({len(prompt):,} chars). Max {MAX_TEXT_CHARS:,} chars."
            }), 400
        model_name = _normalize_model_name(
            request.form.get("model", Model.BASIC_FLASH.model_name),
            default=Model.BASIC_FLASH.model_name,
        )
        data, mime = _read_upload("image")
        if data is None:
            return jsonify({"error": mime}), 400
        if not mime.lower().startswith("image/"):
            return jsonify({"error": "image must be an image file"}), 400
        tmp_path = _save_temp_image(data, mime)
        try:
            result = _run_image_edit_session(tmp_path, prompt, model_name)
        finally:
            tmp_path.unlink(missing_ok=True)
        if "error" in result:
            return jsonify({**result, "mode": "edit", "model": model_name}), 500
        if not result.get("images") and not result.get("text"):
            return jsonify({
                "error": "Gemini could not edit this image. Try a different prompt.",
                "mode": "edit",
                "model": model_name,
                "retryable": False,
            }), 422
        return jsonify({**result, "mode": "edit", "model": model_name})
    except ValueError as e:
        return jsonify({"error": str(e), "mode": "edit"}), 400
    except asyncio.TimeoutError:
        return jsonify({
            "error": "Image editing timed out.",
            "mode": "edit",
            "retryable": True,
        }), 504
    except Exception as e:
        _log("ERROR", f"Image editing failed: {str(e)[:180]}")
        return jsonify({"error": str(e), "mode": "edit"}), 502


@app.route("/api/vision", methods=["POST"])
def api_vision():
    try:
        _track("vision")
        question = request.form.get("question", "").strip()
        if not question:
            return jsonify({"error": "question field is required (form field)"}), 400
        model_name = _normalize_model_name(request.form.get("model", "unspecified"))
        chat_id = request.form.get("chat_id", "").strip() or None
        data, mime = _read_upload("image")
        if data is None:
            return jsonify({"error": mime}), 400
        if not mime.lower().startswith("image/"):
            return jsonify({"error": "image must be an image file"}), 400
        tmp_path = _save_temp_image(data, mime)
        try:
            result = asyncio.run(_run_gemini(
                question,
                model_name,
                files=[tmp_path],
                chat_id=chat_id,
            ))
        finally:
            tmp_path.unlink(missing_ok=True)
        if "error" in result:
            return jsonify(result), 500
        return jsonify({
            "answer": result.get("text", ""),
            "images": result.get("images", []),
            "chat_id": result.get("chat_id") or chat_id,
            "model": model_name,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Deep Research endpoint
# ---------------------------------------------------------------------------

async def _run_deep_research(topic: str, poll_interval: float = 10.0, timeout: float = 600.0) -> dict:
    cookies = load_cookies()
    if not cookies.get("__Secure-1PSID"):
        return {"error": "No cookies found. Add cookies to gaymini.txt"}

    client = _build_client(cookies)
    try:
        await client.init(timeout=60, auto_refresh=False, verbose=False)

        statuses_log: list[dict] = []

        def _on_status(status):
            statuses_log.append({
                "state": status.state,
                "title": status.title,
                "notes": status.notes,
                "done": status.done,
            })

        result = await client.deep_research(
            topic,
            poll_interval=poll_interval,
            timeout=timeout,
            on_status=_on_status,
        )

        plan = result.plan
        final_text = result.text or ""

        return {
            "done": result.done,
            "topic": topic,
            "plan": {
                "research_id": plan.research_id,
                "title": plan.title,
                "query": plan.query,
                "steps": plan.steps,
                "eta_text": plan.eta_text,
            },
            "report": final_text,
            "statuses": statuses_log,
            "status_count": len(statuses_log),
        }
    finally:
        await client.close()


@app.route("/api/deep-research", methods=["POST"])
def api_deep_research():
    """
    Run a full Deep Research cycle on the given topic.

    Request body (JSON):
      topic          — research topic or question (required)
      poll_interval  — seconds between status polls (optional, default: 10)
      timeout        — max seconds to wait for completion (optional, default: 600)

    Response:
      done           — true if research completed within timeout
      topic          — original topic
      plan           — research plan (title, steps, eta_text, research_id, query)
      report         — full research report text
      statuses       — list of status snapshots collected during polling
      status_count   — number of status updates received
    """
    try:
        _track("deep_research")
        data = request.get_json(force=True, silent=True) or {}
        topic = (data.get("topic") or data.get("prompt") or data.get("message") or "").strip()
        if not topic:
            return jsonify({"error": "topic field is required"}), 400

        poll_interval = float(data.get("poll_interval", 10.0))
        timeout = float(data.get("timeout", 600.0))

        poll_interval = max(5.0, min(poll_interval, 60.0))
        timeout = max(60.0, min(timeout, 1800.0))

        result = asyncio.run(_run_deep_research(topic, poll_interval=poll_interval, timeout=timeout))
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# All-in-One endpoint
# ---------------------------------------------------------------------------

_IMAGE_KEYWORDS = {
    "generate", "create", "draw", "make", "paint", "design",
    "illustrate", "render", "sketch", "image", "picture", "photo",
    "artwork", "art", "poster", "wallpaper", "anime", "portrait",
}

_VIDEO_PHRASES = (
    "generate a video", "generate video", "create a video", "create video",
    "make a video", "make video", "video generation", "image to video",
    "image-to-video", "animate this image", "animate the image",
    "animate image", "turn this image into a video", "veo",
)

_MUSIC_PHRASES = (
    "generate music", "create music", "make music", "music generation",
    "generate a song", "create a song", "make a song", "compose a song",
    "compose music", "music track", "soundtrack", "song",
)

_EDIT_KEYWORDS = {
    "edit", "change", "modify", "transform", "convert", "alter",
    "update", "replace", "restyle", "redraw", "remake", "turn",
    "make it", "make him", "make her", "style", "background",
}

_CODE_DEBUG_KEYWORDS = {
    "bug", "error", "fix", "debug", "broken", "wrong", "issue",
    "exception", "traceback", "crash", "fail", "not working",
}

_CODE_EXPLAIN_KEYWORDS = {
    "explain", "what does", "how does", "understand", "meaning",
    "describe this code", "walk me through", "breakdown",
}

_CODE_OPTIMIZE_KEYWORDS = {
    "optimize", "performance", "faster", "efficient", "speed up",
    "improve performance", "refactor", "clean up",
}

_CODE_TEST_KEYWORDS = {
    "write test", "unit test", "test case", "testing", "pytest",
    "unittest", "test coverage", "test this",
}

_CODE_DOCUMENT_KEYWORDS = {
    "document", "docstring", "add comments", "jsdoc", "annotate",
    "add documentation", "comment this",
}

_CODE_COMPLETE_KEYWORDS = {
    "complete", "finish", "fill in", "implement", "todo",
    "placeholder", "missing", "continue this code",
}

_CODE_REVIEW_KEYWORDS = {
    "review", "code review", "check my code", "feedback",
    "look at my code", "critique", "evaluate",
}

_CODE_GENERATE_KEYWORDS = {
    "write code", "write a function", "write a class", "write a script",
    "create a function", "code for", "python code", "javascript code",
    "function that", "class that", "script that",
}

_CODE_CONVERT_KEYWORDS = {
    "convert to", "translate to", "rewrite in", "port to",
    "convert this to", "in javascript", "in python", "in typescript",
}


def _detect_mode(message: str, has_image: bool) -> str:
    """Smarter auto-detect — picks the best mode from message content and image presence."""
    lower = message.lower()
    words = set(lower.split())

    # An explicit video request wins over the generic image/vision mode.
    # This allows multipart image + prompt requests to become image-to-video.
    if has_image and any(phrase in lower for phrase in _VIDEO_PHRASES):
        return "video"

    # Image-based modes first
    if has_image:
        if words & _EDIT_KEYWORDS or any(k in lower for k in _EDIT_KEYWORDS):
            return "edit"
        return "vision"

    # Code modes — check phrase keywords first (more specific)
    if any(k in lower for k in _CODE_CONVERT_KEYWORDS):
        return "convert"
    if any(k in lower for k in _CODE_DEBUG_KEYWORDS) or words & _CODE_DEBUG_KEYWORDS:
        return "debug"
    if any(k in lower for k in _CODE_EXPLAIN_KEYWORDS):
        return "explain"
    if any(k in lower for k in _CODE_OPTIMIZE_KEYWORDS):
        return "optimize"
    if any(k in lower for k in _CODE_TEST_KEYWORDS):
        return "test"
    if any(k in lower for k in _CODE_DOCUMENT_KEYWORDS):
        return "document"
    if any(k in lower for k in _CODE_COMPLETE_KEYWORDS):
        return "complete"
    if any(k in lower for k in _CODE_REVIEW_KEYWORDS):
        return "review"
    if any(k in lower for k in _CODE_GENERATE_KEYWORDS):
        return "generate"

    # Video generation must be checked before the broad image keywords.
    if any(phrase in lower for phrase in _VIDEO_PHRASES):
        return "video"

    # Music requests must win over broad image keywords such as "create".
    if any(phrase in lower for phrase in _MUSIC_PHRASES):
        return "music"

    # Image generation
    if words & _IMAGE_KEYWORDS or any(k in lower for k in _IMAGE_KEYWORDS):
        return "image"

    return "chat"


_CODE_MODES = {"generate", "explain", "debug", "convert", "review", "optimize", "test", "document", "complete", "ask"}
_ALL_MODES = _CODE_MODES | {"auto", "chat", "image", "video", "music", "vision", "edit"}


def _build_code_prompt(mode: str, task: str, language: str, target_lang: str, context: str) -> str:
    """Build a specialized coding prompt for the given mode."""
    ctx_line = f"\nContext/Framework: {context}" if context else ""
    if mode == "generate":
        return (
            f"You are an expert {language} developer. Write clean, well-structured {language} code for the task below.\n"
            f"Include brief inline comments where helpful. Return the code in a fenced code block.{ctx_line}\n\n"
            f"Task: {task}"
        )
    elif mode == "explain":
        return (
            f"Explain the following {language} code clearly and thoroughly.\n"
            f"Cover: what it does, how it works step-by-step, any important patterns or pitfalls.{ctx_line}\n\n"
            f"```{language}\n{task}\n```"
        )
    elif mode == "debug":
        return (
            f"You are a {language} debugging expert. Analyze the code below, identify ALL bugs and issues.\n"
            f"Return:\n1. Fixed code in a fenced code block\n2. A numbered list of what was wrong and what was fixed.{ctx_line}\n\n"
            f"```{language}\n{task}\n```"
        )
    elif mode == "convert":
        return (
            f"Convert the following {language} code to {target_lang}.\n"
            f"Preserve all logic and functionality. Use idiomatic {target_lang} patterns.\n"
            f"Return only the converted code in a fenced code block.{ctx_line}\n\n"
            f"```{language}\n{task}\n```"
        )
    elif mode == "review":
        return (
            f"Do a thorough code review of the following {language} code.{ctx_line}\n\n"
            f"```{language}\n{task}\n```\n\n"
            f"Review sections:\n"
            f"1. **Code Quality** — readability, naming, structure\n"
            f"2. **Bugs & Edge Cases** — potential errors, unhandled cases\n"
            f"3. **Performance** — bottlenecks, inefficiencies\n"
            f"4. **Security** — vulnerabilities if applicable\n"
            f"5. **Improvements** — concrete suggestions with examples"
        )
    elif mode == "optimize":
        return (
            f"Optimize the following {language} code for speed, memory, and readability.{ctx_line}\n\n"
            f"```{language}\n{task}\n```\n\n"
            f"Return:\n1. Optimized code in a fenced code block\n2. A list of optimizations made with explanations"
        )
    elif mode == "test":
        return (
            f"Write comprehensive unit tests for the following {language} code.{ctx_line}\n"
            f"Cover: normal cases, edge cases, and error cases. Use the standard testing framework for {language}.\n\n"
            f"```{language}\n{task}\n```\n\n"
            f"Return the test code in a fenced code block."
        )
    elif mode == "document":
        return (
            f"Add thorough documentation to the following {language} code.{ctx_line}\n"
            f"Include: docstrings/JSDoc/comments for all functions/classes, parameter types, return values, and usage examples.\n\n"
            f"```{language}\n{task}\n```\n\n"
            f"Return the fully documented code in a fenced code block."
        )
    elif mode == "complete":
        return (
            f"Complete the following incomplete {language} code.{ctx_line}\n"
            f"Fill in the missing parts, maintaining the existing style and patterns.\n\n"
            f"```{language}\n{task}\n```\n\n"
            f"Return the completed code in a fenced code block, followed by a brief explanation of what was added."
        )
    else:  # ask / fallback
        if context:
            return f"[Context: {context}]\n\n{task}"
        return task


@app.route("/api/gemini", methods=["POST"])
def api_gemini():
    """
    All-in-One Smart endpoint — auto-detects mode from input.

    Accepts multipart/form-data OR application/json.

    Fields:
      message / prompt  — text input (required)
      image             — file upload (optional, for vision/edit modes)
      chat_id           — continue an existing conversation (optional)
      new_chat          — true to force-start a fresh session (optional)
      model             — model name (optional, default: unspecified)
      language          — programming language for code modes (default: Python)
      target_language   — target language for convert mode (default: JavaScript)
      context           — extra context / framework info for code modes (optional)
      mode              — auto | chat | image | video | music | vision | edit |
                          generate | explain | debug | convert | review |
                          optimize | test | document | complete | ask
                          (default: auto)

    Response:
      text / result  — generated text or code
      mode           — mode that was used (useful when auto-detected)
      chat_id        — session ID for follow-up messages
      images         — list of generated/edited images (when applicable)
      videos         — list of generated videos (when applicable)
    """
    try:
        _track("gemini")

        is_multipart = request.content_type and "multipart" in request.content_type
        if is_multipart:
            message = (request.form.get("message") or request.form.get("prompt") or "").strip()
            chat_id = request.form.get("chat_id", "").strip() or None
            new_chat = request.form.get("new_chat", "").lower() in ("true", "1", "yes")
            model_name = request.form.get("model", "unspecified")
            mode = request.form.get("mode", "auto").strip().lower()
            language = request.form.get("language", "Python").strip()
            target_lang = request.form.get("target_language", "JavaScript").strip()
            context = request.form.get("context", "").strip()
        else:
            body = request.get_json(force=True, silent=True) or {}
            if not isinstance(body, dict):
                return jsonify({"error": "JSON object required"}), 400
            raw_message = body.get("message") or body.get("prompt") or ""
            if not isinstance(raw_message, str):
                return jsonify({"error": "message must be a string"}), 400
            message = raw_message.strip()
            raw_chat_id = body.get("chat_id")
            if raw_chat_id is not None and not isinstance(raw_chat_id, str):
                return jsonify({"error": "chat_id must be a string"}), 400
            chat_id = raw_chat_id.strip() if isinstance(raw_chat_id, str) and raw_chat_id.strip() else None
            new_chat = body.get("new_chat", False)
            if not isinstance(new_chat, bool):
                return jsonify({"error": "new_chat must be a boolean"}), 400
            model_name = body.get("model", "unspecified")
            raw_mode = body.get("mode") or "auto"
            raw_language = body.get("language") or "Python"
            raw_target_lang = body.get("target_language") or "JavaScript"
            raw_context = body.get("context") or ""
            if not all(isinstance(value, str) for value in (
                raw_mode, raw_language, raw_target_lang, raw_context
            )):
                return jsonify({"error": "mode, language, target_language, and context must be strings"}), 400
            mode = raw_mode.strip().lower()
            language = raw_language.strip()
            target_lang = raw_target_lang.strip()
            context = raw_context.strip()

        try:
            model_name = _normalize_model_name(model_name)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        if not message:
            return jsonify({"error": "message or prompt field is required"}), 400
        if len(message) > MAX_TEXT_CHARS:
            return jsonify({"error": f"message too long ({len(message):,} chars). Max {MAX_TEXT_CHARS:,} chars allowed."}), 400

        if mode not in _ALL_MODES:
            return jsonify({
                "error": f"Unknown mode '{mode}'. Valid modes: {', '.join(sorted(_ALL_MODES))}"
            }), 400

        # Force-reset session when requested
        if new_chat:
            chat_id = None

        has_image = is_multipart and "image" in request.files and request.files["image"].filename != ""

        # Auto-detect mode from message content
        if mode == "auto":
            mode = _detect_mode(message, has_image)

        if mode in ("vision", "edit") and not has_image:
            return jsonify({"error": f"mode '{mode}' requires an image file"}), 400

        # --- Image edit ---
        if mode == "edit":
            data, mime = _read_upload("image")
            if data is None:
                return jsonify({"error": mime}), 400
            tmp_path = _save_temp_image(data, mime)
            try:
                result = _run_image_edit_session(tmp_path, message, model_name)
            finally:
                tmp_path.unlink(missing_ok=True)
            if "error" in result:
                return jsonify({**result, "mode": "edit", "model": model_name}), 500
            return jsonify({**result, "mode": "edit", "model": model_name})

        # --- Vision (image Q&A) ---
        elif mode == "vision":
            data, mime = _read_upload("image")
            if data is None:
                return jsonify({"error": mime}), 400
            tmp_path = _save_temp_image(data, mime)
            try:
                result = asyncio.run(_run_gemini(message, model_name, files=[tmp_path], chat_id=chat_id))
            finally:
                tmp_path.unlink(missing_ok=True)
            if "error" in result:
                return jsonify({**result, "mode": "vision", "model": model_name}), 500
            return jsonify({
                "answer": result.get("text", ""),
                "text": result.get("text", ""),
                "images": result.get("images", []),
                "chat_id": result.get("chat_id"),
                "mode": "vision",
                "model": model_name,
            })

        # --- Video generation ---
        elif mode == "video":
            tmp_path = None
            files = None
            if has_image:
                data, mime = _read_upload("image")
                if data is None:
                    return jsonify({"error": mime}), 400
                if not mime.lower().startswith("image/"):
                    return jsonify({"error": "image must be an image file"}), 400
                tmp_path = _save_temp_image(data, mime)
                files = [tmp_path]
            try:
                result = _run_generation_session(
                    "video",
                    f"Generate a video: {message}",
                    model_name,
                    files=files,
                    timeout=VIDEO_GENERATION_TIMEOUT,
                    watchdog_timeout=VIDEO_GENERATION_TIMEOUT,
                )
            finally:
                if tmp_path:
                    tmp_path.unlink(missing_ok=True)
            if "error" in result:
                return jsonify({**result, "mode": "video", "model": model_name}), 500
            videos = result.get("videos", [])
            if not videos:
                return jsonify({
                    "error": "Gemini did not return a video. Video generation may not be available for this account or model.",
                    "text": result.get("text", ""),
                    "videos": [],
                    "mode": "video",
                    "image_attached": bool(files),
                }), 422
            return jsonify({
                "videos": videos,
                "text": result.get("text", ""),
                "chat_id": result.get("chat_id"),
                "mode": "video",
                "model": model_name,
                "image_attached": bool(files),
            })

        # --- Music generation ---
        elif mode == "music":
            result = _run_generation_session(
                "music",
                f"Create an original music track based on this prompt: {message}",
                model_name,
                strict_chat=True,
                timeout=MUSIC_GENERATION_TIMEOUT,
                watchdog_timeout=MUSIC_GENERATION_TIMEOUT,
            )
            if "error" in result:
                return jsonify({**result, "mode": "music", "model": model_name}), 500
            media = result.get("media", [])
            if not media:
                return jsonify({
                    "error": "Gemini did not return an audio track. Music generation may not be available for this account or model.",
                    "text": result.get("text", ""),
                    "media": [],
                    "mode": "music",
                }), 422
            return jsonify({
                "media": media,
                "text": result.get("text", ""),
                "chat_id": result.get("chat_id"),
                "mode": "music",
                "model": model_name,
            })

        # --- Image generation ---
        elif mode == "image":
            result = _run_generation_session(
                "image",
                f"Generate an image: {message}",
                model_name,
            )
            if "error" in result:
                return jsonify({**result, "mode": "image", "model": model_name}), 500
            images = result.get("images", [])
            if not images:
                return jsonify({
                    "error": "Gemini did not return an image.",
                    "text": result.get("text", ""),
                    "images": [],
                    "mode": "image",
                    "model": model_name,
                    "retryable": False,
                }), 422
            return jsonify({
                "images": images,
                "text": result.get("text", ""),
                "chat_id": result.get("chat_id"),
                "mode": "image",
                "model": model_name,
            })

        # --- Code modes (generate, explain, debug, convert, review, optimize, test, document, complete, ask) ---
        elif mode in _CODE_MODES:
            prompt = _build_code_prompt(mode, message, language, target_lang, context)
            result = asyncio.run(_run_gemini(prompt, model_name, chat_id=chat_id))
            if "error" in result:
                return jsonify({**result, "mode": mode, "model": model_name}), 500
            resp = {
                "result": result.get("text", "").strip(),
                "text": result.get("text", "").strip(),
                "mode": mode,
                "language": language,
                "chat_id": result.get("chat_id"),
                "images": result.get("images", []),
                "model": model_name,
            }
            if mode == "convert":
                resp["target_language"] = target_lang
            return jsonify(resp)

        # --- Plain chat ---
        else:
            result = asyncio.run(_run_gemini(message, model_name, chat_id=chat_id))
            if "error" in result:
                return jsonify({**result, "mode": "chat", "model": model_name}), 500
            return jsonify({**result, "mode": "chat", "model": model_name})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Image Style Transfer
# ---------------------------------------------------------------------------

@app.route("/api/image/style", methods=["POST"])
def api_image_style():
    """
    Convert an image to a specific art style.

    Form fields:
      image   — image file (required)
      style   — target style: watercolor | pixel_art | oil_painting | sketch |
                anime | cyberpunk | cartoon | realistic | studio_ghibli |
                comic_book | dark_fantasy | pastel  (default: anime)

    Response:
      images      — list of generated images
      style       — style applied
      edited      — true if image was returned
    """
    try:
        _track("image_style")
        style = request.form.get("style", "anime").strip().lower().replace(" ", "_")
        model_name = _normalize_model_name(
            request.form.get("model", Model.BASIC_FLASH.model_name),
            default=Model.BASIC_FLASH.model_name,
        )
        data, mime = _read_upload("image")
        if data is None:
            return jsonify({"error": mime}), 400
        if not mime.lower().startswith("image/"):
            return jsonify({"error": "image must be an image file"}), 400

        style_descriptions = {
            "watercolor": "soft watercolor painting with flowing colors and visible brush strokes",
            "pixel_art": "retro pixel art style with a limited color palette and blocky pixels",
            "oil_painting": "rich classical oil painting with visible brush texture and deep colors",
            "sketch": "pencil sketch with fine line work, shading, and minimal color",
            "anime": "high-quality anime illustration style with clean lines and vibrant colors",
            "cyberpunk": "neon-lit cyberpunk style with dark tones, glowing lights, and futuristic elements",
            "cartoon": "bold cartoon style with thick outlines and flat vivid colors",
            "realistic": "ultra-realistic photorealistic rendering with detailed textures",
            "studio_ghibli": "warm Studio Ghibli inspired art style with soft colors and detailed backgrounds",
            "comic_book": "classic comic book style with bold lines, halftone dots, and dramatic shading",
            "dark_fantasy": "dark fantasy art style with dramatic lighting and moody atmosphere",
            "pastel": "soft pastel art style with light gentle tones and delicate coloring",
        }
        style_desc = style_descriptions.get(style, f"{style} art style")

        tmp_path = _save_temp_image(data, mime)
        try:
            result = _run_image_edit_session(
                tmp_path,
                f"Redraw this image completely in {style_desc}. Keep the same subject, composition, and elements.",
                model_name,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        if "error" in result:
            return jsonify({**result, "style": style, "mode": "style", "model": model_name}), 500
        return jsonify({**result, "style": style, "mode": "style", "model": model_name})
    except ValueError as e:
        return jsonify({"error": str(e), "mode": "style"}), 400
    except asyncio.TimeoutError:
        return jsonify({
            "error": "Image styling timed out.",
            "mode": "style",
            "retryable": True,
        }), 504
    except Exception as e:
        _log("ERROR", f"Image styling failed: {str(e)[:180]}")
        return jsonify({"error": str(e), "mode": "style"}), 502


async def _run_with_chat_edit(image_path: Path, instruction: str) -> dict:
    """Single-request image style/edit: send image + instruction together using Flash model."""
    cookies = load_cookies()
    if not cookies.get("__Secure-1PSID"):
        return {"error": "No cookies found. Add cookies to gaymini.txt"}
    client = _build_client(cookies)
    try:
        await client.init(timeout=60, auto_refresh=False, verbose=False)
        chat = client.start_chat(model=Model.BASIC_FLASH)
        output = await chat.send_message(instruction, files=[image_path])
        images = await _collect_images(output, client.client)
        text = output.text or ""
        if not images and _is_gemini_refusal(text):
            return {
                "error": "Gemini declined this request due to its content policy.",
                "policy_blocked": True,
                "gemini_message": text,
            }
        return {
            "text": text,
            "images": images,
            "chat_id": output.metadata[0] if output.metadata else None,
            "edited": len(images) > 0,
        }
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

@app.route("/api/translate", methods=["POST"])
def api_translate():
    """
    Translate text into any language.

    JSON body:
      text             — text to translate (required)
      target_language  — e.g. "Bengali", "Spanish", "Japanese" (required)
      source_language  — source language (optional, auto-detect if omitted)
      formal           — true for formal tone (optional, default: false)

    Response:
      translation      — translated text
      target_language  — language used
      detected_source  — detected source language (if auto-detect)
    """
    try:
        _track("translate")
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        target = (data.get("target_language") or data.get("target") or "").strip()
        source = (data.get("source_language") or data.get("source") or "").strip()
        formal = bool(data.get("formal", False))

        if not text:
            return jsonify({"error": "text field is required"}), 400
        if not target:
            return jsonify({"error": "target_language field is required"}), 400
        if len(text) > MAX_TEXT_CHARS:
            return jsonify({"error": f"text too long ({len(text):,} chars). Max {MAX_TEXT_CHARS:,} chars allowed."}), 400

        tone = "formal" if formal else "natural"
        src_clause = f"from {source} " if source else ""
        prompt = (
            f"Translate the following text {src_clause}to {target} in a {tone} tone. "
            f"Return only the translated text with no explanation, no quotes, no labels.\n\n"
            f"{text}"
        )
        model_name = _normalize_model_name(data.get("model", "unspecified"))
        result = asyncio.run(_run_gemini(prompt, model_name))
        if "error" in result:
            return jsonify(result), 500
        return jsonify({
            "translation": result.get("text", "").strip(),
            "target_language": target,
            "source_language": source or "auto-detected",
            "model": model_name,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """
    Summarize any text using Gemini.

    JSON body:
      text    — text to summarize (required)
      length  — "brief" | "medium" | "detailed"  (default: medium)
      format  — "paragraph" | "bullets" | "tldr"  (default: paragraph)
      language — output language (default: same as input)

    Response:
      summary  — summarized text
      length   — length setting used
      format   — format setting used
    """
    try:
        _track("summarize")
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        length = (data.get("length") or "medium").strip().lower()
        fmt = (data.get("format") or "paragraph").strip().lower()
        language = (data.get("language") or "").strip()

        if not text:
            return jsonify({"error": "text field is required"}), 400
        if len(text) > MAX_TEXT_CHARS:
            return jsonify({"error": f"text too long ({len(text):,} chars). Max {MAX_TEXT_CHARS:,} chars allowed."}), 400

        length_map = {
            "brief": "1-2 sentences",
            "medium": "3-5 sentences",
            "detailed": "a thorough paragraph covering all key points",
        }
        length_desc = length_map.get(length, "3-5 sentences")

        fmt_map = {
            "paragraph": "a flowing paragraph",
            "bullets": "concise bullet points",
            "tldr": "a single TL;DR line starting with 'TL;DR:'",
        }
        fmt_desc = fmt_map.get(fmt, "a flowing paragraph")

        lang_clause = f" Respond in {language}." if language else ""
        prompt = (
            f"Summarize the following text in {fmt_desc}, {length_desc} long.{lang_clause} "
            f"Return only the summary with no preamble.\n\n{text}"
        )
        model_name = _normalize_model_name(data.get("model", "unspecified"))
        result = asyncio.run(_run_gemini(prompt, model_name))
        if "error" in result:
            return jsonify(result), 500
        return jsonify({
            "summary": result.get("text", "").strip(),
            "length": length,
            "format": fmt,
            "model": model_name,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Code endpoint
# ---------------------------------------------------------------------------

@app.route("/api/code", methods=["POST"])
def api_code():
    """
    Advanced multi-turn coding assistant with conversation support.

    JSON body:
      task            — what you want to do / the code / the question (required)
      language        — programming language (default: Python)
      mode            — generate | explain | debug | convert | review |
                        optimize | test | document | complete | ask
                        (default: generate)
      target_language — used only when mode is "convert" (e.g. "JavaScript")
      chat_id         — continue an existing coding session (optional)
      new_chat        — true to force-start a fresh session (optional)
      context         — extra context / framework info (optional)

    Response:
      result    — the code or explanation
      mode      — mode used
      language  — language used
      chat_id   — session ID for follow-up messages
    """
    try:
        _track("code")
        data = request.get_json(force=True, silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({"error": "JSON object required"}), 400
        raw_task = next(
            (
                data.get(key)
                for key in ("task", "code", "prompt", "message")
                if data.get(key) not in (None, "")
            ),
            "",
        )
        if not isinstance(raw_task, str):
            return jsonify({"error": "task must be a string"}), 400
        task = raw_task.strip()
        language = data.get("language", "Python")
        mode = data.get("mode", "generate")
        target_lang = data.get("target_language", "JavaScript")
        context = data.get("context", "")
        chat_id = data.get("chat_id")
        new_chat = data.get("new_chat", False)
        if not isinstance(language, str):
            return jsonify({"error": "language must be a string"}), 400
        if not isinstance(mode, str):
            return jsonify({"error": "mode must be a string"}), 400
        if not isinstance(target_lang, str):
            return jsonify({"error": "target_language must be a string"}), 400
        if not isinstance(context, str):
            return jsonify({"error": "context must be a string"}), 400
        if chat_id is not None and not isinstance(chat_id, str):
            return jsonify({"error": "chat_id must be a string"}), 400
        if not isinstance(new_chat, bool):
            return jsonify({"error": "new_chat must be a boolean"}), 400
        language = language.strip()
        mode = mode.strip().lower()
        target_lang = target_lang.strip()
        context = context.strip()
        chat_id = chat_id.strip() or None if isinstance(chat_id, str) else None
        model_name = _normalize_model_name(data.get("model", "unspecified"))

        if not task:
            return jsonify({"error": "task field is required"}), 400
        if len(task) > MAX_TEXT_CHARS:
            return jsonify({"error": f"task too long ({len(task):,} chars). Max {MAX_TEXT_CHARS:,} chars allowed."}), 400
        if len(context) > MAX_TEXT_CHARS:
            return jsonify({"error": f"context too long ({len(context):,} chars). Max {MAX_TEXT_CHARS:,} chars allowed."}), 400
        if not language:
            return jsonify({"error": "language cannot be empty"}), 400
        if mode == "convert" and not target_lang:
            return jsonify({"error": "target_language is required for convert mode"}), 400

        if new_chat:
            chat_id = None

        VALID_MODES = _CODE_MODES
        if mode not in VALID_MODES:
            return jsonify({"error": f"Unknown mode '{mode}'. Valid modes: {', '.join(sorted(VALID_MODES))}"}), 400

        ctx_line = f"\nContext/Framework: {context}" if context else ""

        if mode == "generate":
            prompt = (
                f"You are an expert {language} developer. Write clean, well-structured {language} code for the task below.\n"
                f"Do not add code comments unless the task explicitly asks for them. Return the code in a fenced code block.{ctx_line}\n\n"
                f"Task: {task}"
            )
        elif mode == "explain":
            prompt = (
                f"Explain the following {language} code clearly and thoroughly.\n"
                f"Cover: what it does, how it works step-by-step, any important patterns or pitfalls.{ctx_line}\n\n"
                f"```{language}\n{task}\n```"
            )
        elif mode == "debug":
            prompt = (
                f"You are a {language} debugging expert. Analyze the code below, identify ALL bugs and issues.\n"
                f"Return:\n1. Fixed code in a fenced code block\n2. A numbered list of what was wrong and what was fixed.{ctx_line}\n\n"
                f"```{language}\n{task}\n```"
            )
        elif mode == "convert":
            prompt = (
                f"Convert the following {language} code to {target_lang}.\n"
                f"Preserve all logic and functionality. Use idiomatic {target_lang} patterns.\n"
                f"Return only the converted code in a fenced code block.{ctx_line}\n\n"
                f"```{language}\n{task}\n```"
            )
        elif mode == "review":
            prompt = (
                f"Do a thorough code review of the following {language} code.{ctx_line}\n\n"
                f"```{language}\n{task}\n```\n\n"
                f"Review sections:\n"
                f"1. **Code Quality** — readability, naming, structure\n"
                f"2. **Bugs & Edge Cases** — potential errors, unhandled cases\n"
                f"3. **Performance** — bottlenecks, inefficiencies\n"
                f"4. **Security** — vulnerabilities if applicable\n"
                f"5. **Improvements** — concrete suggestions with examples"
            )
        elif mode == "optimize":
            prompt = (
                f"Optimize the following {language} code for speed, memory, and readability.{ctx_line}\n\n"
                f"```{language}\n{task}\n```\n\n"
                f"Return:\n1. Optimized code in a fenced code block\n2. A list of optimizations made with explanations"
            )
        elif mode == "test":
            prompt = (
                f"Write comprehensive unit tests for the following {language} code.{ctx_line}\n"
                f"Cover: normal cases, edge cases, and error cases. Use the standard testing framework for {language}.\n\n"
                f"```{language}\n{task}\n```\n\n"
                f"Return the test code in a fenced code block."
            )
        elif mode == "document":
            prompt = (
                f"Add thorough documentation to the following {language} code.{ctx_line}\n"
                f"Include: docstrings/JSDoc/comments for all functions/classes, parameter types, return values, and usage examples.\n\n"
                f"```{language}\n{task}\n```\n\n"
                f"Return the fully documented code in a fenced code block."
            )
        elif mode == "complete":
            prompt = (
                f"Complete the following incomplete {language} code.{ctx_line}\n"
                f"Fill in the missing parts, maintaining the existing style and patterns.\n\n"
                f"```{language}\n{task}\n```\n\n"
                f"Return the completed code in a fenced code block, followed by a brief explanation of what was added."
            )
        elif mode == "ask":
            # Free-form question in a coding context — use conversation history
            prompt = task
            if context:
                prompt = f"[Context: {context}]\n\n{task}"

        result = asyncio.run(_run_gemini(prompt, model_name, chat_id=chat_id))
        if "error" in result:
            return jsonify({**result, "mode": mode, "model": model_name}), 500

        return jsonify({
            "result": result.get("text", "").strip(),
            "mode": mode,
            "language": language,
            "chat_id": result.get("chat_id"),
            "model": model_name,
            **({"target_language": target_lang} if mode == "convert" else {}),
        })
    except ValueError as e:
        return jsonify({"error": str(e), "mode": "code"}), 400
    except asyncio.TimeoutError:
        return jsonify({
            "error": "Code generation timed out.",
            "mode": "code",
            "retryable": True,
        }), 504
    except Exception as e:
        _log("ERROR", f"Code generation failed: {str(e)[:180]}")
        return jsonify({"error": str(e), "mode": "code"}), 502


# ---------------------------------------------------------------------------
# Text rewriter / tone changer
# ---------------------------------------------------------------------------

@app.route("/api/rewrite", methods=["POST"])
def api_rewrite():
    """
    Rewrite text in a different tone or style.

    JSON body:
      text    — text to rewrite (required)
      tone    — "formal" | "casual" | "professional" | "friendly" |
                "persuasive" | "simple" | "poetic" | "humorous"  (default: formal)
      language — output language (default: same as input)

    Response:
      rewritten  — rewritten text
      tone       — tone applied
    """
    try:
        _track("rewrite")
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        tone = (data.get("tone") or "formal").strip().lower()
        language = (data.get("language") or "").strip()
        context = (data.get("context") or "").strip()

        if not text:
            return jsonify({"error": "text field is required"}), 400
        if len(text) > MAX_TEXT_CHARS:
            return jsonify({"error": f"text too long ({len(text):,} chars). Max {MAX_TEXT_CHARS:,} chars allowed."}), 400
        if len(context) > MAX_TEXT_CHARS:
            return jsonify({"error": f"context too long ({len(context):,} chars). Max {MAX_TEXT_CHARS:,} chars allowed."}), 400

        lang_clause = f" Write the output in {language}." if language else ""
        prompt = (
            f"{context}\n\n" if context else ""
        ) + (
            f"Rewrite the following text in a {tone} tone.{lang_clause} "
            f"Keep the same meaning. Return only the rewritten text.\n\n{text}"
        )
        model_name = _normalize_model_name(data.get("model", "unspecified"))
        result = asyncio.run(_run_gemini(prompt, model_name))
        if "error" in result:
            return jsonify(result), 500
        return jsonify({
            "rewritten": result.get("text", "").strip(),
            "tone": tone,
            "model": model_name,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# SSE Streaming endpoint
# ---------------------------------------------------------------------------

@app.route("/api/chat/stream")
def api_chat_stream():
    """
    Server-Sent Events streaming chat.
    GET params: message, model (optional), chat_id (optional)

    Event types:
      {"type": "ping"}                          — keep-alive while waiting
      {"type": "chunk", "text": "..."}          — incremental text chunk
      {"type": "image", "url": "...",
       "data_url": "..."}                       — image (if Gemini returns one)
      {"type": "done", "chat_id": "..."}        — stream complete
      {"type": "error", "error": "..."}         — something went wrong
    """
    message = request.args.get("message", "").strip()
    model_name = request.args.get("model", "unspecified")
    chat_id = request.args.get("chat_id", "").strip() or None

    if not message:
        return jsonify({"error": "message query param is required"}), 400

    _track("chat_stream")

    def generate():
        result_holder: list[dict | None] = [None]
        done_event = threading.Event()

        def run_in_thread():
            result_holder[0] = asyncio.run(
                _run_gemini(message, model_name, chat_id=chat_id)
            )
            done_event.set()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(run_in_thread)

        # Send keep-alive pings every 2 s while Gemini works
        while not done_event.wait(timeout=2.0):
            yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        result = result_holder[0] or {"error": "No result returned"}
        executor.shutdown(wait=False)

        if "error" in result:
            yield f"data: {json.dumps({'type': 'error', 'error': result['error']})}\n\n"
            return

        text = result.get("text") or ""
        out_chat_id = result.get("chat_id")
        images = result.get("images", [])

        # Stream text in small word-group chunks for a typewriter effect
        tokens = re.split(r"(\s+)", text)
        chunk_buf = ""
        for token in tokens:
            chunk_buf += token
            if len(chunk_buf) >= 6:
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk_buf})}\n\n"
                chunk_buf = ""
        if chunk_buf:
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk_buf})}\n\n"

        # Stream any images
        for img in images:
            yield f"data: {json.dumps({'type': 'image', 'url': img.get('url', ''), 'data_url': img.get('data_url', '')})}\n\n"

        # Final done event
        yield f"data: {json.dumps({'type': 'done', 'chat_id': out_chat_id})}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    from anthropic_proxy import anthropic_bp
    app.register_blueprint(anthropic_bp)
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    try:
        from anthropic_proxy import anthropic_bp
        app.register_blueprint(anthropic_bp)
    except Exception as _e:
        pass
