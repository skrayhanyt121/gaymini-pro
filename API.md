# Gaymini pro API

Simple HTTP endpoints for chat, media generation, vision, code, and text tools.

## Endpoints

Base URL: the deployed API URL. Each AI request accepts an optional model where supported; `GET /api/models` lists valid model names.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/gemini` | **All-in-One** — auto-detects mode |
| POST | `/api/chat` | Mode-aware chat, image, video, and music generation |
| POST | `/api/chat/history` | Mode-aware multi-turn chat with session |
| GET | `/api/chat/stream` | SSE streaming chat |
| POST | `/api/translate` | Translate text |
| POST | `/api/summarize` | Summarize text |
| POST | `/api/rewrite` | Rewrite / tone change |
| POST | `/api/code` | Code assistant (10 modes) |
| POST | `/api/image` | Generate image |
| POST | `/api/image/edit` | Edit image |
| POST | `/api/image/style` | Style transfer |
| GET | `/api/image/generate` | Generate image (direct file) |
| POST | `/api/video` | Generate video from a text prompt |
| POST | `/api/video/generate` | Alias for `/api/video` |
| GET | `/api/video/proxy` | Stream an authenticated Gemini video URL |
| POST | `/api/music` | Generate music/audio from a text prompt |
| POST | `/api/music/generate` | Alias for `/api/music` |
| GET | `/api/music/proxy` | Stream an authenticated Gemini audio URL |
| POST | `/api/vision` | Vision Q&A |
| POST | `/api/deep-research` | Deep Research |
| GET | `/api/models` | List available models |
| GET | `/api/stats` | Request statistics |

### Chat mode system — `/api/chat`

The chat endpoint now accepts a `mode` field. Use `chat` for normal text, `auto` to detect the request, or explicitly select `image`, `video`, or `music`. Media responses include the same `images`, `videos`, or `media` objects and same-origin proxy URLs as their dedicated endpoints.

```bash
curl -X POST BASE_URL/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Create an anime rock track with taiko drums","mode":"music"}'
```

`POST /api/chat/history` accepts the same `mode` field plus `chat_id` for normal multi-turn text chat. Media modes use their dedicated reusable generation sessions.

### Video generation — `/api/video`

Send JSON for prompt-only video, or multipart form data with an optional image for image-to-video. The optional `model` field defaults to `unspecified`.

```bash
curl -X POST BASE_URL/api/video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A cinematic aerial shot over a misty mountain lake at sunrise"}'
```

For image-to-video:

```bash
curl -X POST BASE_URL/api/video \
  -F "image=@photo.jpg" \
  -F "prompt=Make the subject wave and slowly zoom in"
```

Successful responses include `videos`, each with a Gemini-hosted `url`, same-origin `proxy_url`, `thumbnail`, and `type`, plus the accompanying `text`, `chat_id`, and `image_attached` flag. Use `proxy_url` for browser playback when the direct Gemini URL needs authentication. `/api/video/generate` is an alias. The all-in-one endpoint also supports `"mode": "video"`; auto mode recognizes explicit video phrases even when an image is uploaded.

Video generation can take several minutes. API clients and reverse proxies should allow up to 8 minutes for this request.

### Image generation and editing

`POST /api/image` accepts a JSON object with a required string `prompt` and an optional `model`. Supported model values include `unspecified`, `gemini-3-flash`, and `gemini-3-pro`. The response includes `images`, `text`, `chat_id`, `mode`, and the selected `model`. `GET /api/image/generate` accepts the same prompt and model as query parameters and returns the first image as a file.

```bash
curl -X POST BASE_URL/api/image \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A dragon in a cyberpunk city at night","model":"gemini-3-flash"}'
```

`POST /api/image/edit` and `POST /api/image/style` accept `multipart/form-data` with an `image` file, an instruction or style, and an optional `model`. Editing and style transfer default to `gemini-3-flash`. Model changes start a fresh generation session so a prior model's conversation is not reused.

```bash
curl -X POST BASE_URL/api/image/edit \
  -F "image=@photo.jpg" \
  -F "prompt=Make the background white" \
  -F "model=gemini-3-pro"
```

Image and edit endpoints reject invalid model names, non-image uploads, and overlong prompts with a 4xx response. Generation errors include the mode and retryability where applicable.

### Vision — `/api/vision`

Send `multipart/form-data` with `image` and `question`. Optional `model` selects the model and `chat_id` continues a vision conversation.

### Text endpoints

`POST /api/translate`, `/api/summarize`, and `/api/rewrite` accept an optional `model`. Translation also accepts `formal`; rewriting accepts `context` for a specific rewrite instruction.

### Code assistant — `/api/code`

Send a JSON object with `task`, optional `mode`, `language`, `target_language`, `context`, `chat_id`, `new_chat`, and `model`. The `model` choices use the same values as video generation. All ten code modes are available: `generate`, `explain`, `debug`, `convert`, `review`, `optimize`, `test`, `document`, `complete`, and `ask`.

```bash
curl -X POST BASE_URL/api/code \
  -H "Content-Type: application/json" \
  -d '{"task":"Sort a list of records by date","mode":"generate","language":"Python","context":"Python 3.11","model":"gemini-3-flash"}'
```

The response includes `result`, `mode`, `language`, `chat_id`, and the selected `model`. `target_language` is also returned for `convert` mode. Invalid fields or model names return a 400 response; timeouts return 504 with `retryable: true`.

### Music generation — `/api/music`

Send a JSON body with a music prompt. The optional `model` field defaults to `unspecified`.

```bash
curl -X POST BASE_URL/api/music \
  -H "Content-Type: application/json" \
  -d '{"prompt":"An energetic anime rock anthem with taiko drums and passionate vocals"}'
```

Successful responses include `media`, with `audio_url`, same-origin `audio_proxy_url`, `thumbnail`, `title`, and `type`, plus the accompanying `text`, `chat_id`, and `mode`. Use `audio_proxy_url` for browser playback when the direct Gemini URL needs authentication. `/api/music/generate` is an alias. The all-in-one endpoint also supports `"mode": "music"` and auto mode recognizes music/song/soundtrack prompts.

Music generation may take several minutes. API clients and reverse proxies should allow up to 5 minutes for this request.

### All-in-One — `/api/gemini`
