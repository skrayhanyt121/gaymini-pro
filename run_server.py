import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gemini_api_server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting server on http://{host}:{port}...", flush=True)
    try:
        from waitress import serve
        serve(gemini_api_server.app, host=host, port=port, threads=int(os.environ.get("GAYMINI_THREADS", "16")),
              channel_timeout=900, send_bytes=1, ident="gaymini")
    except ImportError:
        gemini_api_server.app.run(host=host, port=port, debug=False, threaded=True)
