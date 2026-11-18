#!/bin/bash
# Start Gaymini Claude Code Proxy on port 5000
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

export PORT=5000
echo "Starting Gaymini Claude Proxy on http://127.0.0.1:5000 ..."
exec /usr/bin/python3 app.py
