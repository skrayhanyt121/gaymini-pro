#!/usr/bin/env bash
# Quick installation & restore script for Gemini 4 Claude Gateway
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[*] Installing Gemini 4 API Gateway from $DIR..."

# Copy launcher binaries to /usr/local/bin
if [ -d "$DIR/bin" ]; then
    cp "$DIR/bin/cld-argon" /usr/local/bin/
    cp "$DIR/bin/cld-hydroide" /usr/local/bin/
    cp "$DIR/bin/gemini-api" /usr/local/bin/
    chmod +x /usr/local/bin/cld-argon /usr/local/bin/cld-hydroide /usr/local/bin/gemini-api
    sed -i "s|^GAYMINI_DIR=.*|GAYMINI_DIR=\"\${GAYMINI_DIR:-$DIR}\"|" /usr/local/bin/gemini-api
    echo "[✓] Copied launchers to /usr/local/bin"
fi

# Ensure permissions (cookie files hold your live Google session)
chmod 600 "$DIR"/gaymini.txt "$DIR"/cookies.json "$DIR"/imported_cookies.json 2>/dev/null || true
chmod +x "$DIR/gemini_api_server.py"

echo "[✓] Starting Gemini API Server daemon..."
/usr/local/bin/gemini-api restart

echo "[✓] Installation complete! You can now run:"
echo "    - cld-argon     (Launches Claude Code with Gemini 4 Argon Reasoning)"
echo "    - cld-hydroide  (Launches Claude Code with Gemini 4 Hydrogen Fast Coding)"
echo "    - gemini-api    (Manage API gateway service: start|stop|status|logs)"
