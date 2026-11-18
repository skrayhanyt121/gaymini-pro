#!/bin/sh
PORT="${PORT:-8080}"
exec gunicorn app:app --bind "0.0.0.0:$PORT" --workers 1 --timeout 600 --preload --log-level info
