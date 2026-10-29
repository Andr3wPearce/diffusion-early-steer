#!/usr/bin/env bash
# Rewrite data512/status.html every 60 s.
cd "$(dirname "$0")"
while true; do ./.venv/Scripts/python.exe -I status.py; sleep 60; done
