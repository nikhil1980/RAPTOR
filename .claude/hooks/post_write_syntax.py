#!/usr/bin/env python3
"""
PostToolUse hook for Write and Edit tool calls.
Runs a quick Python syntax check on any .py file that was just written or edited.
"""
import json
import subprocess
import sys

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, EOFError):
    sys.exit(0)

file_path = data.get("tool_input", {}).get("file_path", "")

if not file_path.endswith(".py"):
    sys.exit(0)

result = subprocess.run(
    [sys.executable, "-m", "py_compile", file_path],
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print(f"[RAPTOR post-hook] Syntax error in {file_path}:")
    print(result.stderr.strip())
else:
    print(f"[RAPTOR post-hook] Syntax OK: {file_path}")

sys.exit(0)
