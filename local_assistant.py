"""
Launcher shim for the Bubat AI local assistant.

assistant.bat runs this file from the repo root. The real implementation lives in
forex_local_agent/local_assistant.py; this used to be a stale, diverged copy
(it lacked chat/tool logging), so it now just forwards to the canonical module.
"""
import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "forex_local_agent" / "local_assistant.py"

if __name__ == "__main__":
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")
