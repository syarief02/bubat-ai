"""
Persistent, structured, secret-free chat and tool execution logging.
Appends JSONL records to logs/chat_sessions.log with automatic redaction
of credentials, emails, and sensitive identifiers.
"""

import sys
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Pattern matching secret assignments: key=..., password=..., token=...
# Handles quotes, colons/equals, whitespace, and variants like api_key, apikey, auth_token
SECRET_PATTERN = re.compile(
    r"""(?i)(?:['"]?(?:(?:api[_-]?)?key|apikey|(?:access[_-]?|auth[_-]?)?token|password|passwd|pwd)['"]?|\b(?:(?:api[_-]?)?key|apikey|(?:access[_-]?|auth[_-]?)?token|password|passwd|pwd)\b)\s*[:=]\s*(?:['"][^'"]*['"]|[^\s,;]+)"""
)

# Email address pattern
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# Sequences of 8 or more digits (account numbers, numeric IDs, etc.)
DIGITS_PATTERN = re.compile(r"\d{8,}")

DEFAULT_LOG_FILE = Path(__file__).resolve().parent / "logs" / "chat_sessions.log"


def scrub_secrets(text: Optional[str]) -> str:
    """
    Scrubs passwords, API keys, tokens, emails, and sequences of 8+ digits.
    Replaces matched patterns with '[REDACTED]'.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Scrub key=..., password=..., token=... patterns
    text = SECRET_PATTERN.sub("[REDACTED]", text)
    # 2. Scrub email addresses
    text = EMAIL_PATTERN.sub("[REDACTED]", text)
    # 3. Scrub sequences of 8+ digits
    text = DIGITS_PATTERN.sub("[REDACTED]", text)

    return text


def log_chat_event(
    session_id: str,
    role: str,
    content: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_result_preview: Optional[str] = None,
    log_file: Union[str, Path] = DEFAULT_LOG_FILE,
) -> Dict[str, Any]:
    """
    Appends a structured JSONL record to logs/chat_sessions.log.

    Fields:
    - timestamp: ISO 8601 UTC timestamp
    - session_id: unique session UUID string
    - role: 'user', 'assistant', or 'tool'
    - content: message text (scrubbed, truncated to 500 chars)
    - tool_name: tool name if tool call, else None
    - tool_result_preview: tool output (scrubbed, first 200 chars if provided, else None)
    """
    scrubbed_content = (
        scrub_secrets(content)[:500] if content is not None else None
    )
    scrubbed_preview = (
        scrub_secrets(tool_result_preview)[:200]
        if tool_result_preview is not None
        else None
    )
    scrubbed_tool_name = (
        scrub_secrets(str(tool_name)) if tool_name is not None else None
    )

    entry: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": str(session_id),
        "role": str(role),
        "content": scrubbed_content,
        "tool_name": scrubbed_tool_name,
        "tool_result_preview": scrubbed_preview,
    }

    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        sys.stderr.write(f"[ChatLogger Warning] Failed to write log: {e}\n")

    return entry


# Convenience aliases
log_chat_message = log_chat_event
scrub = scrub_secrets
