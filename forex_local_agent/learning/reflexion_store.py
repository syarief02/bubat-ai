"""
Reflexion Store
===============
Quarantine for auto-generated loss post-mortems, plus the persisted set of
closed-trade tickets that have already been reflected on.

Why: the 1.5B reflexion model produced 50 junk rules in one restart (e.g.
"Set a stop-loss level of 208.320 for GBPJPY") and the old loader injected them
into the trading prompt, evicting every curated rule. Candidates now live in
learning/reflexion_candidates.jsonl (gitignored) for the daily audit to review;
only audited rules are written to learned_rules.md.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

MAX_CANDIDATES_BYTES = 2 * 1024 * 1024
PROCESSED_RETENTION_SECONDS = 7 * 24 * 3600

# Things the deterministic engine already owns; a rule about them is noise
_BANNED = (
    "stop-loss", "stop loss", "stoploss", "trailing stop", "take profit", "take-profit", "lot size",
    "implement a", "ensure sufficient", "always ensure", "volume check", "monitor the market",
    "risk management strategy", "verify h1 trend and spread",
)
_PRICE_LEVEL = re.compile(r"\b\d{1,4}\.\d{3,5}\b")


def validate_reflexion_rule(rule: str) -> Tuple[bool, str]:
    text = (rule or "").strip()
    low = text.lower()
    if len(text) < 30:
        return False, "too_short"
    for phrase in _BANNED:
        if phrase in low:
            return False, f"banned:{phrase}"
    if _PRICE_LEVEL.search(text):
        return False, "hardcoded_price_level"
    return True, "ok"


class ReflexionStore:
    def __init__(self, agent_dir: Path):
        agent_dir = Path(agent_dir)
        self.candidates_path = agent_dir / "learning" / "reflexion_candidates.jsonl"
        self.state_path = agent_dir / "state" / "processed_tickets.json"
        self._processed: Dict[str, float] = {}

    def load_processed_tickets(self) -> Set[int]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._processed = {str(k): float(v) for k, v in data.get("tickets", {}).items()}
        except (OSError, ValueError):
            self._processed = {}
        return {int(k) for k in self._processed}

    def mark_processed(self, ticket: int, close_ts: Optional[float] = None):
        self._processed[str(ticket)] = float(close_ts or time.time())

    def save_processed(self):
        cutoff = time.time() - PROCESSED_RETENTION_SECONDS
        self._processed = {k: v for k, v in self._processed.items() if v >= cutoff}
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"tickets": self._processed}), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError:
            pass

    def add_candidate(self, facts: Dict, post_mortem: Dict, valid: bool, reason: str):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "valid": valid,
            "validation": reason,
            "facts": facts,
            "post_mortem": post_mortem,
        }
        try:
            self.candidates_path.parent.mkdir(parents=True, exist_ok=True)
            if self.candidates_path.exists() and self.candidates_path.stat().st_size > MAX_CANDIDATES_BYTES:
                lines = self.candidates_path.read_text(encoding="utf-8").splitlines()
                self.candidates_path.write_text("\n".join(lines[len(lines) // 2:]) + "\n", encoding="utf-8")
            with open(self.candidates_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass
