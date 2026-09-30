"""
Continuous Learner & Dynamic Reflexion Engine
=============================================
Enables the AI agent to permanently learn from user conversations,
trading outcomes, post-mortems, and market sessions.
Rules are persisted both locally (learned_rules.md) and in Supabase Cloud.
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from dotenv import load_dotenv

# Path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
RULES_FILE = ROOT_DIR / "learning" / "learned_rules.md"

try:
    from core.supabase_manager import SupabaseManager
except ImportError:
    try:
        from forex_local_agent.core.supabase_manager import SupabaseManager
    except ImportError:
        SupabaseManager = None


class ContinuousLearner:
    """Manages persistent rule learning and memory retention."""

    def __init__(self):
        self.rules_file = RULES_FILE
        self.supabase = SupabaseManager() if SupabaseManager else None
        self._ensure_rules_file()

    def _ensure_rules_file(self):
        """Ensure learned_rules.md exists with headers."""
        if not self.rules_file.exists():
            self.rules_file.parent.mkdir(parents=True, exist_ok=True)
            header = (
                "# BUBAT AI — LEARNED RULES & CONTINUOUS INTELLIGENCE\n"
                "## Permanently Stored Rules Learned from User Conversations & Trade Reflexions\n"
                "---\n\n"
            )
            self.rules_file.write_text(header, encoding="utf-8")

    def get_all_rules_text(self) -> str:
        """Read all rules as formatted text for system prompt injection."""
        self._ensure_rules_file()
        content = self.rules_file.read_text(encoding="utf-8").strip()
        return content

    def learn_rule(self, rule_text: str, category: str = "STRATEGY", source: str = "user_conversation") -> str:
        """
        Permanently record a new rule to disk and cloud database.
        """
        self._ensure_rules_file()
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        entry = (
            f"\n### [{category.upper()}] Rule #{int(datetime.now().timestamp())}\n"
            f"- **Date**: {now_str}\n"
            f"- **Source**: {source}\n"
            f"- **Directive**: {rule_text.strip()}\n"
        )

        with open(self.rules_file, "a", encoding="utf-8") as f:
            f.write(entry)

        logger.info(f"[ContinuousLearner] Learned new rule: {rule_text.strip()}")

        # Sync to Supabase cloud
        if self.supabase:
            try:
                rule_id = f"RULE_{int(datetime.now().timestamp())}"
                self.supabase.log_learned_rule(
                    rule_id=rule_id,
                    category=category,
                    directive=rule_text.strip(),
                    source=source,
                    metadata={"timestamp": now_str}
                )
                self.supabase.log_telemetry(
                    agent_name="ContinuousLearner",
                    action_type="LEARNED_RULE",
                    details={
                        "rule_id": rule_id,
                        "rule": rule_text.strip(),
                        "category": category,
                        "source": source
                    },
                    status="ACTIVE"
                )
            except Exception as e:
                logger.warning(f"Could not sync learned rule to Supabase: {e}")

        return f"Successfully recorded and learned rule: '{rule_text.strip()}'"

    def auto_detect_and_learn(self, user_text: str) -> Optional[str]:
        """
        Analyze user input to detect corrections, preferences, or directives
        and learn them immediately.
        """
        text = user_text.strip()

        # 1. Malaysian Malay language preference
        if re.search(r"\b(bercakap|bukan berbicara|bahasa melayu|loghat|santai)\b", text, re.IGNORECASE):
            rule = "Use natural conversational Malaysian Malay ('bercakap', santai, mesra) when speaking Malay. Do NOT use formal Indonesian ('berbicara')."
            # Check if not already in file
            existing = self.get_all_rules_text()
            if "bercakap" not in existing.lower():
                return self.learn_rule(rule, category="COMMUNICATION", source="user_correction")

        # 2. Explicit rules ("always...", "never...", "remember to...")
        patterns = [
            (r"\balways\s+(.+)", "DIRECTIVE"),
            (r"\bnever\s+(.+)", "RISK_RESTRICTION"),
            (r"\bremember\s+that\s+(.+)", "STRATEGY"),
            (r"\brule:\s*(.+)", "CORE_RULE")
        ]

        for pat, cat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                directive = match.group(1).strip()
                if len(directive) > 10:
                    return self.learn_rule(directive, category=cat, source="user_instruction")

        return None
