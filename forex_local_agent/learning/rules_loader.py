"""
Curated Rules Loader
====================
Single parser for learning/learned_rules.md, shared by the trading LLM prompt
(core/agent_logic.py) and the chat/assistant prompts (continuous_learner.py).

Accepted block format:
    ### [CATEGORY] Rule #YYYYMMDDHHMM
    - **Date**: ...
    - **Source**: ...
    - **Directive**: ...

Blocks are rendered compactly as "- [CATEGORY] directive" and capped at whole-rule
boundaries (newest rules win when over the cap; a rule is never cut mid-text).
Auto-generated loss reflexions are never injected; they go to
learning/reflexion_candidates.jsonl for human/audit review instead.
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

TRADING_PROMPT_CAP = 2500
CHAT_PROMPT_CAP = 4000

# Never injected into the trading LLM prompt
TRADING_EXCLUDED_CATEGORIES = {"REFLEXION", "COMMUNICATION"}
EXCLUDED_SOURCES = {"loss_reflexion"}

_HEADER = re.compile(r"^###\s*\[([A-Za-z_ -]+)\]\s*(.*)$")
_FIELD = re.compile(r"^-\s*\*\*(\w+)\*\*\s*:\s*(.*)$")


def parse_rule_blocks(text: str) -> List[Dict[str, str]]:
    """Parse '### [CATEGORY]' blocks in file order. Unknown lines end the current block."""
    blocks: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    for raw in text.splitlines():
        line = raw.strip()
        m = _HEADER.match(line)
        if m:
            current = {"category": m.group(1).strip().upper().replace(" ", "_"), "title": m.group(2).strip()}
            blocks.append(current)
            continue
        if current is None:
            continue
        f = _FIELD.match(line)
        if f:
            current[f.group(1).lower()] = f.group(2).strip()
        elif line.startswith("#") or line == "---":
            current = None
    return [b for b in blocks if b.get("directive")]


def select_rules(blocks: List[Dict[str, str]], cap: int,
                 exclude_categories: Iterable[str] = (),
                 exclude_sources: Iterable[str] = EXCLUDED_SOURCES) -> str:
    excl_c = {c.upper() for c in exclude_categories}
    excl_s = {s.lower() for s in exclude_sources}
    seen, eligible = set(), []
    for b in blocks:
        if b["category"] in excl_c or b.get("source", "").lower() in excl_s:
            continue
        line = f"- [{b['category']}] {b['directive']}"
        key = b["directive"].lower()
        if key in seen:
            continue
        seen.add(key)
        eligible.append(line)

    # Newest first until the cap is reached, then restore file order
    kept, used = [], 0
    for line in reversed(eligible):
        cost = len(line) + 1
        if used + cost > cap:
            continue
        kept.append(line)
        used += cost
    kept.reverse()
    return "\n".join(kept)


def load_prompt_rules(rules_path: Path, cap: int = TRADING_PROMPT_CAP,
                      exclude_categories: Iterable[str] = TRADING_EXCLUDED_CATEGORIES) -> str:
    try:
        text = Path(rules_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    return select_rules(parse_rule_blocks(text), cap, exclude_categories)
