"""
What the chat needs from the brain: a status brief for every prompt, a deep-reasoning context,
reply-language detection, and the owner's approve/reject commands (handled in code, never by
the model, so the chat can never approve a proposal on its own).
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from brain.digest import build_digest

REPORTS_DIR = AGENT_DIR / "reports"
STATE_DIR = AGENT_DIR / "state" / "brain"

_MALAY = {
    "apa", "saya", "aku", "kau", "awak", "kita", "kenapa", "mengapa", "macam", "mana", "boleh", "tak", "tidak",
    "nak", "mahu", "ini", "itu", "ni", "tu", "dan", "atau", "untuk", "dengan", "dalam", "ada", "sudah", "dah",
    "belum", "lagi", "patut", "rugi", "untung", "pasaran", "minggu", "hari", "harini", "esok", "tolong", "terangkan",
    "jelaskan", "fikir", "cadangan", "bagus", "teruk", "berapa", "bila", "siapa", "kalau", "jika", "sebab", "pun",
    "je", "jer", "lah", "kan", "ke", "betul", "salah", "beli", "jual", "dagangan", "kawan",
}
_DEEP_HINTS = (
    "why", "should", "explain", "think", "analy", "reason", "compare", "what went wrong", "what do you think",
    "recommend", "worth", "is it good", "kenapa", "mengapa", "patut", "terangkan", "jelaskan", "fikir", "analisis",
    "bandingkan", "cadangkan", "apa pendapat", "macam mana nak",
)
_PERF_HINTS = (
    "bot", "lost", "loss", "losing", "profit", "performance", "week", "proposal", "brain", "result", "win rate",
    "drawdown", "baseline", "rugi", "untung", "prestasi", "cadangan", "keputusan", "minggu",
)
_DECISION = re.compile(r"^\s*(approve|reject|lulus|luluskan|tolak)\s+(p\d+)\b\s*(.*)$", re.IGNORECASE)


def detect_language(text: str) -> str:
    """'Malay' or 'English' for the reply, from the share of common Malay words."""
    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    if not words:
        return "English"
    hits = sum(w in _MALAY for w in words)
    # Words shared with English ("bot", "ya") are left out; still need a real share, not one word
    return "Malay" if hits >= 2 and hits / len(words) >= 0.2 else "English"


def wants_deep(text: str) -> bool:
    low = (text or "").lower()
    return any(h in low for h in _DEEP_HINTS)


def mentions_performance(text: str) -> bool:
    low = (text or "").lower()
    return any(h in low for h in _PERF_HINTS)


def parse_decision(text: str) -> Optional[Tuple[str, str, str]]:
    """('approve'|'reject', 'P5', reason) when the owner typed a decision command."""
    m = _DECISION.match(text or "")
    if not m:
        return None
    action = "approve" if m.group(1).lower() in ("approve", "lulus", "luluskan") else "reject"
    return action, m.group(2).upper(), m.group(3).strip()


def latest_report() -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Newest saved daily report and its age in hours (the chat never builds one: that takes minutes)."""
    files = sorted(REPORTS_DIR.glob("report_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None, None
    age_h = (datetime.now().timestamp() - files[-1].stat().st_mtime) / 3600
    try:
        return json.loads(files[-1].read_text(encoding="utf-8")), round(age_h, 1)
    except (OSError, ValueError):
        return None, None


def _proposals() -> List[Dict[str, Any]]:
    try:
        return json.loads((STATE_DIR / "proposals.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _journal(kinds: tuple, n: int) -> List[Dict[str, Any]]:
    path = STATE_DIR / "journal.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") in kinds:
            out.append(e)
    return out[-n:]


def describe(p: Dict[str, Any]) -> str:
    from brain.agent import _describe
    return _describe(p)


def status_brief() -> str:
    """A few lines for the fast chat model's system prompt: current results, brain view, open proposals."""
    lines = []
    report, age = latest_report()
    if report:
        d = build_digest(report)
        s, b = d["summary"], d["baselines_avg_r"]
        lines.append(
            f"Bot results ({d['window']['start'][:10]} to {d['window']['end'][:10]}, report {age}h old): "
            f"{s.get('trades')} trades, win rate {s.get('win_rate_pct')}%, net {s.get('net_pnl')}, "
            f"profit factor {s.get('profit_factor')}, avg R {s.get('avg_r')}.")
        if "follow_h1_trend" in b and "llm_all_buy_sell" in b:
            lines.append(f"Simulated baselines: follow_h1_trend avg R {b['follow_h1_trend']['avg_r']}, "
                         f"LLM signals avg R {b['llm_all_buy_sell']['avg_r']}.")
    else:
        lines.append("No daily report saved yet.")
    last = _journal(("reflection",), 1)
    if last:
        lines.append(f"Brain's latest assessment ({last[-1]['ts'][:10]}): {last[-1].get('assessment', '')[:400]}")
    open_items = [p for p in _proposals() if p.get("status") == "proposed"]
    if open_items:
        lines.append("Brain proposals waiting for the owner: " +
                     "; ".join(f"{p['id']} [{p.get('risk')}] {describe(p)}" for p in open_items))
    return "\n".join(lines)


def deep_context() -> Dict[str, Any]:
    """Everything the reasoning model gets for a deep question (gathered in code, not by tool calls)."""
    report, age = latest_report()
    ctx: Dict[str, Any] = {"now_utc": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    if report:
        ctx["report_age_hours"] = age
        digest = build_digest(report)
        # Renamed so the meaning is in the key itself: models kept reading these as losses the bot made
        walls = digest.pop("blocked_by_wall_avg_r", {})
        digest["trades_blocked_by_safety_walls_NOT_taken"] = {
            "how_to_read": "simulated result IF these blocked trades had been taken; negative avg_r = the wall "
                           "avoided losses (it is working), positive = it blocked winners",
            "by_wall": walls}
        ctx["performance_digest"] = digest
    items = _proposals()
    ctx["proposals"] = {
        "waiting_for_owner": [{"id": p["id"], "change": describe(p), "risk": p.get("risk"),
                               "rationale": p.get("rationale"), "evidence": p.get("evidence")}
                              for p in items if p.get("status") == "proposed"],
        "decided": [{"id": p["id"], "change": describe(p), "status": p.get("status"), "note": p.get("note", "")[:120]}
                    for p in items if p.get("status") in ("applied", "rejected", "rolled_back")][-6:],
    }
    ctx["brain_memory"] = [
        {"date": e["ts"][:10], "kind": e["kind"],
         "text": (e.get("assessment") or e.get("finding") or "")[:500], "lessons": e.get("lessons", [])[:3]}
        for e in _journal(("reflection", "research"), 5)
    ]
    return ctx


DEEP_SYSTEM = """You are Bubat AI, the owner's assistant for their automated forex trading bot (a small demo
account). The owner is asking a question that needs careful thought. The data below comes from the bot's
own reports and its improvement agent ("the brain"); it is complete for what it covers.

Reason carefully before answering:
- Use only numbers from the data. Never invent figures; say what the data does not show.
- Fewer than 30 trades in a group is noise, not evidence.
- "baselines_avg_r" are simulations (no spread); "summary" is what really happened. Do not mix them up.
- "blocked_by_wall_avg_r" is the simulated result of trades a safety wall BLOCKED: negative = the wall helped.
- Proposals only change the bot when the owner approves them (they type: approve P5 / reject P5 reason).
- Give a clear recommendation when asked, with the reason and the main risk.

Reply in {language}. If Malay, write like a Malaysian friend chatting, casual and short, e.g.
"Bot rugi sebab dia masuk banyak trade lawan trend H1. USDJPY paling teruk: 36 trade, -$15.85."
(not formal: avoid "Kerana", "performa", "berbicara"). Be concise: a short answer first, then the key
numbers behind it."""
