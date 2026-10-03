"""
Per-Session Execution Profiles
==============================
Every session trades, but each one has its own execution profile and its own
daily loss budget, and is re-graded once per UTC day from its recent results.

Sessions (by ENTRY hour, UTC; see trade_analytics.session_for_hour):
    ASIA 22-07, LONDON 07-12, LONDON_NY_OVERLAP 12-16, NEW_YORK 16-21, ROLLOVER 21

Profile (owner baseline in config.json -> session_profiles):
    max_open_trades      open positions that were opened in this session
    loss_budget_pct      net realized loss allowed today (UTC day) from trades opened
                         in this session, % of balance. Budgets add up to
                         daily_loss_limit_pct, so one bad session cannot use up the
                         others' budget.
    max_spread_sl_ratio  spread cost wall for this session
    symbols              allow-list (null = all configured symbols)

Daily grading (trades opened in the session over the last `review_days` days, R = pips / SL pips):
    NORMAL   -> owner baseline
    REDUCED  -> half the slots, USD majors only, spread cap <= 4% of SL
    MINIMAL  -> 1 slot, USD majors only, spread cap <= 3% of SL
    Down: n >= 20 and avg R < -0.15 -> at least REDUCED; n >= 20 and avg R < -0.30 -> MINIMAL.
    Up:   one level per day when n >= 10 and avg R >= 0.
    Never above the owner baseline: the grader only tightens and restores.
"""

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from loguru import logger

try:
    from core.trade_analytics import session_for_hour
except ImportError:
    from forex_local_agent.core.trade_analytics import session_for_hour

SESSIONS = ("ASIA", "LONDON", "LONDON_NY_OVERLAP", "NEW_YORK", "ROLLOVER")
LEVELS = ("NORMAL", "REDUCED", "MINIMAL")
USD_MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
STATE_DIR = Path(__file__).resolve().parent.parent / "state"

DEFAULT_PROFILE = {"max_open_trades": 10, "loss_budget_pct": 1.0, "max_spread_sl_ratio": 0.06, "symbols": None}
GRADING = {"review_days": 5, "reduce_below_r": -0.15, "minimal_below_r": -0.30, "min_trades_down": 20,
           "min_trades_up": 10, "restore_at_or_above_r": 0.0}


def trade_r(t: Dict) -> Optional[float]:
    if not t.get("sl_pips") or t.get("pips") is None:
        return None
    return t["pips"] / t["sl_pips"]


def session_stats(trades: List[Dict]) -> Dict[str, Dict]:
    """{session: {n, avg_r, net}} by entry hour."""
    acc: Dict[str, List] = {s: [] for s in SESSIONS}
    for t in trades:
        if t.get("entry_hour_utc") is None:
            continue
        acc[session_for_hour(int(t["entry_hour_utc"]))].append(t)
    out = {}
    for s, ts in acc.items():
        rs = [r for r in (trade_r(t) for t in ts) if r is not None]
        out[s] = {"n": len(rs), "avg_r": round(sum(rs) / len(rs), 3) if rs else None,
                  "net": round(sum(t.get("net_profit", 0.0) for t in ts), 2)}
    return out


def next_level(current: str, n: int, avg_r: Optional[float], g: Dict = GRADING) -> str:
    """One daily grading step with hysteresis."""
    i = LEVELS.index(current)
    if avg_r is not None and n >= g["min_trades_down"]:
        if avg_r < g["minimal_below_r"]:
            return "MINIMAL"
        if avg_r < g["reduce_below_r"]:
            return LEVELS[max(i, 1)]
    if avg_r is not None and n >= g["min_trades_up"] and avg_r >= g["restore_at_or_above_r"] and i > 0:
        return LEVELS[i - 1]
    return current


def effective_profile(base: Dict, level: str) -> Dict:
    p = {**DEFAULT_PROFILE, **base}
    if level == "REDUCED":
        p["max_open_trades"] = max(1, math.ceil(p["max_open_trades"] / 2))
        p["max_spread_sl_ratio"] = min(p["max_spread_sl_ratio"], 0.04)
        p["symbols"] = [s for s in USD_MAJORS if p["symbols"] is None or s in p["symbols"]]
    elif level == "MINIMAL":
        p["max_open_trades"] = 1
        p["max_spread_sl_ratio"] = min(p["max_spread_sl_ratio"], 0.03)
        p["symbols"] = [s for s in USD_MAJORS if p["symbols"] is None or s in p["symbols"]]
    p["level"] = level
    return p


class SessionManager:
    """Holds per-session baselines + graded levels; answers the session wall for execute_trade()."""

    def __init__(self, config: Dict, state_dir: Path = STATE_DIR):
        cfg = config.get("session_profiles", {})
        self.enabled = bool(cfg) and cfg.get("enabled", True)
        self.grading = {**GRADING, **cfg.get("grading", {})}
        self.base = {s: {**DEFAULT_PROFILE, **cfg.get(s, {})} for s in SESSIONS}
        self.state_file = Path(state_dir) / "session_levels.json"
        state = self._load()
        self.levels: Dict[str, str] = {s: state.get("levels", {}).get(s, "NORMAL") for s in SESSIONS}
        self.reviewed_day: Optional[str] = state.get("reviewed_day")
        self.last_review: Dict = state.get("last_review", {})
        self._today_cache = (0.0, [])

    def _load(self) -> Dict:
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({"levels": self.levels, "reviewed_day": self.reviewed_day,
                                               "last_review": self.last_review}, indent=2), encoding="utf-8")

    def profile(self, session: str) -> Dict:
        return effective_profile(self.base[session], self.levels.get(session, "NORMAL"))

    # daily grading
    def maybe_review(self, fetch_closed: Callable[[datetime], List[Dict]], now_utc: Optional[datetime] = None) -> Optional[Dict]:
        """Grade every session once per UTC day. fetch_closed(since_utc) -> closed trade records."""
        if not self.enabled:
            return None
        now_utc = now_utc or datetime.now(timezone.utc)
        day = now_utc.strftime("%Y-%m-%d")
        if self.reviewed_day == day:
            return None
        trades = fetch_closed(now_utc - timedelta(days=self.grading["review_days"]))
        stats = session_stats(trades)
        review = {}
        for s in SESSIONS:
            old = self.levels.get(s, "NORMAL")
            new = next_level(old, stats[s]["n"], stats[s]["avg_r"], self.grading)
            self.levels[s] = new
            review[s] = {**stats[s], "level_before": old, "level": new}
        self.reviewed_day, self.last_review = day, {"day": day, "sessions": review}
        self._save()
        logger.info("SESSION_REVIEW " + json.dumps(self.last_review))
        return self.last_review

    # session wall
    def _today_closed(self, fetch_closed, now_utc: datetime) -> List[Dict]:
        ts, data = self._today_cache
        if time.time() - ts > 60:
            midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            data = fetch_closed(midnight)
            self._today_cache = (time.time(), data)
        return data

    def check(self, symbol: str, spread_sl_ratio: Optional[float], open_positions_entry_hours: List[int],
              balance: float, fetch_closed: Callable[[datetime], List[Dict]],
              now_utc: Optional[datetime] = None) -> Optional[str]:
        """Return a rejection message, or None if the current session's profile allows the trade."""
        if not self.enabled:
            return None
        now_utc = now_utc or datetime.now(timezone.utc)
        session = session_for_hour(now_utc.hour)
        p = self.profile(session)
        tag = f"{session}/{p['level']}"
        if p["symbols"] is not None and symbol not in p["symbols"]:
            return f"REJECTED: Session profile {tag} does not trade {symbol}"
        in_session = sum(1 for h in open_positions_entry_hours if session_for_hour(h) == session)
        if in_session >= p["max_open_trades"]:
            return f"REJECTED: Session slot cap {tag} reached ({in_session}/{p['max_open_trades']} open)"
        if spread_sl_ratio is not None and spread_sl_ratio > p["max_spread_sl_ratio"]:
            return (f"REJECTED: Spread cost on {symbol} is {spread_sl_ratio:.1%} of SL distance "
                    f"(session {tag} max {p['max_spread_sl_ratio']:.0%})")
        budget = balance * p["loss_budget_pct"] / 100.0
        today = self._today_closed(fetch_closed, now_utc)
        net = sum(t.get("net_profit", 0.0) for t in today
                  if t.get("entry_hour_utc") is not None and session_for_hour(int(t["entry_hour_utc"])) == session)
        if net < 0 and -net > budget:
            return f"REJECTED: Session loss budget {tag} used (${-net:.2f} / ${budget:.2f})"
        return None
