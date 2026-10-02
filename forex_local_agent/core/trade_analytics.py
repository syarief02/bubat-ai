"""
Trade Analytics Helpers
=======================
Pure functions shared by the live engine (mt5_engine.py, main.py) and the daily
post-mortem report (maintenance/daily_report.py). No MT5 calls here, so all of
it is unit-testable offline (tests/test_daily_report.py).
"""

from typing import Dict, List, Optional

import numpy as np

DEAL_REASON = {0: "CLIENT", 1: "MOBILE", 2: "WEB", 3: "EXPERT", 4: "SL", 5: "TP", 6: "STOP_OUT"}
MT5_REASON_SL = 4
MT5_REASON_TP = 5


def pip_size_for(symbol: str, point: float, digits: int) -> float:
    return point * 10 if digits in (3, 5) else point


def session_for_hour(hour_utc: int) -> str:
    if 7 <= hour_utc < 12:
        return "LONDON"
    if 12 <= hour_utc < 16:
        return "LONDON_NY_OVERLAP"
    if 16 <= hour_utc < 21:
        return "NEW_YORK"
    if hour_utc == 21:
        return "ROLLOVER"
    return "ASIA"


def confidence_bucket(conf: Optional[float]) -> str:
    if conf is None:
        return "unknown"
    if conf < 0.80:
        return "<0.80"
    if conf < 0.85:
        return "0.80-0.84"
    if conf < 0.90:
        return "0.85-0.89"
    return ">=0.90"


def spread_bucket(spread_pips: Optional[float]) -> str:
    if spread_pips is None:
        return "unknown"
    for hi, label in ((0.5, "<0.5p"), (1.0, "0.5-1p"), (2.0, "1-2p"), (3.5, "2-3.5p")):
        if spread_pips < hi:
            return label
    return ">=3.5p"


def classify_exit(reason: int, exit_price: float, entry_price: float, orig_sl: float,
                  direction: int, pip: float) -> str:
    """Exit type from MT5 deal reason + prices. direction: +1 BUY, -1 SELL.

    SL-reason exits are split by where the stop sat: at the original SL (full loss),
    near entry (break-even lock), or beyond +3 pips (trailing stop).
    """
    if reason == MT5_REASON_TP:
        return "TP"
    if reason == MT5_REASON_SL:
        if orig_sl and abs(exit_price - orig_sl) <= 0.6 * pip:
            return "SL_FULL"
        pips = (exit_price - entry_price) * direction / pip
        if -0.5 <= pips <= 3.0:
            return "BREAK_EVEN"
        if pips > 3.0:
            return "TRAILING"
        return "SL_MODIFIED"
    return DEAL_REASON.get(reason, f"REASON_{reason}")


def max_drawdown(pnls: List[float]) -> float:
    """Peak-to-trough drawdown of the cumulative P&L curve (starting at 0). Returns <= 0."""
    peak, cum, mdd = 0.0, 0.0, 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return mdd


def simulate_path(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, direction: int,
                  entry: float, sl: float, tp: float) -> Dict:
    """Walk bars forward; return outcome in R. SL wins ties inside one bar (conservative)."""
    risk = abs(entry - sl)
    if risk <= 0 or len(highs) == 0:
        return {"outcome": "NO_DATA", "r": None, "bars": 0}
    if direction > 0:
        hit_sl = lows <= sl
        hit_tp = highs >= tp
    else:
        hit_sl = highs >= sl
        hit_tp = lows <= tp
    i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else None
    i_tp = int(np.argmax(hit_tp)) if hit_tp.any() else None
    if i_sl is not None and (i_tp is None or i_sl <= i_tp):
        return {"outcome": "SL", "r": -1.0, "bars": i_sl + 1}
    if i_tp is not None:
        return {"outcome": "TP", "r": round(abs(tp - entry) / risk, 3), "bars": i_tp + 1}
    mtm = (closes[-1] - entry) * direction / risk
    return {"outcome": "OPEN", "r": round(float(mtm), 3), "bars": len(highs)}


def spread_to_sl_ratio(ask: float, bid: float, sl_distance: float) -> Optional[float]:
    """Spread as a fraction of the SL distance (price units). None if the SL is unknown."""
    if not sl_distance or sl_distance <= 0:
        return None
    return (ask - bid) / sl_distance


def entry_window_open(hour_utc: int, window: Optional[List[int]]) -> bool:
    """window = [start_hour, end_hour) in UTC; None/empty means always open. Wraps midnight if start > end."""
    if not window:
        return True
    start, end = int(window[0]), int(window[1])
    if start <= end:
        return start <= hour_utc < end
    return hour_utc >= start or hour_utc < end


def h1_strength(h1_label: str) -> int:
    """2 = full trend (price and EMA20 on the same side of EMA50), 1 = bias only, 0 = neutral/unknown."""
    lab = (h1_label or "").upper()
    if "BIAS" in lab:
        return 1
    if "BULLISH" in lab or "BEARISH" in lab:
        return 2
    return 0


def quick_reentries(trades: List[Dict], within_seconds: int = 1800) -> List[Dict]:
    """Trades opened on a symbol within `within_seconds` of the previous trade on that symbol closing."""
    by_symbol: Dict[str, List[Dict]] = {}
    for t in sorted(trades, key=lambda t: t["entry_ts"]):
        by_symbol.setdefault(t["symbol"], []).append(t)
    out = []
    for seq in by_symbol.values():
        for prev, cur in zip(seq, seq[1:]):
            if 0 <= cur["entry_ts"] - prev["exit_ts"] < within_seconds:
                out.append(cur)
    return out


def rank_signals(candidates: List[Dict]) -> List[Dict]:
    """Order a cycle's tradeable signals best-first for the limited open-trade slots.

    Full H1 trend before bias-only, then cheapest spread relative to SL. Signals
    without a known spread ratio go last. Stable for ties (config order).
    """
    def key(c):
        ratio = c.get("spread_sl_ratio")
        return (-h1_strength(c.get("h1_trend", "")), ratio if ratio is not None else float("inf"))
    return sorted(candidates, key=key)


_WALLS = (
    ("entry window", "SESSION_WINDOW"), ("calibration cap", "CONFIDENCE_CAP"), ("spread cost", "SPREAD_COST"),
    ("counter-trend", "H1_TREND"), ("daily loss", "DAILY_LOSS_STOP"), ("max open trades", "CAPACITY"),
    ("already open", "DUPLICATE"), ("news blackout", "NEWS_BLACKOUT"), ("currency concentration", "CORRELATION"),
    ("margin", "MARGIN"), ("spread", "SPREAD"), ("xauusd", "GOLD_BALANCE"), ("risk at sl", "RISK_CAP"),
    ("exceeds max drawdown", "RISK_CAP"),
)


def wall_from_message(msg: str) -> str:
    """Map an execute_trade() rejection message to a risk-wall label."""
    m = (msg or "").lower()
    for needle, label in _WALLS:
        if needle in m:
            return label
    return "OTHER"
