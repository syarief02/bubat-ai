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


_WALLS = (
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
