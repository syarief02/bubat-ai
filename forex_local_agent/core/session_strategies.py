"""
Session-Specific Strategies
===========================
The main LLM trend strategy trades only inside `entry_hours_utc` (London). Other
sessions get their own deterministic execution method here.

ASIA: range fade on low-spread majors
-------------------------------------
Asian hours are range-bound with tiny M5 ranges, so instead of following the H1
trend it fades M5 Bollinger-band extremes:

  BUY  when the last completed M5 close < BB(20, dev) lower band and RSI(14) < rsi_low
  SELL when the last completed M5 close > upper band and RSI(14) > 100 - rsi_low
  SL = max(sl_atr * ATR(14), sl_floor_pips); TP = tp_r * SL; force-exit at exit_hour_utc

Backtest (180d M5, 5 majors): +0.05R/trade before costs but about 0R after a
realistic 0.3-1.0 pip cost, so it runs in SHADOW mode only: paper trades are
opened at the live bid/ask (real spread), resolved on M5 bars (SL wins ties),
and logged to state/asia_shadow_trades.jsonl. No order is ever sent from here.
Going live requires the owner's approval once the shadow record clears
`promote_after_trades` / `promote_min_avg_r`.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None

try:
    from core.mt5_time import get_server_utc_offset_seconds, market_open
    from core.trade_analytics import entry_window_open, pip_size_for
except ImportError:
    from forex_local_agent.core.mt5_time import get_server_utc_offset_seconds, market_open
    from forex_local_agent.core.trade_analytics import entry_window_open, pip_size_for

STATE_DIR = Path(__file__).resolve().parent.parent / "state"
# A signal bar older than this means the feed is stale (weekend, holiday, disconnected terminal)
MAX_SIGNAL_BAR_AGE_SECONDS = 15 * 60

DEFAULTS = {
    "enabled": True,
    "mode": "shadow",
    "hours_utc": [0, 6],
    "exit_hour_utc": 7,
    "symbols": ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDJPY"],
    "bb_period": 20,
    "bb_dev": 2.0,
    "rsi_low": 30,
    "sl_atr": 1.5,
    "sl_floor_pips": 8.0,
    "tp_r": 1.0,
    "promote_after_trades": 100,
    "promote_min_avg_r": 0.05,
}


def fade_signal(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                bb_period: int = 20, bb_dev: float = 2.0, rsi_low: float = 30) -> Optional[Dict]:
    """Signal on the LAST element of completed-bar arrays. Returns {direction, atr} or None."""
    if len(closes) < max(bb_period, 15) + 1:
        return None
    c = pd.Series(closes, dtype=float)
    prev = c.shift(1)
    tr = pd.concat([pd.Series(highs) - pd.Series(lows), (pd.Series(highs) - prev).abs(),
                    (pd.Series(lows) - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    rsi = 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)
    mid = c.rolling(bb_period).mean().iloc[-1]
    sd = c.rolling(bb_period).std().iloc[-1]
    last = c.iloc[-1]
    if last < mid - bb_dev * sd and rsi < rsi_low:
        return {"direction": 1, "atr": atr, "rsi": round(float(rsi), 1)}
    if last > mid + bb_dev * sd and rsi > 100 - rsi_low:
        return {"direction": -1, "atr": atr, "rsi": round(float(rsi), 1)}
    return None


def build_paper_trade(symbol: str, direction: int, atr: float, bid: float, ask: float, pip: float,
                      cfg: Dict, now_utc: datetime, bar_time_utc: float) -> Dict:
    """Entry at the live ask (BUY) / bid (SELL), so the real spread is paid."""
    entry = ask if direction > 0 else bid
    sl_d = max(cfg["sl_atr"] * atr, cfg["sl_floor_pips"] * pip)
    return {
        "symbol": symbol, "direction": direction, "opened_at": now_utc.isoformat(),
        "entry_ts": now_utc.timestamp(), "signal_bar_ts": bar_time_utc, "entry": entry,
        "sl": entry - direction * sl_d, "tp": entry + direction * cfg["tp_r"] * sl_d,
        "sl_pips": round(sl_d / pip, 2), "spread_pips": round((ask - bid) / pip, 2),
    }


def resolve_paper_trade(trade: Dict, t_utc: np.ndarray, opens: np.ndarray, highs: np.ndarray,
                        lows: np.ndarray, exit_hour_utc: int) -> Optional[Dict]:
    """Walk M5 bars that OPEN after entry. SL wins ties; force-exit at the open of the exit-hour bar.

    Bars are bid prices: a SELL's exits are approximated on bid (spread already paid at entry).
    Returns {exit_reason, exit_price, r, closed_ts} or None while still open.
    """
    d, entry, sl, tp = trade["direction"], trade["entry"], trade["sl"], trade["tp"]
    risk = abs(entry - sl)
    for k in range(len(t_utc)):
        if t_utc[k] < trade["entry_ts"]:
            continue
        hour = datetime.fromtimestamp(t_utc[k], tz=timezone.utc).hour
        if hour == exit_hour_utc and t_utc[k] - trade["entry_ts"] < 20 * 3600:
            px = float(opens[k])
            return {"exit_reason": "SESSION_END", "exit_price": px, "r": round((px - entry) * d / risk, 3),
                    "closed_ts": float(t_utc[k])}
        adverse = lows[k] if d > 0 else highs[k]
        fav = highs[k] if d > 0 else lows[k]
        if (adverse - sl) * d <= 0:
            return {"exit_reason": "SL", "exit_price": sl, "r": -1.0, "closed_ts": float(t_utc[k])}
        if (fav - tp) * d >= 0:
            return {"exit_reason": "TP", "exit_price": tp, "r": round(abs(tp - entry) / risk, 3),
                    "closed_ts": float(t_utc[k])}
    return None


def summarize_shadow(results: List[Dict], cfg: Dict) -> Dict:
    rs = [r["r"] for r in results if r.get("r") is not None]
    out = {"trades": len(rs), "avg_r": round(float(np.mean(rs)), 3) if rs else None,
           "sum_r": round(float(np.sum(rs)), 2) if rs else 0.0,
           "win_rate_pct": round(100 * float(np.mean([r > 0 for r in rs])), 1) if rs else None}
    out["promotion_ready"] = bool(rs) and len(rs) >= cfg["promote_after_trades"] and out["avg_r"] >= cfg["promote_min_avg_r"]
    return out


class AsiaRangeFadeShadow:
    """Paper-trades the Asia range-fade strategy each cycle. Never sends orders."""

    def __init__(self, config: Dict, state_dir: Path = STATE_DIR):
        self.cfg = {**DEFAULTS, **(config.get("session_strategies", {}).get("asia_range_fade", {}))}
        if self.cfg["mode"] != "shadow":
            logger.warning("asia_range_fade: only 'shadow' mode is implemented; running as shadow.")
            self.cfg["mode"] = "shadow"
        self.state_dir = Path(state_dir)
        self.open_file = self.state_dir / "asia_shadow_open.json"
        self.results_file = self.state_dir / "asia_shadow_trades.jsonl"
        self.open_trades: Dict[str, Dict] = self._load_open()
        self.last_signal_bar: Dict[str, float] = {}

    # persistence
    def _load_open(self) -> Dict[str, Dict]:
        try:
            return json.loads(self.open_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_open(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.open_file.write_text(json.dumps(self.open_trades), encoding="utf-8")

    def _append_result(self, rec: Dict):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.results_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def load_results(self) -> List[Dict]:
        try:
            return [json.loads(x) for x in self.results_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        except Exception:
            return []

    # MT5 access (bars are converted to real-UTC epochs)
    def _bars(self, symbol: str, count: int = 120):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
        if rates is None or len(rates) < 30:
            return None
        off = get_server_utc_offset_seconds()
        return {"t": rates["time"].astype(float) - off, "open": rates["open"], "high": rates["high"],
                "low": rates["low"], "close": rates["close"]}

    def run_cycle(self, now_utc: Optional[datetime] = None) -> Dict:
        """Resolve open paper trades, then (inside the Asia window) open new ones. Returns cycle stats."""
        stats = {"resolved": 0, "opened": 0, "open": 0}
        if not self.cfg["enabled"] or mt5 is None:
            return stats
        now_utc = now_utc or datetime.now(timezone.utc)

        for symbol, tr in list(self.open_trades.items()):
            b = self._bars(symbol, 300)
            if b is None:
                continue
            done = b["t"] < now_utc.timestamp() - 300  # completed bars only
            res = resolve_paper_trade(tr, b["t"][done], b["open"][done], b["high"][done], b["low"][done],
                                      self.cfg["exit_hour_utc"])
            if res:
                rec = {**tr, **res, "strategy": "asia_range_fade", "mode": "shadow"}
                self._append_result(rec)
                del self.open_trades[symbol]
                stats["resolved"] += 1
                logger.info(f"[ASIA-SHADOW] {symbol} paper trade closed: {res['exit_reason']} {res['r']:+.2f}R")

        if market_open(now_utc) and entry_window_open(now_utc.hour, self.cfg["hours_utc"]):
            for symbol in self.cfg["symbols"]:
                if symbol in self.open_trades:
                    continue
                b = self._bars(symbol)
                if b is None:
                    continue
                done = b["t"] < now_utc.timestamp() - 300
                if not done.any() or self.last_signal_bar.get(symbol) == b["t"][done][-1]:
                    continue
                if now_utc.timestamp() - b["t"][done][-1] > MAX_SIGNAL_BAR_AGE_SECONDS:
                    continue
                sig = fade_signal(b["high"][done], b["low"][done], b["close"][done],
                                  self.cfg["bb_period"], self.cfg["bb_dev"], self.cfg["rsi_low"])
                self.last_signal_bar[symbol] = b["t"][done][-1]
                if not sig:
                    continue
                tick = mt5.symbol_info_tick(symbol)
                info = mt5.symbol_info(symbol)
                if tick is None or info is None:
                    continue
                pip = pip_size_for(symbol, info.point, info.digits)
                tr = build_paper_trade(symbol, sig["direction"], sig["atr"], tick.bid, tick.ask, pip,
                                       self.cfg, now_utc, float(b["t"][done][-1]))
                self.open_trades[symbol] = tr
                stats["opened"] += 1
                logger.info(f"[ASIA-SHADOW] {symbol} paper {'BUY' if tr['direction'] > 0 else 'SELL'} @ {tr['entry']} "
                            f"SL {tr['sl_pips']}p spread {tr['spread_pips']}p (RSI {sig['rsi']})")

        self._save_open()
        stats["open"] = len(self.open_trades)
        return stats
