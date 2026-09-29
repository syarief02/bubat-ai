"""
Market Scanner & Multi-Pair Technical Ranking Engine
====================================================
Connects directly to MetaTrader 5, scans all major and minor forex pairs,
calculates indicators (RSI, ATR, EMAs, 24h change), checks active trading sessions,
and produces a ranked leaderboard of the best setups right now.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
from loguru import logger

try:
    from core.web_surfer import WebSurfer
except ImportError:
    from forex_local_agent.core.web_surfer import WebSurfer

DEFAULT_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "NZDUSD", "EURJPY", "GBPJPY", "XAUUSD"
]

class MarketScanner:
    """Scans and ranks live Forex pairs via MetaTrader 5."""

    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.surfer = WebSurfer()

    def scan_and_rank(self) -> Dict[str, Any]:
        """
        Scan all pairs, compute technical metrics, evaluate session context,
        and rank them from best opportunity to lowest.
        """
        session_info = self.surfer.get_current_market_session()
        recommended_pairs = session_info.get("best_pairs_for_session", [])

        if not mt5.initialize():
            logger.error("MarketScanner: Could not initialize MetaTrader 5.")
            return {"status": "error", "message": "MetaTrader 5 terminal not connected."}

        ranked_list = []

        try:
            for sym in self.symbols:
                info = mt5.symbol_info(sym)
                if not info:
                    continue
                if not info.visible:
                    mt5.symbol_select(sym, True)

                tick = mt5.symbol_info_tick(sym)
                if not tick:
                    continue

                # Copy 50 H1 candles for indicator calculation
                rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 0, 50)
                if rates is None or len(rates) < 25:
                    continue

                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")

                # Indicators
                rsi_series = ta.rsi(df["close"], length=14)
                atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
                ema20_series = ta.ema(df["close"], length=20)
                ema50_series = ta.ema(df["close"], length=50)

                current_rsi = round(float(rsi_series.iloc[-1]), 1) if rsi_series is not None and not rsi_series.empty else 50.0
                current_atr = round(float(atr_series.iloc[-1]), 5) if atr_series is not None and not atr_series.empty else 0.001
                last_close = float(df["close"].iloc[-1])
                open_24h_ago = float(df["close"].iloc[0]) if len(df) >= 24 else float(df["close"].iloc[0])
                change_24h_pct = round(((last_close - open_24h_ago) / open_24h_ago) * 100, 2)

                ema20 = float(ema20_series.iloc[-1]) if ema20_series is not None and not ema20_series.empty else last_close
                ema50 = float(ema50_series.iloc[-1]) if ema50_series is not None and not ema50_series.empty else last_close

                # Technical Bias & Structure
                if last_close > ema20 > ema50:
                    bias = "BULLISH (Uptrend above 20/50 EMA)"
                    trend_score = 25
                    trade_bias = "BUY"
                elif last_close < ema20 < ema50:
                    bias = "BEARISH (Downtrend below 20/50 EMA)"
                    trend_score = 25
                    trade_bias = "SELL"
                else:
                    bias = "NEUTRAL (Consolidating / Rangebound)"
                    trend_score = 10
                    trade_bias = "WAIT"

                # Opportunity Score (Base 40)
                score = 40 + trend_score

                # Session alignment bonus (+15)
                if sym in recommended_pairs:
                    score += 15

                # Momentum / Overbought / Oversold bonus (+15)
                if current_rsi <= 35:
                    score += 15
                    trade_bias = "BUY (Oversold Dip)" if "BULLISH" in bias else "SELL (Strong Downside Momentum)"
                elif current_rsi >= 65:
                    score += 15
                    trade_bias = "SELL (Overbought Peak)" if "BEARISH" in bias else "BUY (Strong Upside Momentum)"

                # Strong intraday movement (+10)
                if abs(change_24h_pct) >= 0.25:
                    score += 10

                # Spread penalty (points)
                if info.spread > 30:
                    score -= 10

                ranked_list.append({
                    "symbol": sym,
                    "price": round(tick.bid, 5),
                    "spread_points": info.spread,
                    "rsi": current_rsi,
                    "atr": current_atr,
                    "change_24h_pct": change_24h_pct,
                    "technical_bias": bias,
                    "suggested_action": trade_bias,
                    "opportunity_score": score
                })

        finally:
            mt5.shutdown()

        # Sort by opportunity score descending
        ranked_list.sort(key=lambda x: x["opportunity_score"], reverse=True)
        for rank, item in enumerate(ranked_list, 1):
            item["rank"] = rank

        return {
            "status": "success",
            "session": session_info,
            "total_pairs_scanned": len(ranked_list),
            "rankings": ranked_list
        }

    def format_rankings_text(self, scan_result: Dict[str, Any]) -> str:
        """Format rankings into a readable markdown report."""
        if scan_result.get("status") != "success":
            return f"Market scan failed: {scan_result.get('message', 'Unknown error')}"

        session = scan_result.get("session", {})
        rankings = scan_result.get("rankings", [])

        lines = [
            f"=== LIVE FOREX MARKET SESSION SCAN ===",
            f"UTC Time: {session.get('utc_time')}",
            f"Active Session: {session.get('session_summary')}",
            f"Pairs Evaluated: {scan_result.get('total_pairs_scanned')}",
            "",
            "RANKED FOREX PAIRS (Best Setup to Lowest):"
        ]

        for p in rankings:
            lines.append(
                f"#{p['rank']} {p['symbol']} | Score: {p['opportunity_score']}/100 | "
                f"Action: {p['suggested_action']}\n"
                f"   Price: {p['price']} | RSI(14): {p['rsi']} | ATR: {p['atr']} | "
                f"24h: {p['change_24h_pct']:+.2f}% | Spread: {p['spread_points']} pts\n"
                f"   Structure: {p['technical_bias']}"
            )

        return "\n".join(lines)
