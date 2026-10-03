"""
MT5 Engine — Deterministic Math Calculator & Trade Executor
=============================================================
Separates Qualitative Reasoning (LLM) from Quantitative Math (Python).
The LLM evaluates market sentiment and outputs BUY/SELL/WAIT.
This engine calculates exact Entry, SL, TP, and Lot Size using ATR and account balance.
"""

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import json
import os
import math
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from loguru import logger

try:
    from learning.skills.economic_calendar_filter import is_trade_permitted_by_calendar
except ImportError:
    try:
        from forex_local_agent.learning.skills.economic_calendar_filter import is_trade_permitted_by_calendar
    except ImportError:
        is_trade_permitted_by_calendar = None

try:
    from core.trade_analytics import classify_exit, spread_to_sl_ratio, entry_window_open
    from core.session_profiles import SessionManager
except ImportError:
    from forex_local_agent.core.trade_analytics import classify_exit, spread_to_sl_ratio, entry_window_open
    from forex_local_agent.core.session_profiles import SessionManager

try:
    from core.mt5_time import history_deals_utc, server_epoch_to_utc, get_server_utc_offset_seconds
except ImportError:
    from forex_local_agent.core.mt5_time import history_deals_utc, server_epoch_to_utc, get_server_utc_offset_seconds

try:
    from learning.skills.currency_correlation_filter import is_trade_permitted_by_correlation
except ImportError:
    try:
        from forex_local_agent.learning.skills.currency_correlation_filter import is_trade_permitted_by_correlation
    except ImportError:
        is_trade_permitted_by_correlation = None


class MT5Engine:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        log_dir = self.config_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Note: trades.log logger sink is configured in main.py — do not add here to prevent duplicate entries

        self.tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        # Risk parameters
        risk = self.config.get("risk_parameters", {})
        self.lot_mode = risk.get("lot_mode", "fixed")
        self.fixed_lot = risk.get("fixed_lot", 0.01)
        self.max_lot_size = risk.get("max_lot_size", 0.1)
        self.max_drawdown_pct = risk.get("max_drawdown_pct", 2.0)
        self.max_open_trades = risk.get("max_open_trades", 10)
        self.risk_per_trade_pct = risk.get("risk_per_trade_pct", 1.5)
        self.atr_period = risk.get("atr_period", 14)
        self.atr_multiplier_sl = risk.get("atr_multiplier_sl", 1.5)
        self.atr_multiplier_tp = risk.get("atr_multiplier_tp", 3.0)
        self.min_sl_pips = risk.get("min_sl_pips", 15.0)
        self.max_spread_pips = risk.get("max_spread_pips", 3.5)
        self.trailing_enabled = risk.get("trailing_stop_enabled", True)
        self.breakeven_trigger_pips = risk.get("trailing_breakeven_pips", 10.0)
        self.breakeven_lock_pips = risk.get("trailing_breakeven_lock_pips", 1.0)
        self.trailing_start_pips = risk.get("trailing_start_pips", 15.0)
        self.trailing_distance_pips = risk.get("trailing_distance_pips", 10.0)
        # Minimum SL improvement before a trailing modification is sent
        self.trailing_step_pips = risk.get("trailing_step_pips", 2.0)
        self.daily_loss_limit_pct = risk.get("daily_loss_limit_pct")
        self.global_news_blackout = risk.get("news_blackout_global_tier1", True)
        # Execution-quality walls (cycle #6). None disables each one.
        self.max_spread_sl_ratio = risk.get("max_spread_sl_ratio")
        self.max_confidence = risk.get("max_confidence")
        self.entry_hours_utc = risk.get("entry_hours_utc")
        # Per-session execution profiles + daily grading (core/session_profiles.py)
        self.session_manager = SessionManager(self.config)
        self.min_account_balance_gold = 300.0

        # Trading settings
        trading = self.config.get("trading", {})
        self.default_timeframe = trading.get("timeframe", "H1")
        self.analysis_bars = trading.get("analysis_bars", 100)
        self.order_comment = trading.get("order_comment", "Bubat AI")

    def initialize(self) -> bool:
        # Read MT5 credentials from env vars first, config.json as fallback
        creds = self.config.get("mt5_credentials", {})
        path = os.environ.get("MT5_PATH") or creds.get("path", r"C:\Program Files\MetaTrader 5\terminal64.exe")
        login = os.environ.get("MT5_LOGIN") or creds.get("login")
        server = os.environ.get("MT5_SERVER") or creds.get("server")
        password = os.environ.get("MT5_PASSWORD") or creds.get("password")

        init_kwargs = {}
        if path and Path(path).exists():
            init_kwargs["path"] = path
        if login and login != 0:
            init_kwargs["login"] = int(login)
        if server:
            init_kwargs["server"] = str(server)
        if password:
            init_kwargs["password"] = str(password)

        ok = False
        if init_kwargs:
            ok = mt5.initialize(**init_kwargs)
        if not ok:
            ok = mt5.initialize()

        if not ok:
            logger.error(f"MT5 initialization failed, error code = {mt5.last_error()}")
            return False

        account = mt5.account_info()
        if account:
            logger.info(f"MT5 initialized successfully: Account {account.login} ({account.server}), Balance: {account.balance} {account.currency}")
        else:
            logger.info("MT5 initialized successfully")
        return True

    def shutdown(self):
        mt5.shutdown()
        logger.info("MT5 shutdown")

    def reconnect(self) -> bool:
        for attempt in range(3):
            logger.info(f"Reconnection attempt {attempt + 1}")
            self.shutdown()
            time.sleep(5)
            if self.initialize():
                return True
        logger.error("Failed to reconnect after 3 attempts")
        return False

    def get_account_info(self) -> Dict:
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return {}
        return {
            "balance": float(account_info.balance),
            "equity": float(account_info.equity),
            "margin": float(account_info.margin),
            "free_margin": float(account_info.margin_free),
        }

    def get_technical_data(self, symbol: str, timeframe: str = "H1", bars: int = 100) -> Dict:
        tf = self.tf_map.get(timeframe, mt5.TIMEFRAME_H1)
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None:
            logger.error(f"Failed to get rates for {symbol}")
            return {}

        df = pd.DataFrame(rates)
        # Format time to ISO string to guarantee JSON serializability
        df["time"] = pd.to_datetime(df["time"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")

        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.atr(length=self.atr_period, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)

        last_row = df.iloc[-1]
        close_price = float(last_row["close"])

        macd_cols = [c for c in df.columns if c.startswith("MACD_")]
        macdh_cols = [c for c in df.columns if c.startswith("MACDh_")]
        macds_cols = [c for c in df.columns if c.startswith("MACDs_")]
        rsi_col = [c for c in df.columns if c.startswith("RSI_")]
        atr_col = [c for c in df.columns if c.startswith("ATRr_")]
        ema_20_col = [c for c in df.columns if c.startswith("EMA_20")]
        ema_50_col = [c for c in df.columns if c.startswith("EMA_50")]

        # Safe float conversion
        def to_float(val):
            return float(val) if val is not None and not pd.isna(val) else None

        ema_20_val = to_float(last_row[ema_20_col[0]]) if ema_20_col else None
        ema_50_val = to_float(last_row[ema_50_col[0]]) if ema_50_col else None
        rsi_val = to_float(last_row[rsi_col[0]]) if rsi_col else None
        macd_val = to_float(last_row[macd_cols[0]]) if macd_cols else None
        macds_val = to_float(last_row[macds_cols[0]]) if macds_cols else None
        macdh_val = to_float(last_row[macdh_cols[0]]) if macdh_cols else None

        # 24h change % reference
        open_price_ref = float(df.iloc[0]["open"])
        change_pct = round(((close_price - open_price_ref) / open_price_ref) * 100, 2)

        # Structure analysis
        if ema_20_val and ema_50_val:
            if close_price > ema_20_val and ema_20_val > ema_50_val:
                trend_structure = "STRONG BULLISH (Uptrend above 20 & 50 EMA)"
                bias = "STRONG BULLISH"
            elif close_price < ema_20_val and ema_20_val < ema_50_val:
                trend_structure = "STRONG BEARISH (Downtrend below 20 & 50 EMA)"
                bias = "STRONG BEARISH"
            elif close_price > ema_20_val:
                trend_structure = "BULLISH BIAS (Above 20 EMA)"
                bias = "BULLISH"
            elif close_price < ema_20_val:
                trend_structure = "BEARISH BIAS (Below 20 EMA)"
                bias = "BEARISH"
            else:
                trend_structure = "NEUTRAL (Consolidating around EMAs)"
                bias = "NEUTRAL"
        else:
            trend_structure = "RANGING"
            bias = "NEUTRAL"

        if rsi_val is not None:
            if rsi_val <= 35:
                rsi_condition = f"OVERSOLD / STRONG DOWNSIDE MOMENTUM (RSI = {rsi_val:.1f})"
            elif rsi_val >= 65:
                rsi_condition = f"OVERBOUGHT / STRONG UPSIDE MOMENTUM (RSI = {rsi_val:.1f})"
            elif rsi_val < 50:
                rsi_condition = f"BEARISH PRESSURE (RSI = {rsi_val:.1f})"
            else:
                rsi_condition = f"BULLISH PRESSURE (RSI = {rsi_val:.1f})"
        else:
            rsi_condition = "N/A"

        if macd_val is not None and macds_val is not None:
            if macd_val > macds_val:
                macd_condition = "BULLISH (MACD line above signal line)"
            else:
                macd_condition = "BEARISH (MACD line below signal line)"
        else:
            macd_condition = "N/A"

        last_5 = df.tail(5)[["time", "open", "high", "low", "close"]].to_dict(orient="records")
        for record in last_5:
            record["open"] = float(record["open"])
            record["high"] = float(record["high"])
            record["low"] = float(record["low"])
            record["close"] = float(record["close"])

        # Multi-timeframe trend context: Higher Timeframe (H1) Bias
        htf_trend = "NEUTRAL"
        if timeframe != "H1":
            try:
                h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 60)
                if h1_rates is not None and len(h1_rates) >= 50:
                    h1_df = pd.DataFrame(h1_rates)
                    h1_df.ta.ema(length=20, append=True)
                    h1_df.ta.ema(length=50, append=True)
                    h1_last = h1_df.iloc[-1]
                    h1_close = float(h1_last["close"])
                    h1_cols20 = [c for c in h1_df.columns if c.startswith("EMA_20")]
                    h1_cols50 = [c for c in h1_df.columns if c.startswith("EMA_50")]
                    if h1_cols20 and h1_cols50:
                        h1_e20 = to_float(h1_last[h1_cols20[0]])
                        h1_e50 = to_float(h1_last[h1_cols50[0]])
                        if h1_e20 and h1_e50:
                            if h1_close > h1_e20 and h1_e20 > h1_e50:
                                htf_trend = "BULLISH (H1 Uptrend above 20 & 50 EMA)"
                            elif h1_close < h1_e20 and h1_e20 < h1_e50:
                                htf_trend = "BEARISH (H1 Downtrend below 20 & 50 EMA)"
                            elif h1_close > h1_e20:
                                htf_trend = "BULLISH BIAS (Above H1 20 EMA)"
                            elif h1_close < h1_e20:
                                htf_trend = "BEARISH BIAS (Below H1 20 EMA)"
            except Exception as e:
                logger.debug(f"[{symbol}] Could not fetch H1 HTF trend: {e}")

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": close_price,
            "trend_structure": trend_structure,
            "technical_bias": bias,
            "higher_timeframe_h1": htf_trend,
            "rsi": round(rsi_val, 1) if rsi_val is not None else None,
            "rsi_condition": rsi_condition,
            "macd": round(macd_val, 6) if macd_val is not None else None,
            "macd_signal": round(macds_val, 6) if macds_val is not None else None,
            "macd_hist": round(macdh_val, 6) if macdh_val is not None else None,
            "macd_condition": macd_condition,
            "atr": to_float(last_row[atr_col[0]]) if atr_col else None,
            "change_pct": change_pct,
            "high_range": float(df["high"].max()),
            "low_range": float(df["low"].min()),
            "last_5_candles": last_5,
        }

    def calculate_trade_parameters(self, symbol: str, decision: str) -> Dict:
        """
        Deterministically calculate Entry, SL, TP, and Lot Size using ATR and account data.
        """
        logger.info(f"[MATH ENGINE] Calculating trade parameters for {decision} {symbol}")

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"status": "error", "message": f"Symbol {symbol} not found in MT5"}

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return {"status": "error", "message": f"Failed to select symbol {symbol}"}

        tech_data = self.get_technical_data(symbol, self.default_timeframe, self.analysis_bars)
        atr_value = tech_data.get("atr")
        if atr_value is None or atr_value <= 0:
            return {"status": "error", "message": f"Invalid ATR value: {atr_value}"}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"status": "error", "message": f"Could not get tick for {symbol}"}

        point = symbol_info.point
        digits = symbol_info.digits
        pip_size = point * 10 if digits in (3, 5) else point

        sl_distance = atr_value * self.atr_multiplier_sl
        tp_distance = atr_value * self.atr_multiplier_tp

        # Enforce minimum Stop Loss floor so broker spread and noise don't immediately wipe trades out
        min_sl_dist = self.min_sl_pips * pip_size if symbol != "XAUUSD" else 3.50
        if sl_distance < min_sl_dist:
            logger.info(f"[{symbol}] ATR SL distance ({sl_distance:.5f}) below minimum floor ({min_sl_dist:.5f}). Widening SL to protect against spread noise.")
            sl_distance = min_sl_dist
            tp_distance = max(tp_distance, sl_distance * 1.5)

        # Check broker minimum stops level
        stops_level_points = symbol_info.trade_stops_level
        min_stop_distance = stops_level_points * point

        if min_stop_distance > 0 and sl_distance < min_stop_distance:
            logger.warning(f"ATR SL distance ({sl_distance}) is below broker min stop level ({min_stop_distance}). Widening SL.")
            sl_distance = min_stop_distance

        if min_stop_distance > 0 and tp_distance < min_stop_distance:
            tp_distance = min_stop_distance

        if decision.upper() == "BUY":
            entry_price = float(tick.ask)
            sl_price = round(entry_price - sl_distance, digits)
            tp_price = round(entry_price + tp_distance, digits)
            order_type = mt5.ORDER_TYPE_BUY
        elif decision.upper() == "SELL":
            entry_price = float(tick.bid)
            sl_price = round(entry_price + sl_distance, digits)
            tp_price = round(entry_price - tp_distance, digits)
            order_type = mt5.ORDER_TYPE_SELL
        else:
            return {"status": "error", "message": f"Invalid decision: {decision}"}

        # Lot sizing: Fixed or Dynamic Risk
        account = self.get_account_info()
        if not account:
            return {"status": "error", "message": "Failed to get account info"}

        balance = account["balance"]
        risk_amount = balance * (self.risk_per_trade_pct / 100.0)

        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size

        if tick_size <= 0 or tick_value <= 0:
            return {"status": "error", "message": "Invalid tick size or tick value from broker"}

        sl_distance_ticks = sl_distance / tick_size
        loss_per_lot = sl_distance_ticks * tick_value

        vol_min = symbol_info.volume_min
        vol_max = symbol_info.volume_max
        vol_step = symbol_info.volume_step

        if self.lot_mode == "fixed":
            final_lot = max(vol_min, min(self.fixed_lot, vol_max, self.max_lot_size))
        else:
            if loss_per_lot <= 0:
                return {"status": "error", "message": "Invalid loss calculation per lot"}
            raw_lot = risk_amount / loss_per_lot
            if vol_step > 0:
                raw_lot = math.floor(raw_lot / vol_step) * vol_step
            final_lot = max(vol_min, min(raw_lot, vol_max, self.max_lot_size))

        final_lot = round(final_lot, 2)
        # Actual money lost if the SL is hit at this lot size (what the risk cap must check)
        actual_risk = round(loss_per_lot * final_lot, 2)

        return {
            "status": "calculated",
            "symbol": symbol,
            "action": decision.upper(),
            "order_type": order_type,
            "entry": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "lot": final_lot,
            "atr": round(atr_value, digits),
            "sl_distance": round(sl_distance, digits),
            "tp_distance": round(tp_distance, digits),
            "risk_amount": actual_risk,
            "risk_budget": round(risk_amount, 2),
            "risk_reward_ratio": round(tp_distance / sl_distance, 2) if sl_distance > 0 else None,
        }

    def current_spread_sl_ratio(self, symbol: str, sl_distance: float) -> Optional[float]:
        """Live spread as a fraction of the SL distance; used to rank a cycle's signals."""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            return spread_to_sl_ratio(tick.ask, tick.bid, sl_distance)
        except Exception:
            return None

    def execute_trade(self, trade_params: Dict) -> Dict:
        symbol = trade_params["symbol"]
        action = trade_params["action"]
        order_type = trade_params["order_type"]
        entry_price = trade_params["entry"]
        sl = trade_params["sl"]
        tp = trade_params["tp"]
        final_lot = trade_params["lot"]

        logger.info(f"[EXECUTION] Executing {action} {symbol} lot={final_lot} entry={entry_price} sl={sl} tp={tp}")

        # Entry Window Wall: only open new trades inside the configured UTC hours
        hour_utc = datetime.now(timezone.utc).hour
        if not entry_window_open(hour_utc, self.entry_hours_utc):
            msg = (f"REJECTED: Outside entry window for {symbol} "
                   f"({self.entry_hours_utc[0]:02d}:00-{self.entry_hours_utc[1]:02d}:00 UTC, now {hour_utc:02d}h)")
            logger.warning(msg)
            return {"status": "rejected", "message": msg}

        # Confidence Calibration Cap: LLM confidence above this has been anti-predictive
        confidence = trade_params.get("confidence")
        if self.max_confidence is not None and confidence is not None and confidence > self.max_confidence:
            msg = f"REJECTED: Confidence {confidence:.2f} on {symbol} exceeds calibration cap {self.max_confidence:.2f}"
            logger.warning(msg)
            return {"status": "rejected", "message": msg}

        # H1 MTF Trend Confirmation Wall
        tech_h1 = self.get_technical_data(symbol, "M5", 60)
        h1_trend = tech_h1.get("higher_timeframe_h1", "")
        if action == "BUY" and "BEARISH" in h1_trend:
            msg = f"REJECTED: Counter-trend BUY blocked on {symbol}. H1 trend is BEARISH."
            logger.warning(msg)
            return {"status": "rejected", "message": msg}
        elif action == "SELL" and "BULLISH" in h1_trend:
            msg = f"REJECTED: Counter-trend SELL blocked on {symbol}. H1 trend is BULLISH."
            logger.warning(msg)
            return {"status": "rejected", "message": msg}

        # Daily Loss Stop Wall (real UTC day, net of swap/commission)
        daily_check = self.check_daily_loss_limit()
        if daily_check is not None:
            return daily_check

        open_positions = self.get_open_positions()
        if len(open_positions) >= self.max_open_trades:
            msg = f"REJECTED: Max open trades ({self.max_open_trades}) reached"
            logger.warning(msg)
            return {"status": "rejected", "message": msg}

        # Prevent duplicate / stacking positions on the same pair
        if any(p.get("symbol") == symbol for p in open_positions):
            msg = f"REJECTED: Position already open for {symbol}. One active position per pair enforced."
            logger.warning(msg)
            return {"status": "rejected", "message": msg}

        # Economic Calendar High-Impact News Blackout Check
        if is_trade_permitted_by_calendar:
            try:
                pre_buffer = self.config.get("risk_parameters", {}).get("news_blackout_pre_mins", 30)
                post_buffer = self.config.get("risk_parameters", {}).get("news_blackout_post_mins", 15)
                permitted, reason = is_trade_permitted_by_calendar(
                    symbol, pre_buffer_mins=pre_buffer, post_buffer_mins=post_buffer,
                    global_tier1=self.global_news_blackout,
                )
                if not permitted:
                    msg = f"REJECTED: News blackout active for {symbol} ({reason})"
                    logger.warning(msg)
                    return {"status": "rejected", "message": msg}
            except Exception as e:
                logger.debug(f"Calendar check exception: {e}")

        # Currency Correlation & Portfolio Concentration Wall
        if is_trade_permitted_by_correlation:
            try:
                max_curr_exp = self.config.get("risk_parameters", {}).get("max_currency_exposure", 3)
                permitted, reason = is_trade_permitted_by_correlation(
                    symbol=symbol,
                    action=action,
                    open_positions=open_positions,
                    max_currency_exposure=max_curr_exp
                )
                if not permitted:
                    msg = f"REJECTED: Currency concentration limit reached for {symbol} ({reason})"
                    logger.warning(msg)
                    return {"status": "rejected", "message": msg}
            except Exception as e:
                logger.debug(f"Currency correlation check exception: {e}")

        account = self.get_account_info()
        if not account:
            return {"status": "error", "message": "Could not get account info"}

        balance = account.get("balance", 0)

        # Commodity Account Balance Guard: Protect small accounts from XAUUSD volatility
        if symbol == "XAUUSD" and balance < self.min_account_balance_gold:
            msg = f"REJECTED: Account balance (${balance:.2f}) below minimum ${self.min_account_balance_gold:.0f} threshold for high-beta XAUUSD."
            logger.warning(msg)
            return {"status": "rejected", "message": msg}

        risk_amount = trade_params.get("risk_amount", 0)
        max_allowed_risk = balance * (self.max_drawdown_pct / 100.0)

        if risk_amount > max_allowed_risk:
            msg = f"REJECTED: Risk at SL ${risk_amount:.2f} exceeds per-trade cap ${max_allowed_risk:.2f} ({self.max_drawdown_pct}% of balance)"
            logger.warning(msg)
            return {"status": "rejected", "message": msg}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"status": "error", "message": f"Failed to get live tick for {symbol}"}

        # Broker Spread Protection Wall
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            point = symbol_info.point
            pip_size = point * 10 if symbol_info.digits in (3, 5) else point
            spread_pips = (tick.ask - tick.bid) / pip_size
            if spread_pips > self.max_spread_pips:
                msg = f"REJECTED: Spread on {symbol} is {spread_pips:.1f} pips (exceeds max allowed {self.max_spread_pips} pips)"
                logger.warning(msg)
                return {"status": "rejected", "message": msg}

        # Spread Cost Wall: spread relative to the SL distance (cost in R paid on entry)
        ratio = spread_to_sl_ratio(tick.ask, tick.bid, abs(entry_price - sl))
        if self.max_spread_sl_ratio is not None and ratio is not None and ratio > self.max_spread_sl_ratio:
            msg = (f"REJECTED: Spread cost on {symbol} is {ratio:.1%} of SL distance "
                   f"(max {self.max_spread_sl_ratio:.0%})")
            logger.warning(msg)
            return {"status": "rejected", "message": msg}

        # Session Profile Wall: per-session symbols, slots, spread cap and loss budget
        try:
            offset = get_server_utc_offset_seconds()
            entry_hours = [server_epoch_to_utc(p["time"], offset).hour for p in open_positions if p.get("time")]
            session_msg = self.session_manager.check(symbol, ratio, entry_hours, balance, self.check_closed_trades)
        except Exception as e:
            logger.warning(f"Session profile check failed for {symbol}: {e}")
            session_msg = f"REJECTED: Session profile check unavailable for {symbol} ({type(e).__name__})"
        if session_msg:
            logger.warning(session_msg)
            return {"status": "rejected", "message": session_msg}

        live_price = float(tick.ask if action == "BUY" else tick.bid)

        # Margin Check: Ensure account has sufficient free margin
        free_margin = account.get("free_margin", 0)
        if free_margin <= 0:
            msg = f"REJECTED: Insufficient free margin (${free_margin:.2f} available). Cannot open new trades."
            logger.warning(msg)
            return {"status": "rejected", "message": msg}

        req_margin = mt5.order_calc_margin(order_type, symbol, float(final_lot), live_price)
        if req_margin is not None and req_margin > free_margin:
            msg = f"REJECTED: Required margin (${req_margin:.2f}) exceeds free margin (${free_margin:.2f})."
            logger.warning(msg)
            return {"status": "rejected", "message": msg}

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(final_lot),
            "type": order_type,
            "price": live_price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": self.order_comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else "None"
            comment = result.comment if result else "No response"
            logger.error(f"Order failed, retcode={retcode} ({comment})")
            return {"status": "error", "message": f"Order failed with retcode {retcode}: {comment}"}

        logger.info(f"Order successful: ticket={result.order}")
        return {
            "status": "success",
            "ticket": result.order,
            "price": result.price,
            "lot": result.volume,
            "sl": sl,
            "tp": tp,
        }

    def get_realized_pnl_today_utc(self, now_utc: Optional[datetime] = None) -> float:
        """Net realized P&L (profit + swap + commission + fee) of deals closed since 00:00 real UTC."""
        now_utc = now_utc or datetime.now(timezone.utc)
        start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        deals = history_deals_utc(start, now_utc + timedelta(minutes=1))
        return float(sum(d.profit + d.swap + d.commission + getattr(d, "fee", 0.0)
                         for d in deals if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)))

    def check_daily_loss_limit(self) -> Optional[Dict]:
        """Return a rejection dict when today's realized loss exceeds daily_loss_limit_pct of balance."""
        if self.daily_loss_limit_pct is None:
            return None
        account_info = mt5.account_info()
        if account_info is None:
            return None
        realized_pnl = self.get_realized_pnl_today_utc()
        if realized_pnl >= 0:
            return None
        realized_loss = abs(realized_pnl)
        limit = account_info.balance * (self.daily_loss_limit_pct / 100.0)
        if realized_loss > limit:
            msg = f"REJECTED: Daily loss limit reached (${realized_loss:.2f} / ${limit:.2f})"
            logger.warning(msg)
            return {"status": "rejected", "message": msg}
        if realized_loss > (limit * 0.5):
            logger.warning(f"WARNING: Daily loss is at {realized_loss:.2f}, approaching limit of {limit:.2f}!")
        return None

    def get_open_positions(self) -> List[Dict]:
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [p._asdict() for p in positions]

    def check_closed_trades(self, since: datetime) -> List[Dict]:
        """Exit deals closed since `since` (real UTC), enriched with entry/SL/TP facts.

        Adds: close_time_utc (ISO), close_ts, entry_price, direction, orig_sl, orig_tp,
        exit_type (TP / SL_FULL / BREAK_EVEN / TRAILING / MANUAL), pips, hold_minutes.
        """
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        offset = get_server_utc_offset_seconds()
        deals = history_deals_utc(since, datetime.now(timezone.utc) + timedelta(minutes=1), offset)
        out = []
        for d in deals:
            if d.entry != mt5.DEAL_ENTRY_OUT:
                continue
            rec = d._asdict()
            close_dt = server_epoch_to_utc(d.time, offset)
            rec["close_time_utc"] = close_dt.isoformat()
            rec["close_ts"] = close_dt.timestamp()
            try:
                rec.update(self._closed_trade_context(d, offset))
            except Exception as e:
                logger.debug(f"Could not enrich closed deal {d.ticket}: {e}")
            out.append(rec)
        return out

    def _closed_trade_context(self, exit_deal, offset: int) -> Dict:
        pos_deals = mt5.history_deals_get(position=exit_deal.position_id) or []
        entry = next((x for x in pos_deals if x.entry == mt5.DEAL_ENTRY_IN), None)
        if entry is None:
            return {}
        orders = mt5.history_orders_get(position=exit_deal.position_id) or []
        open_order = next((o for o in orders if o.ticket == entry.order), None)
        info = mt5.symbol_info(exit_deal.symbol)
        pip = (info.point * 10 if info.digits in (3, 5) else info.point) if info else 0.0001
        direction = 1 if entry.type == mt5.DEAL_TYPE_BUY else -1
        orig_sl = float(open_order.sl) if open_order else 0.0
        orig_tp = float(open_order.tp) if open_order else 0.0
        pips = (exit_deal.price - entry.price) * direction / pip
        exit_type = classify_exit(exit_deal.reason, exit_deal.price, entry.price, orig_sl, direction, pip)
        entry_dt = server_epoch_to_utc(entry.time, offset)
        return {
            "direction": "BUY" if direction > 0 else "SELL",
            "entry_price": entry.price,
            "entry_time_utc": entry_dt.isoformat(),
            "orig_sl": orig_sl,
            "orig_tp": orig_tp,
            "sl_pips": round(abs(entry.price - orig_sl) / pip, 1) if orig_sl else None,
            "pips": round(pips, 1),
            "exit_type": exit_type,
            "hold_minutes": round((exit_deal.time - entry.time) / 60, 1),
            "entry_hour_utc": entry_dt.hour,
            "net_profit": round(sum(x.profit + x.swap + x.commission + getattr(x, "fee", 0.0) for x in pos_deals), 2),
        }

    def manage_trailing_stops(self) -> List[Dict]:
        """
        Actively manage all open positions in MT5:
        1. Break-Even Protection: When profit >= breakeven_trigger_pips (default 10 pips),
           automatically move Stop Loss to entry price + breakeven_lock_pips (1 pip) to guarantee a risk-free trade.
        2. Trailing Stop: When profit >= trailing_start_pips (default 15 pips),
           trail Stop Loss behind live price by trailing_distance_pips (default 10 pips).
        Returns a list of modification event dicts.
        """
        modifications = []
        if not self.trailing_enabled:
            return modifications
        open_positions = self.get_open_positions()
        if not open_positions:
            return modifications

        for p in open_positions:
            try:
                ticket = p.get("ticket")
                symbol = p.get("symbol")
                ptype = p.get("type")  # 0 = BUY, 1 = SELL
                open_price = float(p.get("price_open", 0.0))
                current_price = float(p.get("price_current", 0.0))
                current_sl = float(p.get("sl", 0.0))
                current_tp = float(p.get("tp", 0.0))

                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    continue

                point = symbol_info.point
                digits = symbol_info.digits
                pip_size = point * 10 if digits in (3, 5) else point

                if symbol == "XAUUSD":
                    floating_pips = (current_price - open_price) if ptype == 0 else (open_price - current_price)
                    be_trigger = 2.50  # $2.50 on Gold
                    be_lock = 0.50
                    trail_start = 4.00
                    trail_dist = 2.50
                    step_size = 0.50
                else:
                    floating_pips = (current_price - open_price) / pip_size if ptype == 0 else (open_price - current_price) / pip_size
                    be_trigger = self.breakeven_trigger_pips
                    be_lock = self.breakeven_lock_pips * pip_size
                    trail_start = self.trailing_start_pips
                    trail_dist = self.trailing_distance_pips * pip_size
                    step_size = self.trailing_step_pips * pip_size

                new_sl = None
                action_name = None

                if ptype == 0:  # BUY
                    # 1. Break-Even Check
                    if floating_pips >= be_trigger:
                        lock_price = round(open_price + be_lock, digits)
                        if current_sl < lock_price:
                            new_sl = lock_price
                            action_name = "BREAK_EVEN"

                    # 2. Trailing Stop Check
                    if floating_pips >= trail_start:
                        candidate_sl = round(current_price - trail_dist, digits)
                        if new_sl is None and candidate_sl > current_sl + step_size:
                            new_sl = candidate_sl
                            action_name = "TRAILING_STOP"
                        elif new_sl is not None and candidate_sl > new_sl:
                            new_sl = candidate_sl
                            action_name = "TRAILING_STOP"

                elif ptype == 1:  # SELL
                    # 1. Break-Even Check
                    if floating_pips >= be_trigger:
                        lock_price = round(open_price - be_lock, digits)
                        if current_sl == 0 or current_sl > lock_price:
                            new_sl = lock_price
                            action_name = "BREAK_EVEN"

                    # 2. Trailing Stop Check
                    if floating_pips >= trail_start:
                        candidate_sl = round(current_price + trail_dist, digits)
                        if new_sl is None and (current_sl == 0 or candidate_sl < current_sl - step_size):
                            new_sl = candidate_sl
                            action_name = "TRAILING_STOP"
                        elif new_sl is not None and candidate_sl < new_sl:
                            new_sl = candidate_sl
                            action_name = "TRAILING_STOP"

                # If an update is warranted and meets broker limits
                if new_sl is not None and new_sl != current_sl:
                    stops_level = symbol_info.trade_stops_level * point
                    dist_to_price = abs(current_price - new_sl)
                    if dist_to_price >= stops_level:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": ticket,
                            "symbol": symbol,
                            "sl": new_sl,
                            "tp": current_tp,
                        }
                        result = mt5.order_send(request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"[{symbol}] {action_name} APPLIED #{ticket}: SL moved {current_sl} -> {new_sl} (+{floating_pips:.1f} pips profit)")
                            modifications.append({
                                "ticket": ticket,
                                "symbol": symbol,
                                "action": action_name,
                                "old_sl": current_sl,
                                "new_sl": new_sl,
                                "floating_pips": round(floating_pips, 1),
                            })
                        else:
                            retcode = result.retcode if result else "None"
                            logger.warning(f"[{symbol}] Failed to apply {action_name} on #{ticket}: retcode {retcode}")

            except Exception as e:
                logger.error(f"Error managing position #{p.get('ticket')}: {e}")

        return modifications

