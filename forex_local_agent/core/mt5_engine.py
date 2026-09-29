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
import math
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from loguru import logger


class MT5Engine:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        log_dir = self.config_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / "trades.log", rotation="10 MB")

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
        self.max_lot_size = risk.get("max_lot_size", 0.1)
        self.max_drawdown_pct = risk.get("max_drawdown_pct", 2.0)
        self.max_open_trades = risk.get("max_open_trades", 10)
        self.risk_per_trade_pct = risk.get("risk_per_trade_pct", 1.5)
        self.atr_period = risk.get("atr_period", 14)
        self.atr_multiplier_sl = risk.get("atr_multiplier_sl", 1.5)
        self.atr_multiplier_tp = risk.get("atr_multiplier_tp", 3.0)

        # Trading settings
        trading = self.config.get("trading", {})
        self.default_timeframe = trading.get("timeframe", "H1")
        self.analysis_bars = trading.get("analysis_bars", 100)
        self.order_comment = trading.get("order_comment", "Bubat AI")

    def initialize(self) -> bool:
        creds = self.config.get("mt5_credentials", {})
        path = creds.get("path", r"C:\Program Files\MetaTrader 5\terminal64.exe")
        login = creds.get("login")
        server = creds.get("server")
        password = creds.get("password")

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

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "current_price": close_price,
            "trend_structure": trend_structure,
            "technical_bias": bias,
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

        sl_distance = atr_value * self.atr_multiplier_sl
        tp_distance = atr_value * self.atr_multiplier_tp

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

        # Check broker minimum stops level
        stops_level_points = symbol_info.trade_stops_level
        min_stop_distance = stops_level_points * point

        if min_stop_distance > 0 and sl_distance < min_stop_distance:
            logger.warning(f"ATR SL distance ({sl_distance}) is below broker min stop level ({min_stop_distance}). Widening SL.")
            sl_distance = min_stop_distance
            if decision.upper() == "BUY":
                sl_price = round(entry_price - sl_distance, digits)
            else:
                sl_price = round(entry_price + sl_distance, digits)

        if min_stop_distance > 0 and tp_distance < min_stop_distance:
            tp_distance = min_stop_distance
            if decision.upper() == "BUY":
                tp_price = round(entry_price + tp_distance, digits)
            else:
                tp_price = round(entry_price - tp_distance, digits)

        # Dynamic lot sizing
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

        if loss_per_lot <= 0:
            return {"status": "error", "message": "Invalid loss calculation per lot"}

        raw_lot = risk_amount / loss_per_lot

        vol_min = symbol_info.volume_min
        vol_max = symbol_info.volume_max
        vol_step = symbol_info.volume_step

        if vol_step > 0:
            raw_lot = math.floor(raw_lot / vol_step) * vol_step

        final_lot = max(vol_min, min(raw_lot, vol_max, self.max_lot_size))
        final_lot = round(final_lot, 2)

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
            "risk_amount": round(risk_amount, 2),
            "risk_reward_ratio": round(self.atr_multiplier_tp / self.atr_multiplier_sl, 2),
        }

    def execute_trade(self, trade_params: Dict) -> Dict:
        symbol = trade_params["symbol"]
        action = trade_params["action"]
        order_type = trade_params["order_type"]
        entry_price = trade_params["entry"]
        sl = trade_params["sl"]
        tp = trade_params["tp"]
        final_lot = trade_params["lot"]

        logger.info(f"[EXECUTION] Executing {action} {symbol} lot={final_lot} entry={entry_price} sl={sl} tp={tp}")

        open_positions = self.get_open_positions()
        if len(open_positions) >= self.max_open_trades:
            msg = f"REJECTED: Max open trades ({self.max_open_trades}) reached"
            logger.warning(msg)
            return {"status": "rejected", "message": msg}

        account = self.get_account_info()
        if not account:
            return {"status": "error", "message": "Could not get account info"}

        balance = account.get("balance", 0)
        risk_amount = trade_params.get("risk_amount", 0)
        max_allowed_risk = balance * (self.max_drawdown_pct / 100.0)

        if risk_amount > max_allowed_risk:
            msg = f"REJECTED: Risk ${risk_amount:.2f} exceeds max drawdown allowed ${max_allowed_risk:.2f}"
            logger.warning(msg)
            return {"status": "rejected", "message": msg}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"status": "error", "message": f"Failed to get live tick for {symbol}"}

        live_price = float(tick.ask if action == "BUY" else tick.bid)

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
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed, retcode={result.retcode}")
            return {"status": "error", "message": f"Order failed with retcode {result.retcode}"}

        logger.info(f"Order successful: ticket={result.order}")
        return {
            "status": "success",
            "ticket": result.order,
            "price": result.price,
            "lot": result.volume,
            "sl": sl,
            "tp": tp,
        }

    def get_open_positions(self) -> List[Dict]:
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [p._asdict() for p in positions]

    def check_closed_trades(self, since: datetime) -> List[Dict]:
        deals = mt5.history_deals_get(since, datetime.now())
        if deals is None:
            return []
        return [d._asdict() for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]

