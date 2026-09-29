import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from loguru import logger

class MT5Engine:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
        log_dir = self.config_path.parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / 'trades.log', rotation="10 MB")
        
        self.tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }

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

    def get_account_info(self) -> Dict:
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return {}
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free
        }

    def get_technical_data(self, symbol: str, timeframe: str = 'H1', bars: int = 100) -> Dict:
        tf = self.tf_map.get(timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None:
            logger.error(f"Failed to get rates for {symbol}")
            return {}
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.atr(length=14, append=True)
        
        last_row = df.iloc[-1]
        
        macd_cols = [c for c in df.columns if c.startswith('MACD_')]
        macdh_cols = [c for c in df.columns if c.startswith('MACDh_')]
        macds_cols = [c for c in df.columns if c.startswith('MACDs_')]
        rsi_col = [c for c in df.columns if c.startswith('RSI_')]
        atr_col = [c for c in df.columns if c.startswith('ATRr_')]

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': last_row['close'],
            'rsi': last_row[rsi_col[0]] if rsi_col else None,
            'macd': last_row[macd_cols[0]] if macd_cols else None,
            'macd_signal': last_row[macds_cols[0]] if macds_cols else None,
            'macd_hist': last_row[macdh_cols[0]] if macdh_cols else None,
            'atr': last_row[atr_col[0]] if atr_col else None,
            'high_range': df['high'].max(),
            'low_range': df['low'].min(),
            'last_5_candles': df.tail(5)[['time', 'open', 'high', 'low', 'close']].to_dict(orient='records')
        }

    def execute_trade(self, symbol: str, action: str, suggested_lot: float, sl_points: float, tp_points: float) -> Dict:
        logger.info(f"Trade requested: {action} {symbol} lot={suggested_lot} sl={sl_points} tp={tp_points}")
        
        max_lot = self.config.get('risk_parameters', {}).get('max_lot_size', 0.1)
        final_lot = min(suggested_lot, max_lot)
        
        account_info = self.get_account_info()
        if not account_info:
            return {"status": "error", "message": "Could not get account info"}
            
        balance = account_info.get('balance', 0)
        
        max_open = self.config.get('risk_parameters', {}).get('max_open_trades', 5)
        open_positions = self.get_open_positions()
        if len(open_positions) >= max_open:
            msg = f"REJECTED: Max open trades ({max_open}) reached"
            logger.warning(msg)
            return {"status": "rejected", "message": msg}
            
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"status": "error", "message": f"Symbol {symbol} not found"}
            
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return {"status": "error", "message": f"Failed to select {symbol}"}
                
        point = symbol_info.point
        price = mt5.symbol_info_tick(symbol).ask if action.lower() == 'buy' else mt5.symbol_info_tick(symbol).bid
        
        if action.lower() == 'buy':
            sl = price - (sl_points * point)
            tp = price + (tp_points * point)
            order_type = mt5.ORDER_TYPE_BUY
        else:
            sl = price + (sl_points * point)
            tp = price - (tp_points * point)
            order_type = mt5.ORDER_TYPE_SELL
            
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        if tick_size > 0:
            loss_value = (sl_points * point / tick_size) * tick_value * final_lot
            max_drawdown_pct = self.config.get('risk_parameters', {}).get('max_drawdown_pct', 0.05)
            if loss_value > (balance * max_drawdown_pct):
                msg = f"REJECTED: SL risk {loss_value} exceeds max drawdown allowed {balance * max_drawdown_pct}"
                logger.warning(msg)
                return {"status": "rejected", "message": msg}

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(final_lot),
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": "python_script",
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
            "tp": tp
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

    def reconnect(self) -> bool:
        for attempt in range(3):
            logger.info(f"Reconnection attempt {attempt + 1}")
            self.shutdown()
            time.sleep(5)
            if self.initialize():
                return True
        logger.error("Failed to reconnect after 3 attempts")
        return False
