"""
maintenance/daily_report.py
============================
24h post-mortem metrics, computed in code (never by hand).

Sources: MT5 deal/order/tick/bar history, Supabase decision + execution
tables, logs/system_errors.log, logs/trades.log (incl. rotated files) and
learning/learned_rules.md.

All MT5 timestamps are broker server time; they are converted to real UTC via
core/mt5_time.py before any windowing.

Counterfactual simulations ("what if the trade had been taken / left
unmanaged") walk M5 bars forward from the signal with the bot's own SL/TP
geometry. When SL and TP are both inside one bar the SL is assumed to hit
first (conservative). Spread is not charged in simulated R.

Usage:
    python maintenance/daily_report.py [--hours 24] [--no-supabase] [--no-sim]
Saves JSON to reports/report_<UTC date>_<HHMM>.json.
"""
import argparse
import json
import re
import sys
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not installed.")
    sys.exit(1)

from core.mt5_time import get_server_utc_offset_seconds, server_epoch_to_utc, history_deals_utc
from core.trade_analytics import (DEAL_REASON, pip_size_for, session_for_hour, confidence_bucket, spread_bucket,
                                  classify_exit, max_drawdown, simulate_path, wall_from_message)

LOCAL_TZ = datetime.now().astimezone().tzinfo  # loguru writes local wall-clock time
LOGS = AGENT_DIR / "logs"
REPORTS = AGENT_DIR / "reports"

# ── Aggregation ─────────────────────────────────────────────────────────────

def summarize(trades: List[Dict], hours: float = 24.0) -> Dict:
    n = len(trades)
    nets = [t["net"] for t in trades]
    wins = [x for x in nets if x > 0.005]
    losses = [x for x in nets if x < -0.005]
    gross_win, gross_loss = sum(wins), sum(losses)
    avg_w = gross_win / len(wins) if wins else 0.0
    avg_l = gross_loss / len(losses) if losses else 0.0
    ordered = sorted(trades, key=lambda t: t["exit_utc"])
    rs = [t["r"] for t in trades if t.get("r") is not None]
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "breakevens": n - len(wins) - len(losses),
        "win_rate_pct": round(100 * len(wins) / n, 1) if n else 0.0,
        "gross_pnl": round(sum(t["profit"] for t in trades), 2),
        "swap": round(sum(t["swap"] for t in trades), 2),
        "commission": round(sum(t["commission"] for t in trades), 2),
        "net_pnl": round(sum(nets), 2),
        "avg_winner": round(avg_w, 3),
        "avg_loser": round(avg_l, 3),
        "payoff_ratio": round(abs(avg_w / avg_l), 2) if avg_l else None,
        "profit_factor": round(abs(gross_win / gross_loss), 2) if gross_loss else None,
        "expectancy_per_trade": round(sum(nets) / n, 4) if n else 0.0,
        "max_drawdown": round(max_drawdown([t["net"] for t in ordered]), 2),
        "avg_hold_min": round(sum(t["hold_min"] for t in trades) / n, 1) if n else 0.0,
        "trades_per_hour": round(n / hours, 2) if hours else None,
        "avg_r": round(sum(rs) / len(rs), 3) if rs else None,
    }


def breakdown(trades: List[Dict], key: str) -> Dict:
    groups = defaultdict(list)
    for t in trades:
        groups[str(t.get(key))].append(t)
    out = {}
    for k, ts in sorted(groups.items(), key=lambda kv: sum(t["net"] for t in kv[1])):
        s = summarize(ts)
        out[k] = {f: s[f] for f in ("trades", "wins", "losses", "breakevens", "win_rate_pct", "net_pnl", "avg_r", "payoff_ratio")}
    return out


# ── MT5 data access ─────────────────────────────────────────────────────────

class BarCache:
    """Per-symbol M5 bars as numpy arrays keyed by real-UTC epoch."""

    def __init__(self, offset: int):
        self.offset = offset
        self.cache: Dict[str, Dict[str, np.ndarray]] = {}

    def load(self, symbol: str, start_utc: datetime, end_utc: datetime):
        if symbol in self.cache:
            return self.cache[symbol]
        mt5.symbol_select(symbol, True)
        frm = datetime.fromtimestamp(start_utc.timestamp() + self.offset, tz=timezone.utc)
        to = datetime.fromtimestamp(end_utc.timestamp() + self.offset, tz=timezone.utc)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, frm, to)
        if rates is None or len(rates) == 0:
            self.cache[symbol] = None
            return None
        df = pd.DataFrame(rates)
        df["t_utc"] = df["time"] - self.offset
        # ATR(14) Wilder (pandas_ta default RMA), same as the live engine
        prev_close = df["close"].shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
        self.cache[symbol] = {c: df[c].to_numpy() for c in ("t_utc", "open", "high", "low", "close", "atr")}
        return self.cache[symbol]

    def h1_trend_series(self, symbol: str, start_utc: datetime, end_utc: datetime):
        key = f"{symbol}#H1"
        if key in self.cache:
            return self.cache[key]
        frm = datetime.fromtimestamp(start_utc.timestamp() + self.offset - 80 * 3600, tz=timezone.utc)
        to = datetime.fromtimestamp(end_utc.timestamp() + self.offset, tz=timezone.utc)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, frm, to)
        if rates is None or len(rates) < 50:
            self.cache[key] = None
            return None
        df = pd.DataFrame(rates)
        df["t_utc"] = df["time"] - self.offset
        self.cache[key] = df[["t_utc", "close"]].reset_index(drop=True)
        return self.cache[key]

    def h1_trend_at(self, symbol: str, t_utc: float, price_now: float, start_utc, end_utc) -> str:
        """Re-create the live engine's H1 label: last 60 H1 bars, forming bar close = price_now."""
        df = self.h1_trend_series(symbol, start_utc, end_utc)
        if df is None:
            return "UNKNOWN"
        hist = df[df["t_utc"] <= t_utc].tail(60)
        if len(hist) < 50:
            return "UNKNOWN"
        closes = hist["close"].to_numpy().copy()
        closes[-1] = price_now
        s = pd.Series(closes)
        e20 = _ema_sma_seed(s, 20)
        e50 = _ema_sma_seed(s, 50)
        if e20 is None or e50 is None:
            return "UNKNOWN"
        c = closes[-1]
        if c > e20 and e20 > e50:
            return "BULLISH"
        if c < e20 and e20 < e50:
            return "BEARISH"
        if c > e20:
            return "BULLISH_BIAS"
        if c < e20:
            return "BEARISH_BIAS"
        return "NEUTRAL"


def _ema_sma_seed(s: pd.Series, n: int) -> Optional[float]:
    """pandas_ta-style EMA: SMA seed over first n values, then recursive EMA."""
    if len(s) < n:
        return None
    alpha = 2 / (n + 1)
    val = float(s.iloc[:n].mean())
    for x in s.iloc[n:]:
        val = alpha * float(x) + (1 - alpha) * val
    return val


def spread_at(symbol: str, server_epoch: int, pip: float) -> Optional[float]:
    frm = datetime.fromtimestamp(server_epoch - 30, tz=timezone.utc)
    to = datetime.fromtimestamp(server_epoch + 1, tz=timezone.utc)
    ticks = mt5.copy_ticks_range(symbol, frm, to, mt5.COPY_TICKS_INFO)
    if ticks is None or len(ticks) == 0:
        return None
    last = ticks[-1]
    if last["ask"] <= 0 or last["bid"] <= 0:
        return None
    return round((last["ask"] - last["bid"]) / pip, 2)


def collect_trades(start_utc: datetime, end_utc: datetime, offset: int, bars: BarCache, sim: bool) -> List[Dict]:
    exits = [d for d in history_deals_utc(start_utc, end_utc, offset)
             if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)]
    trades = []
    for x in exits:
        pos_deals = mt5.history_deals_get(position=x.position_id) or []
        entry = next((d for d in pos_deals if d.entry == mt5.DEAL_ENTRY_IN), None)
        if entry is None:
            continue
        info = mt5.symbol_info(x.symbol)
        pip = pip_size_for(x.symbol, info.point, info.digits) if info else 0.0001
        direction = 1 if entry.type == mt5.DEAL_TYPE_BUY else -1
        orders = mt5.history_orders_get(position=x.position_id) or []
        open_order = next((o for o in orders if o.ticket == entry.order), None)
        orig_sl = float(open_order.sl) if open_order else 0.0
        orig_tp = float(open_order.tp) if open_order else 0.0
        entry_utc = server_epoch_to_utc(entry.time, offset)
        exit_utc = server_epoch_to_utc(x.time, offset)
        commission = sum(d.commission + getattr(d, "fee", 0.0) for d in pos_deals)
        swap = sum(d.swap for d in pos_deals)
        profit = sum(d.profit for d in pos_deals)
        sl_pips = abs(entry.price - orig_sl) / pip if orig_sl else None
        pips = (x.price - entry.price) * direction / pip
        exit_type = classify_exit(x.reason, x.price, entry.price, orig_sl, direction, pip)
        t = {
            "position_id": x.position_id,
            "symbol": x.symbol,
            "direction": "BUY" if direction > 0 else "SELL",
            "volume": entry.volume,
            "entry_utc": entry_utc.isoformat(),
            "exit_utc": exit_utc.isoformat(),
            "entry_ts": entry_utc.timestamp(),
            "exit_ts": exit_utc.timestamp(),
            "hold_min": round((x.time - entry.time) / 60, 1),
            "entry_price": entry.price,
            "exit_price": x.price,
            "orig_sl": orig_sl,
            "orig_tp": orig_tp,
            "sl_pips": round(sl_pips, 1) if sl_pips else None,
            "pips": round(pips, 1),
            "r": round(pips / sl_pips, 3) if sl_pips else None,
            "profit": profit,
            "swap": swap,
            "commission": commission,
            "net": profit + swap + commission,
            "exit_type": exit_type,
            "exit_reason": DEAL_REASON.get(x.reason, x.reason),
            "hour_utc": entry_utc.hour,
            "session": session_for_hour(entry_utc.hour),
            "spread_pips": spread_at(x.symbol, entry.time, pip),
        }
        t["spread_bucket"] = spread_bucket(t["spread_pips"])
        # pip value for this trade's volume (current tick value; approximate for crosses)
        if info and info.trade_tick_size > 0:
            t["usd_per_pip"] = info.trade_tick_value * (pip / info.trade_tick_size) * entry.volume
        else:
            t["usd_per_pip"] = None

        b = bars.load(x.symbol, start_utc - timedelta(hours=48), end_utc + timedelta(hours=1))
        if b is not None:
            t["h1_at_entry"] = bars.h1_trend_at(x.symbol, entry_utc.timestamp(), entry.price, start_utc, end_utc)
            trend_dir = 1 if "BULLISH" in t["h1_at_entry"] else (-1 if "BEARISH" in t["h1_at_entry"] else 0)
            t["h1_alignment"] = "ALIGNED" if trend_dir == direction else ("COUNTER" if trend_dir == -direction else "NEUTRAL")
            if sim and orig_sl and orig_tp:
                # bars strictly after the entry bar; MFE during the actual hold; unmanaged counterfactual
                after = b["t_utc"] > entry_utc.timestamp() - 300
                held = after & (b["t_utc"] <= exit_utc.timestamp())
                if held.any():
                    fav = (b["high"][held] - entry.price) if direction > 0 else (entry.price - b["low"][held])
                    t["mfe_pips"] = round(float(fav.max()) / pip, 1)
                horizon = after & (b["t_utc"] <= min(end_utc.timestamp(), entry_utc.timestamp() + 48 * 3600))
                cf = simulate_path(b["high"][horizon], b["low"][horizon], b["close"][horizon], direction,
                                   entry.price, orig_sl, orig_tp)
                t["unmanaged_outcome"] = cf["outcome"]
                t["unmanaged_r"] = cf["r"]
        else:
            t["h1_at_entry"] = "UNKNOWN"
            t["h1_alignment"] = "UNKNOWN"
        trades.append(t)
    return trades


# ── Supabase ────────────────────────────────────────────────────────────────

def fetch_supabase(table: str, start_utc: datetime, end_utc: datetime, columns: str = "*") -> Optional[List[Dict]]:
    try:
        from core.supabase_manager import SupabaseManager
        client = SupabaseManager().client
        if client is None:
            return None
        rows, page = [], 0
        while True:
            res = (client.table(table).select(columns).gte("created_at", start_utc.isoformat())
                   .lte("created_at", end_utc.isoformat()).order("created_at").range(page * 1000, page * 1000 + 999).execute())
            rows.extend(res.data or [])
            if not res.data or len(res.data) < 1000:
                break
            page += 1
        return rows
    except Exception as e:
        print(f"[WARN] Supabase fetch failed for {table}: {type(e).__name__}: {str(e)[:120]}")
        return None


def _parse_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def attach_confidence(trades: List[Dict], decisions: List[Dict]):
    by_sym = defaultdict(list)
    for d in decisions:
        if d.get("decision") in ("BUY", "SELL"):
            by_sym[d["symbol"]].append((_parse_ts(d["created_at"]), d))
    for t in trades:
        best, best_dt = None, None
        for ts, d in by_sym.get(t["symbol"], []):
            dt = ts - t["entry_ts"]
            if d["decision"] == t["direction"] and -120 <= dt <= 180 and (best_dt is None or abs(dt) < abs(best_dt)):
                best, best_dt = d, dt
        conf = None
        if best and best.get("confidence") is not None:
            conf = float(best["confidence"])
            conf = conf / 100 if conf > 1 else conf
        t["confidence"] = conf
        t["confidence_bucket"] = confidence_bucket(conf)


def simulate_signals(signals: List[Dict], bars: BarCache, start_utc, end_utc, offset: int) -> List[Dict]:
    """Each signal: symbol, ts, direction (+1/-1), optional entry/sl/tp. Geometry mirrors calculate_trade_parameters."""
    out = []
    for s in signals:
        b = bars.load(s["symbol"], start_utc - timedelta(hours=48), end_utc + timedelta(hours=1))
        if b is None:
            continue
        info = mt5.symbol_info(s["symbol"])
        if info is None:
            continue
        pip = pip_size_for(s["symbol"], info.point, info.digits)
        idx = np.searchsorted(b["t_utc"], s["ts"], side="right") - 1  # bar containing the signal
        if idx < 15 or idx >= len(b["t_utc"]) - 1:
            continue
        # The bar containing the signal is still forming at decision time; its close is
        # future information. Enter at the next bar's open (first price actually tradable).
        entry = s.get("entry") or float(b["open"][idx + 1])
        sl, tp = s.get("sl"), s.get("tp")
        if not sl or not tp:
            atr = float(b["atr"][idx - 1])  # last completed bar
            sl_d = atr * 1.5
            tp_d = atr * 3.0
            floor = 3.50 if s["symbol"] == "XAUUSD" else 15.0 * pip
            if sl_d < floor:
                sl_d = floor
                tp_d = max(tp_d, sl_d * 1.5)
            sl = entry - s["direction"] * sl_d
            tp = entry + s["direction"] * tp_d
        fwd = slice(idx + 1, None)
        horizon = b["t_utc"][fwd] <= min(end_utc.timestamp(), s["ts"] + 24 * 3600)
        res = simulate_path(b["high"][fwd][horizon], b["low"][fwd][horizon], b["close"][fwd][horizon],
                            s["direction"], entry, sl, tp)
        if res["r"] is None:
            continue
        out.append({**s, **res})
    return out


def dedupe_hourly(signals: List[Dict], key=("symbol", "direction")) -> List[Dict]:
    seen, out = set(), []
    for s in sorted(signals, key=lambda x: x["ts"]):
        k = tuple(s.get(k) for k in key) + (int(s["ts"] // 3600),)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def r_stats(sim: List[Dict]) -> Dict:
    rs = [s["r"] for s in sim]
    if not rs:
        return {"n": 0}
    c = Counter(s["outcome"] for s in sim)
    return {"n": len(rs), "avg_r": round(float(np.mean(rs)), 3), "sum_r": round(float(np.sum(rs)), 1),
            "tp": c.get("TP", 0), "sl": c.get("SL", 0), "open": c.get("OPEN", 0),
            "tp_rate_pct": round(100 * c.get("TP", 0) / max(1, c.get("TP", 0) + c.get("SL", 0)), 1)}


# ── Logs ────────────────────────────────────────────────────────────────────

LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| (\w+)\s*\| ([^ ]+) - (.*)$")


def _log_files(prefix: str) -> List[Path]:
    return sorted(LOGS.glob(f"{prefix}*.log"))


def scan_log(prefix: str, start_utc: datetime, end_utc: datetime):
    for path in _log_files(prefix):
        try:
            if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < start_utc:
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = LOG_LINE.match(line.rstrip("\n"))
                    if not m:
                        continue
                    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc)
                    if start_utc <= ts <= end_utc:
                        yield ts, m.group(2), m.group(3), m.group(4)
        except OSError:
            continue


def error_signatures(start_utc, end_utc) -> Dict:
    sigs = Counter()
    for _, level, where, msg in scan_log("system_errors", start_utc, end_utc):
        norm = re.sub(r"\[[A-Z]{6}\]", "[SYM]", msg)
        norm = re.sub(r"\d+(\.\d+)?", "N", norm)[:120]
        sigs[f"{level} {where} {norm}"] += 1
    return dict(sigs.most_common(25))


def trade_log_walls(start_utc, end_utc) -> Dict:
    walls = Counter()
    for _, _, _, msg in scan_log("trades", start_utc, end_utc):
        if "Trade result" not in msg:
            continue
        try:
            payload = json.loads(msg.split("Trade result: ", 1)[1])
        except Exception:
            continue
        if payload.get("status") == "rejected":
            walls[wall_from_message(payload.get("message", ""))] += 1
        else:
            walls[f"STATUS_{payload.get('status', '?').upper()}"] += 1
    return dict(walls.most_common())


def reflexion_rules_in_window(start_utc, end_utc) -> Dict:
    path = AGENT_DIR / "learning" / "learned_rules.md"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.findall(r"### \[([A-Z_]+)\][^\n]*\n- \*\*Date\*\*: ([0-9-]+ [0-9:]+) UTC\n- \*\*Source\*\*: ([^\n]+)", text)
    in_win = Counter()
    for cat, date, source in blocks:
        ts = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if start_utc <= ts <= end_utc:
            in_win[f"{cat}/{source.strip()}"] += 1
    return {"total_blocks": len(blocks), "in_window": dict(in_win), "file_bytes": path.stat().st_size}


# ── Exposure ────────────────────────────────────────────────────────────────

def exposure_stats(trades: List[Dict], open_positions: List[Dict], now_ts: float) -> Dict:
    intervals = [(t["entry_ts"], t["exit_ts"], t["symbol"], t["direction"]) for t in trades]
    intervals += [(p["entry_ts"], now_ts, p["symbol"], p["type"]) for p in open_positions]
    events = sorted({ts for a, b, _, _ in intervals for ts in (a, b)})
    max_total, max_ccy = 0, Counter()
    for ts in events:
        live = [(s, d) for a, b, s, d in intervals if a <= ts < b]
        max_total = max(max_total, len(live))
        ccy = Counter()
        for s, _ in live:
            if len(s) == 6:
                ccy[s[:3]] += 1
                ccy[s[3:]] += 1
        for c, n in ccy.items():
            max_ccy[c] = max(max_ccy[c], n)
    return {"max_concurrent_positions": max_total, "max_concurrent_per_currency": dict(max_ccy.most_common())}


# ── Previous cycles ─────────────────────────────────────────────────────────

def previous_cycles(limit: int = 3) -> List[Dict]:
    out = []
    log = LOGS / "code_evolution.log"
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"## \[([^\]]+)\] 24-Hour Post-Mortem Cycle #(\d+)(.*?)(?=\n## \[|\Z)", text, re.S):
            body = m.group(3)

            def grab(pat):
                g = re.search(pat, body)
                return g.group(1) if g else None
            out.append({
                "cycle": int(m.group(2)), "logged": m.group(1),
                "trades": grab(r"Closed Trades: (\d+)"), "win_rate": grab(r"Win Rate: ([\d.]+)%"),
                "net": grab(r"Net(?: P&L)?: (-?\+?\$?-?[\d.]+)"), "payoff": grab(r"Payoff Ratio: ([\d.]+)"),
                "max_dd": grab(r"Max Drawdown: (-?\$?-?[\d.]+)"),
            })
    return out[-limit:]


# ── Main ────────────────────────────────────────────────────────────────────

def collect_report(hours: float = 24.0, use_supabase: bool = True, sim: bool = True) -> Dict:
    if not mt5.initialize():
        return {"error": f"MT5 init failed: {mt5.last_error()}"}
    try:
        offset = get_server_utc_offset_seconds(force=True)
        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(hours=hours)
        bars = BarCache(offset)

        acc = mt5.account_info()
        account = {"balance": acc.balance, "equity": acc.equity, "margin": acc.margin,
                   "free_margin": acc.margin_free, "margin_level": acc.margin_level or None, "currency": acc.currency}

        positions = []
        for p in mt5.positions_get() or []:
            positions.append({"symbol": p.symbol, "type": "BUY" if p.type == 0 else "SELL", "volume": p.volume,
                              "profit": round(p.profit, 2), "swap": round(p.swap, 2),
                              "entry_ts": server_epoch_to_utc(p.time, offset).timestamp(),
                              "sl": p.sl, "tp": p.tp, "price_open": p.price_open})

        trades = collect_trades(start_utc, end_utc, offset, bars, sim)

        decisions = fetch_supabase("forex_trade_decisions", start_utc, end_utc,
                                   "created_at,symbol,decision,confidence,reasoning,entry_price,stop_loss,take_profit,metadata") if use_supabase else None
        executions = fetch_supabase("forex_executed_trades", start_utc, end_utc,
                                    "created_at,symbol,order_type,status,open_price,stop_loss,take_profit,execution_result") if use_supabase else None
        rules_sb = fetch_supabase("forex_learned_rules", start_utc, end_utc, "created_at,rule_id,category,source") if use_supabase else None
        telemetry = fetch_supabase("ai_agent_telemetry", start_utc, end_utc, "created_at,agent_name,action_type,status") if use_supabase else None

        attach_confidence(trades, decisions or [])

        report = {
            "generated_at": end_utc.isoformat(),
            "window_start": start_utc.isoformat(),
            "window_end": end_utc.isoformat(),
            "server_utc_offset_hours": offset / 3600,
            "account": account,
            "open_positions": positions,
            "floating_pnl": round(sum(p["profit"] + p["swap"] for p in positions), 2),
            "summary": summarize(trades, hours),
        }
        s = report["summary"]
        report["breakdowns"] = {k: breakdown(trades, k) for k in
                                ("symbol", "direction", "session", "hour_utc", "confidence_bucket", "exit_type",
                                 "spread_bucket", "h1_alignment")}

        # Payoff asymmetry & management counterfactual
        managed = [t for t in trades if t.get("unmanaged_r") is not None and t.get("r") is not None]
        report["payoff_asymmetry"] = {
            "exit_type_counts": dict(Counter(t["exit_type"] for t in trades)),
            "avg_win_r": round(float(np.mean([t["r"] for t in trades if t["r"] is not None and t["net"] > 0.005])), 3) if any(t["net"] > 0.005 and t["r"] is not None for t in trades) else None,
            "avg_loss_r": round(float(np.mean([t["r"] for t in trades if t["r"] is not None and t["net"] < -0.005])), 3) if any(t["net"] < -0.005 and t["r"] is not None for t in trades) else None,
            "avg_mfe_pips_winners": round(float(np.mean([t["mfe_pips"] for t in trades if t.get("mfe_pips") is not None and t["net"] > 0.005])), 1) if any(t.get("mfe_pips") is not None and t["net"] > 0.005 for t in trades) else None,
            "avg_sl_pips": round(float(np.mean([t["sl_pips"] for t in trades if t["sl_pips"]])), 1) if any(t["sl_pips"] for t in trades) else None,
            "actual_sum_r": round(sum(t["r"] for t in managed), 2),
            "unmanaged_sum_r": round(sum(t["unmanaged_r"] for t in managed), 2),
            "n_compared": len(managed),
            "be_trail_exits_that_would_hit_tp": sum(1 for t in managed if t["exit_type"] in ("BREAK_EVEN", "TRAILING") and t["unmanaged_outcome"] == "TP"),
            "be_trail_exits_that_would_hit_sl": sum(1 for t in managed if t["exit_type"] in ("BREAK_EVEN", "TRAILING") and t["unmanaged_outcome"] == "SL"),
            "be_trail_exits_total": sum(1 for t in managed if t["exit_type"] in ("BREAK_EVEN", "TRAILING")),
        }

        # Cost drag
        spread_cost = sum((t["spread_pips"] or 0) * (t["usd_per_pip"] or 0) for t in trades)
        gross_wins = sum(t["profit"] for t in trades if t["profit"] > 0)
        report["cost_drag"] = {
            "est_spread_cost": round(spread_cost, 2),
            "swap": s["swap"], "commission": s["commission"],
            "total_cost": round(spread_cost - s["swap"] - s["commission"], 2),
            "cost_pct_of_gross_wins": round(100 * (spread_cost - s["swap"] - s["commission"]) / gross_wins, 1) if gross_wins else None,
            "median_spread_pips": float(np.median([t["spread_pips"] for t in trades if t["spread_pips"] is not None])) if any(t["spread_pips"] is not None for t in trades) else None,
            "avg_spread_as_pct_of_sl": round(100 * float(np.mean([t["spread_pips"] / t["sl_pips"] for t in trades if t["spread_pips"] is not None and t["sl_pips"]])), 1) if any(t["spread_pips"] is not None and t["sl_pips"] for t in trades) else None,
        }
        report["exposure"] = exposure_stats(trades, positions, end_utc.timestamp())

        # LLM health from Supabase decisions
        if decisions is not None:
            mix = Counter(d["decision"] for d in decisions)
            confs = [float(d["confidence"]) / 100 if float(d["confidence"] or 0) > 1 else float(d["confidence"] or 0) for d in decisions]
            defensive = [d for d in decisions if "defensive WAIT" in (d.get("reasoning") or "")]
            breaker = [d for d in decisions if "circuit breaker" in (d.get("reasoning") or "").lower()]
            # cycle latency: span of decisions within each 5-min candle bucket
            buckets = defaultdict(list)
            for d in decisions:
                ts = _parse_ts(d["created_at"])
                buckets[int(ts // 300)].append(ts)
            spans = [max(v) - min(v) for v in buckets.values() if len(v) > 1]
            report["llm_health"] = {
                "decisions": len(decisions),
                "mix": dict(mix),
                "actionable_ge_threshold": sum(1 for d, c in zip(decisions, confs) if d["decision"] in ("BUY", "SELL") and c >= 0.80),
                "confidence_hist": dict(Counter(confidence_bucket(c) if d["decision"] != "WAIT" else "WAIT" for d, c in zip(decisions, confs))),
                "defensive_wait_fallbacks": len(defensive),
                "defensive_wait_symbols": dict(Counter(d["symbol"] for d in defensive)),
                "circuit_breaker_decisions": len(breaker),
                "cycles_observed": len(buckets),
                "cycle_span_sec_median": round(float(np.median(spans)), 1) if spans else None,
                "cycle_span_sec_p90": round(float(np.percentile(spans, 90)), 1) if spans else None,
                "decisions_with_metadata": sum(1 for d in decisions if d.get("metadata")),
            }
        else:
            report["llm_health"] = {"error": "Supabase unavailable"}

        # Risk walls (Supabase rows + trades.log cross-check)
        if executions is not None:
            rej = [e for e in executions if e.get("status") == "REJECTED"]
            report["walls_supabase"] = dict(Counter(wall_from_message((e.get("execution_result") or {}).get("message", "")) for e in rej).most_common())
            report["executions_status"] = dict(Counter(e.get("status") for e in executions))
        report["walls_trades_log"] = trade_log_walls(start_utc, end_utc)
        report["errors"] = error_signatures(start_utc, end_utc)
        report["reflexion_rules"] = reflexion_rules_in_window(start_utc, end_utc)
        report["supabase_rules_in_window"] = [(r["rule_id"], r["category"], r["source"]) for r in (rules_sb or [])]
        report["telemetry_actions"] = dict(Counter(f"{t['agent_name']}/{t['action_type']}/{t['status']}" for t in (telemetry or [])))

        # Counterfactuals: walls + baselines
        if sim and decisions is not None:
            if executions is not None:
                rej_signals = []
                for e in executions:
                    if e.get("status") != "REJECTED" or e.get("order_type") not in ("BUY", "SELL"):
                        continue
                    rej_signals.append({"symbol": e["symbol"], "ts": _parse_ts(e["created_at"]),
                                        "direction": 1 if e["order_type"] == "BUY" else -1,
                                        "entry": e.get("open_price"), "sl": e.get("stop_loss"), "tp": e.get("take_profit"),
                                        "wall": wall_from_message((e.get("execution_result") or {}).get("message", ""))})
                rej_dedup = dedupe_hourly(rej_signals, key=("symbol", "direction", "wall"))
                sims = simulate_signals(rej_dedup, bars, start_utc, end_utc, offset)
                by_wall = defaultdict(list)
                for x in sims:
                    by_wall[x["wall"]].append(x)
                report["wall_counterfactuals_hourly_dedup"] = {w: r_stats(v) for w, v in by_wall.items()}

            llm_signals, h1_signals, all_points = [], [], []
            for d, c in zip(decisions, confs):
                ts = _parse_ts(d["created_at"])
                all_points.append({"symbol": d["symbol"], "ts": ts, "decision": d["decision"], "conf": c})
            all_points = dedupe_hourly(all_points, key=("symbol",))
            for p in all_points:
                b = bars.load(p["symbol"], start_utc - timedelta(hours=48), end_utc + timedelta(hours=1))
                if b is None:
                    continue
                idx = np.searchsorted(b["t_utc"], p["ts"], side="right") - 1
                if idx < 0:
                    continue
                trend = bars.h1_trend_at(p["symbol"], p["ts"], float(b["close"][idx - 1]) if idx > 0 else float(b["open"][idx]), start_utc, end_utc)
                p["h1"] = trend
                tdir = 1 if "BULLISH" in trend else (-1 if "BEARISH" in trend else 0)
                if tdir:
                    h1_signals.append({"symbol": p["symbol"], "ts": p["ts"], "direction": tdir, "direction_label": trend})
                if p["decision"] in ("BUY", "SELL"):
                    llm_signals.append({"symbol": p["symbol"], "ts": p["ts"], "direction": 1 if p["decision"] == "BUY" else -1,
                                        "conf": p["conf"], "aligned": tdir == (1 if p["decision"] == "BUY" else -1)})
            llm_sim = simulate_signals(llm_signals, bars, start_utc, end_utc, offset)
            h1_sim = simulate_signals(h1_signals, bars, start_utc, end_utc, offset)
            report["baselines_hourly_dedup"] = {
                "decision_points": len(all_points),
                "always_wait": {"n": len(all_points), "avg_r": 0.0, "sum_r": 0.0},
                "follow_h1_trend": r_stats(h1_sim),
                "llm_all_buy_sell": r_stats(llm_sim),
                "llm_ge_0.80": r_stats([x for x in llm_sim if x["conf"] >= 0.80]),
                "llm_ge_0.80_h1_aligned": r_stats([x for x in llm_sim if x["conf"] >= 0.80 and x["aligned"]]),
                "llm_by_conf_bucket": {b: r_stats([x for x in llm_sim if confidence_bucket(x["conf"]) == b])
                                       for b in ("<0.80", "0.80-0.84", "0.85-0.89", ">=0.90")},
            }

        report["previous_cycles"] = previous_cycles()
        report["trades"] = trades
        return report
    finally:
        mt5.shutdown()


def print_report(r: Dict):
    if "error" in r:
        print(f"ERROR: {r['error']}")
        return
    s = r["summary"]
    print(f"Window (UTC): {r['window_start'][:16]} -> {r['window_end'][:16]}  | broker server offset UTC{r['server_utc_offset_hours']:+.0f}")
    a = r["account"]
    print(f"Balance ${a['balance']:.2f} | Equity ${a['equity']:.2f} | Free margin ${a['free_margin']:.2f} | Open {len(r['open_positions'])} (floating ${r['floating_pnl']})")
    print(f"Trades {s['trades']} ({s['wins']}W/{s['losses']}L/{s['breakevens']}BE) WR {s['win_rate_pct']}% | "
          f"Gross ${s['gross_pnl']} Swap ${s['swap']} Comm ${s['commission']} Net ${s['net_pnl']}")
    print(f"Avg W ${s['avg_winner']} | Avg L ${s['avg_loser']} | Payoff {s['payoff_ratio']} | PF {s['profit_factor']} | "
          f"Exp ${s['expectancy_per_trade']}/trade | MaxDD ${s['max_drawdown']} | Hold {s['avg_hold_min']}m | {s['trades_per_hour']}/h | avg R {s['avg_r']}")
    for k, rows in r["breakdowns"].items():
        print(f"\n[{k}]")
        for name, v in rows.items():
            print(f"  {name:18s} n={v['trades']:3d} {v['wins']}W/{v['losses']}L/{v['breakevens']}BE WR={v['win_rate_pct']:5.1f}% net=${v['net_pnl']:7.2f} avgR={v['avg_r']}")
    for sec in ("payoff_asymmetry", "cost_drag", "exposure", "llm_health", "walls_supabase", "walls_trades_log",
                "wall_counterfactuals_hourly_dedup", "baselines_hourly_dedup", "errors", "reflexion_rules",
                "supabase_rules_in_window", "telemetry_actions", "previous_cycles"):
        if sec in r:
            print(f"\n[{sec}]\n{json.dumps(r[sec], indent=1, default=str)}")


def save_report(r: Dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"report_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M')}.json"
    path.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    print(f"\nReport saved: {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--no-supabase", action="store_true")
    ap.add_argument("--no-sim", action="store_true")
    args = ap.parse_args()
    rep = collect_report(args.hours, use_supabase=not args.no_supabase, sim=not args.no_sim)
    print_report(rep)
    if "error" not in rep:
        save_report(rep)
