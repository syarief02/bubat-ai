"""
Health Check
============
Is the bot WORKING (not: is it profitable)? Read-only checks, each OK / WARN / FAULT:

- agent:      the trading agent is running (its webhook port answers) and not stuck
- activity:   analysis cycles ran while the market was open, without long gaps
- errors:     ERROR lines in system_errors.log, grouped (order failures count most)
- ollama:     the server answers and the trading model is installed
- mt5:        terminal connected, algo trading allowed
- positions:  every open bot position has SL and TP, lot within limits, not too many open
- loss_wall:  no trade was opened after the day's loss passed daily_loss_limit_pct
- brain:      the brain ran recently
- disk:       enough free space for logs and reports

Writes state/health/latest.json (the chat shows it at start) and appends to logs/health.log.
Usage:
    python maintenance/health_check.py [--hours 24]
Exit code: 0 OK, 1 WARN, 2 FAULT.
"""

import argparse
import json
import re
import shutil
import socket
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

from core.mt5_time import market_open  # noqa: E402

LOGS_DIR = AGENT_DIR / "logs"
HEALTH_DIR = AGENT_DIR / "state" / "health"
BOT_MAGIC = 234000            # mt5_engine.execute_trade
MAX_CYCLE_GAP_MIN = 30        # cycles run every M5 candle; a longer gap while open means stuck
BRAIN_MAX_AGE_H = 50
MIN_FREE_DISK_GB = 2.0

OK, WARN, FAULT = "OK", "WARN", "FAULT"
_RANK = {OK: 0, WARN: 1, FAULT: 2}
_LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| (\w+)\s*\| ([^ ]+) - (.*)$")


def check(name: str, status: str, detail: str) -> Dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def overall(checks: List[Dict[str, str]]) -> str:
    return max((c["status"] for c in checks), key=_RANK.get, default=OK)


# ── Log parsing (log times are the PC's local time, written by loguru) ──────────

def parse_log_lines(lines: Iterable[str]) -> List[Tuple[datetime, str, str, str]]:
    """(utc time, level, source, message) for each first line of a log record."""
    out = []
    for line in lines:
        m = _LOG_LINE.match(line.rstrip("\n"))
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").astimezone(timezone.utc)
            out.append((ts, m.group(2), m.group(3), m.group(4)))
    return out


def read_logs(pattern: str, since_utc: datetime) -> List[Tuple[datetime, str, str, str]]:
    """Records since `since_utc` from a log and its rotated copies (e.g. agent.log, agent.2026-...log)."""
    since_local = since_utc.timestamp()
    records = []
    for path in sorted(LOGS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime):
        if path.stat().st_mtime < since_local:
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            records += [r for r in parse_log_lines(f) if r[0] >= since_utc]
    return records


def open_minutes(start_utc: datetime, end_utc: datetime) -> int:
    """Minutes the FX market was open in the window (5-minute steps)."""
    t, n = start_utc, 0
    while t < end_utc:
        n += 5 if market_open(t) else 0
        t += timedelta(minutes=5)
    return n


def check_activity(cycles: List[datetime], start_utc: datetime, end_utc: datetime,
                   log_start: Optional[datetime] = None) -> Dict[str, str]:
    """log_start: oldest record in the agent logs; the window can't be judged before it."""
    note = ""
    if log_start and log_start - start_utc > timedelta(minutes=MAX_CYCLE_GAP_MIN):
        start_utc = log_start
        note = f" (logs only go back to {log_start.astimezone():%a %H:%M} local)"
    open_min = open_minutes(start_utc, end_utc)
    closed = sum(not market_open(t) for t in cycles)
    if closed:
        return check("activity", WARN, f"{closed} pair analyses ran while the market was CLOSED "
                                       f"(the weekend pause is not working?)" + note)
    if open_min < 60:
        return check("activity", OK, f"market open only {open_min} min in the window; {len(cycles)} pair analyses" + note)
    if not cycles:
        return check("activity", FAULT, f"market was open {open_min // 60}h but no analysis cycle ran" + note)
    # Gaps while the market was open (start and end of the window count as edges)
    edges = [start_utc] + sorted(cycles) + [end_utc]
    gaps = []
    for a, b in zip(edges, edges[1:]):
        mins = (b - a).total_seconds() / 60
        if mins > MAX_CYCLE_GAP_MIN and market_open(a + (b - a) / 2):
            gaps.append((a, mins))
    if gaps:
        worst = max(gaps, key=lambda g: g[1])
        return check("activity", WARN,
                     f"{len(cycles)} pair analyses, but {len(gaps)} gap(s) over {MAX_CYCLE_GAP_MIN} min while open; "
                     f"longest {worst[1]:.0f} min from {worst[0].astimezone():%a %H:%M} local" + note)
    return check("activity", OK, f"{len(cycles)} pair analyses over {open_min // 60}h of open market, no long gaps" + note)


def _signature(message: str) -> str:
    sig = re.sub(r"\[[A-Z]{6}\]\s*", "", message)
    sig = re.sub(r"\d+(\.\d+)?", "#", sig)
    return sig[:90]


def check_errors(errors: List[Tuple[datetime, str, str, str]], hours: float = 24.0) -> Dict[str, str]:
    if not errors:
        return check("errors", OK, "no errors logged")
    sigs = Counter(_signature(e[3]) for e in errors)
    top = "; ".join(f"{n}x {s}" for s, n in sigs.most_common(3))
    order_failures = sum(n for s, n in sigs.items() if s.startswith("Order failed"))
    if order_failures >= 5:
        return check("errors", FAULT, f"{order_failures} failed orders. Top: {top}")
    parse_failures = sum(n for s, n in sigs.items() if "TradeDecision" in s)
    other = len(errors) - parse_failures
    # The model sometimes returns unparseable JSON and the bot safely WAITs; only a flood matters
    if other == 0 and parse_failures < 50 * hours / 24:
        return check("errors", OK, f"{parse_failures} unparseable model replies (bot waited safely)")
    return check("errors", WARN, f"{len(errors)} errors. Top: {top}")


# ── Live checks ─────────────────────────────────────────────────────────────────

def check_agent(cfg: Dict, last_record: Optional[datetime], now_utc: datetime) -> Dict[str, str]:
    port = int(cfg.get("openclaw_webhook_port", 5055))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3):
            running = True
    except OSError:
        running = False
    is_open = market_open(now_utc)
    if not running:
        soon = market_open(now_utc + timedelta(hours=2))
        return check("agent", FAULT if (is_open or soon) else WARN,
                     f"trading agent is not running (nothing on port {port}). Start run_agent.bat")
    if is_open and last_record and (now_utc - last_record) > timedelta(minutes=MAX_CYCLE_GAP_MIN):
        mins = (now_utc - last_record).total_seconds() / 60
        return check("agent", FAULT, f"running but silent for {mins:.0f} min while the market is open (stuck?)")
    return check("agent", OK, "running" + ("" if is_open else " (market closed, waiting for the open)"))


def check_ollama(cfg: Dict) -> Dict[str, str]:
    base = cfg.get("ollama_base_url", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as r:
            names = {m.get("name") for m in json.load(r).get("models", [])}
    except Exception as e:
        return check("ollama", FAULT, f"Ollama is not answering at {base} ({e.__class__.__name__})")
    model = cfg.get("active_model", "")
    wanted = {model, model if ":" in model else f"{model}:latest"}
    if not names & wanted:
        return check("ollama", FAULT, f"trading model {model!r} is not installed in Ollama")
    return check("ollama", OK, f"answering, {model} installed")


def check_positions(positions: List[Dict], risk: Dict) -> Dict[str, str]:
    bot = [p for p in positions if p.get("magic") == BOT_MAGIC]
    problems, warnings = [], []
    max_lot = float(risk.get("max_lot_size", 0.1))
    fixed = float(risk.get("fixed_lot", 0.01)) if risk.get("lot_mode") == "fixed" else None
    for p in bot:
        tag = f"{p.get('symbol')} #{p.get('ticket')}"
        if not p.get("sl"):
            problems.append(f"{tag} has NO stop loss")
        if not p.get("tp"):
            problems.append(f"{tag} has no take profit")
        if p.get("volume", 0) > max_lot + 1e-9:
            problems.append(f"{tag} lot {p.get('volume')} > max {max_lot}")
        elif fixed is not None and abs(p.get("volume", 0) - fixed) > 1e-9:
            warnings.append(f"{tag} lot {p.get('volume')} (fixed lot is {fixed})")
    limit = int(risk.get("max_open_trades", 10))
    if len(bot) > limit:
        problems.append(f"{len(bot)} bot positions open, limit is {limit}")
    others = len(positions) - len(bot)
    note = f"{len(bot)} bot position(s) open" + (f", {others} manual" if others else "")
    if problems:
        return check("positions", FAULT, f"{note}. " + "; ".join(problems))
    if warnings:
        return check("positions", WARN, f"{note}. " + "; ".join(warnings))
    return check("positions", OK, f"{note}, all with SL and TP, lots within limits")


def check_loss_wall(deals: List[Dict], balance_now: float, limit_pct: Optional[float],
                    since_utc: datetime) -> Dict[str, str]:
    """deals: dicts with time_utc, entry ('in'/'out'/'other'), magic, pnl, symbol, sorted by time,
    from 00:00 UTC of the first day checked until now ('other' = deposits etc., they move the balance).

    Mirrors mt5_engine.check_daily_loss_limit: a day's realized loss (OUT deals since 00:00 UTC)
    against daily_loss_limit_pct of the balance at that moment. The wall only blocks NEW trades,
    so trades already open can push the loss past the limit (WARN); a bot trade opened after the
    limit was passed means the wall failed (FAULT). Only trades inside the window are judged.
    """
    if limit_pct is None:
        return check("loss_wall", WARN, "daily_loss_limit_pct is not set")
    # Balance just before each deal, rebuilt backwards from the current balance
    balance_before, running = [], balance_now
    for d in reversed(deals):
        running -= d["pnl"]
        balance_before.append(running)
    balance_before.reverse()

    breaches, overruns = [], []
    day, loss, hit_at, limit, last_day_time = None, 0.0, None, 0.0, None

    def close_day():
        if day and last_day_time and last_day_time >= since_utc and loss > limit:
            overruns.append(f"{day}: lost ${loss:.2f} (limit ${limit:.2f})")

    for d, bal in zip(deals, balance_before):
        d_day = d["time_utc"].strftime("%Y-%m-%d")
        if d_day != day:
            close_day()
            day, loss, hit_at = d_day, 0.0, None
        last_day_time = d["time_utc"]
        limit = max(bal, 0.0) * limit_pct / 100.0
        if d["entry"] == "in" and d["magic"] == BOT_MAGIC and hit_at and d["time_utc"] >= since_utc:
            breaches.append(f"{day}: {d['symbol']} opened at {d['time_utc']:%H:%M} UTC, after the "
                            f"limit was passed at {hit_at:%H:%M}")
        if d["entry"] == "out":
            loss -= d["pnl"]
            if hit_at is None and loss > limit:
                hit_at = d["time_utc"]
    close_day()
    if breaches:
        return check("loss_wall", FAULT, f"daily loss wall failed ({len(breaches)} trade(s)): "
                     + "; ".join(breaches[:3]))
    if overruns:
        return check("loss_wall", WARN, "loss went past the daily limit through trades that were already open: "
                     + "; ".join(overruns))
    return check("loss_wall", OK, "no trade opened after a daily loss limit was hit")


def mt5_checks(cfg: Dict, start_utc: datetime) -> List[Dict[str, str]]:
    try:
        import MetaTrader5 as mt5
        from core.mt5_time import get_server_utc_offset_seconds, history_deals_utc, server_epoch_to_utc
    except ImportError as e:
        return [check("mt5", FAULT, f"MetaTrader5 package missing: {e}")]
    if not mt5.initialize():
        return [check("mt5", FAULT, f"cannot connect to the MT5 terminal ({mt5.last_error()}). Is MT5 open?")]
    try:
        out = []
        acct, term = mt5.account_info(), mt5.terminal_info()
        if acct is None or term is None:
            return [check("mt5", FAULT, "connected but cannot read the account (logged out?)")]
        if not term.connected:
            out.append(check("mt5", FAULT, "terminal has no connection to the broker server"))
        elif not term.trade_allowed:
            out.append(check("mt5", FAULT, "Algo Trading is switched OFF in MT5 (toolbar button)"))
        else:
            out.append(check("mt5", OK, f"account {acct.login} ({acct.server}), balance {acct.balance:.2f} "
                                        f"{acct.currency}, equity {acct.equity:.2f}"))
        risk = cfg.get("risk_parameters", {})
        out.append(check_positions([p._asdict() for p in (mt5.positions_get() or [])], risk))

        offset = get_server_utc_offset_seconds()
        day_start = start_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        deals = []
        for d in history_deals_utc(day_start, datetime.now(timezone.utc) + timedelta(minutes=1), offset):
            if d.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL):
                entry = "other"            # deposit, withdrawal, credit: moves the balance only
            elif d.entry == mt5.DEAL_ENTRY_IN:
                entry = "in"
            elif d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT):
                entry = "out"
            else:
                continue
            deals.append({"time_utc": server_epoch_to_utc(d.time, offset), "entry": entry, "magic": d.magic,
                          "symbol": d.symbol,
                          "pnl": d.profit + d.swap + d.commission + getattr(d, "fee", 0.0)})
        deals.sort(key=lambda d: d["time_utc"])
        out.append(check_loss_wall(deals, acct.balance, risk.get("daily_loss_limit_pct"), start_utc))
        return out
    finally:
        mt5.shutdown()


def check_brain(cfg: Dict, now_utc: datetime) -> Dict[str, str]:
    if not cfg.get("brain", {}).get("enabled"):
        return check("brain", OK, "disabled in config")
    path = AGENT_DIR / "state" / "brain" / "journal.jsonl"
    last = None
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") == "run":
                last = e
    if not last:
        return check("brain", WARN, "the brain has never completed a run")
    age_h = (now_utc - datetime.fromisoformat(last["ts"])).total_seconds() / 3600
    status = last.get("status")
    if age_h > BRAIN_MAX_AGE_H:
        return check("brain", WARN, f"last run {age_h:.0f}h ago (status {status})")
    if status != "ok":
        return check("brain", WARN, f"last run {age_h:.0f}h ago ended with status {status!r}")
    return check("brain", OK, f"last run {age_h:.0f}h ago, ok")


def check_disk() -> Dict[str, str]:
    free_gb = shutil.disk_usage(AGENT_DIR).free / 1e9
    if free_gb < MIN_FREE_DISK_GB:
        return check("disk", WARN, f"only {free_gb:.1f} GB free")
    return check("disk", OK, f"{free_gb:.0f} GB free")


def run(hours: float) -> Dict:
    cfg = json.loads((AGENT_DIR / "config.json").read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    agent_records = read_logs("agent*.log", start)
    cycles = [r[0] for r in agent_records if "Starting analysis cycle" in r[3]]
    errors = [r for r in read_logs("system_errors*.log", start) if r[1] in ("ERROR", "CRITICAL")]

    checks = [check_agent(cfg, agent_records[-1][0] if agent_records else None, now),
              check_activity(cycles, start, now, agent_records[0][0] if agent_records else None),
              check_errors(errors, hours),
              check_ollama(cfg)]
    checks += mt5_checks(cfg, start)
    checks += [check_brain(cfg, now), check_disk()]
    return {"checked_at": now.isoformat(timespec="seconds"), "window_hours": hours,
            "status": overall(checks), "checks": checks}


def save(result: Dict):
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    (HEALTH_DIR / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(LOGS_DIR / "health.log", "a", encoding="utf-8") as f:
        for c in result["checks"]:
            f.write(f"{result['checked_at']} | {c['status']:<5} | {c['name']:<9} | {c['detail']}\n")


def print_result(result: Dict):
    colour = {OK: "\033[92m", WARN: "\033[93m", FAULT: "\033[91m"}
    reset, bold = "\033[0m", "\033[1m"
    local = datetime.fromisoformat(result["checked_at"]).astimezone()
    print(f"\n{bold}BUBAT AI HEALTH CHECK{reset}  {local:%a %Y-%m-%d %H:%M} (last {result['window_hours']:g}h)\n")
    for c in result["checks"]:
        print(f"  {colour[c['status']]}{c['status']:<5}{reset}  {c['name']:<9}  {c['detail']}")
    s = result["status"]
    verdict = {OK: "All good.", WARN: "Working, but look at the WARN lines.",
               FAULT: "Something is broken: fix the FAULT lines, or ask Claude to audit the bot."}[s]
    print(f"\n  {colour[s]}{bold}{s}{reset}  {verdict}\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()
    result = run(args.hours)
    save(result)
    print_result(result)
    sys.exit(_RANK[result["status"]])


if __name__ == "__main__":
    main()
