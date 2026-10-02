"""
maintenance/daily_report.py
============================
Computes 24h trading performance metrics from MT5 deal history.
Saves JSON to reports/ and prints a summary to stdout.
Reusable across daily post-mortem cycles.
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not installed.")
    sys.exit(1)


def collect_24h_report(hours: int = 24) -> dict:
    """Collect and compute all 24h performance metrics from MT5."""
    if not mt5.initialize():
        return {"error": f"MT5 init failed: {mt5.last_error()}"}

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(hours=hours)

    acc = mt5.account_info()
    account = {
        "balance": acc.balance,
        "equity": acc.equity,
        "free_margin": acc.margin_free,
        "margin_level": acc.margin_level if acc.margin_level else None,
        "open_positions": mt5.positions_total(),
    }

    deals = mt5.history_deals_get(start_utc, now_utc)
    if not deals:
        mt5.shutdown()
        return {
            "window_start": start_utc.isoformat(),
            "window_end": now_utc.isoformat(),
            "account": account,
            "total_deals": 0,
            "closed_trades": 0,
        }

    # Build position entry map
    entry_map = {}
    wins, losses, breakevens = 0, 0, 0
    total_profit, total_swap, total_commission = 0.0, 0.0, 0.0
    win_amounts, loss_amounts = [], []
    running_pnl, max_dd = 0.0, 0.0
    hold_times = []
    buys_w, buys_l, sells_w, sells_l = 0, 0, 0, 0
    entry_deals, exit_deals = 0, 0

    # Per-symbol and per-hour breakdowns
    sym_stats = defaultdict(lambda: {"w": 0, "l": 0, "be": 0, "pnl": 0.0})
    hour_stats = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0.0})

    for d in deals:
        if d.entry == 0:  # Entry
            entry_map[d.position_id] = d
            entry_deals += 1
        elif d.entry == 1:  # Exit
            exit_deals += 1
            profit = d.profit
            total_profit += profit
            total_swap += d.swap
            total_commission += d.commission
            running_pnl += profit + d.swap + d.commission
            if running_pnl < max_dd:
                max_dd = running_pnl

            # Hold time
            if d.position_id in entry_map:
                ed = entry_map[d.position_id]
                hold_times.append(d.time - ed.time)

            # Exit hour
            exit_hour = datetime.fromtimestamp(d.time, tz=timezone.utc).hour

            if profit > 0.005:
                wins += 1
                win_amounts.append(profit)
                sym_stats[d.symbol]["w"] += 1
                hour_stats[exit_hour]["w"] += 1
                if d.position_id in entry_map:
                    if entry_map[d.position_id].type == 0:
                        buys_w += 1
                    else:
                        sells_w += 1
            elif profit < -0.005:
                losses += 1
                loss_amounts.append(profit)
                sym_stats[d.symbol]["l"] += 1
                hour_stats[exit_hour]["l"] += 1
                if d.position_id in entry_map:
                    if entry_map[d.position_id].type == 0:
                        buys_l += 1
                    else:
                        sells_l += 1
            else:
                breakevens += 1
                sym_stats[d.symbol]["be"] += 1

            sym_stats[d.symbol]["pnl"] += profit

    total_closed = wins + losses + breakevens
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
    avg_win = sum(win_amounts) / len(win_amounts) if win_amounts else 0
    avg_loss = sum(loss_amounts) / len(loss_amounts) if loss_amounts else 0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    gross_win = sum(win_amounts)
    gross_loss = sum(loss_amounts)
    profit_factor = abs(gross_win / gross_loss) if gross_loss != 0 else float("inf")
    net_pnl = total_profit + total_swap + total_commission
    expectancy = net_pnl / total_closed if total_closed > 0 else 0
    avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0
    trades_per_hour = total_closed / hours

    # Active positions
    positions = mt5.positions_get()
    active = []
    floating_pnl = 0.0
    if positions:
        for p in positions:
            ptype = "BUY" if p.type == 0 else "SELL"
            active.append({
                "symbol": p.symbol,
                "type": ptype,
                "volume": p.volume,
                "profit": round(p.profit, 2),
                "sl": p.sl,
                "tp": p.tp,
                "ticket": p.ticket,
            })
            floating_pnl += p.profit

    mt5.shutdown()

    report = {
        "generated_at": now_utc.isoformat(),
        "window_start": start_utc.isoformat(),
        "window_end": now_utc.isoformat(),
        "account": account,
        "total_deals": len(deals),
        "entry_deals": entry_deals,
        "exit_deals": exit_deals,
        "closed_trades": total_closed,
        "wins": wins,
        "losses": losses,
        "breakevens": breakevens,
        "win_rate_pct": round(win_rate, 1),
        "gross_pnl": round(total_profit, 2),
        "swap": round(total_swap, 2),
        "commission": round(total_commission, 2),
        "net_pnl": round(net_pnl, 2),
        "avg_winner": round(avg_win, 2),
        "avg_loser": round(avg_loss, 2),
        "payoff_ratio": round(payoff, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "expectancy_per_trade": round(expectancy, 4),
        "max_drawdown": round(max_dd, 2),
        "avg_hold_time_min": round(avg_hold / 60, 1),
        "trades_per_hour": round(trades_per_hour, 1),
        "direction": {
            "buy_wins": buys_w,
            "buy_losses": buys_l,
            "sell_wins": sells_w,
            "sell_losses": sells_l,
        },
        "per_symbol": {
            s: dict(v) for s, v in sorted(sym_stats.items(), key=lambda x: x[1]["pnl"])
        },
        "per_hour": {str(h): dict(v) for h, v in sorted(hour_stats.items())},
        "active_positions": active,
        "floating_pnl": round(floating_pnl, 2),
    }
    return report


def print_report(report: dict):
    """Print a human-readable summary."""
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return

    print(f"Window: {report['window_start'][:16]}Z to {report['window_end'][:16]}Z")
    print(f"Balance: ${report['account']['balance']:.2f} | Equity: ${report['account']['equity']:.2f}")
    print(f"\nClosed: {report['closed_trades']} ({report['wins']}W/{report['losses']}L/{report['breakevens']}BE)")
    print(f"Win Rate: {report['win_rate_pct']}%")
    print(f"Net P&L: ${report['net_pnl']} (Gross: ${report['gross_pnl']}, Swap: ${report['swap']}, Comm: ${report['commission']})")
    print(f"Avg Winner: ${report['avg_winner']} | Avg Loser: ${report['avg_loser']} | Payoff: {report['payoff_ratio']}")
    print(f"Profit Factor: {report['profit_factor']} | Expectancy: ${report['expectancy_per_trade']}/trade")
    print(f"Max DD: ${report['max_drawdown']} | Avg Hold: {report['avg_hold_time_min']}min | Trades/hr: {report['trades_per_hour']}")
    d = report["direction"]
    print(f"BUY: {d['buy_wins']}W/{d['buy_losses']}L | SELL: {d['sell_wins']}W/{d['sell_losses']}L")

    print(f"\nPer-Symbol (worst to best):")
    for sym, st in report["per_symbol"].items():
        total = st["w"] + st["l"] + st.get("be", 0)
        wr = st["w"] / (st["w"] + st["l"]) * 100 if (st["w"] + st["l"]) > 0 else 0
        print(f"  {sym:8s} {st['w']}W/{st['l']}L WR={wr:.0f}% P&L=${st['pnl']:.2f}")

    print(f"\nActive: {len(report['active_positions'])} positions, floating ${report['floating_pnl']}")


def save_report(report: dict):
    """Save report JSON to reports/ directory."""
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filepath = reports_dir / f"report_{date_str}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {filepath}")
    return filepath


if __name__ == "__main__":
    report = collect_24h_report()
    print_report(report)
    save_report(report)
