"""Condense a daily_report JSON (~300 KB) into the few numbers the brain reasons over (fits an 8K context)."""
import re
from typing import Any, Dict, List

SUMMARY_KEYS = ("trades", "win_rate_pct", "net_pnl", "profit_factor", "payoff_ratio", "avg_r",
                "expectancy_per_trade", "max_drawdown", "trades_per_hour")
GROUP_KEYS = ("trades", "win_rate_pct", "net_pnl", "avg_r")


def _pick(d: Dict, keys) -> Dict:
    return {k: d.get(k) for k in keys if k in d}


def _ranked(group: Dict[str, Dict], n: int) -> Dict[str, List]:
    rows = [(name, _pick(v, GROUP_KEYS)) for name, v in group.items() if isinstance(v, dict)]
    rows.sort(key=lambda r: r[1].get("net_pnl") or 0)
    return {"worst": rows[:n], "best": rows[-n:][::-1] if len(rows) > n else []}


def build_digest(report: Dict[str, Any]) -> Dict[str, Any]:
    br = report.get("breakdowns", {})
    base = report.get("baselines_hourly_dedup", {})
    digest = {
        "window": {"start": report.get("window_start"), "end": report.get("window_end")},
        "account": _pick(report.get("account", {}), ("balance", "equity", "currency")),
        "open_positions": len(report.get("open_positions", [])),
        "summary": _pick(report.get("summary", {}), SUMMARY_KEYS),
        "by_session": {k: _pick(v, GROUP_KEYS) for k, v in br.get("session", {}).items()},
        "by_symbol": _ranked(br.get("symbol", {}), 4),
        "by_direction": {k: _pick(v, GROUP_KEYS) for k, v in br.get("direction", {}).items()},
        "by_h1_alignment": {k: _pick(v, GROUP_KEYS) for k, v in br.get("h1_alignment", {}).items()},
        "by_confidence": {k: _pick(v, GROUP_KEYS) for k, v in br.get("confidence_bucket", {}).items()},
        "exit_types": report.get("payoff_asymmetry", {}).get("exit_type_counts", {}),
        "cost": _pick(report.get("cost_drag", {}), ("total_cost", "cost_pct_of_gross_wins", "avg_spread_as_pct_of_sl")),
        # Simulated avg R of signals each risk wall blocked: negative = the wall saved money
        "blocked_by_wall_avg_r": {k: {"n": v.get("n"), "avg_r": v.get("avg_r")}
                                  for k, v in report.get("wall_counterfactuals_hourly_dedup", {}).items()},
        # Simulated avg R of simple strategies over the same decision points
        "baselines_avg_r": {k: {"n": v.get("n"), "avg_r": v.get("avg_r")}
                            for k, v in base.items() if isinstance(v, dict) and "avg_r" in v},
        "llm_parse_failures": report.get("llm_health", {}).get("defensive_wait_fallbacks"),
        "top_errors": dict(sorted(report.get("errors", {}).items(), key=lambda kv: -kv[1])[:3]),
        "sample_sizes": sample_sizes(report),
    }
    total = digest["sample_sizes"]["total"]
    if total < MIN_SAMPLE:
        digest["note"] = (f"Only {total} closed trades in this window: too few to justify any config, symbol "
                          f"or rule change (minimum {MIN_SAMPLE}). Research and ideas only.")
    return digest


MIN_SAMPLE = 30


def sample_sizes(report: Dict[str, Any]) -> Dict[str, Any]:
    """Closed-trade counts the proposal validator uses to refuse changes based on noise."""
    br = report.get("breakdowns", {})
    return {"total": int(report.get("summary", {}).get("trades") or 0),
            "by_symbol": {k: v.get("trades", 0) for k, v in br.get("symbol", {}).items() if isinstance(v, dict)},
            "by_session": {k: v.get("trades", 0) for k, v in br.get("session", {}).items() if isinstance(v, dict)}}


def headline(digest: Dict[str, Any]) -> Dict[str, Any]:
    """Few numbers stored with a proposal so its effect can be judged later."""
    s = digest.get("summary", {})
    return {k: s.get(k) for k in ("trades", "win_rate_pct", "net_pnl", "profit_factor", "avg_r")}


def _money(x) -> str:
    """-147.79 -> "-$147.79" (account currency is USD), so net P&L is never read as R."""
    try:
        return f"{'-' if float(x) < 0 else ''}${abs(float(x)):.2f}"
    except (TypeError, ValueError):
        return str(x)


def _groups(group: Dict[str, Dict], label: str) -> List[str]:
    """Rank groups with enough trades by avg R; list the rest as too small to judge."""
    big = {k: v for k, v in group.items() if (v.get("trades") or 0) >= MIN_SAMPLE and v.get("avg_r") is not None}
    small = {k: v.get("trades") for k, v in group.items() if 0 < (v.get("trades") or 0) < MIN_SAMPLE}
    out = []
    if big:
        ranked = sorted(big.items(), key=lambda kv: kv[1]["avg_r"])
        out.append(f"{label} with >= {MIN_SAMPLE} trades, worst to best by avg R: " + ", ".join(
            f"{k} {v['avg_r']:+.2f}R ({v['trades']} trades, net {_money(v.get('net_pnl'))})" for k, v in ranked))
    if small:
        out.append(f"{label} with too few trades to judge: " + ", ".join(f"{k} ({n})" for k, n in small.items()))
    return out


def key_facts(digest: Dict[str, Any]) -> List[str]:
    """Conclusions computed in code, so the model does not have to read raw tables correctly."""
    s, base = digest.get("summary", {}), digest.get("baselines_avg_r", {})
    facts = [f"REAL results: {s.get('trades')} trades, win rate {s.get('win_rate_pct')}%, net {_money(s.get('net_pnl'))}, "
             f"profit factor {s.get('profit_factor')}, avg {s.get('avg_r')}R per trade, "
             f"max drawdown {_money(s.get('max_drawdown'))}."]
    llm, h1 = base.get("llm_all_buy_sell", {}).get("avg_r"), base.get("follow_h1_trend", {}).get("avg_r")
    if llm is not None and h1 is not None:
        facts.append(f"SIMULATED over the same decision points (no spread, no trade management): LLM signals "
                     f"{llm:+.3f}R, simple 'follow the H1 trend' {h1:+.3f}R, always wait 0R. The LLM "
                     f"{'beats' if llm > h1 else 'does not beat'} the H1-trend baseline. Real results "
                     f"({s.get('avg_r')}R) are far below the simulation, so costs, timing and trade "
                     f"management lose money beyond the signal itself.")
    for wall, w in digest.get("blocked_by_wall_avg_r", {}).items():
        if (w.get("n") or 0) >= 20 and w.get("avg_r") is not None:
            # No R figure here: models kept quoting it as a loss the bot made
            verdict = ("they would have lost money, so the wall avoided losses (working)" if w["avg_r"] < 0
                       else "they would have won, so the wall blocked winning trades (costing money)")
            facts.append(f"{wall} wall BLOCKED {w['n']} trades that were never taken: {verdict}.")
    facts += _groups(digest.get("by_session", {}), "Sessions")
    sym = digest.get("by_symbol", {})
    facts += _groups({k: v for k, v in sym.get("worst", []) + sym.get("best", [])}, "Symbols (worst/best only)")
    facts += _groups(digest.get("by_h1_alignment", {}), "H1 trend alignment")
    facts += _groups(digest.get("by_confidence", {}), "LLM confidence buckets")
    facts += _groups(digest.get("by_direction", {}), "Direction")
    cost = digest.get("cost", {})
    if cost:
        facts.append(f"Trading costs: {cost.get('total_cost')} total, {cost.get('cost_pct_of_gross_wins')}% of "
                     f"gross wins; spread averages {cost.get('avg_spread_as_pct_of_sl')}% of the stop-loss.")
    return facts


_BUCKET_FLOOR = {"<0.80": 0.0, "0.80-0.84": 0.80, "0.85-0.89": 0.85, ">=0.90": 0.90}


def _pool(rows: List[Dict]) -> Dict[str, Any]:
    n = sum(r.get("trades") or 0 for r in rows)
    if not n:
        return {"trades": 0, "avg_r": None, "net_pnl": 0}
    avg = sum((r.get("avg_r") or 0) * (r.get("trades") or 0) for r in rows) / n
    return {"trades": n, "avg_r": round(avg, 3), "net_pnl": round(sum(r.get("net_pnl") or 0 for r in rows), 2)}


def proposal_check(p: Dict[str, Any], digest: Dict[str, Any]) -> str:
    """What the report itself says about a proposal, computed in code.

    The brain's own `evidence` text is model output and has misread the tables (P3 cited a
    "<0.80" bucket that holds no trades), so the chat and the next reflection judge proposals on this.
    """
    total = digest.get("summary", {}).get("trades") or 0
    key = p.get("key", "")
    if p.get("kind") == "config" and key.endswith("confidence_threshold"):
        new = float(p.get("value", 0))
        buckets = digest.get("by_confidence", {})
        if new not in _BUCKET_FLOOR.values():
            return f"The report's confidence buckets cannot show the effect of a {new} threshold."
        skipped = _pool([v for k, v in buckets.items() if k in _BUCKET_FLOOR and _BUCKET_FLOOR[k] < new])
        kept = _pool([v for k, v in buckets.items() if k in _BUCKET_FLOOR and _BUCKET_FLOOR[k] >= new])
        pct = round(100 * skipped["trades"] / total) if total else 0
        thin = " (too few to judge)" if kept["trades"] < MIN_SAMPLE else ""
        return (f"A {new} threshold would have skipped {skipped['trades']} of {total} trades ({pct}%), which "
                f"averaged {skipped['avg_r']}R (net {_money(skipped['net_pnl'])}). The {kept['trades']} trades it "
                f"keeps averaged {kept['avg_r']}R (net {_money(kept['net_pnl'])}){thin}.")
    if p.get("kind") == "remove_symbol":
        sym = p.get("symbol")
        rows = dict(digest.get("by_symbol", {}).get("worst", []) + digest.get("by_symbol", {}).get("best", []))
        if sym in rows:
            r = rows[sym]
            thin = " (too few to judge)" if (r.get("trades") or 0) < MIN_SAMPLE else ""
            return (f"{sym}: {r.get('trades')} trades{thin}, avg {r.get('avg_r')}R, net {_money(r.get('net_pnl'))}, "
                    f"win rate {r.get('win_rate_pct')}%. Removing it would have avoided that result.")
        return f"{sym} is not among the report's worst or best symbols, so the report gives no direct evidence."
    m = re.fullmatch(r"session_profiles\.([A-Z_]+)\.[a-z_]+", key)
    if p.get("kind") == "config" and m:
        r = digest.get("by_session", {}).get(m.group(1))
        if r:
            return (f"{m.group(1)} session: {r.get('trades')} trades, avg {r.get('avg_r')}R, net {_money(r.get('net_pnl'))}.")
    if p.get("kind") == "config" and "spread" in key:
        w = digest.get("blocked_by_wall_avg_r", {}).get("SPREAD", {})
        c = digest.get("cost", {})
        return (f"Spread averages {c.get('avg_spread_as_pct_of_sl')}% of the stop-loss. The current spread wall "
                f"already blocks the worst spreads ({w.get('n')} trades, which would have lost money). The report "
                f"does not show how many taken trades a tighter cap would also have blocked.")
    return "The report has no direct evidence for this change."
