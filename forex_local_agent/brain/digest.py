"""Condense a daily_report JSON (~300 KB) into the few numbers the brain reasons over (fits an 8K context)."""
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
