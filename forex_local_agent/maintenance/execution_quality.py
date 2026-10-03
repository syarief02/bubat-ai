"""
Execution Quality Report
========================
Measures where R is lost between the signal and the realised trade:

- entry timing: real fill (real SL/TP, unmanaged) vs a shadow entry at the next
  M5 bar open after the fill, same direction, signal geometry (ATR SL/TP)
- trade management: actual R vs the same trade unmanaged
- spread cost: actual R by spread / SL-distance bucket
- quick re-entries: same symbol re-opened within 30 min of a close
- Asia range-fade shadow (paper) record vs its promotion bar

Read-only (history + bars). Usage:
    python maintenance/execution_quality.py --hours 24
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))

import MetaTrader5 as mt5  # noqa: E402

from core.mt5_time import get_server_utc_offset_seconds  # noqa: E402
from core.session_strategies import AsiaRangeFadeShadow, summarize_shadow  # noqa: E402
from core.trade_analytics import quick_reentries  # noqa: E402
from maintenance.daily_report import BarCache, collect_trades, simulate_signals  # noqa: E402

SPREAD_BUCKETS = ((0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, float("inf")))


def _stats(xs: List[float]) -> Dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "avg_r": round(float(np.mean(xs)), 3), "sum_r": round(float(np.sum(xs)), 1)}


def build_report(hours: float) -> Dict:
    if not mt5.initialize():
        return {"error": f"MT5 init failed: {mt5.last_error()}"}
    try:
        offset = get_server_utc_offset_seconds(force=True)
        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(hours=hours)
        bars = BarCache(offset)
        trades = [t for t in collect_trades(start_utc, end_utc, offset, bars, sim=True) if t.get("r") is not None]

        signals = [{"symbol": t["symbol"], "ts": t["entry_ts"], "position_id": t["position_id"],
                    "direction": 1 if t["direction"] == "BUY" else -1} for t in trades]
        shadow = {s["position_id"]: s["r"] for s in simulate_signals(signals, bars, start_utc, end_utc, offset)}
        paired = [t for t in trades if t["position_id"] in shadow and t.get("unmanaged_r") is not None]

        spread_rows = {}
        for lo, hi in SPREAD_BUCKETS:
            g = [t for t in trades if t.get("spread_pips") is not None and t.get("sl_pips")
                 and lo <= t["spread_pips"] / t["sl_pips"] < hi]
            spread_rows[f"{lo:.2f}-{hi:.2f}"] = _stats([t["r"] for t in g])

        reentries = quick_reentries(trades)
        asia = AsiaRangeFadeShadow(json.loads((AGENT_DIR / "config.json").read_text(encoding="utf-8")))
        asia_results = asia.load_results()
        return {
            "generated_at": end_utc.isoformat(),
            "window_start": start_utc.isoformat(),
            "window_end": end_utc.isoformat(),
            "trades": len(trades),
            "actual": _stats([t["r"] for t in trades]),
            "entry_timing": {
                "shadow_next_bar_open": _stats([shadow[t["position_id"]] for t in paired]),
                "real_fill_unmanaged": _stats([t["unmanaged_r"] for t in paired]),
                "note": "shadow ignores spread; real fill includes it",
            },
            "management": {
                "unmanaged": _stats([t["unmanaged_r"] for t in paired]),
                "actual": _stats([t["r"] for t in paired]),
            },
            "spread_to_sl": spread_rows,
            "quick_reentries_30m": {**_stats([t["r"] for t in reentries]),
                                    "share_pct": round(100 * len(reentries) / len(trades), 1) if trades else 0.0},
            "asia_shadow_all_time": {**summarize_shadow(asia_results, asia.cfg),
                                     "open_paper_trades": len(asia.open_trades),
                                     "promotion_bar": {"trades": asia.cfg["promote_after_trades"],
                                                       "min_avg_r": asia.cfg["promote_min_avg_r"]}},
        }
    finally:
        mt5.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()
    report = build_report(args.hours)
    print(json.dumps(report, indent=2))
    out_dir = AGENT_DIR / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"execution_quality_{datetime.now(timezone.utc):%Y-%m-%d_%H%M}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
