"""
Model benchmark: compare Ollama models on the real AgentLogic prompt (no MT5, no orders).

Five synthetic scenarios with an unambiguous expected answer, run twice each:
clear uptrend (BUY), clear downtrend (SELL), M5/H1 conflict (WAIT), ranging (WAIT),
uptrend 15 min before NFP (WAIT). Run before changing `active_model`:

    python maintenance/model_benchmark.py agent-brain:32k agent-brain-8b:8k

Passing only shows the model follows the prompt's rules; it says nothing about edge.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))
os.chdir(AGENT_DIR)

from core.agent_logic import AgentLogic


def tech(sym, price, trend, bias, h1, rsi, rsi_c, macd_h, macd_c, chg):
    return {"symbol": sym, "timeframe": "M5", "current_price": price, "trend_structure": trend,
            "technical_bias": bias, "higher_timeframe_h1": h1, "rsi": rsi, "rsi_condition": rsi_c,
            "macd": macd_h * 2, "macd_signal": macd_h, "macd_hist": macd_h, "macd_condition": macd_c,
            "atr": 0.00042, "change_pct": chg, "high_range": price * 1.002, "low_range": price * 0.997}


NO_NEWS = {"headlines": [], "economic_calendar_summary": "No high-impact events in the next 4 hours."}
SCENARIOS = [
    ("clear uptrend, H1 bullish", "BUY",
     tech("EURUSD", 1.1255, "Price above EMA20 > EMA50 (uptrend)", "BULLISH", "BULLISH BIAS", 61.0, "BULLISH MOMENTUM", 0.00008, "BULLISH CROSSOVER", 0.21), NO_NEWS),
    ("clear downtrend, H1 bearish", "SELL",
     tech("GBPUSD", 1.3010, "Price below EMA20 < EMA50 (downtrend)", "BEARISH", "BEARISH BIAS", 38.0, "BEARISH MOMENTUM", -0.00009, "BEARISH CROSSOVER", -0.25), NO_NEWS),
    ("M5 up but H1 bearish (conflict)", "WAIT",
     tech("AUDUSD", 0.6610, "Price above EMA20 > EMA50 (uptrend)", "BULLISH", "BEARISH BIAS", 58.0, "BULLISH MOMENTUM", 0.00004, "BULLISH", 0.08), NO_NEWS),
    ("flat / ranging", "WAIT",
     tech("EURCHF", 0.9350, "EMAs flat and intertwined (ranging)", "NEUTRAL", "NEUTRAL", 50.2, "NEUTRAL", 0.000001, "NEUTRAL", 0.01), NO_NEWS),
    ("uptrend but NFP in 15 min", "WAIT",
     tech("USDJPY", 147.80, "Price above EMA20 > EMA50 (uptrend)", "BULLISH", "BULLISH BIAS", 63.0, "BULLISH MOMENTUM", 0.02, "BULLISH CROSSOVER", 0.18),
     {"headlines": [], "economic_calendar_summary": "HIGH IMPACT: USD Non-Farm Payrolls in 15 minutes."}),
]


async def run(model, repeats=2):
    a = AgentLogic("config.json")
    a.default_model = model
    rows = []
    # Warm-up load (not timed)
    await a.query_ollama("ping", 'Reply with {"ok": true}')
    for name, want, t, n in SCENARIOS:
        for _ in range(repeats):
            s = time.time()
            d = await a.get_trade_decision(t, n, [])
            rows.append({"scenario": name, "want": want, "got": d.decision, "conf": d.confidence_score,
                         "fallback": a.last_decision_meta.get("fallback"), "secs": round(time.time() - s, 1),
                         "reason": d.reasoning[:110]})
    return rows


async def main():
    for model in sys.argv[1:]:
        rows = await run(model)
        ok = sum(r["got"] == r["want"] for r in rows)
        print(f"\n===== {model}: {ok}/{len(rows)} correct, "
              f"fallbacks={sum(bool(r['fallback']) for r in rows)}, "
              f"avg {sum(r['secs'] for r in rows) / len(rows):.1f}s/decision")
        for r in rows:
            mark = "OK " if r["got"] == r["want"] else "XX "
            print(f"{mark}{r['scenario']:<34} want={r['want']:<4} got={r['got']:<4} conf={r['conf']:.2f} {r['secs']}s | {r['reason']}")


if __name__ == "__main__":
    asyncio.run(main())
