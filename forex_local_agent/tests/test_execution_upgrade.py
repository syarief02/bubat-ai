"""
Regression tests for the cycle #6 execution upgrade (offline: MT5 is mocked, no orders are sent).

Run: python tests/test_execution_upgrade.py
"""

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
WORKSPACE_DIR = AGENT_DIR.parent
for p in (str(AGENT_DIR), str(WORKSPACE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.trade_analytics import (entry_window_open, h1_strength, quick_reentries, rank_signals,
                                  spread_to_sl_ratio, wall_from_message)
from core.mt5_engine import MT5Engine
from maintenance import daily_report

CONFIG = AGENT_DIR / "config.json"


class TestPureHelpers(unittest.TestCase):
    def test_spread_to_sl_ratio(self):
        self.assertAlmostEqual(spread_to_sl_ratio(1.10010, 1.10000, 0.00150), 0.0667, places=3)
        self.assertIsNone(spread_to_sl_ratio(1.1001, 1.1, 0))

    def test_entry_window(self):
        self.assertTrue(entry_window_open(7, [7, 13]))
        self.assertTrue(entry_window_open(12, [7, 13]))
        self.assertFalse(entry_window_open(13, [7, 13]))
        self.assertFalse(entry_window_open(3, [7, 13]))
        self.assertTrue(entry_window_open(23, [22, 2]))   # wraps midnight
        self.assertTrue(entry_window_open(1, [22, 2]))
        self.assertFalse(entry_window_open(5, [22, 2]))
        self.assertTrue(entry_window_open(3, None))

    def test_h1_strength_matches_engine_labels(self):
        self.assertEqual(h1_strength("BULLISH (H1 Uptrend above 20 & 50 EMA)"), 2)
        self.assertEqual(h1_strength("BEARISH (H1 Downtrend below 20 & 50 EMA)"), 2)
        self.assertEqual(h1_strength("BULLISH BIAS (Above H1 20 EMA)"), 1)
        self.assertEqual(h1_strength("NEUTRAL"), 0)

    def test_rank_full_trend_then_cheapest_spread(self):
        c = [
            {"symbol": "EURUSD", "h1_trend": "BULLISH BIAS (Above H1 20 EMA)", "spread_sl_ratio": 0.01},
            {"symbol": "NZDCHF", "h1_trend": "BEARISH (H1 Downtrend below 20 & 50 EMA)", "spread_sl_ratio": 0.09},
            {"symbol": "GBPUSD", "h1_trend": "BULLISH (H1 Uptrend above 20 & 50 EMA)", "spread_sl_ratio": 0.02},
            {"symbol": "AUDNZD", "h1_trend": "BULLISH (H1 Uptrend above 20 & 50 EMA)", "spread_sl_ratio": None},
        ]
        self.assertEqual([x["symbol"] for x in rank_signals(c)], ["GBPUSD", "NZDCHF", "AUDNZD", "EURUSD"])

    def test_quick_reentries(self):
        trades = [
            {"symbol": "EURUSD", "entry_ts": 0, "exit_ts": 600},
            {"symbol": "EURUSD", "entry_ts": 900, "exit_ts": 2000},     # 5 min after close -> quick
            {"symbol": "EURUSD", "entry_ts": 9000, "exit_ts": 9500},    # 2 h later -> not quick
            {"symbol": "GBPUSD", "entry_ts": 700, "exit_ts": 800},
        ]
        self.assertEqual([t["entry_ts"] for t in quick_reentries(trades)], [900])

    def test_new_wall_labels(self):
        self.assertEqual(wall_from_message("REJECTED: Spread cost on NZDCHF is 9.1% of SL distance (max 6%)"), "SPREAD_COST")
        self.assertEqual(wall_from_message("REJECTED: Spread on X is 4.0 pips (exceeds max allowed 3.5 pips)"), "SPREAD")
        self.assertEqual(wall_from_message("REJECTED: Outside entry window for X (07:00-13:00 UTC, now 03h)"), "SESSION_WINDOW")
        self.assertEqual(wall_from_message("REJECTED: Confidence 0.95 on X exceeds calibration cap 0.89"), "CONFIDENCE_CAP")


class TestExecutionWalls(unittest.TestCase):
    PARAMS = {"symbol": "EURUSD", "action": "BUY", "order_type": 0, "entry": 1.1, "sl": 1.0985,
              "tp": 1.10225, "lot": 0.01, "risk_amount": 1.5, "confidence": 0.82}

    def setUp(self):
        self.engine = MT5Engine(str(CONFIG))
        self.send = patch("core.mt5_engine.mt5.order_send", side_effect=AssertionError("order_send called"))
        self.send.start()

    def tearDown(self):
        self.send.stop()

    def test_config_keys_wired(self):
        risk = json.loads(CONFIG.read_text())["risk_parameters"]
        self.assertEqual(self.engine.max_spread_sl_ratio, risk["max_spread_sl_ratio"])
        self.assertEqual(self.engine.max_confidence, risk["max_confidence"])
        self.assertEqual(self.engine.entry_hours_utc, risk["entry_hours_utc"])

    def test_outside_entry_window_rejected_before_any_mt5_call(self):
        hour = datetime.now(timezone.utc).hour
        self.engine.entry_hours_utc = [(hour + 1) % 24, (hour + 2) % 24]
        with patch.object(self.engine, "get_technical_data") as tech:
            res = self.engine.execute_trade(dict(self.PARAMS))
        self.assertEqual(res["status"], "rejected")
        self.assertEqual(wall_from_message(res["message"]), "SESSION_WINDOW")
        tech.assert_not_called()

    def test_overconfident_signal_rejected(self):
        self.engine.entry_hours_utc = None
        with patch.object(self.engine, "get_technical_data") as tech:
            res = self.engine.execute_trade({**self.PARAMS, "confidence": 0.95})
        self.assertEqual(wall_from_message(res["message"]), "CONFIDENCE_CAP")
        tech.assert_not_called()

    def test_current_spread_sl_ratio(self):
        with patch("core.mt5_engine.mt5.symbol_info_tick", return_value=MagicMock(ask=1.10010, bid=1.10000)):
            self.assertAlmostEqual(self.engine.current_spread_sl_ratio("EURUSD", 0.0015), 0.0667, places=3)
        with patch("core.mt5_engine.mt5.symbol_info_tick", return_value=None):
            self.assertIsNone(self.engine.current_spread_sl_ratio("EURUSD", 0.0015))


class TestSignalSimulationNoLookahead(unittest.TestCase):
    def test_entry_is_next_bar_open_not_forming_bar_close(self):
        n = 40
        t = np.arange(n) * 300.0
        opens = np.full(n, 1.1000)
        closes = np.full(n, 1.1000)
        highs = np.full(n, 1.1010)
        lows = np.full(n, 1.0990)
        closes[20] = 1.1050          # forming bar's close: future information at decision time
        highs[20] = 1.1055
        highs[25] = 1.1200           # TP hit later, SL never hit
        bars = MagicMock()
        bars.load.return_value = {"t_utc": t, "open": opens, "high": highs, "low": lows,
                                  "close": closes, "atr": np.full(n, 0.001)}
        sig = [{"symbol": "EURUSD", "ts": 20 * 300 + 120, "direction": 1, "sl": 1.0900, "tp": 1.1200}]
        with patch.object(daily_report.mt5, "symbol_info", return_value=MagicMock(point=0.00001, digits=5)):
            out = daily_report.simulate_signals(sig, bars, datetime.fromtimestamp(0, tz=timezone.utc),
                                                datetime.fromtimestamp(10 ** 6, tz=timezone.utc), 0)
        self.assertEqual(len(out), 1)
        # entry 1.1000 (next open): R = 0.0200 / 0.0100 = 2.0; the old lookahead entry 1.1050 gave 1.0
        self.assertAlmostEqual(out[0]["r"], 2.0, places=3)


class TestRankedCycle(unittest.TestCase):
    """main.ForexAgent defers tradeable signals and executes them best-first (all subsystems mocked)."""

    def _agent(self):
        from unittest.mock import AsyncMock
        import main as main_mod
        a = object.__new__(main_mod.ForexAgent)
        a.config = {"risk_parameters": {"auto_approve": True}, "alerts": {}, "memory": {}}
        a.timeframe, a.confidence_threshold, a.rank_signals = "M5", 0.80, True
        a.symbol_cooldowns, a.cycle_stats, a.approval_timeout = {}, {}, 300
        h1 = {"AAA": "BULLISH BIAS (Above H1 20 EMA)", "BBB": "BULLISH (H1 Uptrend above 20 & 50 EMA)"}
        spread = {"AAA": 0.01, "BBB": 0.05}
        eng = MagicMock()
        eng.get_open_positions.return_value = []
        eng.get_technical_data.side_effect = lambda s, timeframe=None: {"higher_timeframe_h1": h1[s], "current_price": 1.0}
        eng.calculate_trade_parameters.side_effect = lambda symbol, decision: {
            "status": "calculated", "symbol": symbol, "action": "BUY", "order_type": 0, "entry": 1.0, "sl": 0.9985,
            "tp": 1.00225, "lot": 0.01, "atr": 0.001, "risk_amount": 1.5, "risk_reward_ratio": 1.5, "sl_distance": 0.0015}
        eng.current_spread_sl_ratio.side_effect = lambda s, d: spread[s]
        eng.execute_trade.side_effect = lambda p: {"status": "success", "ticket": 1, "lot": 0.01}
        a.mt5_engine = eng
        a.sentiment_engine = MagicMock(get_live_news=AsyncMock(return_value={"headlines": []}))
        a.memory = MagicMock(query_similar=AsyncMock(return_value=[]), store_episode=AsyncMock())
        a.memory._format_episode_for_prompt.return_value = ""
        decision = MagicMock(decision="BUY", confidence_score=0.82, market_sentiment="BULLISH", reasoning="r")
        decision.model_dump.return_value = {}
        a.agent_logic = MagicMock(get_trade_decision=AsyncMock(return_value=decision), last_decision_meta={})
        a.openclaw = MagicMock(send_alert=AsyncMock())
        a.supabase = MagicMock()
        return a, main_mod

    def test_signals_deferred_then_executed_best_first(self):
        import asyncio
        a, main_mod = self._agent()

        async def cycle():
            pending = [ (await a.run_analysis_cycle(s))["pending"] for s in ("AAA", "BBB") ]
            a.mt5_engine.execute_trade.assert_not_called()          # nothing executes during analysis
            return [await a._approve_execute_and_record(c) for c in main_mod.rank_signals(pending)]

        results = asyncio.run(cycle())
        executed = [c.args[0]["symbol"] for c in a.mt5_engine.execute_trade.call_args_list]
        self.assertEqual(executed, ["BBB", "AAA"])                    # full H1 trend first
        self.assertTrue(all(r["status"].startswith("EXECUTED") for r in results))
        self.assertEqual(a.mt5_engine.execute_trade.call_args_list[0].args[0]["confidence"], 0.82)
        self.assertEqual(a.supabase.log_decision.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
