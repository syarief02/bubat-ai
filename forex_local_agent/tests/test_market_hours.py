"""
Regression tests: the agent pauses while the FX market is closed (offline: MT5 is mocked, no orders are sent).

Saturday 2026-10-03: the agent kept analysing the frozen weekend feed and queued 7 SELL
signals that only the spread wall (8-19 pip weekend spreads) stopped.

Run: python tests/test_market_hours.py
"""

import asyncio
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
WORKSPACE_DIR = AGENT_DIR.parent
for p in (str(AGENT_DIR), str(WORKSPACE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.mt5_time import market_open, next_market_open


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


class TestMarketHours(unittest.TestCase):
    def test_market_open_boundaries(self):
        self.assertTrue(market_open(utc(2026, 10, 2, 20, 59)))    # Fri just before close
        self.assertFalse(market_open(utc(2026, 10, 2, 21, 0)))    # Fri close
        self.assertFalse(market_open(utc(2026, 10, 3, 14, 0)))    # Sat
        self.assertFalse(market_open(utc(2026, 10, 4, 20, 59)))   # Sun before open
        self.assertTrue(market_open(utc(2026, 10, 4, 21, 0)))     # Sun open
        self.assertTrue(market_open(utc(2026, 10, 7, 3, 0)))      # Wed

    def test_next_market_open(self):
        sunday_open = utc(2026, 10, 4, 21, 0)
        self.assertEqual(next_market_open(utc(2026, 10, 2, 21, 30)), sunday_open)   # Fri after close
        self.assertEqual(next_market_open(utc(2026, 10, 3, 14, 0)), sunday_open)    # Sat
        self.assertEqual(next_market_open(utc(2026, 10, 4, 20, 0)), sunday_open)    # Sun before open
        now = utc(2026, 10, 6, 10, 0)
        self.assertEqual(next_market_open(now), now)                                # already open


class TestMainLoopPause(unittest.TestCase):
    def _agent(self):
        from loguru import logger
        import main as main_mod
        logger.remove()  # main adds the live agent/trade log sinks at import; keep test output out of them
        a = object.__new__(main_mod.ForexAgent)
        a.running, a.timeframe, a.symbols, a.cycle_stats = True, "M5", ["EURUSD"], {}
        a.mt5_engine = MagicMock()
        a.mt5_engine.initialize.return_value = True
        a.mt5_engine.get_account_info.return_value = {}
        a.openclaw = MagicMock(send_alert=AsyncMock())
        a.supabase = MagicMock()
        a.run_analysis_cycle = AsyncMock()
        a.shutdown = AsyncMock()
        return a, main_mod

    def test_wait_is_capped_and_logged_once(self):
        a, main_mod = self._agent()
        sat = utc(2026, 10, 3, 14, 0)
        self.assertEqual(a.market_closed_wait(sat), main_mod.MARKET_CLOSED_POLL_SECONDS)
        self.assertTrue(a._market_closed_logged)
        # 10 minutes before the open: sleep only until the open
        self.assertEqual(a.market_closed_wait(utc(2026, 10, 4, 20, 50)), 600)
        self.assertIsNone(a.market_closed_wait(utc(2026, 10, 5, 8, 0)))
        self.assertFalse(a._market_closed_logged)

    def test_closed_market_skips_analysis(self):
        a, _ = self._agent()
        calls = {"n": 0}

        def closed(now_utc=None):
            calls["n"] += 1
            if calls["n"] >= 3:
                a.running = False
            return 0.01

        a.market_closed_wait = closed
        asyncio.run(a.main_loop())
        a.run_analysis_cycle.assert_not_called()
        a.mt5_engine.execute_trade.assert_not_called()
        self.assertGreaterEqual(calls["n"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
