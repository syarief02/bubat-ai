"""
Tests for the Asia range-fade shadow strategy (offline: MT5 is mocked, nothing can be sent).

Run: python tests/test_session_strategies.py
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
for p in (str(AGENT_DIR), str(AGENT_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core import session_strategies as ss

CFG = {**ss.DEFAULTS}
PIP = 0.0001


def _wave(n=60, base=1.1000, amp=0.0003):
    x = base + amp * np.sin(np.arange(n) / 3.0)
    return x + 0.0001, x - 0.0001, x.copy()


class TestFadeSignal(unittest.TestCase):
    def test_sharp_drop_gives_buy(self):
        h, l, c = _wave()
        c[-4:] = [1.0990, 1.0980, 1.0970, 1.0960]
        h[-4:], l[-4:] = c[-4:] + 0.0001, c[-4:] - 0.0001
        sig = ss.fade_signal(h, l, c)
        self.assertEqual(sig["direction"], 1)
        self.assertLess(sig["rsi"], 30)

    def test_sharp_rise_gives_sell(self):
        h, l, c = _wave()
        c[-4:] = [1.1010, 1.1020, 1.1030, 1.1040]
        h[-4:], l[-4:] = c[-4:] + 0.0001, c[-4:] - 0.0001
        self.assertEqual(ss.fade_signal(h, l, c)["direction"], -1)

    def test_inside_bands_no_signal(self):
        self.assertIsNone(ss.fade_signal(*_wave()))

    def test_too_few_bars(self):
        h, l, c = _wave(10)
        self.assertIsNone(ss.fade_signal(h, l, c))


class TestPaperTrade(unittest.TestCase):
    NOW = datetime(2026, 10, 5, 2, 7, tzinfo=timezone.utc)

    def test_buy_pays_spread_and_uses_sl_floor(self):
        t = ss.build_paper_trade("EURUSD", 1, atr=0.0002, bid=1.1000, ask=1.10005, pip=PIP, cfg=CFG,
                                 now_utc=self.NOW, bar_time_utc=0.0)
        self.assertEqual(t["entry"], 1.10005)                       # BUY at ask
        self.assertAlmostEqual(t["sl_pips"], 8.0)                   # 1.5*ATR = 3p < 8p floor
        self.assertAlmostEqual(t["tp"] - t["entry"], t["entry"] - t["sl"])  # tp_r = 1.0
        self.assertAlmostEqual(t["spread_pips"], 0.5)

    def _trade(self, d=1):
        return {"direction": d, "entry": 1.1000, "sl": 1.1000 - d * 0.0008, "tp": 1.1000 + d * 0.0008,
                "entry_ts": self.NOW.timestamp()}

    def _bars(self, rows, start=None):
        start = start or self.NOW.timestamp() + 180
        t = np.array([start + 300 * i for i in range(len(rows))], dtype=float)
        o, h, l = (np.array([r[i] for r in rows]) for i in range(3))
        return t, o, h, l

    def test_tp_hit(self):
        t, o, h, l = self._bars([(1.1000, 1.1003, 1.0998), (1.1003, 1.1009, 1.1001)])
        res = ss.resolve_paper_trade(self._trade(), t, o, h, l, 7)
        self.assertEqual((res["exit_reason"], res["r"]), ("TP", 1.0))

    def test_sl_wins_tie(self):
        t, o, h, l = self._bars([(1.1000, 1.1009, 1.0991)])
        self.assertEqual(ss.resolve_paper_trade(self._trade(), t, o, h, l, 7)["exit_reason"], "SL")

    def test_bars_before_entry_ignored_and_still_open(self):
        t, o, h, l = self._bars([(1.1000, 1.1009, 1.0991), (1.1000, 1.1002, 1.0998)],
                                start=self.NOW.timestamp() - 300)
        t[1] = self.NOW.timestamp() + 60
        self.assertIsNone(ss.resolve_paper_trade(self._trade(), t, o, h, l, 7))

    def test_session_end_exit_at_open(self):
        seven = datetime(2026, 10, 5, 7, 0, tzinfo=timezone.utc).timestamp()
        t = np.array([self.NOW.timestamp() + 180, seven])
        o, h, l = np.array([1.1000, 1.1004]), np.array([1.1002, 1.1006]), np.array([1.0999, 1.1003])
        res = ss.resolve_paper_trade(self._trade(), t, o, h, l, 7)
        self.assertEqual(res["exit_reason"], "SESSION_END")
        self.assertAlmostEqual(res["r"], 0.5)

    def test_summary_promotion(self):
        cfg = {**CFG, "promote_after_trades": 3, "promote_min_avg_r": 0.05}
        self.assertFalse(ss.summarize_shadow([{"r": 1.0}, {"r": -1.0}], cfg)["promotion_ready"])
        self.assertTrue(ss.summarize_shadow([{"r": 1.0}, {"r": -1.0}, {"r": 1.0}], cfg)["promotion_ready"])


class TestShadowRunner(unittest.TestCase):
    def _fake_mt5(self, closes, t0):
        n = len(closes)
        rates = np.zeros(n, dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8")])
        rates["time"] = [t0 + 300 * i for i in range(n)]
        rates["close"] = closes
        rates["open"] = closes
        rates["high"] = np.array(closes) + 0.0001
        rates["low"] = np.array(closes) - 0.0001
        fake = MagicMock()
        fake.copy_rates_from_pos.return_value = rates
        fake.symbol_info_tick.return_value = MagicMock(bid=closes[-1], ask=closes[-1] + 0.00002)
        fake.symbol_info.return_value = MagicMock(point=0.00001, digits=5)
        fake.order_send.side_effect = AssertionError("shadow strategy must never send orders")
        return fake

    def test_opens_paper_trade_in_window_and_persists(self):
        now = datetime(2026, 10, 5, 2, 7, tzinfo=timezone.utc)
        _, _, c = _wave()
        c[-5:-1] = [1.0990, 1.0980, 1.0970, 1.0960]       # last completed bar is the drop; [-1] is forming
        fake = self._fake_mt5(list(c), int(now.timestamp()) - 300 * (len(c) - 1) - 120)
        with tempfile.TemporaryDirectory() as d, patch.object(ss, "mt5", fake), \
                patch.object(ss, "get_server_utc_offset_seconds", return_value=0):
            runner = ss.AsiaRangeFadeShadow({"session_strategies": {"asia_range_fade": {"symbols": ["EURUSD"]}}}, Path(d))
            st = runner.run_cycle(now)
            self.assertEqual(st["opened"], 1)
            saved = json.loads((Path(d) / "asia_shadow_open.json").read_text())
            self.assertEqual(saved["EURUSD"]["direction"], 1)
            again = ss.AsiaRangeFadeShadow({"session_strategies": {"asia_range_fade": {"symbols": ["EURUSD"]}}}, Path(d))
            self.assertIn("EURUSD", again.open_trades)                # survives restart
        fake.order_send.assert_not_called()

    def test_no_new_trades_outside_window(self):
        now = datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc)
        _, _, c = _wave()
        c[-5:-1] = [1.0990, 1.0980, 1.0970, 1.0960]
        fake = self._fake_mt5(list(c), int(now.timestamp()) - 300 * (len(c) - 1))
        with tempfile.TemporaryDirectory() as d, patch.object(ss, "mt5", fake), \
                patch.object(ss, "get_server_utc_offset_seconds", return_value=0):
            st = ss.AsiaRangeFadeShadow({}, Path(d)).run_cycle(now)
        self.assertEqual(st["opened"], 0)

    def test_no_trades_on_weekend_or_stale_feed(self):
        _, _, c = _wave()
        c[-5:-1] = [1.0990, 1.0980, 1.0970, 1.0960]
        saturday = datetime(2026, 10, 3, 2, 7, tzinfo=timezone.utc)
        monday = datetime(2026, 10, 5, 2, 7, tzinfo=timezone.utc)
        for now, t0 in ((saturday, int(saturday.timestamp()) - 300 * (len(c) - 1) - 120),
                        (monday, int(monday.timestamp()) - 300 * (len(c) - 1) - 3600)):  # bars 1h old
            fake = self._fake_mt5(list(c), t0)
            with tempfile.TemporaryDirectory() as d, patch.object(ss, "mt5", fake),                     patch.object(ss, "get_server_utc_offset_seconds", return_value=0):
                st = ss.AsiaRangeFadeShadow({}, Path(d)).run_cycle(now)
            self.assertEqual(st["opened"], 0, now)

    def test_live_mode_is_forced_to_shadow(self):
        with tempfile.TemporaryDirectory() as d:
            r = ss.AsiaRangeFadeShadow({"session_strategies": {"asia_range_fade": {"mode": "live"}}}, Path(d))
        self.assertEqual(r.cfg["mode"], "shadow")

    def test_live_config_block(self):
        cfg = json.loads((AGENT_DIR / "config.json").read_text())["session_strategies"]["asia_range_fade"]
        self.assertEqual(cfg["mode"], "shadow")
        self.assertEqual(cfg["hours_utc"], [0, 6])


if __name__ == "__main__":
    unittest.main(verbosity=2)
