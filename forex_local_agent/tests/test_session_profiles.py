"""
Tests for per-session execution profiles and daily grading (offline, no MT5 calls, no orders).

Run: python tests/test_session_profiles.py
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
for p in (str(AGENT_DIR), str(AGENT_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core import session_profiles as sp
from core.trade_analytics import wall_from_message

CONFIG = json.loads((AGENT_DIR / "config.json").read_text())


def _t(hour, pips, sl=15.0, net=None):
    return {"entry_hour_utc": hour, "pips": pips, "sl_pips": sl, "net_profit": net if net is not None else pips / 10}


class TestGrading(unittest.TestCase):
    def test_stats_attributed_by_entry_hour(self):
        st = sp.session_stats([_t(2, -15), _t(23, 15), _t(8, 15), _t(14, -15), _t(18, -15), _t(21, -15)])
        self.assertEqual(st["ASIA"]["n"], 2)
        self.assertEqual(st["ASIA"]["avg_r"], 0.0)
        self.assertEqual((st["LONDON"]["n"], st["LONDON_NY_OVERLAP"]["n"], st["NEW_YORK"]["n"], st["ROLLOVER"]["n"]),
                         (1, 1, 1, 1))

    def test_next_level(self):
        self.assertEqual(sp.next_level("NORMAL", 25, -0.20), "REDUCED")
        self.assertEqual(sp.next_level("NORMAL", 25, -0.40), "MINIMAL")
        self.assertEqual(sp.next_level("MINIMAL", 25, -0.20), "MINIMAL")    # never loosens while losing
        self.assertEqual(sp.next_level("NORMAL", 10, -0.90), "NORMAL")      # too few trades to judge
        self.assertEqual(sp.next_level("MINIMAL", 12, 0.05), "REDUCED")     # one step up per day
        self.assertEqual(sp.next_level("REDUCED", 12, 0.05), "NORMAL")
        self.assertEqual(sp.next_level("NORMAL", 50, 0.50), "NORMAL")       # never above baseline
        self.assertEqual(sp.next_level("REDUCED", 5, 0.50), "REDUCED")      # not enough evidence to restore

    def test_effective_profile(self):
        base = {"max_open_trades": 8, "max_spread_sl_ratio": 0.06, "symbols": None, "loss_budget_pct": 1.0}
        r = sp.effective_profile(base, "REDUCED")
        self.assertEqual((r["max_open_trades"], r["max_spread_sl_ratio"]), (4, 0.04))
        self.assertEqual(r["symbols"], sp.USD_MAJORS)
        m = sp.effective_profile(base, "MINIMAL")
        self.assertEqual((m["max_open_trades"], m["max_spread_sl_ratio"]), (1, 0.03))
        self.assertEqual(sp.effective_profile(base, "NORMAL")["max_open_trades"], 8)


class TestSessionWall(unittest.TestCase):
    LONDON = datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc)
    ASIA = datetime(2026, 10, 5, 3, 0, tzinfo=timezone.utc)

    def _mgr(self, d, levels=None):
        m = sp.SessionManager(CONFIG, Path(d))
        if levels:
            m.levels.update(levels)
        return m

    def test_allows_normal_trade(self):
        with tempfile.TemporaryDirectory() as d:
            msg = self._mgr(d).check("EURUSD", 0.02, [], 150.0, lambda since: [], self.LONDON)
        self.assertIsNone(msg)

    def test_minimal_level_majors_only_and_one_slot(self):
        with tempfile.TemporaryDirectory() as d:
            m = self._mgr(d, {"ASIA": "MINIMAL"})
            self.assertEqual(wall_from_message(m.check("GBPNZD", 0.01, [], 150.0, lambda s: [], self.ASIA)),
                             "SESSION_SYMBOLS")
            self.assertEqual(wall_from_message(m.check("EURUSD", 0.01, [2], 150.0, lambda s: [], self.ASIA)),
                             "SESSION_SLOTS")
            # a London position does not use an Asia slot
            self.assertIsNone(m.check("EURUSD", 0.01, [9], 150.0, lambda s: [], self.ASIA))

    def test_session_spread_cap(self):
        with tempfile.TemporaryDirectory() as d:
            msg = self._mgr(d).check("EURUSD", 0.05, [], 150.0, lambda s: [], self.ASIA)  # Asia baseline 4%
        self.assertEqual(wall_from_message(msg), "SPREAD_COST")

    def test_loss_budget_is_per_session(self):
        asia_losses = [_t(1, -15, net=-1.0), _t(2, -15, net=-1.0)]          # $2 > 1% of $150
        with tempfile.TemporaryDirectory() as d:
            m = self._mgr(d)
            self.assertEqual(wall_from_message(m.check("EURUSD", 0.01, [], 150.0, lambda s: asia_losses, self.ASIA)),
                             "SESSION_BUDGET")
            m._today_cache = (0.0, [])
            # Asia's losses do not block London
            self.assertIsNone(m.check("EURUSD", 0.01, [], 150.0, lambda s: asia_losses, self.LONDON))

    def test_review_once_per_day_and_persisted(self):
        calls = []
        bad_ny = [_t(18, -15) for _ in range(25)]

        def fetch(since):
            calls.append(since)
            return bad_ny
        with tempfile.TemporaryDirectory() as d:
            m = self._mgr(d)
            review = m.maybe_review(fetch, self.LONDON)
            self.assertEqual(review["sessions"]["NEW_YORK"]["level"], "MINIMAL")
            self.assertEqual(review["sessions"]["LONDON"]["level"], "NORMAL")
            self.assertIsNone(m.maybe_review(fetch, self.LONDON))            # same UTC day
            self.assertEqual(len(calls), 1)
            self.assertEqual(self._mgr(d).levels["NEW_YORK"], "MINIMAL")      # survives restart


class TestConfig(unittest.TestCase):
    def test_every_session_open_and_budgets_sum_to_daily_limit(self):
        self.assertIsNone(CONFIG["risk_parameters"]["entry_hours_utc"])
        prof = CONFIG["session_profiles"]
        total = sum(prof[s]["loss_budget_pct"] for s in sp.SESSIONS)
        self.assertAlmostEqual(total, CONFIG["risk_parameters"]["daily_loss_limit_pct"])
        for s in sp.SESSIONS:
            self.assertLessEqual(prof[s]["max_open_trades"], CONFIG["risk_parameters"]["max_open_trades"])

    def test_engine_has_session_manager(self):
        from core.mt5_engine import MT5Engine
        e = MT5Engine(str(AGENT_DIR / "config.json"))
        self.assertTrue(e.session_manager.enabled)
        self.assertIsNone(e.entry_hours_utc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
