"""
Regression tests for post-mortem cycle #5 fixes (offline: MT5 is mocked, no orders are sent).

Run: python tests/test_cycle5_regressions.py
"""

import json
import sys
import tempfile
import unittest
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
WORKSPACE_DIR = AGENT_DIR.parent
for p in (str(AGENT_DIR), str(WORKSPACE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from learning.rules_loader import parse_rule_blocks, select_rules, load_prompt_rules
from learning.reflexion_store import ReflexionStore, validate_reflexion_rule
from core import mt5_time
from core.trade_analytics import classify_exit, max_drawdown, simulate_path, wall_from_message
from core.mt5_engine import MT5Engine
from learning.skills.economic_calendar_filter import EconomicCalendarFilter

CONFIG = AGENT_DIR / "config.json"
Deal = namedtuple("Deal", "entry profit swap commission fee")


def _block(cat, rid, source, directive):
    return f"\n### [{cat}] Rule #{rid}\n- **Date**: 2026-10-02 00:00:00 UTC\n- **Source**: {source}\n- **Directive**: {directive}\n"


class TestRulesLoader(unittest.TestCase):
    """Bug: junk loss reflexions were injected and evicted every curated rule."""

    def test_reflexion_and_communication_excluded_from_trading_prompt(self):
        text = "# Learned Rules\n---\n"
        text += _block("STRATEGY", 1, "user_instruction", "Trade all 28 pairs.")
        text += _block("COMMUNICATION", 2, "user_correction", "Use Malaysian Malay.")
        for i in range(50):
            text += _block("REFLEXION", 100 + i, "loss_reflexion", f"[GBPJPY] Set a stop-loss level of 208.{i:03d}.")
        out = select_rules(parse_rule_blocks(text), cap=2500, exclude_categories={"REFLEXION", "COMMUNICATION"})
        self.assertIn("Trade all 28 pairs.", out)
        self.assertNotIn("stop-loss level", out)
        self.assertNotIn("Malay", out)

    def test_loss_reflexion_source_excluded_even_with_curated_category(self):
        text = _block("RISK_MANAGEMENT", 1, "loss_reflexion", "junk rule text here")
        self.assertEqual(select_rules(parse_rule_blocks(text), cap=2500), "")

    def test_cap_keeps_whole_rules_newest_first(self):
        text = "".join(_block("STRATEGY", i, "audit", f"rule-{i} " + "x" * 300) for i in range(20))
        out = select_rules(parse_rule_blocks(text), cap=1000)
        self.assertLessEqual(len(out), 1000)
        self.assertIn("rule-19 ", out)
        self.assertNotIn("rule-0 ", out)
        for line in out.splitlines():
            self.assertTrue(line.endswith("x"), "a rule was truncated mid-text")

    def test_live_rules_file_within_cap_and_clean(self):
        out = load_prompt_rules(AGENT_DIR / "learning" / "learned_rules.md")
        self.assertTrue(0 < len(out) <= 2500)
        self.assertNotIn("[REFLEXION]", out)


class TestReflexionStore(unittest.TestCase):
    def test_validator(self):
        self.assertFalse(validate_reflexion_rule("Set a stop-loss level of 208.320 for GBPJPY trades.")[0])
        self.assertFalse(validate_reflexion_rule("Avoid GBPJPY entries near 208.320 when RSI is high")[0])
        self.assertFalse(validate_reflexion_rule("short")[0])
        self.assertTrue(validate_reflexion_rule(
            "Avoid SELL entries on EURCHF during 12:00-13:00 UTC when a USD tier-1 release is due")[0])

    def test_processed_tickets_persist_across_restart(self):
        with tempfile.TemporaryDirectory() as d:
            s1 = ReflexionStore(Path(d))
            s1.load_processed_tickets()
            s1.mark_processed(123, datetime.now(timezone.utc).timestamp())
            s1.save_processed()
            self.assertIn(123, ReflexionStore(Path(d)).load_processed_tickets())

    def test_candidates_go_to_jsonl_not_rules_file(self):
        with tempfile.TemporaryDirectory() as d:
            s = ReflexionStore(Path(d))
            s.add_candidate({"symbol": "EURJPY"}, {"new_rule": "x"}, False, "too_short")
            lines = (Path(d) / "learning" / "reflexion_candidates.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0])["facts"]["symbol"], "EURJPY")
            self.assertFalse((Path(d) / "learning" / "learned_rules.md").exists())


class TestServerTimeOffset(unittest.TestCase):
    """Bug: MT5 history queries used real UTC against server-time (UTC+3) stamps, missing the last 3h."""

    def test_offset_from_fresh_tick(self):
        now = 1_790_962_369.0
        self.assertEqual(mt5_time.offset_from_tick_time(int(now) + 10800 - 4, now), 10800)

    def test_stale_tick_gives_none(self):
        now = 1_790_962_369.0
        self.assertIsNone(mt5_time.offset_from_tick_time(int(now) + 10800 - 3000, now))

    def test_history_query_is_shifted(self):
        fake = MagicMock()
        fake.history_deals_get.return_value = []
        start = datetime(2026, 10, 2, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 10, 2, 17, 0, tzinfo=timezone.utc)
        with patch.object(mt5_time, "mt5", fake):
            mt5_time.history_deals_utc(start, end, offset=10800)
        args = fake.history_deals_get.call_args[0]
        self.assertEqual(args[0], start + timedelta(hours=3))
        self.assertEqual(args[1], end + timedelta(hours=3))

    def test_weekend_gives_none(self):
        sat = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc).timestamp()
        self.assertIsNone(mt5_time.offset_from_tick_time(int(sat) + 10800, sat))

    def test_server_epoch_to_utc(self):
        self.assertEqual(mt5_time.server_epoch_to_utc(10800 + 60, offset=10800),
                         datetime(1970, 1, 1, 0, 1, tzinfo=timezone.utc))


class TestDailyLossWall(unittest.TestCase):
    def setUp(self):
        self.engine = MT5Engine(str(CONFIG))
        self.engine.daily_loss_limit_pct = 5.0

    def test_realized_pnl_is_net_and_exits_only(self):
        deals = [Deal(1, -2.0, -0.5, -0.1, 0.0), Deal(0, 0.0, 0.0, -0.1, 0.0), Deal(1, 1.0, 0.0, 0.0, 0.0)]
        with patch("core.mt5_engine.history_deals_utc", return_value=deals):
            self.assertAlmostEqual(self.engine.get_realized_pnl_today_utc(), -1.6)

    def test_rejects_when_loss_exceeds_limit(self):
        with patch("core.mt5_engine.history_deals_utc", return_value=[Deal(1, -8.0, 0, 0, 0)]), \
             patch("core.mt5_engine.mt5.account_info", return_value=MagicMock(balance=144.0)):
            res = self.engine.check_daily_loss_limit()
        self.assertEqual(res["status"], "rejected")

    def test_allows_when_within_limit(self):
        with patch("core.mt5_engine.history_deals_utc", return_value=[Deal(1, -3.0, 0, 0, 0)]), \
             patch("core.mt5_engine.mt5.account_info", return_value=MagicMock(balance=144.0)):
            self.assertIsNone(self.engine.check_daily_loss_limit())

    def test_window_starts_at_real_utc_midnight(self):
        now = datetime(2026, 10, 2, 17, 30, tzinfo=timezone.utc)
        with patch("core.mt5_engine.history_deals_utc", return_value=[]) as h:
            self.engine.get_realized_pnl_today_utc(now)
        self.assertEqual(h.call_args[0][0], datetime(2026, 10, 2, tzinfo=timezone.utc))
        self.assertGreaterEqual(h.call_args[0][1], now)


class TestTrailingConfig(unittest.TestCase):
    """Bug: trailing_stop_enabled / trailing_step_pips were never read."""

    def test_config_keys_wired(self):
        risk = json.loads(CONFIG.read_text(encoding="utf-8"))["risk_parameters"]
        e = MT5Engine(str(CONFIG))
        self.assertEqual(e.trailing_enabled, risk["trailing_stop_enabled"])
        self.assertEqual(e.trailing_step_pips, risk["trailing_step_pips"])
        self.assertEqual(e.trailing_start_pips, risk["trailing_start_pips"])
        self.assertEqual(e.trailing_distance_pips, risk["trailing_distance_pips"])

    def test_disabled_does_nothing(self):
        e = MT5Engine(str(CONFIG))
        e.trailing_enabled = False
        with patch.object(e, "get_open_positions") as gp:
            self.assertEqual(e.manage_trailing_stops(), [])
            gp.assert_not_called()


class TestTier1GlobalBlackout(unittest.TestCase):
    """Bug: NFP blacked out USD pairs only; 17 non-USD crosses lost -$15.17 around the release."""

    def setUp(self):
        self.f = EconomicCalendarFilter.__new__(EconomicCalendarFilter)
        self.f._cached_events = [
            {"title": "Non-Farm Employment Change", "country": "USD", "date": "2026-10-02T08:30:00-04:00", "impact": "High"},
            {"title": "CPI Flash Estimate y/y", "country": "EUR", "date": "2026-10-02T05:00:00-04:00", "impact": "Medium"},
        ]
        self.f.refresh_calendar = lambda force=False: self.f._cached_events

    def test_cross_blocked_after_nfp(self):
        now = datetime(2026, 10, 2, 12, 35, tzinfo=timezone.utc)
        ok, why = self.f.is_trade_permitted_by_calendar("EURCHF", 30, 15, global_tier1=True, now=now)
        self.assertFalse(ok)
        self.assertIn("tier-1 global", why)

    def test_cross_blocked_before_nfp(self):
        now = datetime(2026, 10, 2, 12, 10, tzinfo=timezone.utc)
        self.assertFalse(self.f.is_trade_permitted_by_calendar("AUDNZD", 30, 15, global_tier1=True, now=now)[0])

    def test_flag_off_keeps_old_behaviour(self):
        now = datetime(2026, 10, 2, 12, 35, tzinfo=timezone.utc)
        self.assertTrue(self.f.is_trade_permitted_by_calendar("EURCHF", 30, 15, global_tier1=False, now=now)[0])

    def test_outside_window_clear(self):
        now = datetime(2026, 10, 2, 13, 0, tzinfo=timezone.utc)
        self.assertTrue(self.f.is_trade_permitted_by_calendar("EURCHF", 30, 15, global_tier1=True, now=now)[0])


class TestTradeAnalytics(unittest.TestCase):
    def test_classify_exit(self):
        pip = 0.0001
        self.assertEqual(classify_exit(5, 1.1030, 1.1000, 1.0985, 1, pip), "TP")
        self.assertEqual(classify_exit(4, 1.0985, 1.1000, 1.0985, 1, pip), "SL_FULL")
        self.assertEqual(classify_exit(4, 1.1001, 1.1000, 1.0985, 1, pip), "BREAK_EVEN")
        self.assertEqual(classify_exit(4, 1.1008, 1.1000, 1.0985, 1, pip), "TRAILING")
        self.assertEqual(classify_exit(4, 1.0992, 1.1000, 1.1015, -1, pip), "TRAILING")

    def test_max_drawdown_is_peak_to_trough(self):
        # old report used min(cumulative) = -1; true peak-to-trough is 5 -> 1 = -4
        self.assertEqual(max_drawdown([5, -4, 1]), -4)
        self.assertEqual(max_drawdown([1, 1]), 0)

    def test_simulate_path_ties_go_to_sl(self):
        h, l, c = np.array([1.2]), np.array([0.8]), np.array([1.0])
        self.assertEqual(simulate_path(h, l, c, 1, 1.0, 0.9, 1.1)["outcome"], "SL")
        self.assertEqual(simulate_path(np.array([1.05, 1.2]), np.array([0.95, 1.0]), np.array([1.0, 1.1]),
                                       1, 1.0, 0.9, 1.1)["r"], 1.0)

    def test_wall_labels(self):
        self.assertEqual(wall_from_message("REJECTED: Daily loss limit reached ($35.36 / $7.22)"), "DAILY_LOSS_STOP")
        self.assertEqual(wall_from_message("REJECTED: Counter-trend BUY blocked on GBPCHF."), "H1_TREND")
        self.assertEqual(wall_from_message("REJECTED: Max open trades (10) reached"), "CAPACITY")


class TestDecisionOutcome(unittest.TestCase):
    def test_classify_outcome(self):
        from main import ForexAgent
        d = MagicMock(decision="BUY")
        self.assertEqual(ForexAgent._classify_outcome(d, {"status": "calculated"}, True,
                                                      {"status": "rejected", "message": "REJECTED: Spread on X"}),
                         "REJECTED:SPREAD")
        self.assertEqual(ForexAgent._classify_outcome(MagicMock(decision="WAIT"), None, False, None), "WAIT")
        self.assertEqual(ForexAgent._classify_outcome(d, None, False, None), "BELOW_THRESHOLD")


class TestChatbot(unittest.TestCase):
    def test_root_assistant_is_shim(self):
        src = (WORKSPACE_DIR / "local_assistant.py").read_text(encoding="utf-8")
        self.assertIn("runpy.run_path", src)
        self.assertLess(len(src), 2000)

    def test_package_assistant_paths(self):
        import forex_local_agent.local_assistant as la
        self.assertEqual(la.WORKSPACE_DIR, WORKSPACE_DIR)
        self.assertTrue((la.AGENT_DIR / "config.json").exists())

    def test_chat_log_path_is_absolute(self):
        from forex_local_agent.chat_logger import DEFAULT_LOG_FILE
        self.assertTrue(Path(DEFAULT_LOG_FILE).is_absolute())

    def test_account_status_has_no_none_placeholders(self):
        import forex_local_agent.chat as chat
        ex = chat.ChatToolExecutor.__new__(chat.ChatToolExecutor)
        ex.config_path = CONFIG
        fake = MagicMock()
        fake.initialize.return_value = True
        fake.get_account_info.return_value = {"balance": 144.36, "equity": 142.54, "margin": 16.0, "free_margin": 128.1}
        fake.get_open_positions.return_value = [{"symbol": "EURUSD", "type": 0, "volume": 0.01, "profit": -0.5, "swap": 0.0}]
        fake.get_realized_pnl_today_utc.return_value = -29.63
        with patch.object(chat, "MT5Engine", return_value=fake):
            out = ex._tool_get_account_status({})
        self.assertNotIn("None", out)
        self.assertIn("144.36", out)
        self.assertIn("-29.63", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
