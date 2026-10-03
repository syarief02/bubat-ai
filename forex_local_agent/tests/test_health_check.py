"""Health check verdicts (pure logic; no MT5, Ollama or log files needed)."""
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maintenance import health_check as hc  # noqa: E402

WED = datetime(2026, 9, 30, 0, 0, tzinfo=timezone.utc)      # market open all day
SAT = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)     # market closed


def deal(minute, entry, pnl=0.0, symbol="EURUSD", magic=hc.BOT_MAGIC):
    return {"time_utc": WED + timedelta(minutes=minute), "entry": entry, "pnl": pnl, "symbol": symbol,
            "magic": magic}


class TestLossWall(unittest.TestCase):
    def test_trade_after_limit_is_a_fault(self):
        # balance 100 now after -6 -> 106 at the start, 5% limit ~5.3
        deals = [deal(0, "in"), deal(10, "out", -6.0), deal(20, "in", symbol="GBPUSD")]
        r = hc.check_loss_wall(deals, 94.0, 5.0, WED)
        self.assertEqual(r["status"], hc.FAULT)
        self.assertIn("GBPUSD", r["detail"])

    def test_open_trades_running_past_limit_is_only_a_warning(self):
        deals = [deal(0, "in"), deal(1, "in"), deal(10, "out", -4.0), deal(11, "out", -4.0)]
        self.assertEqual(hc.check_loss_wall(deals, 92.0, 5.0, WED)["status"], hc.WARN)

    def test_breach_before_the_window_is_ignored(self):
        deals = [deal(0, "in"), deal(10, "out", -6.0), deal(20, "in")]
        r = hc.check_loss_wall(deals, 94.0, 5.0, WED + timedelta(hours=1))
        self.assertEqual(r["status"], hc.OK)

    def test_limit_uses_the_balance_at_the_time(self):
        # A deposit later in the day must not shrink the limit that applied before it
        deals = [deal(0, "in"), deal(10, "out", -6.0), deal(20, "in", symbol="GBPUSD")]
        self.assertEqual(hc.check_loss_wall(deals, 50.0 - 6.0, 5.0, WED)["status"], hc.FAULT)  # 50: limit 2.5
        self.assertEqual(hc.check_loss_wall(deals, 200.0 - 6.0, 5.0, WED)["status"], hc.OK)    # 200: limit 10

    def test_manual_trades_are_not_blamed_on_the_bot(self):
        deals = [deal(0, "in"), deal(10, "out", -6.0), deal(20, "in", magic=0)]
        self.assertEqual(hc.check_loss_wall(deals, 94.0, 5.0, WED)["status"], hc.WARN)


class TestPositions(unittest.TestCase):
    RISK = {"lot_mode": "fixed", "fixed_lot": 0.01, "max_lot_size": 0.1, "max_open_trades": 2}

    def pos(self, **kw):
        p = {"magic": hc.BOT_MAGIC, "symbol": "EURUSD", "ticket": 1, "sl": 1.1, "tp": 1.2, "volume": 0.01}
        p.update(kw)
        return p

    def test_all_good(self):
        self.assertEqual(hc.check_positions([self.pos()], self.RISK)["status"], hc.OK)

    def test_missing_stop_loss_is_a_fault(self):
        r = hc.check_positions([self.pos(sl=0.0)], self.RISK)
        self.assertEqual(r["status"], hc.FAULT)
        self.assertIn("NO stop loss", r["detail"])

    def test_oversized_lot_and_too_many_positions(self):
        self.assertEqual(hc.check_positions([self.pos(volume=0.5)], self.RISK)["status"], hc.FAULT)
        self.assertEqual(hc.check_positions([self.pos()] * 3, self.RISK)["status"], hc.FAULT)

    def test_manual_positions_are_ignored(self):
        self.assertEqual(hc.check_positions([self.pos(magic=0, sl=0.0)], self.RISK)["status"], hc.OK)


class TestActivity(unittest.TestCase):
    def test_no_cycles_while_open_is_a_fault(self):
        self.assertEqual(hc.check_activity([], WED, WED + timedelta(hours=6))["status"], hc.FAULT)

    def test_steady_cycles_are_ok(self):
        cycles = [WED + timedelta(minutes=5 * i) for i in range(72)]
        self.assertEqual(hc.check_activity(cycles, WED, WED + timedelta(hours=6))["status"], hc.OK)

    def test_long_gap_while_open_is_a_warning(self):
        cycles = [WED + timedelta(minutes=5 * i) for i in range(12)] + [WED + timedelta(hours=5)]
        self.assertEqual(hc.check_activity(cycles, WED, WED + timedelta(hours=5, minutes=5))["status"], hc.WARN)

    def test_analysing_a_closed_market_is_a_warning(self):
        r = hc.check_activity([SAT], SAT - timedelta(hours=1), SAT + timedelta(hours=1))
        self.assertEqual(r["status"], hc.WARN)
        self.assertIn("CLOSED", r["detail"])

    def test_closed_market_without_cycles_is_ok(self):
        self.assertEqual(hc.check_activity([], SAT, SAT + timedelta(hours=6))["status"], hc.OK)


class TestErrorsAndLogs(unittest.TestCase):
    def rec(self, msg):
        return (WED, "ERROR", "x", msg)

    def test_a_few_parse_failures_are_ok(self):
        errs = [self.rec("[EURUSD] All 3 attempts failed to obtain TradeDecision: ...")] * 10
        self.assertEqual(hc.check_errors(errs)["status"], hc.OK)

    def test_order_failures_are_a_fault(self):
        errs = [self.rec("Order failed, retcode=10019 (No money)")] * 5
        self.assertEqual(hc.check_errors(errs)["status"], hc.FAULT)

    def test_parse_log_lines(self):
        lines = ["2026-10-03 02:11:48.112 | INFO     | __main__:run_analysis_cycle:120 - ═══ Starting analysis cycle",
                 "Traceback (most recent call last):"]
        recs = hc.parse_log_lines(lines)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0][1], "INFO")
        self.assertIsNotNone(recs[0][0].tzinfo)

    def test_overall_is_the_worst(self):
        checks = [hc.check("a", hc.OK, ""), hc.check("b", hc.WARN, ""), hc.check("c", hc.OK, "")]
        self.assertEqual(hc.overall(checks), hc.WARN)


if __name__ == "__main__":
    unittest.main()
