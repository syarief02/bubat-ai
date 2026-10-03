"""
Offline tests for the chat upgrade: language, deep routing, bot status, owner decisions
(no MT5, no Ollama, no web; brain state lives in a temp dir).

Run: python tests/test_chat_upgrade.py
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
WORKSPACE_DIR = AGENT_DIR.parent
for p in (str(AGENT_DIR), str(WORKSPACE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from brain import chat_support as cs


class TestHelpers(unittest.TestCase):
    def test_language(self):
        self.assertEqual(cs.detect_language("My bot lost money this week. Should I stop USDJPY?"), "English")
        self.assertEqual(cs.detect_language("kenapa bot aku rugi minggu ni?"), "Malay")
        self.assertEqual(cs.detect_language("patut tak aku stop trade USDJPY"), "Malay")
        self.assertEqual(cs.detect_language("ok"), "English")
        self.assertEqual(cs.detect_language("is the bot ya ok"), "English")   # one stray word is not Malay

    def test_deep_routing_hints(self):
        self.assertTrue(cs.wants_deep("Why did the bot lose money?"))
        self.assertTrue(cs.wants_deep("Should I approve P5?"))
        self.assertTrue(cs.wants_deep("kenapa USDJPY teruk"))
        self.assertFalse(cs.wants_deep("show me the EURUSD rsi"))

    def test_parse_decision(self):
        self.assertEqual(cs.parse_decision("approve P5"), ("approve", "P5", ""))
        self.assertEqual(cs.parse_decision("reject p3 not enough evidence"), ("reject", "P3", "not enough evidence"))
        self.assertEqual(cs.parse_decision("lulus P4"), ("approve", "P4", ""))
        self.assertEqual(cs.parse_decision("tolak P4 belum cukup data"), ("reject", "P4", "belum cukup data"))
        self.assertIsNone(cs.parse_decision("should I approve P5?"))   # a question, not a decision


class TempBrainState(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "reports").mkdir()
        (self.tmp / "brain").mkdir()
        report = {"window_start": "2026-09-28T00:00:00+00:00", "window_end": "2026-10-03T00:00:00+00:00",
                  "summary": {"trades": 419, "win_rate_pct": 36.0, "net_pnl": -199.64, "profit_factor": 0.5,
                              "avg_r": -0.29},
                  "breakdowns": {"symbol": {"USDJPY": {"trades": 36, "net_pnl": -15.85}}},
                  "baselines_hourly_dedup": {"follow_h1_trend": {"n": 1793, "avg_r": 0.027},
                                             "llm_all_buy_sell": {"n": 1435, "avg_r": 0.052}}}
        (self.tmp / "reports" / "report_2026-10-03_1652.json").write_text(json.dumps(report), encoding="utf-8")
        (self.tmp / "brain" / "proposals.json").write_text(json.dumps([
            {"id": "P5", "kind": "remove_symbol", "symbol": "USDJPY", "risk": "SAFER", "status": "proposed",
             "rationale": "36 trades, -0.367R", "evidence": "USDJPY trades 36"},
            {"id": "P1", "kind": "remove_symbol", "symbol": "GBPCHF", "risk": "SAFER", "status": "rejected"},
        ]), encoding="utf-8")
        (self.tmp / "brain" / "journal.jsonl").write_text(json.dumps(
            {"ts": "2026-10-03T17:01:00+00:00", "kind": "reflection", "assessment": "Losing; USDJPY worst.",
             "lessons": ["USDJPY loses"]}) + "\n", encoding="utf-8")
        self.patches = [patch.object(cs, "REPORTS_DIR", self.tmp / "reports"),
                        patch.object(cs, "STATE_DIR", self.tmp / "brain")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestBotKnowledge(TempBrainState):
    def test_status_brief_has_current_numbers(self):
        brief = cs.status_brief()
        self.assertIn("419 trades", brief)
        self.assertIn("net -199.64", brief)
        self.assertIn("follow_h1_trend avg R 0.027", brief)
        self.assertIn("Losing; USDJPY worst.", brief)
        self.assertIn("P5 [SAFER] stop trading USDJPY", brief)
        self.assertNotIn("GBPCHF", brief)                      # rejected proposals are not "waiting"

    def test_deep_context(self):
        ctx = cs.deep_context()
        self.assertEqual(ctx["performance_digest"]["summary"]["trades"], 419)
        self.assertEqual(ctx["proposals"]["waiting_for_owner"][0]["id"], "P5")
        self.assertEqual(ctx["proposals"]["decided"][0]["status"], "rejected")
        self.assertEqual(ctx["brain_memory"][0]["text"], "Losing; USDJPY worst.")
        self.assertNotIn("blocked_by_wall_avg_r", ctx["performance_digest"])
        self.assertIn("NOT_taken", json.dumps(ctx))


class TestChatRouting(TempBrainState):
    def _chat(self):
        import chat
        c = object.__new__(chat.IntelligentForexChat)
        c.model, c.session_id, c.log_file = "fast", "s", self.tmp / "chat.log"
        c.conversation_history, c.plain_history = [{"role": "system", "content": "sys"}], []
        c.executor = MagicMock()
        c.executor.scanner.symbols = ["EURUSD", "USDJPY"]
        c.executor.learner.auto_detect_and_learn.return_value = None
        c.executor._tool_get_pair_technicals.return_value = '{"rsi": 52}'
        c.deep_llm = MagicMock(model="gpt-oss:20b", last_stats={"think_used": "medium", "seconds": 40})
        c.deep_llm.gpu_layers.return_value = 0
        c._call_ollama = MagicMock(return_value="fast answer")
        return c, chat

    def test_why_question_goes_deep_with_data(self):
        c, _ = self._chat()
        c.deep_llm.ask_chat.return_value = "deep answer"
        self.assertEqual(c.chat_turn("Why should I stop trading USDJPY?"), "deep answer")
        c._call_ollama.assert_not_called()
        messages = c.deep_llm.ask_chat.call_args.args[0]
        self.assertIn("Reply in English", messages[0]["content"])
        self.assertIn('"trades": 419', messages[-1]["content"])          # real report numbers
        self.assertIn("USDJPY", messages[-1]["content"])                  # live technicals for the named pair
        self.assertEqual(c.plain_history[-1]["content"], "deep answer")

    def test_malay_deep_question(self):
        c, _ = self._chat()
        c.deep_llm.ask_chat.return_value = "jawapan"
        c.chat_turn("kenapa bot aku rugi minggu ni?")
        self.assertIn("Reply in Malay", c.deep_llm.ask_chat.call_args.args[0][0]["content"])

    def test_quick_question_stays_fast_and_names_language(self):
        c, _ = self._chat()
        self.assertEqual(c.chat_turn("hello there"), "fast answer")
        c.deep_llm.ask_chat.assert_not_called()
        self.assertIn("[Reply in English.]", c.conversation_history[-2]["content"])

    def test_forced_modes_and_fallback(self):
        c, _ = self._chat()
        c.deep_llm.ask_chat.return_value = None                          # deep model unavailable
        self.assertEqual(c.chat_turn("show eurusd", mode="deep"), "fast answer")
        c.deep_llm.ask_chat.reset_mock()
        c.chat_turn("why is that", mode="fast")
        c.deep_llm.ask_chat.assert_not_called()

    def test_model_has_no_way_to_approve(self):
        import chat
        handlers = chat.ChatToolExecutor.execute.__code__.co_consts
        self.assertFalse(any("approve" in str(x) for x in handlers))


class TestOwnerDecision(TempBrainState):
    def _run(self, action, answer, status="proposed"):
        import chat
        brain = MagicMock()
        brain.proposals.get.return_value = {"id": "P5", "kind": "remove_symbol", "symbol": "USDJPY",
                                            "risk": "SAFER", "status": status, "rationale": "r", "evidence": "e"}
        brain.proposals.approve.return_value = (True, "8 suites passed")
        brain.journal.recent.return_value = []
        agent = MagicMock()
        with patch("brain.agent.Brain", return_value=brain), patch("builtins.input", return_value=answer):
            chat.handle_decision(agent, action, "P5", "too early")
        return brain

    def test_approve_needs_yes(self):
        self.assertFalse(self._run("approve", "n").proposals.approve.called)
        b = self._run("approve", "y")
        b.proposals.approve.assert_called_once()
        self.assertEqual(b.journal.add.call_args.kwargs["via"], "chat")

    def test_reject_keeps_reason(self):
        b = self._run("reject", "ya")
        b.proposals.reject.assert_called_once_with("P5", "too early")

    def test_already_decided(self):
        b = self._run("approve", "y", status="applied")
        b.proposals.approve.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
