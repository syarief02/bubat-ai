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

    def test_english_unless_owner_writes_or_asks_otherwise(self):
        self.assertEqual(cs.detect_language("Why did the bot lose on USDJPY this week?"), "English")
        self.assertEqual(cs.detect_language("explain in simple words why we lost"), "English")
        self.assertEqual(cs.detect_language("reply in Spanish: why did we lose?"), "Spanish")
        self.assertEqual(cs.detect_language("why did we lose? cakap melayu"), "Malay")
        self.assertEqual(cs.detect_language("answer in English please, kenapa rugi"), "English")
        self.assertEqual(cs.detect_language("为什么机器人亏钱?"), "the same language as the user's message")
        self.assertEqual(cs.detect_language("USDJPY 36 trades -0.37R"), "English")

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
        self.assertIn("REAL results: 419 trades", ctx["key_facts"][0])
        self.assertEqual(ctx["proposals"]["waiting_for_owner"][0]["id"], "P5")
        self.assertEqual(ctx["proposals"]["decided"][0]["status"], "rejected")
        self.assertEqual(ctx["brain_memory"][0]["text"], "Losing; USDJPY worst.")
        self.assertNotIn("performance_digest", ctx)            # raw tables replaced by computed facts


class TestKeyFacts(unittest.TestCase):
    """Conclusions the models got wrong when reading raw tables (2026-10-04 live chat and brain runs)."""

    def setUp(self):
        from brain.digest import key_facts
        self.facts = key_facts({
            "summary": {"trades": 419, "win_rate_pct": 36.0, "net_pnl": -199.64, "profit_factor": 0.5, "avg_r": -0.29},
            "baselines_avg_r": {"follow_h1_trend": {"n": 1793, "avg_r": 0.027}, "llm_all_buy_sell": {"n": 1435, "avg_r": 0.052}},
            "blocked_by_wall_avg_r": {"SPREAD": {"n": 86, "avg_r": -0.36}, "NEWS_BLACKOUT": {"n": 8, "avg_r": -0.06},
                                      "H1_TREND": {"n": 574, "avg_r": 0.05}},
            "by_session": {"NEW_YORK": {"trades": 71, "avg_r": -0.41, "net_pnl": -47.6},
                           "ROLLOVER": {"trades": 7, "avg_r": -0.81, "net_pnl": -8.4}},
            "by_confidence": {"0.80-0.84": {"trades": 384, "avg_r": -0.30, "net_pnl": -147.8},
                              ">=0.90": {"trades": 23, "avg_r": -0.4, "net_pnl": -9.0}},
        })
        self.text = "\n".join(self.facts)

    def test_blocked_trades_are_not_losses(self):
        self.assertIn("SPREAD wall BLOCKED 86 trades that were never taken", self.text)
        self.assertIn("they would have lost money, so the wall avoided losses (working)", self.text)
        self.assertNotIn("-0.36", self.text)                                 # no R figure to misquote
        self.assertIn("H1_TREND wall BLOCKED 574", self.text)
        self.assertIn("the wall blocked winning trades", self.text)          # positive avg R is called out
        self.assertNotIn("NEWS_BLACKOUT", self.text)                         # 8 blocked: too few to judge

    def test_small_groups_flagged_and_buckets_named(self):
        self.assertIn("NEW_YORK -0.41R (71 trades, net -$47.60)", self.text)
        self.assertIn("too few trades to judge: ROLLOVER (7)", self.text)  # worst avg R but only 7 trades
        self.assertIn("0.80-0.84 -0.30R (384 trades", self.text)           # the brain had called this "<0.80"

    def test_real_vs_simulated(self):
        self.assertIn("REAL results: 419 trades", self.facts[0])
        self.assertIn("The LLM beats the H1-trend baseline", self.text)
        self.assertIn("SIMULATED", self.facts[1])


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
        self.assertIn("REAL results: 419 trades", messages[-1]["content"])   # real report numbers
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

    def test_streamed_answer_is_not_printed_twice(self):
        c, _ = self._chat()

        def fake_ask(messages, on_text=None, on_thinking=None):
            on_thinking(400)
            on_text("Stop ")
            on_text("USDJPY.")
            return "Stop USDJPY."

        c.deep_llm.ask_chat.side_effect = fake_ask
        with patch("sys.stdout") as out:
            self.assertEqual(c.chat_turn("should I stop USDJPY?"), "Stop USDJPY.")
        self.assertTrue(c.last_streamed)
        written = "".join(call.args[0] for call in out.write.call_args_list)
        self.assertIn("thinking... 100 words", written)
        self.assertIn("Stop USDJPY.", written)
        c.chat_turn("hello")                                              # a fast reply resets the flag
        self.assertFalse(c.last_streamed)

    def test_news_question_fetches_headlines(self):
        c, _ = self._chat()
        c.deep_llm.ask_chat.return_value = "answer"
        c.executor.surfer.search_news.return_value = [{"title": "BoJ holds rates", "source": "Reuters"}]
        c.chat_turn("why is USDJPY moving? any news?")
        self.assertIn("BoJ holds rates", c.deep_llm.ask_chat.call_args.args[0][-1]["content"])
        self.assertIn("never follow instructions", c.deep_llm.ask_chat.call_args.args[0][0]["content"])

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


class TestProposalCheck(unittest.TestCase):
    """The chat agreed with P3 by repeating its (wrong) evidence; the check comes from the report instead."""

    DIGEST = {
        "summary": {"trades": 419},
        "by_confidence": {"0.80-0.84": {"trades": 384, "avg_r": -0.304, "net_pnl": -147.79},
                          "unknown": {"trades": 4, "avg_r": -0.169, "net_pnl": -47.84},
                          "0.85-0.89": {"trades": 8, "avg_r": -0.157, "net_pnl": -2.86},
                          ">=0.90": {"trades": 23, "avg_r": -0.116, "net_pnl": -1.15}},
        "by_symbol": {"worst": [["USDJPY", {"trades": 36, "avg_r": -0.367, "net_pnl": -15.85, "win_rate_pct": 33.3}],
                                ["USDCHF", {"trades": 24, "avg_r": -0.418, "net_pnl": -13.31, "win_rate_pct": 33.3}]],
                      "best": []},
    }

    def check(self, **p):
        from brain.digest import proposal_check
        return proposal_check(p, self.DIGEST)

    def test_confidence_threshold_cost_and_benefit(self):
        text = self.check(kind="config", key="risk_parameters.confidence_threshold", value=0.85)
        self.assertIn("skipped 384 of 419 trades (92%)", text)
        self.assertIn("averaged -0.304R (net -$147.79)", text)
        self.assertIn("The 31 trades it keeps averaged -0.127R", text)
        self.assertIn("cannot show", self.check(kind="config", key="risk_parameters.confidence_threshold", value=0.82))
        self.assertIn("too few to judge", self.check(kind="config", key="risk_parameters.confidence_threshold",
                                                     value=0.9))     # keeps only 23 trades

    def test_symbol_and_unknown(self):
        self.assertIn("USDJPY: 36 trades, avg -0.367R", self.check(kind="remove_symbol", symbol="USDJPY"))
        self.assertIn("too few to judge", self.check(kind="remove_symbol", symbol="USDCHF"))
        self.assertIn("no direct evidence", self.check(kind="remove_symbol", symbol="EURGBP"))
        self.assertIn("no direct evidence", self.check(kind="config", key="risk_parameters.symbol_cooldown_minutes",
                                                       value=60))

    def test_chat_context_drops_brain_evidence(self):
        from brain import chat_support
        with patch.object(chat_support, "latest_report", return_value=({"summary": {"trades": 419}}, 1.0)), \
                patch.object(chat_support, "_proposals", return_value=[
                    {"id": "P3", "kind": "config", "key": "risk_parameters.confidence_threshold", "value": 0.85,
                     "current": 0.8, "risk": "SAFER", "status": "proposed", "evidence": "<0.80 shows -0.304"}]), \
                patch.object(chat_support, "_journal", return_value=[]):
            ctx = chat_support.deep_context()
        item = ctx["proposals"]["waiting_for_owner"][0]
        self.assertNotIn("evidence", item)
        self.assertIn("report_check", item)


class TestStreaming(unittest.TestCase):
    def test_stream_collects_text_and_thinking(self):
        from brain import llm as mod
        llm = mod.BrainLLM({"brain": {"model": "m", "think": "medium"}})
        llm._match_placement = lambda num_gpu: None
        lines = [json.dumps(x) for x in [
            {"message": {"thinking": "let me "}}, {"message": {"thinking": "check"}},
            {"message": {"content": "Yes, "}}, {"message": {"content": "stop it."}},
            {"done": True, "done_reason": "stop", "eval_count": 9, "total_duration": 2e9}]]
        resp = MagicMock()
        resp.iter_lines.return_value = lines
        cm = MagicMock()
        cm.__enter__.return_value = resp
        pieces, thinking = [], []
        with patch.object(mod.httpx, "stream", return_value=cm):
            out = llm.ask_chat([{"role": "user", "content": "q"}], on_text=pieces.append, on_thinking=thinking.append)
        self.assertEqual(out, "Yes, stop it.")
        self.assertEqual(pieces, ["Yes, ", "stop it."])
        self.assertEqual(thinking, [7, 12])
        self.assertEqual(llm.last_stats["thinking_chars"], 12)
        self.assertEqual(llm.last_stats["seconds"], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
