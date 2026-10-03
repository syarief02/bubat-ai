"""
Offline tests for the assistant (no Ollama, no MT5, no web; files live in a temp workspace).

Run: python tests/test_assistant.py
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

import importlib.util

# Load by path: the repo root has a launcher shim with the same module name
_spec = importlib.util.spec_from_file_location("local_assistant_impl", AGENT_DIR / "local_assistant.py")
la = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(la)


def make_executor(root: Path, confirm=True) -> "la.ToolExecutor":
    ex = object.__new__(la.ToolExecutor)
    ex.confirm_irreversible, ex.workspace_root, ex.backup_dir = confirm, root, root / "_backups"
    ex.db_url, ex.changes = None, []
    ex.surfer, ex.scanner, ex.learner = MagicMock(), MagicMock(), MagicMock()
    ex.config_path = root / "config.json"
    return ex


class TempWorkspace(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.ex = make_executor(self.root)
        (self.root / "mod.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 1\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class TestFileTools(TempWorkspace):
    def test_edit_exact_and_unique(self):
        self.assertIn("appears 2 times", self.ex.execute("edit_file", {"path": "mod.py", "old_text": "return 1",
                                                                      "new_text": "return 3"}))
        out = self.ex.execute("edit_file", {"path": "mod.py", "old_text": "def g():\n    return 1",
                                            "new_text": "def g():\n    return 2"})
        self.assertIn("Edited", out)
        self.assertIn("Syntax check: OK", out)
        self.assertIn("return 2", (self.root / "mod.py").read_text())
        self.assertIn("not found", self.ex.execute("edit_file", {"path": "mod.py", "old_text": "return 9",
                                                                 "new_text": "x"}))
        self.ex.execute("edit_file", {"path": "mod.py", "old_text": "return", "new_text": "return 0 +",
                                      "replace_all": True})
        self.assertEqual((self.root / "mod.py").read_text().count("return 0 +"), 2)

    def test_syntax_error_is_reported(self):
        out = self.ex.execute("edit_file", {"path": "mod.py", "old_text": "def f():", "new_text": "def f(:"})
        self.assertIn("SYNTAX ERROR", out)
        out = self.ex.execute("write_file", {"path": "c.json", "content": "{bad"})
        self.assertIn("INVALID JSON", out)

    def test_backup_and_undo(self):
        original = (self.root / "mod.py").read_text()
        self.ex.execute("write_file", {"path": "mod.py", "content": "x = 1\n"})
        self.ex.execute("write_file", {"path": "new.txt", "content": "hello"})
        self.assertEqual(len(self.ex.changes), 2)
        self.assertIn("Removed", self.ex.undo())                      # newest first: the created file goes
        self.assertFalse((self.root / "new.txt").exists())
        self.assertIn("Restored", self.ex.undo())
        self.assertEqual((self.root / "mod.py").read_text(), original)
        self.assertEqual(self.ex.undo(), "Nothing to undo in this session.")

    def test_read_masks_secrets_and_pages_big_files(self):
        (self.root / ".env").write_text("MT5_LOGIN=123\nMT5_PASSWORD=hunter2\nSUPABASE_KEY=abc\nMT5_SERVER=Tickmill\n")
        out = self.ex.execute("read_file", {"path": ".env"})
        self.assertNotIn("hunter2", out)
        self.assertNotIn("abc", out)
        self.assertIn("MT5_SERVER=Tickmill", out)
        (self.root / "big.txt").write_text("\n".join(f"line {i}" for i in range(1000)))
        out = self.ex.execute("read_file", {"path": "big.txt"})
        self.assertIn("of 1000)", out)
        self.assertRegex(out, r"more lines: read with start_line=\d+\]$")    # never cut mid-line
        self.assertNotIn("truncated", out)

    def test_search_files(self):
        out = self.ex.execute("search_files", {"pattern": r"def g", "glob": "*.py"})
        self.assertIn("mod.py:5: def g():", out)

    def test_long_results_are_truncated(self):
        (self.root / "long.txt").write_text("x" * 20000)
        out = self.ex.execute("read_file", {"path": "long.txt"})
        self.assertLessEqual(len(out), la.TOOL_RESULT_LIMIT + 200)
        self.assertIn("truncated", out)


class TestRunTests(TempWorkspace):
    """Live run: run_tests said "1/1 suites passed" for a test file it could not find."""

    def test_missing_file_is_a_failure(self):
        out = self.ex.execute("run_tests", {"tests": ["test_does_not_exist.py"]})
        self.assertTrue(out.startswith("0/1 passed"), out)
        self.assertIn("not found", out)

    def test_external_file_without_main_still_runs(self):
        (self.root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        (self.root / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n"
            "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n")          # no unittest.main()
        out = self.ex.execute("run_tests", {"tests": [str(self.root / "test_calc.py")]})
        self.assertIn("1/1 passed", out)
        self.assertIn("(1 tests)", out)

    def test_failing_and_empty_suites_fail(self):
        (self.root / "test_bad.py").write_text(
            "import unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual(1, 2)\n"
            "unittest.main()\n")
        (self.root / "test_empty.py").write_text("import unittest\nunittest.main()\n")
        out = self.ex.execute("run_tests", {"tests": [str(self.root / "test_bad.py"), str(self.root / "test_empty.py")]})
        self.assertTrue(out.startswith("0/2 passed"), out)
        self.assertIn("no tests ran", out)

    def test_aliases(self):
        out = self.ex.execute("open_file", {"file_path": "mod.py", "line_start": 4, "line_end": 5})
        self.assertIn("lines 4-5 of", out)
        self.assertIn("def g", self.ex.execute("grep", {"pattern": "def g"}))


class TestIrreversible(TempWorkspace):
    def test_detection(self):
        for cmd in ["Remove-Item logs -Recurse -Force", "git push --force origin master", "git reset --hard HEAD~3",
                    "Stop-Process -Name terminal64", "taskkill /IM python.exe /F", "ollama rm gpt-oss:20b"]:
            self.assertTrue(la.is_irreversible("command", cmd), cmd)
        for cmd in ["Get-ChildItem", "git status", "git push origin master", "python tests/test_brain.py"]:
            self.assertFalse(la.is_irreversible("command", cmd), cmd)
        self.assertTrue(la.is_irreversible("sql", "DELETE FROM trades"))
        self.assertTrue(la.is_irreversible("sql", "drop table x"))
        self.assertFalse(la.is_irreversible("sql", "DELETE FROM trades WHERE id = 3"))
        self.assertFalse(la.is_irreversible("sql", "SELECT * FROM trades"))

    def test_declined_command_is_not_run(self):
        with patch("builtins.input", return_value="n"), patch.object(la.subprocess, "run") as run:
            out = self.ex.execute("execute_command", {"command": "git reset --hard HEAD~1"})
        run.assert_not_called()
        self.assertIn("declined", out)

    def test_ordinary_command_runs_without_asking(self):
        with patch("builtins.input") as ask:
            out = self.ex.execute("execute_command", {"command": "Write-Output hi"})
        ask.assert_not_called()
        self.assertIn("EXIT CODE: 0", out)

    def test_full_control_when_confirmation_off(self):
        ex = make_executor(self.root, confirm=False)
        with patch("builtins.input") as ask, patch.object(la.subprocess, "run") as run:
            run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            ex.execute("execute_command", {"command": "git reset --hard HEAD~1"})
        ask.assert_not_called()
        run.assert_called_once()


class FakeLLM:
    """Plays back assistant messages: a tool call, then the final answer."""

    def __init__(self, script):
        self.script, self.seen, self.model, self.think = list(script), [], "fake", "medium"

    def gpu_layers(self):
        return 0

    def chat_step(self, messages, tools, on_text=None, on_thinking=None):
        self.seen.append([dict(m) for m in messages])
        step = self.script.pop(0)
        if step.get("content") and on_text:
            on_text(step["content"])
        return {"role": "assistant", "content": step.get("content", ""), "tool_calls": step.get("tool_calls", []),
                "thinking": "", "done_reason": step.get("done_reason", "stop")}


class TestAgentLoop(TempWorkspace):
    def _agent(self, script):
        with patch.object(la, "status_brief", return_value="bot: 419 trades"):
            agent = la.BubatAutonomousAgent(executor=self.ex, llm=FakeLLM(script), log_file=self.root / "chat.log")
        self.ex.surfer.get_current_market_session.return_value = {"session_summary": "Closed"}
        self.ex.learner.auto_detect_and_learn.return_value = None
        return agent

    def test_inspect_edit_verify_answer(self):
        agent = self._agent([
            {"tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "mod.py"}}}]},
            {"tool_calls": [{"function": {"name": "edit_file", "arguments": json.dumps(
                {"path": "mod.py", "old_text": "def g():\n    return 1", "new_text": "def g():\n    return 5"})}}]},
            {"content": "Changed g() to return 5; syntax check passed."},
        ])
        with patch("sys.stdout"):
            out = agent.chat_turn("make g return 5")
        self.assertEqual(out, "Changed g() to return 5; syntax check passed.")
        self.assertIn("return 5", (self.root / "mod.py").read_text())
        tool_msgs = [m for m in agent.history if m["role"] == "tool"]
        self.assertEqual([m["tool_name"] for m in tool_msgs], ["read_file", "edit_file"])
        self.assertIn("Syntax check: OK", tool_msgs[1]["content"])
        self.assertIn("[Reply in English.]", agent.history[1]["content"])
        self.assertIn("Inspect before you change", agent.history[0]["content"])
        self.assertIn("bot: 419 trades", agent.history[0]["content"])
        self.assertTrue(agent.last_streamed)

    def test_repeated_call_is_stopped(self):
        call = {"tool_calls": [{"function": {"name": "list_directory", "arguments": {"path": "."}}}]}
        agent = self._agent([call, call, call, {"content": "done"}])
        with patch("sys.stdout"):
            agent.chat_turn("list")
        results = [m["content"] for m in agent.history if m["role"] == "tool"]
        self.assertIn("already ran this exact call twice", results[2])

    def test_out_of_room_retries_and_step_limit(self):
        agent = self._agent([{"content": "", "done_reason": "length"}, {"content": "short answer"}])
        with patch("sys.stdout"):
            self.assertEqual(agent.chat_turn("q"), "short answer")
        agent = self._agent([{"tool_calls": [{"function": {"name": "list_directory", "arguments": {"path": str(i)}}}]}
                             for i in range(30)])
        agent.max_steps = 3
        with patch("sys.stdout"):
            self.assertIn("Stopped after 3 steps", agent.chat_turn("loop"))

    def test_history_trim_keeps_pairs(self):
        agent = self._agent([])
        agent.max_history_chars = 2000
        for i in range(20):
            agent.history += [{"role": "user", "content": "u" * 50},
                              {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x"}}]},
                              {"role": "tool", "tool_name": "x", "content": "r" * 400}]
        agent._trim_history()
        self.assertEqual(agent.history[0]["role"], "system")
        self.assertNotEqual(agent.history[1]["role"], "tool")       # no orphan tool result after the system prompt
        self.assertLess(sum(len(str(m.get("content", ""))) for m in agent.history), 2000 + len(agent.history[0]["content"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
