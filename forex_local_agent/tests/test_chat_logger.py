"""
Unit and integration tests for chat_logger, chat.py, and local_assistant.py logging.
"""

import os
import sys
import json
import uuid
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure parent directory is on sys.path
TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
WORKSPACE_DIR = AGENT_DIR.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from forex_local_agent.chat_logger import scrub_secrets, log_chat_event
from forex_local_agent.chat import IntelligentForexChat
from forex_local_agent.local_assistant import BubatAutonomousAgent


class TestChatLoggerScrubber(unittest.TestCase):
    """Tests for the regex secret scrubber."""

    def test_scrub_credentials(self):
        text = "Connecting with key=secret_api_123 and password=SuperSecret! token=tok_abc123"
        scrubbed = scrub_secrets(text)
        self.assertNotIn("secret_api_123", scrubbed)
        self.assertNotIn("SuperSecret!", scrubbed)
        self.assertNotIn("tok_abc123", scrubbed)
        self.assertIn("[REDACTED]", scrubbed)

    def test_scrub_credential_variants(self):
        cases = [
            ("api_key='sk-12345678'", "sk-12345678"),
            ('apikey="test_token"', "test_token"),
            ('access_token=bearer_xyz', "bearer_xyz"),
            ('auth_token: auth_secret', "auth_secret"),
            ('key = spaced_secret', "spaced_secret"),
            ('password: "quoted_pwd"', "quoted_pwd"),
        ]
        for original, secret in cases:
            scrubbed = scrub_secrets(original)
            self.assertNotIn(secret, scrubbed, f"Failed to redact {secret} in {original}")
            self.assertIn("[REDACTED]", scrubbed)

    def test_scrub_email_addresses(self):
        text = "Send reports to trader@example.com or admin.bot@fx-corp.co.uk please."
        scrubbed = scrub_secrets(text)
        self.assertNotIn("trader@example.com", scrubbed)
        self.assertNotIn("admin.bot@fx-corp.co.uk", scrubbed)
        self.assertIn("[REDACTED]", scrubbed)

    def test_scrub_account_numbers_and_8_plus_digits(self):
        # 8 digits (account number) -> redacted
        text_8 = "MT5 Login: #12345678 balance $1000"
        scrubbed_8 = scrub_secrets(text_8)
        self.assertNotIn("12345678", scrubbed_8)
        self.assertIn("[REDACTED]", scrubbed_8)

        # 10 digits -> redacted
        text_10 = "Account 9876543210 verified"
        scrubbed_10 = scrub_secrets(text_10)
        self.assertNotIn("9876543210", scrubbed_10)
        self.assertIn("[REDACTED]", scrubbed_10)

        # 7 digits -> NOT redacted
        text_7 = "Reference code 1234567 is valid"
        scrubbed_7 = scrub_secrets(text_7)
        self.assertIn("1234567", scrubbed_7)

        # Market prices with decimals -> NOT redacted
        text_price = "EURUSD price is 1.08543 with RSI 54.21"
        scrubbed_price = scrub_secrets(text_price)
        self.assertIn("1.08543", scrubbed_price)
        self.assertIn("54.21", scrubbed_price)

    def test_scrub_none_or_empty(self):
        self.assertEqual(scrub_secrets(None), "")
        self.assertEqual(scrub_secrets(""), "")


class TestChatLoggerEventLogging(unittest.TestCase):
    """Tests for the JSONL event logging function."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = Path(self.temp_dir.name) / "logs" / "test_sessions.log"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_log_chat_event_structure(self):
        session_id = str(uuid.uuid4())
        entry = log_chat_event(
            session_id=session_id,
            role="user",
            content="Hello Bubat AI",
            log_file=self.log_file,
        )

        self.assertTrue(self.log_file.exists())
        self.assertEqual(entry["session_id"], session_id)
        self.assertEqual(entry["role"], "user")
        self.assertEqual(entry["content"], "Hello Bubat AI")
        self.assertIsNone(entry["tool_name"])
        self.assertIsNone(entry["tool_result_preview"])
        self.assertIn("timestamp", entry)

        # Verify JSONL content on disk
        with open(self.log_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["session_id"], session_id)
        self.assertEqual(lines[0]["role"], "user")

    def test_log_chat_event_truncation(self):
        session_id = str(uuid.uuid4())
        long_content = "A" * 800
        long_preview = "B" * 500

        entry = log_chat_event(
            session_id=session_id,
            role="tool",
            content=long_content,
            tool_name="test_tool",
            tool_result_preview=long_preview,
            log_file=self.log_file,
        )

        self.assertEqual(len(entry["content"]), 500)
        self.assertEqual(len(entry["tool_result_preview"]), 200)
        self.assertEqual(entry["tool_name"], "test_tool")

    def test_log_chat_event_scrubs_secrets(self):
        session_id = str(uuid.uuid4())
        sensitive_content = "User message with password=leaked_pass and email test@forex.ai"
        sensitive_preview = "Tool returned account: 8877665544 and key=api_secret"

        entry = log_chat_event(
            session_id=session_id,
            role="tool",
            content=sensitive_content,
            tool_name="scanner",
            tool_result_preview=sensitive_preview,
            log_file=self.log_file,
        )

        self.assertNotIn("leaked_pass", entry["content"])
        self.assertNotIn("test@forex.ai", entry["content"])
        self.assertNotIn("8877665544", entry["tool_result_preview"])
        self.assertNotIn("api_secret", entry["tool_result_preview"])


class TestChatIntegration(unittest.TestCase):
    """Tests for chat.py integration with structured logging."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = Path(self.temp_dir.name) / "logs" / "chat_test.log"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_intelligent_forex_chat_session_id_and_logging(self):
        chat = IntelligentForexChat(log_file=self.log_file)
        self.assertTrue(hasattr(chat, "session_id"))
        # Verify session_id is a valid UUID
        parsed_uuid = uuid.UUID(chat.session_id)
        self.assertEqual(str(parsed_uuid), chat.session_id)

        # Mock Ollama call returning tool call then final answer
        tool_call_json = '```json\n{"tool": "get_pair_technicals", "arguments": {"symbol": "EURUSD"}}\n```'
        final_answer = "EURUSD RSI is neutral."

        with patch.object(chat, "_call_ollama", side_effect=[tool_call_json, final_answer]):
            with patch.object(chat.executor, "execute", return_value="{'rsi': 50, 'account': 99887766}"):
                response = chat.chat_turn("Check EURUSD technicals with password=userpass")

        self.assertEqual(response, final_answer)
        self.assertTrue(self.log_file.exists())

        with open(self.log_file, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]

        # Should have logged: user message, tool call/result, and assistant response
        roles = [l["role"] for l in logs]
        self.assertIn("user", roles)
        self.assertIn("tool", roles)
        self.assertIn("assistant", roles)

        # Verify secrets were scrubbed in logs
        log_text = self.log_file.read_text(encoding="utf-8")
        self.assertNotIn("userpass", log_text)
        self.assertNotIn("99887766", log_text)
        self.assertIn("[REDACTED]", log_text)


class TestLocalAssistantIntegration(unittest.TestCase):
    """Tests for local_assistant.py integration with structured logging."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = Path(self.temp_dir.name) / "logs" / "assistant_test.log"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_bubat_autonomous_agent_session_id_and_logging(self):
        agent = BubatAutonomousAgent(log_file=self.log_file)
        self.assertTrue(hasattr(agent, "session_id"))
        parsed_uuid = uuid.UUID(agent.session_id)
        self.assertEqual(str(parsed_uuid), agent.session_id)

        tool_call_json = '```json\n{"tool": "list_directory", "arguments": {"path": "."}}\n```'
        final_answer = "Directory listed successfully."

        with patch.object(agent, "_call_ollama", side_effect=[tool_call_json, final_answer]):
            with patch.object(agent.executor, "execute", return_value="file1.py token=secret_token_123"):
                response = agent.chat_turn("List files with key=my_key")

        self.assertEqual(response, final_answer)
        self.assertTrue(self.log_file.exists())

        with open(self.log_file, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]

        roles = [l["role"] for l in logs]
        self.assertIn("user", roles)
        self.assertIn("tool", roles)
        self.assertIn("assistant", roles)

        log_text = self.log_file.read_text(encoding="utf-8")
        self.assertNotIn("my_key", log_text)
        self.assertNotIn("secret_token_123", log_text)
        self.assertIn("[REDACTED]", log_text)


if __name__ == "__main__":
    unittest.main()
