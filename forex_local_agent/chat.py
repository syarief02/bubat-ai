"""
Bubat AI — Interactive Forex Intelligence & Live Market Agent
=============================================================
Autonomous chat interface equipped with live web browsing, real-time MT5
market scanning, session ranking, and continuous learning memory.
"""

import sys
import os
import json
import re
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Ensure UTF-8 on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(WORKSPACE_DIR))

from core.web_surfer import WebSurfer
from core.market_scanner import MarketScanner
from learning.continuous_learner import ContinuousLearner

try:
    from forex_local_agent.chat_logger import log_chat_event, scrub_secrets
except ImportError:
    from chat_logger import log_chat_event, scrub_secrets

try:
    from core.mt5_engine import MT5Engine
except ImportError:
    MT5Engine = None

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

OLLAMA_API_BASE = "http://localhost:11434"
DEFAULT_MODEL = "agent-brain:32k"
MAX_AGENT_STEPS = 6


class ChatToolExecutor:
    """Executes live web, MT5 scanning, and learning tools for the chat agent."""

    def __init__(self):
        self.surfer = WebSurfer()
        self.scanner = MarketScanner()
        self.learner = ContinuousLearner()
        self.config_path = ROOT_DIR / "config.json"

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        handlers = {
            "scan_market_pairs": self._tool_scan_market_pairs,
            "search_live_web": self._tool_search_live_web,
            "scrape_webpage": self._tool_scrape_webpage,
            "get_pair_technicals": self._tool_get_pair_technicals,
            "learn_new_rule": self._tool_learn_new_rule,
            "get_account_status": self._tool_get_account_status,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return f"Error: Tool '{tool_name}' not recognized."

        try:
            return handler(arguments)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"

    def _tool_scan_market_pairs(self, args: Dict[str, Any]) -> str:
        """Scan all MT5 pairs and rank them for the current session."""
        scan_res = self.scanner.scan_and_rank()
        return self.scanner.format_rankings_text(scan_res)

    def _tool_search_live_web(self, args: Dict[str, Any]) -> str:
        """Search Google News RSS and live web for financial headlines."""
        query = args.get("query", "forex market session").strip()
        news = self.surfer.search_news(query, max_results=5)
        if not news:
            web_results = self.surfer.search_web(query, max_results=4)
            if web_results:
                return f"Live Web Results for '{query}':\n" + "\n".join([f"- {r['snippet']}" for r in web_results])
            return f"No news found for '{query}'."

        formatted = [f"- [{n['source']}] {n['title']}\n  Snippet: {n['snippet']}\n  Link: {n['url']}" for n in news]
        return f"Live Financial News for '{query}':\n" + "\n".join(formatted)

    def _tool_scrape_webpage(self, args: Dict[str, Any]) -> str:
        url = args.get("url", "").strip()
        if not url:
            return "Error: No URL provided."
        return self.surfer.scrape_webpage(url)

    def _tool_get_pair_technicals(self, args: Dict[str, Any]) -> str:
        symbol = args.get("symbol", "EURUSD").upper().strip()
        if not MT5Engine:
            return "MT5Engine not available."
        engine = MT5Engine(str(self.config_path))
        if not engine.initialize():
            return "Could not connect to MT5 terminal."
        try:
            tech = engine.get_technical_data(symbol)
            return json.dumps(tech, indent=2, default=str)
        finally:
            engine.shutdown()

    def _tool_learn_new_rule(self, args: Dict[str, Any]) -> str:
        rule_text = args.get("rule_text", "").strip()
        category = args.get("category", "USER_DIRECTIVE").strip()
        if not rule_text:
            return "Error: No rule text provided."
        return self.learner.learn_rule(rule_text, category=category, source="chat_interaction")

    def _tool_get_account_status(self, args: Dict[str, Any]) -> str:
        if not MT5Engine:
            return "MT5Engine not available."
        engine = MT5Engine(str(self.config_path))
        if not engine.initialize():
            return "Could not connect to MT5 terminal."
        try:
            acct = engine.get_account_info()
            positions = engine.get_open_positions()
            return (
                f"MT5 Account: #{acct.get('login')} ({acct.get('server')})\n"
                f"Balance: ${acct.get('balance'):.2f} | Equity: ${acct.get('equity'):.2f} | Free Margin: ${acct.get('free_margin'):.2f}\n"
                f"Open Positions: {len(positions)}"
            )
        finally:
            engine.shutdown()


class IntelligentForexChat:
    """Conversational Forex AI that browses the web, scans MT5, and learns."""

    def __init__(self, model: str = DEFAULT_MODEL, log_file: Optional[Union[str, Path]] = None):
        self.model = model
        self.session_id = str(uuid.uuid4())
        self.log_file = log_file or "logs/chat_sessions.log"
        self.executor = ChatToolExecutor()
        self.conversation_history: List[Dict[str, str]] = []
        self._build_system_prompt()

    def _build_system_prompt(self):
        session = self.executor.surfer.get_current_market_session()
        rules = self.executor.learner.get_all_rules_text()

        sys_prompt = f"""You are Bubat AI, an elite autonomous Forex Intelligence Agent.
You are running locally on the user's high-performance hardware (RTX 4060 GPU) with DIRECT real-time access to MetaTrader 5 and the live internet.

ACTIVE ENVIRONMENT:
- Current UTC Time: {session.get('utc_time')}
- Active Market Session: {session.get('session_summary')}
- High Liquidity Pairs for this Session: {', '.join(session.get('best_pairs_for_session', []))}

STORED LEARNED RULES & MEMORY:
{rules}

ABSOLUTE OPERATIONAL MANDATES:
1. NEVER say "As an AI I do not have access to real-time data or the web". You DO have direct access to live MT5 quotes and live web search!
2. When asked about which pairs to trade, current session setups, or ranking pairs, evaluate the live MT5 scan and news data provided in context.
3. LANGUAGE DIRECTIVE: If the user speaks Malay, chat in friendly, natural Malaysian Malay ("Bahasa Melayu santai/Malaysia", e.g. "Beres boss, kita bercakap santai-santai", "jom kita tengok market harini", "setup ni nampak cun"). NEVER use formal Indonesian ("berbicara").
   Example:
   User: "kita bercakap. bukan berbicara."
   Response: "Beres boss! Jom kita bercakap santai-santai. Ada apa-apa pair forex yang nak kita kaji malam ni?"
4. When asked to remember or learn something, call `learn_new_rule` to store it permanently.

AVAILABLE TOOLS:
- scan_market_pairs(): Scan all live MT5 pairs (EURUSD, USDJPY, GBPUSD, AUDUSD, NZDUSD, EURJPY, GBPJPY, XAUUSD), compute RSI, ATR, EMAs, 24h change %, and rank them by opportunity for the active session.
- search_live_web(query: str): Search live financial news, Google News RSS, and economic headlines.
- scrape_webpage(url: str): Read any web page in full text.
- get_pair_technicals(symbol: str): Deep dive into indicators for a specific pair.
- get_account_status(): Check MT5 account balance, equity, and open positions.
- learn_new_rule(rule_text: str, category: str): Permanently save a rule into disk memory and Supabase cloud.

TOOL CALL FORMAT:
When you need to call a tool, respond with a JSON code block:
```json
{{"tool": "tool_name", "arguments": {{"param": "value"}}}}
```
When you receive the tool result, synthesize the findings into a clear, structured response for the user.
"""
        self.conversation_history = [
            {"role": "system", "content": sys_prompt}
        ]

    def reset(self):
        self._build_system_prompt()

    def chat_turn(self, user_prompt: str) -> str:
        # Log user message
        log_chat_event(
            session_id=self.session_id,
            role="user",
            content=user_prompt,
            log_file=self.log_file,
        )

        # 1. Continuous Learning: Check if user prompt teaches a rule or preference
        learned_notice = self.executor.learner.auto_detect_and_learn(user_prompt)
        if learned_notice:
            print(f"\n{GREEN}{BOLD}[BRAIN UPDATE]{RESET} {learned_notice}")
            # Refresh system prompt with new learned rule
            self._build_system_prompt()

        # 2. Context Auto-Enrichment:
        # If user asks about market, pairs, session, rankings, trades, or news,
        # fetch real-time MT5 scan and live financial news immediately!
        market_keywords = [
            "pair", "session", "rank", "trade", "best", "setup", "market",
            "eurusd", "usdjpy", "gbpusd", "gold", "xauusd", "rsi", "indicator",
            "news", "economic", "calendar", "pasaran", "mata wang", "pilihan"
        ]
        is_market_query = any(k in user_prompt.lower() for k in market_keywords)

        augmented_prompt = user_prompt
        if is_market_query:
            print(f"\n{CYAN}{BOLD}▶ [REAL-TIME ENGINE]{RESET} {DIM}Scanning live MT5 pairs & financial news...{RESET}")
            try:
                scan_data = self.executor.scanner.scan_and_rank()
                scan_text = self.executor.scanner.format_rankings_text(scan_data)
                news_items = self.executor.surfer.search_news("forex market", max_results=3)
                news_text = "\n".join([f"- {n['title']} ({n.get('snippet', '')})" for n in news_items])

                augmented_prompt = (
                    f"{user_prompt}\n\n"
                    f"--- LIVE REAL-TIME MT5 MARKET FEED & SESSION DATA ---\n"
                    f"{scan_text}\n\n"
                    f"--- LIVE WEB FINANCIAL NEWS HEADLINES ---\n"
                    f"{news_text}\n\n"
                    f"[Instruction: Synthesize this real-time MT5 scan and news directly to give the user an accurate, ranked answer.]"
                )
            except Exception as e:
                print(f"{YELLOW}[Warning] Auto-scan error: {e}{RESET}")

        self.conversation_history.append({"role": "user", "content": augmented_prompt})

        called_tools = []
        step = 0

        while step < MAX_AGENT_STEPS:
            step += 1

            response_text = self._call_ollama()
            if not response_text:
                err_msg = f"Error: Unable to connect to Ollama server at {OLLAMA_API_BASE}."
                log_chat_event(
                    session_id=self.session_id,
                    role="assistant",
                    content=err_msg,
                    log_file=self.log_file,
                )
                return f"{RED}{err_msg}{RESET}"

            tool_call = self._extract_tool_call(response_text)

            if not tool_call:
                log_chat_event(
                    session_id=self.session_id,
                    role="assistant",
                    content=response_text,
                    log_file=self.log_file,
                )
                self.conversation_history.append({"role": "assistant", "content": response_text})
                return response_text

            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            # Loop detection
            call_key = (tool_name, json.dumps(tool_args, sort_keys=True))
            if call_key in called_tools[-2:]:
                self.conversation_history.append({"role": "assistant", "content": response_text})
                self.conversation_history.append({
                    "role": "user",
                    "content": "You have already executed this tool. Please synthesize the data you collected and give the final answer to the user now."
                })
                final_res = self._call_ollama()
                final_text = final_res or "Analysis complete."
                log_chat_event(
                    session_id=self.session_id,
                    role="assistant",
                    content=final_text,
                    log_file=self.log_file,
                )
                self.conversation_history.append({"role": "assistant", "content": final_res})
                return final_text

            called_tools.append(call_key)

            print(f"\n{CYAN}{BOLD}▶ Calling Tool:{RESET} {GREEN}{tool_name}{RESET}")
            if tool_args:
                arg_str = json.dumps(tool_args)
                if len(arg_str) > 100:
                    arg_str = arg_str[:100] + "..."
                print(f"  {DIM}Args: {arg_str}{RESET}")

            tool_output = self.executor.execute(tool_name, tool_args)

            lines = tool_output.strip().splitlines()
            preview = lines[0] if lines else "(empty)"
            if len(preview) > 90:
                preview = preview[:90] + "..."
            print(f"  {DIM}Result: {preview} ({len(tool_output)} chars){RESET}")

            # Log tool call and result
            log_chat_event(
                session_id=self.session_id,
                role="tool",
                content=json.dumps(tool_args),
                tool_name=tool_name,
                tool_result_preview=tool_output,
                log_file=self.log_file,
            )

            self.conversation_history.append({"role": "assistant", "content": response_text})
            self.conversation_history.append({
                "role": "user",
                "content": f"[TOOL RESULT for {tool_name}]:\n{tool_output}\n\nTask: Synthesize this data and answer the user's question with clarity. If more info is needed, invoke the next tool."
            })

        log_chat_event(
            session_id=self.session_id,
            role="assistant",
            content="Analysis completed.",
            log_file=self.log_file,
        )
        return "Analysis completed."

    def _call_ollama(self) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 16384
            }
        }

        req = urllib.request.Request(
            f"{OLLAMA_API_BASE}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"{RED}[Ollama Error] {e}{RESET}")
            return None

    def _extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        # Look for ```json ... ``` blocks
        matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        for m in matches:
            try:
                parsed = json.loads(m)
                tool_name = parsed.get("tool") or parsed.get("name")
                if tool_name:
                    return {
                        "name": tool_name,
                        "arguments": parsed.get("arguments") or parsed.get("parameters", {})
                    }
            except Exception:
                continue

        # Look for raw JSON object
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                tool_name = parsed.get("tool") or parsed.get("name")
                if tool_name:
                    return {
                        "name": tool_name,
                        "arguments": parsed.get("arguments") or parsed.get("parameters", {})
                    }
            except Exception:
                pass

        return None


def print_banner(agent: IntelligentForexChat):
    session = agent.executor.surfer.get_current_market_session()
    print("\n" + "=" * 72)
    print(f"{CYAN}{BOLD}  BUBAT AI — LOCAL FOREX INTELLIGENCE & MARKET AGENT{RESET}")
    print(f"{DIM}  Model: {agent.model}  |  Server: localhost:11434  |  GPU: RTX 4060{RESET}")
    print("=" * 72)
    print(f"  Live Market Session: {GREEN}{BOLD}{session.get('session_summary')}{RESET}")
    print(f"  Active UTC Time:     {DIM}{session.get('utc_time')}{RESET}")
    print("-" * 72)
    print("  Capabilities:")
    print(f"   • {GREEN}Live MT5 Multi-Pair Scanner{RESET} -> Ranks EURUSD, GBPUSD, USDJPY, Gold, etc.")
    print(f"   • {GREEN}Real-Time Web Surfing{RESET}       -> Google News Financial RSS, live macro news")
    print(f"   • {GREEN}Continuous Learning Memory{RESET}  -> Remembers rules in learned_rules.md & Supabase")
    print(f"   • {GREEN}Bilingual Intelligence{RESET}      -> Natural Malaysian Malay & English")
    print("-" * 72)
    print(f"  Commands: {YELLOW}clear{RESET} (reset memory), {YELLOW}rules{RESET} (view learned rules), {YELLOW}exit{RESET}")
    print("=" * 72 + "\n")


def main():
    agent = IntelligentForexChat()
    print_banner(agent)

    while True:
        try:
            user_input = input(f"\n{BOLD}You ❯{RESET} ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print(f"\n{CYAN}Bubat AI standing by. Jumpa lagi!{RESET}")
                break

            if user_input.lower() == "clear":
                agent.reset()
                print(f"{YELLOW}[+] Conversation memory cleared.{RESET}")
                continue

            if user_input.lower() == "rules":
                rules = agent.executor.learner.get_all_rules_text()
                print(f"\n{CYAN}{rules}{RESET}")
                continue

            # Run intelligent agent turn
            start_time = time.time()
            response = agent.chat_turn(user_input)
            duration = time.time() - start_time

            print(f"\n{MAGENTA}{BOLD}Bubat AI ❯{RESET} {response}")
            print(f"{DIM}[Response in {duration:.2f}s]{RESET}")

        except KeyboardInterrupt:
            print(f"\n{YELLOW}Interrupted. Type 'exit' to quit or enter a new request.{RESET}")
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")


if __name__ == "__main__":
    main()
