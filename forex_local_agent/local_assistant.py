"""
================================================================================
BUBAT AI — LOCAL AUTONOMOUS SYSTEM & CODING AGENT
================================================================================
Empowers your local Ollama model (config.json `active_model`) with
autonomous tool-calling capabilities to act directly on your Windows machine:
- Execute PowerShell commands
- Inspect, read, create, and edit files
- Explore directories and codebases
- Directly query and manage your Supabase PostgreSQL database
- Live multi-pair MT5 market scanning and ranking
- Real-time web surfing and financial news retrieval
- Continuous learning memory (learned_rules.md & Supabase)

Usage:
    python local_assistant.py [--model agent-brain-8b:8k] [--auto-confirm]
================================================================================
"""

import os
import sys
import json
import re
import time
import subprocess
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import urllib.request
import urllib.error

# Ensure UTF-8 output in Windows PowerShell / CMD
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths & Environment
_HERE = Path(__file__).resolve().parent
# Canonical copy lives in forex_local_agent/; the repo-root local_assistant.py is a launcher shim
WORKSPACE_DIR = _HERE.parent if _HERE.name == "forex_local_agent" else _HERE
AGENT_DIR = WORKSPACE_DIR / "forex_local_agent"
sys.path.insert(0, str(WORKSPACE_DIR))
sys.path.insert(0, str(AGENT_DIR))

try:
    from forex_local_agent.chat_logger import log_chat_event, scrub_secrets, DEFAULT_LOG_FILE
except ImportError:
    from chat_logger import log_chat_event, scrub_secrets, DEFAULT_LOG_FILE

from dotenv import load_dotenv

env_file = WORKSPACE_DIR / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

# Subsystems
try:
    from forex_local_agent.core.web_surfer import WebSurfer
    from forex_local_agent.core.market_scanner import MarketScanner
    from forex_local_agent.learning.continuous_learner import ContinuousLearner
except ImportError:
    from core.web_surfer import WebSurfer
    from core.market_scanner import MarketScanner
    from learning.continuous_learner import ContinuousLearner

try:
    from forex_local_agent.core.mt5_engine import MT5Engine
except ImportError:
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


def _load_model_settings() -> tuple[str, Optional[bool]]:
    """Share the trading agent's model (config.json) so Ollama keeps a single copy in VRAM."""
    try:
        with open(AGENT_DIR / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("active_model", "agent-brain:32k"), cfg.get("ollama_think")
    except Exception:
        return "agent-brain:32k", None


DEFAULT_MODEL, OLLAMA_THINK = _load_model_settings()
MAX_AGENT_STEPS = 8


# ==============================================================================
# TOOL EXECUTOR (The "Hands")
# ==============================================================================

class ToolExecutor:
    """Executes actions locally on Windows, MT5, Supabase, and the live web."""

    def __init__(self, auto_confirm: bool = False):
        self.auto_confirm = auto_confirm
        self.workspace_root = WORKSPACE_DIR
        self.db_url = os.getenv("DATABASE_URL")
        self.surfer = WebSurfer()
        self.scanner = MarketScanner()
        self.learner = ContinuousLearner()
        self.config_path = AGENT_DIR / "config.json"

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        handlers = {
            "execute_command": self._tool_execute_command,
            "read_file": self._tool_read_file,
            "write_file": self._tool_write_file,
            "list_directory": self._tool_list_directory,
            "query_database": self._tool_query_database,
            "get_system_status": self._tool_get_system_status,
            "scan_market_pairs": self._tool_scan_market_pairs,
            "search_live_web": self._tool_search_live_web,
            "scrape_webpage": self._tool_scrape_webpage,
            "learn_new_rule": self._tool_learn_new_rule,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return f"Error: Tool '{tool_name}' is not recognized. Available tools: {list(handlers.keys())}"

        try:
            return handler(arguments)
        except Exception as e:
            return f"Execution error in tool '{tool_name}': {str(e)}"

    def _tool_execute_command(self, args: Dict[str, Any]) -> str:
        command = args.get("command", "").strip()
        timeout = int(args.get("timeout_seconds", 45))

        if not command:
            return "Error: No command provided."

        destructive = [
            r"\brmdir\s+/s\b", r"\bdel\s+/[sfq]\b", r"\bRemove-Item\b.*-Recurse",
            r"\bformat\b", r"\bdrop\s+table\b", r"\bdrop\s+database\b", r"\bshutdown\b"
        ]
        if any(re.search(pat, command, re.IGNORECASE) for pat in destructive) and not self.auto_confirm:
            print(f"\n{YELLOW}{BOLD}[SAFETY WARNING] Command may be destructive:{RESET} {command}")
            if input(f"{YELLOW}Authorize this command? (y/N): {RESET}").strip().lower() != "y":
                return "Execution cancelled by user."

        try:
            process = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                cwd=str(self.workspace_root),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace"
            )

            stdout = process.stdout.strip()
            stderr = process.stderr.strip()
            parts = []
            if stdout:
                parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                parts.append(f"STDERR:\n{stderr}")
            if not stdout and not stderr:
                parts.append("(Command completed with no output)")
            parts.append(f"EXIT CODE: {process.returncode}")

            res = "\n".join(parts)
            return res[:2500] + ("\n... [truncated]" if len(res) > 2500 else "")
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds."
        except Exception as e:
            return f"Error: {e}"

    def _tool_read_file(self, args: Dict[str, Any]) -> str:
        raw_path = args.get("path", "").strip()
        if not raw_path:
            return "Error: No file path provided."

        target = Path(raw_path)
        if not target.is_absolute():
            target = (self.workspace_root / target).resolve()

        if not target.exists():
            return f"Error: File '{raw_path}' does not exist."
        if target.is_dir():
            return f"Error: Path '{raw_path}' is a directory."

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            start = args.get("start_line")
            end = args.get("end_line")
            start_idx = max(0, (start - 1) if start else 0)
            end_idx = min(len(lines), end if end else len(lines))
            sliced = lines[start_idx:end_idx]
            formatted = [f"{i + start_idx + 1:4d}: {l}" for i, l in enumerate(sliced)]
            return f"File: {raw_path} (Lines {start_idx + 1}-{end_idx} of {len(lines)})\n" + "\n".join(formatted)
        except Exception as e:
            return f"Error reading '{raw_path}': {e}"

    def _tool_write_file(self, args: Dict[str, Any]) -> str:
        raw_path = args.get("path", "").strip()
        content = args.get("content", "")
        mode = args.get("mode", "write").lower()

        if not raw_path:
            return "Error: No file path provided."

        target = Path(raw_path)
        if not target.is_absolute():
            target = (self.workspace_root / target).resolve()

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            write_mode = "a" if (mode == "append" and target.exists()) else "w"
            with open(target, write_mode, encoding="utf-8") as f:
                f.write(content)
            return f"Success: Saved {len(content)} characters to '{raw_path}'."
        except Exception as e:
            return f"Error writing '{raw_path}': {e}"

    def _tool_list_directory(self, args: Dict[str, Any]) -> str:
        raw_path = str(args.get("path", ".")).strip()
        if raw_path in ["/", "\\", ".", "", "./"]:
            target = self.workspace_root
        else:
            target = Path(raw_path)
            if not target.is_absolute():
                target = (self.workspace_root / target).resolve()

        if not target.exists():
            return f"Error: Directory '{raw_path}' does not exist."

        try:
            entries = []
            for p in sorted(target.iterdir()):
                if p.name in [".git", "__pycache__", "node_modules"]:
                    continue
                entries.append(f"[DIR]  {p.name}/" if p.is_dir() else f"[FILE] {p.name} ({p.stat().st_size:,} bytes)")
            return f"Directory '{raw_path}' ({len(entries)} items):\n" + "\n".join(entries[:100])
        except Exception as e:
            return f"Error: {e}"

    def _tool_query_database(self, args: Dict[str, Any]) -> str:
        sql = args.get("sql", "").strip()
        if not sql:
            return "Error: No SQL query provided."
        if not self.db_url:
            return "Error: DATABASE_URL not set in .env."

        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url, connect_timeout=8)
            cur = conn.cursor()
            cur.execute(sql)

            if cur.description:
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                conn.commit()
                conn.close()
                formatted = [dict(zip(cols, [str(c) if c is not None else "NULL" for c in r])) for r in rows[:25]]
                return f"Query returned {len(rows)} rows:\n" + json.dumps(formatted, indent=2, default=str)
            else:
                affected = cur.rowcount
                conn.commit()
                conn.close()
                return f"Query executed successfully. Rows affected: {affected}"
        except Exception as e:
            return f"Database error: {e}"

    def _tool_get_system_status(self, args: Dict[str, Any]) -> str:
        parts = []
        try:
            total, used, free = shutil.disk_usage(str(self.workspace_root))
            parts.append(f"Disk Free: {free // (2**30)} GB of {total // (2**30)} GB")
        except Exception:
            pass

        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                acct = mt5.account_info()
                positions = mt5.positions_total()
                if acct:
                    parts.append(f"MT5 Account ({acct.server}) | Balance: ${acct.balance:.2f} | Equity: ${acct.equity:.2f} | Open Positions: {positions}")
                mt5.shutdown()
            else:
                parts.append("MT5 Terminal: Not connected")
        except Exception as e:
            parts.append(f"MT5: {e}")

        parts.append("Supabase DB: Connected" if self.db_url else "Supabase DB: Not configured")
        return "System Status:\n- " + "\n- ".join(parts)

    def _tool_scan_market_pairs(self, args: Dict[str, Any]) -> str:
        scan_res = self.scanner.scan_and_rank()
        return self.scanner.format_rankings_text(scan_res)

    def _tool_search_live_web(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "forex market").strip()
        news = self.surfer.search_news(query, max_results=5)
        if not news:
            web = self.surfer.search_web(query, max_results=4)
            return "\n".join([f"- {r['snippet']}" for r in web]) if web else f"No results for '{query}'."
        return "\n".join([f"- [{n['source']}] {n['title']} ({n['snippet']})" for n in news])

    def _tool_scrape_webpage(self, args: Dict[str, Any]) -> str:
        url = args.get("url", "").strip()
        return self.surfer.scrape_webpage(url)

    def _tool_learn_new_rule(self, args: Dict[str, Any]) -> str:
        rule_text = args.get("rule_text", "").strip()
        cat = args.get("category", "USER_DIRECTIVE").strip()
        return self.learner.learn_rule(rule_text, category=cat, source="assistant_interaction")


# ==============================================================================
# AUTONOMOUS AGENT LOOP (The "Brain")
# ==============================================================================

class BubatAutonomousAgent:
    """Agentic orchestrator that reasons, calls tools, and reports results."""

    def __init__(self, model: str = DEFAULT_MODEL, auto_confirm: bool = False, log_file: Optional[Union[str, Path]] = None):
        self.model = model
        self.session_id = str(uuid.uuid4())
        self.log_file = log_file or DEFAULT_LOG_FILE
        self.executor = ToolExecutor(auto_confirm=auto_confirm)
        self.conversation_history: List[Dict[str, str]] = []
        self._init_system_prompt()

    def _init_system_prompt(self):
        rules = self.executor.learner.get_all_rules_text()
        session = self.executor.surfer.get_current_market_session()

        system_instruction = f"""You are Bubat AI, an elite autonomous system engineer, coder, and Forex Intelligence Agent running locally on Windows (RTX 4060 GPU).
You have DIRECT real-time access to MetaTrader 5, the live web, PowerShell, and your Supabase PostgreSQL database.

CURRENT ENVIRONMENT:
- UTC Time: {session.get('utc_time')}
- Active Market Session: {session.get('session_summary')}
- High Liquidity Pairs: {', '.join(session.get('best_pairs_for_session', []))}

STORED LEARNED RULES & MEMORY:
{rules}

OPERATIONAL DIRECTIVES:
1. NEVER say "As an AI I do not have access to real-time data or the internet". You DO have direct access via your tools!
2. When asked about forex pairs, market sessions, setups, or rankings, evaluate the live MT5 scan and live financial news provided in context.
3. If the user speaks Malay, use natural, friendly Malaysian Malay ("Bahasa Melayu santai/Malaysia", e.g. "Beres boss, jom kita bercakap", "setup ni nampak cun"). NEVER use formal Indonesian ("berbicara").
4. If a tool fails, analyze the error output and adjust your approach autonomously.

AVAILABLE TOOLS:
- execute_command(command: str): Run PowerShell commands on Windows.
- read_file(path: str, start_line: int, end_line: int): Read files.
- write_file(path: str, content: str, mode: "write"|"append"): Create or edit files.
- list_directory(path: str): List files and folders.
- query_database(sql: str): Execute SQL on Supabase PostgreSQL.
- get_system_status(): Check MT5 account balance, equity, disk space.
- scan_market_pairs(): Scan all live MT5 pairs, compute RSI, ATR, EMAs, 24h change %, and rank best setups.
- search_live_web(query: str): Search Google News RSS and live web for financial/technical info.
- scrape_webpage(url: str): Read any web URL in full text.
- learn_new_rule(rule_text: str, category: str): Permanently store a rule to disk and cloud.

TOOL FORMAT:
Output ONLY a JSON block when invoking a tool:
```json
{{"tool": "tool_name", "arguments": {{"param": "value"}}}}
```
"""
        self.conversation_history = [
            {"role": "system", "content": system_instruction}
        ]

    def reset(self):
        self._init_system_prompt()

    def chat_turn(self, user_prompt: str) -> str:
        # Log user message
        log_chat_event(
            session_id=self.session_id,
            role="user",
            content=user_prompt,
            log_file=self.log_file,
        )

        # 1. Continuous Learning: Auto-detect rules or language preference
        learned = self.executor.learner.auto_detect_and_learn(user_prompt)
        if learned:
            print(f"\n{GREEN}{BOLD}[BRAIN UPDATE]{RESET} {learned}")
            self._init_system_prompt()

        # 2. Auto-enrichment for market queries
        market_keywords = [
            "pair", "session", "rank", "trade", "best", "setup", "market",
            "eurusd", "usdjpy", "gbpusd", "gold", "xauusd", "rsi", "indicator",
            "pasaran", "mata wang", "pilihan"
        ]
        augmented_prompt = user_prompt
        if any(k in user_prompt.lower() for k in market_keywords):
            print(f"\n{CYAN}{BOLD}▶ [REAL-TIME ENGINE]{RESET} {DIM}Scanning live MT5 pairs & financial news...{RESET}")
            try:
                scan_data = self.executor.scanner.scan_and_rank()
                scan_text = self.executor.scanner.format_rankings_text(scan_data)
                news = self.executor.surfer.search_news("forex market", max_results=3)
                news_text = "\n".join([f"- {n['title']} ({n.get('snippet', '')})" for n in news])

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

            response_content = self._call_ollama()
            if not response_content:
                err_msg = f"Error: Unable to connect to Ollama at {OLLAMA_API_BASE}."
                log_chat_event(
                    session_id=self.session_id,
                    role="assistant",
                    content=err_msg,
                    log_file=self.log_file,
                )
                return f"{RED}{err_msg}{RESET}"

            tool_call = self._extract_tool_call(response_content)

            if not tool_call:
                log_chat_event(
                    session_id=self.session_id,
                    role="assistant",
                    content=response_content,
                    log_file=self.log_file,
                )
                self.conversation_history.append({"role": "assistant", "content": response_content})
                return response_content

            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            # Loop guard
            call_sig = (tool_name, json.dumps(tool_args, sort_keys=True))
            if call_sig in called_tools[-2:]:
                self.conversation_history.append({"role": "assistant", "content": response_content})
                self.conversation_history.append({
                    "role": "user",
                    "content": "You have already executed this action. Please synthesize your findings and give the final answer to the user now."
                })
                final_answer = self._call_ollama()
                final_text = final_answer or "Task completed."
                log_chat_event(
                    session_id=self.session_id,
                    role="assistant",
                    content=final_text,
                    log_file=self.log_file,
                )
                self.conversation_history.append({"role": "assistant", "content": final_answer})
                return final_text

            called_tools.append(call_sig)

            print(f"\n{CYAN}{BOLD}▶ Calling Tool:{RESET} {GREEN}{tool_name}{RESET}")
            if tool_args:
                arg_preview = json.dumps(tool_args)
                if len(arg_preview) > 100:
                    arg_preview = arg_preview[:100] + "..."
                print(f"  {DIM}Args: {arg_preview}{RESET}")

            tool_result = self.executor.execute(tool_name, tool_args)

            lines = tool_result.strip().splitlines()
            preview = lines[0] if lines else "(empty)"
            if len(preview) > 80:
                preview = preview[:80] + "..."
            print(f"  {DIM}Result: {preview} ({len(tool_result)} chars){RESET}")

            # Log tool call and result
            log_chat_event(
                session_id=self.session_id,
                role="tool",
                content=json.dumps(tool_args),
                tool_name=tool_name,
                tool_result_preview=tool_result,
                log_file=self.log_file,
            )

            self.conversation_history.append({"role": "assistant", "content": response_content})
            self.conversation_history.append({
                "role": "user",
                "content": f"[TOOL RESULT for {tool_name}]:\n{tool_result}\n\nTask: Analyze this output. If complete, answer the user directly. If more action is needed, invoke the next tool."
            })

        log_chat_event(
            session_id=self.session_id,
            role="assistant",
            content="Agent reached maximum step limit.",
            log_file=self.log_file,
        )
        return "Agent reached maximum step limit."

    def _call_ollama(self) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": False,
            # num_ctx comes from the Modelfile; a different value here would load a second copy of the model
            "options": {
                "temperature": 0.2,
            }
        }
        if OLLAMA_THINK is not None:
            payload["think"] = OLLAMA_THINK

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


# ==============================================================================
# INTERACTIVE CLI SHELL
# ==============================================================================

def print_banner(agent: BubatAutonomousAgent):
    session = agent.executor.surfer.get_current_market_session()
    print("\n" + "=" * 74)
    print(f"{CYAN}{BOLD}  BUBAT AI — LOCAL AUTONOMOUS SYSTEM & FOREX INTELLIGENCE AGENT{RESET}")
    print(f"{DIM}  Model: {agent.model}  |  Host: localhost:11434  |  GPU: RTX 4060{RESET}")
    print("=" * 74)
    print(f"  Live Session: {GREEN}{BOLD}{session.get('session_summary')}{RESET}")
    print(f"  Active Time:  {DIM}{session.get('utc_time')}{RESET}")
    print("-" * 74)
    print("  Capabilities:")
    print(f"   • {GREEN}Live MT5 Market Scanner{RESET}  -> Real-time prices, RSI, ATR, and pair rankings")
    print(f"   • {GREEN}Real-Time Web Surfer{RESET}     -> Live Google News RSS, web search & scrapers")
    print(f"   • {GREEN}Continuous Learning{RESET}      -> Stores rules permanently to disk & Supabase")
    print(f"   • {GREEN}PowerShell / Code Engine{RESET} -> Execute commands, read/write/edit code files")
    print(f"   • {GREEN}Supabase Database Hub{RESET}    -> Direct PostgreSQL queries & telemetry sync")
    print("-" * 74)
    print(f"  Commands: {YELLOW}/clear{RESET} (reset memory), {YELLOW}/rules{RESET} (view learned rules), {YELLOW}/status{RESET} (check system), {YELLOW}exit{RESET}")
    print("=" * 74 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bubat AI Local Autonomous Assistant")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--auto-confirm", action="store_true", help="Auto-confirm potentially risky commands")
    args = parser.parse_args()

    agent = BubatAutonomousAgent(model=args.model, auto_confirm=args.auto_confirm)
    print_banner(agent)

    while True:
        try:
            user_input = input(f"\n{BOLD}You ❯{RESET} ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", ":q"):
                print(f"\n{CYAN}Session ended. Bubat AI standing by.{RESET}")
                break

            if user_input.lower() == "/clear":
                agent.reset()
                print(f"{YELLOW}Conversation memory reset.{RESET}")
                continue

            if user_input.lower() == "/rules":
                rules = agent.executor.learner.get_all_rules_text()
                print(f"\n{CYAN}{rules}{RESET}")
                continue

            if user_input.lower() == "/status":
                status = agent.executor.execute("get_system_status", {})
                print(f"\n{CYAN}{status}{RESET}")
                continue

            # Run autonomous turn
            start_time = time.time()
            response = agent.chat_turn(user_input)
            duration = time.time() - start_time

            print(f"\n{MAGENTA}{BOLD}Bubat AI ❯{RESET} {response}")
            print(f"{DIM}[Completed in {duration:.2f}s]{RESET}")

        except KeyboardInterrupt:
            print(f"\n{YELLOW}Interrupted. Type 'exit' to quit or enter a new request.{RESET}")
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")


if __name__ == "__main__":
    main()
