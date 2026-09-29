"""
================================================================================
BUBAT AI - LOCAL AUTONOMOUS SYSTEM & CODING AGENT
================================================================================
Empowers your local Ollama models (agent-brain:32k, qwen2.5-coder) with
autonomous tool-calling capabilities to act directly on your Windows machine:
- Execute PowerShell commands
- Inspect, read, create, and edit files
- Explore directories and codebases
- Directly query and manage your Supabase PostgreSQL database
- Monitor MetaTrader 5 accounts and open positions
- Search the web for live documentation and news

Usage:
    python local_assistant.py [--model agent-brain:32k] [--auto-confirm]
================================================================================
"""

import os
import sys
import json
import re
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import urllib.request
import urllib.error

# Ensure UTF-8 output in Windows PowerShell / CMD
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment variables (.env)
from dotenv import load_dotenv

workspace_dir = Path(__file__).resolve().parent
env_file = workspace_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

# ANSI Color codes for rich terminal formatting
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
MAX_AGENT_STEPS = 8


# ==============================================================================
# TOOL IMPLEMENTATIONS (The "Hands")
# ==============================================================================

class ToolExecutor:
    """Executes agent actions locally with safety checks and output sanitization."""

    def __init__(self, auto_confirm: bool = False):
        self.auto_confirm = auto_confirm
        self.workspace_root = workspace_dir
        self.db_url = os.getenv("DATABASE_URL")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch tool call by name."""
        handlers = {
            "execute_command": self._tool_execute_command,
            "read_file": self._tool_read_file,
            "write_file": self._tool_write_file,
            "list_directory": self._tool_list_directory,
            "query_database": self._tool_query_database,
            "get_system_status": self._tool_get_system_status,
            "web_search": self._tool_web_search,
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

        # Safety confirmation for potentially destructive operations
        destructive_patterns = [
            r"\brmdir\s+/s\b", r"\bdel\s+/[sfq]\b", r"\bRemove-Item\b.*-Recurse",
            r"\bformat\b", r"\bdrop\s+table\b", r"\bdrop\s+database\b", r"\bshutdown\b"
        ]
        is_risky = any(re.search(pat, command, re.IGNORECASE) for pat in destructive_patterns)

        if is_risky and not self.auto_confirm:
            print(f"\n{YELLOW}{BOLD}[SAFETY WARNING] Command may be destructive:{RESET} {command}")
            confirm = input(f"{YELLOW}Authorize this command? (y/N): {RESET}").strip().lower()
            if confirm != "y":
                return "Execution cancelled by user."

        try:
            # Use PowerShell for rich Windows script support
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
            exit_code = process.returncode

            output_parts = []
            if stdout:
                output_parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                output_parts.append(f"STDERR:\n{stderr}")
            if not stdout and not stderr:
                output_parts.append("(Command completed with no output)")

            output_parts.append(f"EXIT CODE: {exit_code}")
            full_output = "\n".join(output_parts)

            # Limit output length to prevent context explosion
            if len(full_output) > 4000:
                full_output = full_output[:2000] + f"\n\n... [TRUNCATED {len(full_output) - 4000} CHARS] ...\n\n" + full_output[-2000:]

            return full_output

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds."
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _tool_read_file(self, args: Dict[str, Any]) -> str:
        raw_path = args.get("path", "")
        if not raw_path:
            return "Error: No file path provided."

        target_path = Path(raw_path)
        if not target_path.is_absolute():
            target_path = (self.workspace_root / target_path).resolve()

        if not target_path.exists():
            return f"Error: File '{raw_path}' does not exist."

        if target_path.is_dir():
            return f"Error: Path '{raw_path}' is a directory. Use list_directory instead."

        try:
            content = target_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total_lines = len(lines)

            start = args.get("start_line")
            end = args.get("end_line")

            if start is not None or end is not None:
                start_idx = max(0, (start - 1) if start else 0)
                end_idx = min(total_lines, end if end else total_lines)
                sliced_lines = lines[start_idx:end_idx]
                formatted = [f"{i + start_idx + 1:4d}: {line}" for i, line in enumerate(sliced_lines)]
                header = f"File: {raw_path} (Lines {start_idx + 1}-{end_idx} of {total_lines})\n"
                return header + "\n".join(formatted)

            # If small enough, return whole file with line numbers
            if total_lines <= 150:
                formatted = [f"{i + 1:4d}: {line}" for i, line in enumerate(lines)]
                return f"File: {raw_path} ({total_lines} lines)\n" + "\n".join(formatted)
            else:
                head = [f"{i + 1:4d}: {lines[i]}" for i in range(80)]
                tail = [f"{total_lines - 40 + i + 1:4d}: {lines[total_lines - 40 + i]}" for i in range(40)]
                return (
                    f"File: {raw_path} ({total_lines} lines - TRUNCATED)\n"
                    + "\n".join(head)
                    + f"\n\n... [{total_lines - 120} lines omitted] ...\n\n"
                    + "\n".join(tail)
                )

        except Exception as e:
            return f"Error reading file '{raw_path}': {str(e)}"

    def _tool_write_file(self, args: Dict[str, Any]) -> str:
        raw_path = args.get("path", "")
        content = args.get("content", "")
        mode = args.get("mode", "write").lower()

        if not raw_path:
            return "Error: No file path provided."

        target_path = Path(raw_path)
        if not target_path.is_absolute():
            target_path = (self.workspace_root / target_path).resolve()

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append" and target_path.exists():
                with open(target_path, "a", encoding="utf-8") as f:
                    f.write(content)
                return f"Success: Appended {len(content)} characters to '{raw_path}'."
            else:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"Success: Wrote {len(content)} characters to '{raw_path}'."
        except Exception as e:
            return f"Error writing to '{raw_path}': {str(e)}"

    def _tool_list_directory(self, args: Dict[str, Any]) -> str:
        raw_path = args.get("path", ".")
        recursive = bool(args.get("recursive", False))

        raw_path_str = str(raw_path).strip()
        if raw_path_str in ["/", "\\", ".", "", "./"]:
            target_path = self.workspace_root
        else:
            target_path = Path(raw_path)
            if not target_path.is_absolute():
                target_path = (self.workspace_root / target_path).resolve()

        if not target_path.exists():
            return f"Error: Directory '{raw_path}' does not exist."

        try:
            entries = []
            if recursive:
                for p in target_path.rglob("*"):
                    if ".git" in p.parts or "__pycache__" in p.parts or "node_modules" in p.parts:
                        continue
                    rel = p.relative_to(target_path)
                    if p.is_dir():
                        entries.append(f"[DIR]  {rel}")
                    else:
                        entries.append(f"[FILE] {rel} ({p.stat().st_size:,} bytes)")
            else:
                for p in sorted(target_path.iterdir()):
                    if p.name == ".git":
                        continue
                    if p.is_dir():
                        entries.append(f"[DIR]  {p.name}/")
                    else:
                        entries.append(f"[FILE] {p.name} ({p.stat().st_size:,} bytes)")

            summary = f"Directory listing of '{raw_path}' ({len(entries)} items):\n"
            return summary + "\n".join(entries[:100]) + ("\n... [truncated]" if len(entries) > 100 else "")
        except Exception as e:
            return f"Error listing directory '{raw_path}': {str(e)}"

    def _tool_query_database(self, args: Dict[str, Any]) -> str:
        sql = args.get("sql", "").strip()
        if not sql:
            return "Error: No SQL query provided."

        if not self.db_url:
            return "Error: DATABASE_URL is not configured in .env."

        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url, connect_timeout=8)
            cur = conn.cursor()
            cur.execute(sql)

            if cur.description:
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                conn.commit()
                conn.close()

                if not rows:
                    return f"Query returned 0 rows.\nColumns: {', '.join(columns)}"

                formatted_rows = [dict(zip(columns, [str(c) if c is not None else "NULL" for c in row])) for row in rows[:25]]
                return f"Query successful ({len(rows)} rows returned, showing {min(len(rows), 25)}):\n" + json.dumps(formatted_rows, indent=2, default=str)
            else:
                rowcount = cur.rowcount
                conn.commit()
                conn.close()
                return f"Query executed successfully. Rows affected: {rowcount}"

        except Exception as e:
            return f"Database error: {str(e)}"

    def _tool_get_system_status(self, args: Dict[str, Any]) -> str:
        status_info = []

        # 1. Disk Space
        try:
            total, used, free = shutil.disk_usage(str(self.workspace_root))
            status_info.append(f"Disk Free: {free // (2**30)} GB of {total // (2**30)} GB")
        except Exception:
            pass

        # 2. MetaTrader 5 status
        try:
            import MetaTrader5 as mt5
            initialized = mt5.initialize()
            if initialized:
                acc_info = mt5.account_info()
                positions = mt5.positions_total()
                if acc_info:
                    status_info.append(
                        f"MT5 Account: #{acc_info.login} ({acc_info.server}) | "
                        f"Balance: ${acc_info.balance:.2f} | Equity: ${acc_info.equity:.2f} | "
                        f"Open Positions: {positions}"
                    )
                else:
                    status_info.append("MT5 Terminal: Connected, but account details unavailable.")
            else:
                status_info.append("MT5 Terminal: Not running or initialization failed.")
        except Exception as e:
            status_info.append(f"MT5 Check Error: {e}")

        # 3. Supabase DB status
        if self.db_url:
            status_info.append("Supabase DB: Connected via PostgreSQL (.env)")
        else:
            status_info.append("Supabase DB: Missing credentials in .env")

        return "System Status:\n- " + "\n- ".join(status_info)

    def _tool_web_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "").strip()
        if not query:
            return "Error: No search query provided."

        try:
            import urllib.parse
            import xml.etree.ElementTree as ET

            encoded_query = urllib.parse.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            req = urllib.request.Request(
                rss_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            items = root.findall(".//item")
            results = []
            for item in items[:5]:
                title = item.find("title").text if item.find("title") is not None else ""
                link = item.find("link").text if item.find("link") is not None else ""
                results.append(f"- {title}\n  URL: {link}")

            if results:
                return f"Live Search Results for '{query}':\n" + "\n".join(results)
            else:
                return f"No search results found for '{query}'."

        except Exception as e:
            return f"Search error: {str(e)}"


# ==============================================================================
# AUTONOMOUS AGENT LOOP (The "Brain")
# ==============================================================================

SYSTEM_PROMPT = """You are Bubat AI, an autonomous system and coding agent running locally on the user's Windows PC.
You have direct access to tools to execute PowerShell commands, read/write files, inspect directories, query the Supabase PostgreSQL database, check MT5 status, and search the web.

AVAILABLE TOOLS:
- execute_command(command: str): Run a shell command in PowerShell. (e.g. dir, git status, python test.py)
- read_file(path: str, start_line: int, end_line: int): Read the contents of a file.
- write_file(path: str, content: str, mode: "write"|"append"): Create or edit a file.
- list_directory(path: str, recursive: bool): List files and folders.
- query_database(sql: str): Execute a SQL query on the Supabase PostgreSQL database.
- get_system_status(): Check MT5 account balance, disk space, and system telemetry.
- web_search(query: str): Search Google News for live financial/technical information.

DATABASE SCHEMA REFERENCE:
- forex_trade_decisions: (id, created_at, symbol, decision, confidence, market_sentiment, reasoning, entry_price, stop_loss, take_profit, lot_size, atr, risk_reward_ratio, approved, executed, metadata)
- forex_executed_trades: (id, created_at, symbol, ticket_id, order_type, volume, open_price, stop_loss, take_profit, close_price, profit, status, execution_result)
- ai_agent_telemetry: (id, created_at, agent_name, action_type, details, status)
Always use "created_at" (not "timestamp") when ordering tables by date/time.

INSTRUCTIONS:
1. When you need to take an action or gather information, output a JSON code block with the tool and arguments:
```json
{"tool": "tool_name", "arguments": {"param1": "value1"}}
```
2. When the tool output is provided to you:
   - If you have enough information, answer the user directly in concise, natural language.
   - If further action is needed, call the next tool.
3. DO NOT output placeholder schema definitions. Put actual string values in arguments.
"""

class BubatAutonomousAgent:
    """Agentic orchestrator that reasons, calls tools, and reports results."""

    def __init__(self, model: str = DEFAULT_MODEL, auto_confirm: bool = False):
        self.model = model
        self.executor = ToolExecutor(auto_confirm=auto_confirm)
        self.conversation_history: List[Dict[str, str]] = []
        self._init_system_prompt()

    def _init_system_prompt(self):
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def reset(self):
        """Reset conversation memory."""
        self._init_system_prompt()

    def chat_turn(self, user_prompt: str) -> str:
        """Execute one complete user turn with iterative tool-calling loop."""
        self.conversation_history.append({"role": "user", "content": user_prompt})

        called_tools_history = []
        step = 0

        while step < MAX_AGENT_STEPS:
            step += 1

            # Inference request to Ollama
            response_content = self._call_ollama()
            if not response_content:
                return f"{RED}Error: Unable to get response from Ollama at {OLLAMA_API_BASE}. Ensure server is running.{RESET}"

            # Check if model wants to call a tool
            tool_call = self._extract_tool_call(response_content)

            if not tool_call:
                # No tool called; this is the final answer
                self.conversation_history.append({"role": "assistant", "content": response_content})
                return response_content

            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            # Loop prevention: Detect identical sequential tool calls
            call_sig = (tool_name, json.dumps(tool_args, sort_keys=True))
            if call_sig in called_tools_history[-2:]:
                # Force final synthesis
                self.conversation_history.append({"role": "assistant", "content": response_content})
                self.conversation_history.append({
                    "role": "user",
                    "content": "You have already executed this action. Please synthesize your findings and provide the final answer to the user now."
                })
                final_answer = self._call_ollama()
                self.conversation_history.append({"role": "assistant", "content": final_answer})
                return final_answer or "Task completed."

            called_tools_history.append(call_sig)

            # Print tool invocation feedback in terminal
            print(f"\n{CYAN}{BOLD}▶ Calling Tool:{RESET} {GREEN}{tool_name}{RESET}")
            if tool_args:
                arg_preview = json.dumps(tool_args)
                if len(arg_preview) > 100:
                    arg_preview = arg_preview[:100] + "..."
                print(f"  {DIM}Args: {arg_preview}{RESET}")

            # Execute tool locally
            tool_result = self.executor.execute(tool_name, tool_args)

            # Print brief preview
            lines = tool_result.strip().splitlines()
            preview = lines[0] if lines else "(empty)"
            if len(preview) > 80:
                preview = preview[:80] + "..."
            print(f"  {DIM}Result: {preview} ({len(tool_result)} chars){RESET}")

            # Append to history with guidance prompt
            self.conversation_history.append({"role": "assistant", "content": response_content})
            self.conversation_history.append({
                "role": "user",
                "content": f"[TOOL RESULT for {tool_name}]:\n{tool_result}\n\nTask: Analyze this output. If complete, answer the user directly. If more action is needed, invoke the next tool."
            })

        return "Agent reached maximum step limit."

    def _call_ollama(self) -> Optional[str]:
        """Call Ollama /api/chat."""
        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 16384,
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
        except urllib.error.URLError as e:
            print(f"{RED}[Ollama Error] Connection failed: {e}{RESET}")
            return None
        except Exception as e:
            print(f"{RED}[Ollama Error] Exception: {e}{RESET}")
            return None

    def _extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract tool call JSON from text response."""
        if not text:
            return None

        # 1. Search for ```json ... ``` blocks
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

        # 2. Search for raw JSON object with "tool" or "name"
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

        # 3. Search for <tool_call> tags
        tc_matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        for m in tc_matches:
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

        return None


# ==============================================================================
# INTERACTIVE CLI SHELL
# ==============================================================================

def print_banner(agent: BubatAutonomousAgent):
    print("\n" + "=" * 70)
    print(f"{CYAN}{BOLD}  BUBAT AI - LOCAL AUTONOMOUS SYSTEM & CODING AGENT{RESET}")
    print(f"{DIM}  Model: {agent.model}  |  Host: localhost:11434  |  GPU: RTX 4060{RESET}")
    print("=" * 70)
    print("  Capabilities enabled:")
    print(f"   • {GREEN}execute_command{RESET}  -> Run PowerShell commands on Windows")
    print(f"   • {GREEN}read_file / write_file{RESET} -> Inspect, edit, and create code")
    print(f"   • {GREEN}list_directory{RESET}   -> Explore files and folder structures")
    print(f"   • {GREEN}query_database{RESET}   -> Direct Supabase PostgreSQL management")
    print(f"   • {GREEN}get_system_status{RESET}-> MT5 terminal & account balance")
    print(f"   • {GREEN}web_search{RESET}       -> Live Google News / technical search")
    print("-" * 70)
    print(f"  Commands: {YELLOW}/clear{RESET} (reset memory), {YELLOW}/status{RESET} (check system), {YELLOW}exit{RESET}")
    print("=" * 70 + "\n")


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

            if user_input.lower() == "/status":
                status = agent.executor.execute("get_system_status", {})
                print(f"\n{CYAN}{status}{RESET}")
                continue

            # Run autonomous agent turn
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
