"""
Interactive Chat Interface with Agent-Brain
===========================================
Chat directly with the local Forex AI agent from any terminal window.
Can query live MT5 technicals, inspect learned rules, check trades log, or discuss market strategy.
"""

import sys
import json
import asyncio
import httpx
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

try:
    from core.mt5_engine import MT5Engine
except ImportError:
    MT5Engine = None

CONFIG_FILE = ROOT_DIR / "config.json"
RULES_FILE = ROOT_DIR / "learning" / "learned_rules.md"
TRADES_LOG = ROOT_DIR / "logs" / "trades.log"

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

async def query_model(messages: list, ollama_url: str, model: str) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(f"{ollama_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

def get_context_snapshot():
    context_parts = []
    
    # Check learned rules
    if RULES_FILE.exists():
        rules = RULES_FILE.read_text(encoding="utf-8").strip()
        if rules:
            context_parts.append(f"--- CURRENT LEARNED TRADING RULES ---\n{rules}")
            
    # Check recent trades
    if TRADES_LOG.exists():
        lines = TRADES_LOG.read_text(encoding="utf-8").strip().splitlines()
        recent = "\n".join(lines[-10:])
        if recent:
            context_parts.append(f"--- RECENT LOGGED ACTIVITY ---\n{recent}")
            
    return "\n\n".join(context_parts)

async def main():
    config = load_config()
    ollama_url = config.get("ollama_base_url", "http://localhost:11434")
    model = config.get("active_model", "agent-brain:32k")

    print("=" * 64)
    print("      BUBAT AI — LOCAL FOREX AGENT INTERACTIVE CHAT")
    print(f"      Model: {model} | Server: {ollama_url}")
    print("=" * 64)
    print("Commands:")
    print("  'live'    - Fetch live MT5 indicators for EURUSD")
    print("  'rules'   - Show learned trading rules from memory")
    print("  'clear'   - Reset conversation history")
    print("  'exit'    - Quit chat")
    print("-" * 64)

    # Verify Ollama is reachable
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{ollama_url}/")
            if r.status_code != 200:
                print(f"[!] Warning: Ollama returned status {r.status_code}")
    except Exception:
        print("[!] Ollama server is not running on localhost:11434.")
        print("    Please start it first by running: ollama serve")
        print("-" * 64)

    system_prompt = (
        "You are the resident AI Forex Intelligence Agent for Bubat AI. "
        "You analyze market structure, news sentiment, and risk parameters. "
        "You explain trading concepts, review recent trade episodes, and follow strict risk guidelines. "
        "Remember: All quantitative math is calculated deterministically by Python using ATR; "
        "your strength is macroeconomic reasoning, technical pattern recognition, and self-learning reflexion."
    )

    context = get_context_snapshot()
    if context:
        system_prompt += f"\n\nActive Context from disk:\n{context}"

    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if cmd == "clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("[+] Conversation memory cleared.")
            continue

        if cmd == "rules":
            if RULES_FILE.exists():
                print("\n--- LEARNED RULES ---")
                print(RULES_FILE.read_text(encoding="utf-8"))
            else:
                print("[!] No learned rules recorded yet.")
            continue

        if cmd == "live":
            print("[*] Connecting to MT5 and fetching live market indicators...")
            try:
                engine = MT5Engine(str(CONFIG_FILE))
                if engine.initialize():
                    data = engine.get_technical_data("EURUSD")
                    acct = engine.get_account_info()
                    engine.shutdown()
                    print(f"  Account Balance: ${acct.get('balance')} | Free Margin: ${acct.get('free_margin')}")
                    print(f"  EURUSD Price: {data.get('current_price')} | RSI(14): {data.get('rsi')} | ATR(14): {data.get('atr')}")
                    print(f"  MACD: {data.get('macd')} | Signal: {data.get('macd_signal')} | Hist: {data.get('macd_hist')}")
                    user_input = (
                        f"Here is the live market data for EURUSD right now:\n"
                        f"Price: {data.get('current_price')}, RSI: {data.get('rsi')}, MACD: {data.get('macd')}, "
                        f"ATR: {data.get('atr')}, High Range: {data.get('high_range')}, Low Range: {data.get('low_range')}.\n"
                        f"Give me your quick qualitative assessment of this setup."
                    )
                else:
                    print("[!] Could not initialize MT5. Ensure MT5 is running.")
                    continue
            except Exception as e:
                print(f"[!] MT5 query failed: {e}")
                continue

        messages.append({"role": "user", "content": user_input})
        print("\nAgent > Thinking...", end="\r", flush=True)

        try:
            reply = await query_model(messages, ollama_url, model)
            print("Agent > " + " " * 15)  # Clear the 'Thinking...' line
            print(reply)
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"\n[!] Error querying Ollama: {e}")

if __name__ == "__main__":
    asyncio.run(main())
