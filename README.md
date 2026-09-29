# 🤖 Local Autonomous Forex AI Agent (Deterministic Math Engine)

> **A 100% local, self-hosted, private, and autonomous institutional-grade Forex trading system with strict separation of Qualitative LLM Sentiment and Quantitative Python Math Execution.**

---

## 📌 Executive Summary & Architecture

Modern Large Language Models (LLMs) excel at qualitative reasoning, news synthesis, and macro narrative analysis, but they **hallucinate floating-point arithmetic** and cannot be trusted to calculate live broker prices, pip distances, or dynamic position sizing.

This project solves this fundamental architectural flaw with a **strict two-tier separation of concerns**:

```
 ┌─────────────────────────────────────────────────────────────┐
 │            QUALITATIVE TIER (Ollama AI Brain)               │
 │ • Reads live news headlines & macro sentiment               │
 │ • Evaluates technical indicator structure (RSI, MACD, ATR) │
 │ • Determines MARKET DIRECTION ONLY: BUY / SELL / WAIT       │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Direction + Confidence
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │       QUANTITATIVE TIER (Deterministic Python Engine)       │
 │ • Calculates ATR volatility-based Stop Loss & Take Profit   │
 │ • Enforces broker minimum stops distance (TRADE_STOPS_LEVEL)│
 │ • Calculates dynamic lot size from %-of-balance risk formula │
 │ • Clamps lots between broker VOLUME_MIN and config MAX      │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Formatted Proposal
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │            HUMAN-IN-THE-LOOP (OpenClaw Bridge)              │
 │ • Dispatches exact computed trade parameters to WhatsApp    │
 │ • Waits for trader approval ("YES" / "NO")                  │
 └──────────────────────────────┬──────────────────────────────┘
                                │ If "YES"
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │         DETERMINISTIC RISK WALL (MetaTrader 5 Engine)       │
 │ • Maximum open trades gatekeeper (rejects if exceeded)      │
 │ • Maximum drawdown safety barrier                           │
 │ • Dispatches atomic order to MT5 via order_send()           │
 └─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Directory Map

```
bubat AI/
├── .env                                  # Workspace environment variables (Supabase, API keys)
├── .gitignore                            # Protection against committing secrets, logs, cache
├── README.md                             # Complete documentation and user guide
├── assistant.bat                         # Desktop 1-click launcher for Local Autonomous Agent
├── run_agent.bat                         # Desktop 1-click launcher for Forex Trading Loop
├── chat.bat                              # Desktop 1-click launcher for Interactive Forex Chat
├── stop_all.bat                          # Desktop 1-click launcher to kill all agents & Ollama
├── local_assistant.py                    # Local Autonomous System & Coding Agent (Tool Engine)
├── Local_Autonomous_Forex_AI_Master_Blueprint.txt # Original system specification
└── forex_local_agent/                    # Core agent codebase
    ├── config.json                       # Central system configuration
    ├── Modelfile                         # Custom Ollama model specification (32k context)
    ├── requirements.txt                  # Python dependencies
    ├── main.py                           # Master orchestration loop & candle monitor
    ├── chat.py                           # Forex market analysis CLI chat
    ├── local_assistant.py                # Autonomous agent copy
    │
    ├── core/
    │   ├── agent_logic.py                # Qualitative LLM reasoning (Ollama + Pydantic)
    │   ├── mt5_engine.py                 # MT5 driver + Deterministic ATR Math Calculator
    │   ├── openclaw_bridge.py            # WhatsApp approval gateway & email alerts
    │   ├── sentiment_engine.py           # SearXNG news engine + live RSS fallback
    │   └── supabase_manager.py           # Cloud Supabase & PostgreSQL telemetry sync
    │
    ├── learning/
    │   ├── memory_manager.py             # ChromaDB / Mem0 episodic memory & recall
    │   ├── skill_factory.py              # Dynamic self-coding tool generator
    │   ├── learned_rules.md              # Auto-appended trading rules from loss reflexion
    │   └── skills/
    │       ├── template_skill.py         # Base template for dynamic skills
    │       └── skills_index.json         # Registry of active skills
    │
    ├── maintenance/
    │   └── model_updater.py              # Automated weekly model discovery & hot-swapper
    │
    └── logs/
        ├── trades.log                    # Historical log of all proposed/executed trades
        ├── system_errors.log             # Exception traces and critical diagnostics
        └── model_auditions.log           # Benchmark results from model upgrade tests
```

---

## ⚙️ Core Subsystems

### 1. Deterministic Math Engine (`core/mt5_engine.py`)
Computes trade geometry mathematically without LLM hallucination:
- **BUY Orders**:
  $$\text{Entry Price} = \text{Ask}$$
  $$\text{Stop Loss} = \text{Entry} - (\text{ATR}_{14} \times \text{Multiplier}_{\text{SL}})$$
  $$\text{Take Profit} = \text{Entry} + (\text{ATR}_{14} \times \text{Multiplier}_{\text{TP}})$$
- **SELL Orders**:
  $$\text{Entry Price} = \text{Bid}$$
  $$\text{Stop Loss} = \text{Entry} + (\text{ATR}_{14} \times \text{Multiplier}_{\text{SL}})$$
  $$\text{Take Profit} = \text{Entry} - (\text{ATR}_{14} \times \text{Multiplier}_{\text{TP}})$$
- **Dynamic Lot Sizing**:
  $$\text{Monetary Risk} = \text{Account Balance} \times \left(\frac{\text{risk\_per\_trade\_pct}}{100}\right)$$
  $$\text{Loss Per Lot} = \left(\frac{|\text{Entry} - \text{SL}|}{\text{Tick Size}}\right) \times \text{Tick Value}$$
  $$\text{Calculated Lot} = \frac{\text{Monetary Risk}}{\text{Loss Per Lot}}$$
- **Broker Stops Guard**:
  Inspects the broker's `SYMBOL_TRADE_STOPS_LEVEL`. If the ATR-calculated stop loss sits closer than the broker's minimum allowable distance, the stop is widened automatically to satisfy broker constraints.
- **Volume Step Clamping**:
  Floors volume to the broker's `volume_step`, bounded by `volume_min`, `volume_max`, and `max_lot_size`.

### 2. Qualitative Sentiment Engine (`core/agent_logic.py`)
- Interfaces with local **Ollama** (`agent-brain:32k`).
- Receives sanitized technical summaries (RSI, MACD, ATR values, recent candle price action) and fundamental news headlines.
- Returns strictly structured JSON validated by Pydantic:
  ```json
  {
    "market_sentiment": "BULLISH",
    "decision": "BUY",
    "confidence_score": 0.85,
    "reasoning": "Strong bullish divergence on RSI accompanied by positive macro data."
  }
  ```
- Any parse failures trigger an automatic 3-retry repair loop.

### 3. Resilient News Pipeline (`core/sentiment_engine.py`)
- **Primary Search**: Queries a local SearXNG instance on `http://localhost:8080/search`.
- **Automatic Fallback**: If SearXNG or Docker is offline, it automatically switches to live Google Financial RSS feeds and ForexFactory weekly calendar events, ensuring the analysis loop never stalls.
- **Article Scraping**: Uses `crawl4ai` (or an optimized `httpx` + `BeautifulSoup` + `html2text` fallback) to parse clean markdown for news context.

### 4. Human-In-The-Loop Approval (`core/openclaw_bridge.py`)
- High-confidence signals ($\ge 80\%$) trigger a WhatsApp proposal via the OpenClaw gateway.
- Example WhatsApp proposal:
  ```
  PROPOSAL: BUY EURUSD
  Lot: 0.10 | Entry: 1.08720
  SL: 1.08450 | TP: 1.09260
  ATR: 0.00180 | R:R 1:2.0
  Risk: $22.18 | Confidence: 85%
  Reason: Bullish momentum with positive European macroeconomic data.
  Reply YES to execute or NO to abort.
  ```
- Listens on webhook `http://localhost:5055/webhook/approval`. If `YES` is received within 300 seconds, the trade executes; otherwise, it is cancelled.

### 5. Reflexion Self-Learning Engine (`learning/memory_manager.py`)
- At the end of every candle cycle, historical deal history is checked.
- If a closed trade resulted in a loss, a post-mortem is dispatched to Ollama to isolate the root cause.
- A concise rule is extracted and automatically appended to `learning/learned_rules.md`.
- These rules are injected into every future trading prompt so the AI never repeats the same mistake twice.

---

## 🛠️ System Requirements

| Component | Minimum | Recommended (Tested Configuration) |
|---|---|---|
| **OS** | Windows 10/11 64-bit | Windows 11 64-bit |
| **CPU** | 6-core Intel / AMD | Intel Core i5 / i7 / Ryzen 5 / 7 |
| **GPU** | NVIDIA GTX 1660 (6GB) | NVIDIA RTX 4060 (8GB VRAM) or higher |
| **RAM** | 16 GB | 32 GB DDR4 / DDR5 |
| **Python** | 3.10+ | Python 3.13.2 |
| **Broker Terminal** | MetaTrader 5 Build 4000+ | MetaTrader 5 Build 5.0.6231 |

---

## 🚀 Complete Setup & Installation Guide

### Step 1: Ensure Python is on PATH
Verify Python is available:
```powershell
python --version
```

### Step 2: Install Python Dependencies
Open PowerShell in the project directory and install the packages:
```powershell
cd "c:\Users\User\OneDrive\Desktop\bubat AI\forex_local_agent"
python -m pip install -r requirements.txt
```

### Step 3: Configure Ollama & Build the Model
1. Ensure Ollama is running in the background:
   ```powershell
   ollama serve
   ```
   *(If not running, you can launch it in a separate terminal or let the system daemon run it).*
2. Verify Ollama is listening:
   ```powershell
   curl http://localhost:11434/
   # Should return: "Ollama is running"
   ```
3. Pull the base model and create the 32K context model:
   ```powershell
   ollama pull qwen2.5-coder:1.5b
   # Or for 8GB VRAM: ollama pull qwen2.5-coder:7b
   ollama create agent-brain:32k -f Modelfile
   ```
4. Verify the model exists:
   ```powershell
   ollama list
   ```

### Step 4: Configure MetaTrader 5
1. Launch your MetaTrader 5 terminal: `C:\Program Files\MetaTrader 5\terminal64.exe`.
2. Log into your broker account (e.g. **Tickmill-Demo**).
3. **CRITICAL STEP**: Enable Automated Trading in MT5:
   - In MT5 top menu: **Tools** $\rightarrow$ **Options** $\rightarrow$ **Expert Advisors**.
   - Check **"Allow Algo Trading"**.
   - Check **"Allow DLL imports"**.
   - Click **OK**.
   - Make sure the green **"Algo Trading"** button on the toolbar is enabled.

### Step 5: Verify Central Configuration (`config.json`)
Open `forex_local_agent/config.json` and review the settings:
```json
{
  "active_model": "agent-brain:32k",
  "ollama_base_url": "http://localhost:11434",
  "searxng_url": "http://localhost:8080",
  "open_webui_url": "http://localhost:3000",
  "openclaw_webhook_port": 5055,
  "mt5_credentials": {
    "login": 25360772,
    "password": "",
    "server": "Tickmill-Demo",
    "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
  },
  "risk_parameters": {
    "max_lot_size": 0.1,
    "max_drawdown_pct": 2.0,
    "max_open_trades": 3,
    "confidence_threshold": 0.80,
    "approval_timeout_seconds": 300,
    "risk_per_trade_pct": 1.5,
    "atr_period": 14,
    "atr_multiplier_sl": 1.5,
    "atr_multiplier_tp": 3.0
  },
  "trading": {
    "symbols": ["EURUSD"],
    "timeframe": "H1",
    "analysis_bars": 100,
    "max_news_tokens": 4000
  },
  "alerts": {
    "admin_email": "hello@syariefazman.com",
    "whatsapp_enabled": true
  }
}
```

---

## 🏃 Running the Agent

### Start the Live Agent Loop
Run the orchestrator from PowerShell:
```powershell
cd "c:\Users\User\OneDrive\Desktop\bubat AI\forex_local_agent"
python main.py
```

### What Happens on Startup:
1. **Background Webhook**: Starts a FastAPI server on port `5055` to listen for WhatsApp approval responses.
2. **MT5 Handshake**: Connects to the local MT5 terminal and verifies account balance and equity.
3. **Immediate Analysis**: Executes an initial full market cycle on launch so you don't have to wait for the next candle close.
4. **Candle Synchronizer**: Calculates the remaining seconds until the next H1 candle close (top of the hour) and sleeps in resilient non-blocking intervals.

---

## 🧪 Testing Subsystems Independently

You can verify any component in isolation using single-line Python commands:

### 1. Test MT5 Connection & Indicator Calculation
```powershell
python -c "from core.mt5_engine import MT5Engine; e = MT5Engine('config.json'); e.initialize(); print(e.get_technical_data('EURUSD')); e.shutdown()"
```

### 2. Test Live News Scraper & Fallback
```powershell
python -c "import asyncio; from core.sentiment_engine import SentimentEngine; asyncio.run(SentimentEngine('config.json').get_live_news('EURUSD'))"
```

### 3. Test Ollama AI Reasoning
```powershell
python -c "import asyncio; from core.agent_logic import AgentLogic; al = AgentLogic('config.json'); print(asyncio.run(al.query_ollama('Ping test', 'You are an AI assistant.')))"
```

### 4. Test Single Trading Cycle Dry Run
```powershell
python -c "import asyncio; from main import ForexAgent; a = ForexAgent(); a.mt5_engine.initialize(); asyncio.run(a.run_analysis_cycle('EURUSD')); a.mt5_engine.shutdown()"
```

---

## 🛡️ Risk Management & Safety Limits

| Parameter | Default | Purpose |
|---|---|---|
| `risk_per_trade_pct` | `1.5%` | Account equity percentage risked per trade |
| `max_drawdown_pct` | `2.0%` | Hard stop threshold; orders exceeding this are aborted |
| `max_open_trades` | `3` | Maximum concurrent positions allowed |
| `max_lot_size` | `0.10` | Hard cap on lot size regardless of calculation |
| `confidence_threshold`| `0.80` | Minimum confidence required to propose a trade |
| `approval_timeout` | `300s` | Auto-aborts proposals if WhatsApp approval is not received within 5 minutes |

---

## ❓ Frequently Asked Questions (FAQ)

#### Q: Does this require an active Internet connection?
**A:** MT5 needs an internet connection to reach your broker server, and the news scraper fetches live macro headlines. However, **all AI model inference is 100% local** via Ollama on your RTX 4060 GPU with zero external LLM API costs or rate limits.

#### Q: How does the agent handle Docker if I don't have it?
**A:** Docker is optional. If local SearXNG is not found, the agent automatically falls back to live financial RSS news streams and the ForexFactory economic calendar.

#### Q: Where are the logs stored?
**A:**
- Trade decisions & execution results: `forex_local_agent/logs/trades.log`
- System errors & exceptions: `forex_local_agent/logs/system_errors.log`
- Reflexion rules: `forex_local_agent/learning/learned_rules.md`

#### Q: How do I stop the agent gracefully?
**A:** Press `Ctrl + C` in the PowerShell terminal, or double-click `stop_all.bat`.

---

## ⚡ 1-Click Desktop Batch Runners

For zero-friction operation, double-clickable `.bat` scripts are provided in the root directory:

| Script | Purpose | Description |
|---|---|---|
| `assistant.bat` | **Autonomous Agent** | Launches the tool-enabled autonomous system assistant with PowerShell, File, DB, and MT5 tools. |
| `run_agent.bat` | **Forex Trading Loop** | Starts Ollama server and launches the live autonomous trading loop (`main.py`). |
| `chat.bat` | **Forex Interactive Chat** | Opens interactive CLI chat with `agent-brain:32k` to ask questions about markets and live MT5 technicals. |
| `stop_all.bat` | **Bulletproof Terminator** | Kills all background Ollama servers, Python trading agents, and assistant processes. |

---

## 🦾 Bubat AI Local Autonomous System & Coding Agent (`local_assistant.py`)

The workspace includes a **fully autonomous agentic loop** powered by your local Ollama model (`agent-brain:32k`) running on your NVIDIA RTX 4060 GPU.

Unlike standard chatbots that can only output text, this assistant possesses **"hands" (tools)** to directly inspect and manipulate your machine:

### Available Tools:
1. `execute_command(command)`: Runs shell commands via PowerShell on Windows (e.g. `dir`, `git status`, `python script.py`).
2. `read_file(path, start_line, end_line)`: Inspects code and configuration files with line slicing.
3. `write_file(path, content, mode)`: Autonomously creates or edits files.
4. `list_directory(path, recursive)`: Explores file and directory trees.
5. `query_database(sql)`: Executes SQL statements directly on your Supabase PostgreSQL database.
6. `get_system_status()`: Checks MT5 terminal connection, account equity, balance, open positions, and disk space.
7. `web_search(query)`: Performs live web search via Google News RSS for live documentation or macro news.

### Example Prompts you can ask in `assistant.bat`:
- *"Check if MetaTrader 5 is connected and show my account balance and equity."*
- *"Query the database to see the recent trade decisions from the forex_trade_decisions table."*
- *"Check the git status of this repository and list any uncommitted files."*
- *"Inspect `forex_local_agent/config.json` and tell me what symbols are active."*
- *"Create a test python script that checks my disk space and run it."*

---

## 🗄️ Supabase Cloud & PostgreSQL Database Integration

All trade signals, executions, and agent telemetry are synchronized directly to your cloud Supabase database:

### Tables Managed:
- **`forex_trade_decisions`**: Stores qualitative LLM decisions, ATR math calculations (SL, TP, lot size, confidence, sentiment), and approval states.
- **`forex_executed_trades`**: Stores MT5 execution records with broker ticket IDs, open/close prices, lot volumes, and execution results.
- **`ai_agent_telemetry`**: Stores system health audits, error logs, and diagnostic telemetry.

Both the PostgREST REST client (`supabase-py`) and direct PostgreSQL connection (`psycopg2`) are integrated with automatic failover in `forex_local_agent/core/supabase_manager.py`.

