# 🤖 Bubat AI — Local Autonomous Institutional Forex AI Agent

> **A 100% local, self-hosted, private, and autonomous institutional-grade Forex trading system with strict separation of Qualitative LLM Sentiment and Quantitative Deterministic Python Math Execution.**

---

## 📌 Table of Contents
1. [Executive Architecture & Philosophy](#-executive-architecture--philosophy)
2. [Supported Instruments (All 28 Pairs + Gold)](#-supported-instruments-all-28-pairs--gold)
3. [Project Directory & File Structure](#-project-directory--file-structure)
4. [Subsystem Breakdown](#-subsystem-breakdown)
   - [1. Deterministic Math Engine (`core/mt5_engine.py`)](#1-deterministic-math-engine-coremt5_enginepy)
   - [2. Qualitative LLM Reasoner (`core/agent_logic.py`)](#2-qualitative-llm-reasoner-coreagent_logicpy)
   - [3. Live WebSurfer Financial Pipeline (`core/web_surfer.py` & `sentiment_engine.py`)](#3-live-websurfer-financial-pipeline)
   - [4. Multi-Pair Market Scanner (`core/market_scanner.py`)](#4-multi-pair-market-scanner-coremarket_scannerpy)
   - [5. Continuous Learning & Reflexion Engine (`learning/`)](#5-continuous-learning--reflexion-engine-learning)
   - [6. Human-In-The-Loop WhatsApp Gateway (`core/openclaw_bridge.py`)](#6-human-in-the-loop-whatsapp-gateway)
   - [7. Supabase Cloud Database Telemetry (`core/supabase_manager.py`)](#7-supabase-cloud-database-telemetry)
5. [Autonomous System & Coding Agent (`local_assistant.py`)](#-autonomous-system--coding-agent-local_assistantpy)
6. [System Requirements & Hardware Setup](#-system-requirements--hardware-setup)
7. [Installation & Quick Start Guide](#-installation--quick-start-guide)
8. [Configuration Guide (`config.json`)](#-configuration-guide-configjson)
9. [1-Click Desktop Batch Runners](#-1-click-desktop-batch-runners)
10. [Troubleshooting & Common Scenarios](#-troubleshooting--common-scenarios)

---

## 🏛️ Executive Architecture & Philosophy

Modern Large Language Models (LLMs) excel at qualitative reasoning, news synthesis, and macroeconomic narrative analysis. However, **LLMs hallucinate floating-point arithmetic** and cannot be trusted to calculate live broker prices, pip distances, or dynamic position sizing.

Bubat AI solves this fundamental flaw with a **strict two-tier separation of concerns**:

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                   QUALITATIVE TIER (Ollama AI Brain)                   │
 │ • Reads real-time financial news headlines & macro sentiment           │
 │ • Evaluates market regime & technical structure (RSI, ATR, EMAs)       │
 │ • Generates qualitative direction ONLY: BUY / SELL / WAIT + Confidence │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Direction + Confidence
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │            QUANTITATIVE TIER (Deterministic Python Math Engine)        │
 │ • ATR-based volatility Stop Loss & Take Profit calculation             │
 │ • Broker minimum stops enforcement (TRADE_STOPS_LEVEL guard)           │
 │ • Dynamic lot sizing based on account equity risk percentage           │
 │ • Volume clamping bounded by broker VOLUME_MIN, VOLUME_STEP, & MAX     │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Formatted Proposal
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                 HUMAN-IN-THE-LOOP (OpenClaw Bridge)                    │
 │ • Dispatches exact trade geometry to WhatsApp via OpenClaw             │
 │ • Waits for trader approval ("YES" / "NO") within timeout window       │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ If "YES"
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │              DETERMINISTIC RISK WALL (MetaTrader 5 Engine)             │
 │ • Maximum open trades barrier (strict limit, default: 3)               │
 │ • Maximum account drawdown gatekeeper (hard emergency halt)            │
 │ • Atomic order execution via MT5 API order_send()                      │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Record Decisions & Executions
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │             CLOUD SUPABASE & POSTGRESQL SYNCHRONIZATION                │
 │ • Real-time logging of decisions, rationale, trade outcomes, telemetry │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 🌐 Supported Instruments (All 28 Pairs + Gold)

The bot natively supports, scans, and trades **all 28 major and cross currency pairs plus Gold (`XAUUSD`)** (total 29 instruments) with automatic tick-value conversion to the account's base currency (USD):

| Category | Pairs | Description |
|---|---|---|
| **7 Major Pairs** | `EURUSD`, `GBPUSD`, `USDJPY`, `USDCHF`, `AUDUSD`, `NZDUSD`, `USDCAD` | Highest global liquidity, tightest broker spreads |
| **6 Euro Crosses** | `EURGBP`, `EURJPY`, `EURCHF`, `EURAUD`, `EURNZD`, `EURCAD` | European macro drivers & liquidity flows |
| **5 Pound Crosses** | `GBPJPY`, `GBPCHF`, `GBPAUD`, `GBPNZD`, `GBPCAD` | High-volatility session runners |
| **4 Aussie Crosses** | `AUDJPY`, `AUDCHF`, `AUDNZD`, `AUDCAD` | Asian / Pacific commodity proxies |
| **3 Kiwi Crosses** | `NZDJPY`, `NZDCHF`, `NZDCAD` | High-beta Pacific currency crosses |
| **3 Yen / Swiss Crosses**| `CADJPY`, `CADCHF`, `CHFJPY` | Carry trade & European safe-haven plays |
| **Commodity / Metal** | `XAUUSD` | Spot Gold vs US Dollar |

---

## 📁 Project Directory & File Structure

```
bubat AI/
├── .env                                  # Workspace environment variables (Supabase URL & Keys)
├── .gitignore                            # Protection against committing logs, cache, or credentials
├── README.md                             # Comprehensive technical documentation & user guide
├── assistant.bat                         # 1-click launcher for Local Autonomous Assistant
├── run_agent.bat                         # 1-click launcher for 24/7 Forex Trading Loop
├── chat.bat                              # 1-click launcher for Interactive Market Intelligence Chat
├── stop_all.bat                          # 1-click bulletproof process terminator
├── local_assistant.py                    # Local Autonomous System & Coding Agent (Tool Calling)
└── forex_local_agent/                    # Core trading agent package
    ├── config.json                       # Central system configuration (timeframe, pairs, risk)
    ├── Modelfile                         # Custom Ollama model definition (32K context)
    ├── requirements.txt                  # Python dependencies
    ├── main.py                           # Master orchestration loop & candle synchronizer
    ├── chat.py                           # Interactive market analysis chat with live MT5 scan
    │
    ├── core/
    │   ├── agent_logic.py                # Qualitative LLM reasoning (Ollama + Pydantic validation)
    │   ├── mt5_engine.py                 # MT5 API connection + Deterministic ATR Math Calculator
    │   ├── market_scanner.py             # 29-pair live technical scanner & session opportunity ranker
    │   ├── web_surfer.py                 # High-speed live financial news & web intelligence engine
    │   ├── sentiment_engine.py           # News aggregation pipeline with fast-routing & RSS fallback
    │   ├── openclaw_bridge.py            # WhatsApp approval gateway & email alerts
    │   └── supabase_manager.py           # Supabase REST & PostgreSQL telemetry sync
    │
    ├── learning/
    │   ├── continuous_learner.py         # Autonomous rule capture engine for preferences & directives
    │   ├── memory_manager.py             # ChromaDB / JSON episodic memory & historical recall
    │   ├── skill_factory.py              # Dynamic trading skill code generator
    │   ├── learned_rules.md              # Auto-appended trading rules from reflexion & user instructions
    │   └── skills/
    │       ├── template_skill.py         # Base template for dynamic self-contained skills
    │       └── skills_index.json         # Registry of active skills
    │
    ├── maintenance/
    │   └── model_updater.py              # Automated weekly model discovery, benchmark & hot-swap
    │
    └── logs/
        ├── trades.log                    # Historical log of all proposed and executed trades
        ├── system_errors.log             # Exception traces and critical diagnostics
        └── model_auditions.log           # Audition logs from model upgrade benchmarks
```

---

## ⚙️ Subsystem Breakdown

### 1. Deterministic Math Engine (`core/mt5_engine.py`)
Computes trade geometry mathematically without LLM hallucination:
* **BUY Orders**:
  $$\text{Entry Price} = \text{Ask}$$
  $$\text{Stop Loss} = \text{Entry} - (\text{ATR}_{14} \times \text{Multiplier}_{\text{SL}})$$
  $$\text{Take Profit} = \text{Entry} + (\text{ATR}_{14} \times \text{Multiplier}_{\text{TP}})$$
* **SELL Orders**:
  $$\text{Entry Price} = \text{Bid}$$
  $$\text{Stop Loss} = \text{Entry} + (\text{ATR}_{14} \times \text{Multiplier}_{\text{SL}})$$
  $$\text{Take Profit} = \text{Entry} - (\text{ATR}_{14} \times \text{Multiplier}_{\text{TP}})$$
* **Dynamic Lot Sizing (Risk Percentage Formula)**:
  $$\text{Monetary Risk} = \text{Account Balance} \times \left(\frac{\text{risk\_per\_trade\_pct}}{100}\right)$$
  $$\text{Loss Per Lot} = \left(\frac{|\text{Entry} - \text{SL}|}{\text{Tick Size}}\right) \times \text{Tick Value}$$
  $$\text{Calculated Lot} = \frac{\text{Monetary Risk}}{\text{Loss Per Lot}}$$
* **Broker Stops Guard**:
  Inspects the broker's `SYMBOL_TRADE_STOPS_LEVEL`. If the ATR-calculated stop loss sits closer than the broker's minimum allowable distance, the stop is widened automatically to satisfy broker constraints.
* **Volume Step Clamping**:
  Floors volume to the broker's `volume_step`, bounded by `volume_min`, `volume_max`, and `max_lot_size`.

---

### 2. Qualitative LLM Reasoner (`core/agent_logic.py`)
* Interfaces with local **Ollama** (`agent-brain:32k`) running locally on your GPU.
* Receives technical indicators (RSI, ATR, MACD, recent 5 candles) and real-time news headlines.
* Produces strictly structured JSON validated by Pydantic:
  ```json
  {
    "market_sentiment": "BULLISH",
    "decision": "BUY",
    "confidence_score": 0.85,
    "reasoning": "Strong bullish EMA breakout with RSI momentum and positive macro data."
  }
  ```
* Automatic 3-retry repair loop in case of malformed LLM responses.

---

### 3. Live WebSurfer Financial Pipeline
* Built into [`core/web_surfer.py`](file:///c:/Users/User/OneDrive/Desktop/bubat%20AI/forex_local_agent/core/web_surfer.py) and [`core/sentiment_engine.py`](file:///c:/Users/User/OneDrive/Desktop/bubat%20AI/forex_local_agent/core/sentiment_engine.py).
* **Zero SearXNG Delay**: Automatically probes SearXNG on port 8080 once (0.6s). If Docker is offline, it immediately routes directly to live WebSurfer (Google News RSS & Investing.com RSS) with **zero retry spam and zero delay**.
* Fetches live news per pair in **~0.3 to 0.7 seconds**.
* Scrapes article context in parallel using `asyncio.gather` with strict timeouts.

---

### 4. Multi-Pair Market Scanner (`core/market_scanner.py`)
* Scans all 29 instruments simultaneously via MetaTrader 5 in **under 2 seconds**.
* Calculates real-time RSI(14), ATR(14), 20/50 EMAs, 24-hour price change %, and broker spread points.
* Evaluates active trading sessions (London, New York, Tokyo, Sydney) and calculates an **Opportunity Score (0 to 100+)** to rank the best trading opportunities.

---

### 5. Continuous Learning & Reflexion Engine (`learning/`)
* **Directive Learning (`continuous_learner.py`)**: Automatically captures user instructions, communication preferences (e.g. Malaysian Malay format), and trading rules into `learning/learned_rules.md` and Supabase.
* **Loss Reflexion**: At the end of every candle cycle, closed trades are inspected. Any loss triggers an LLM post-mortem to isolate the root cause, create a new rule, and prevent the bot from repeating the same mistake.

---

### 6. Human-In-The-Loop WhatsApp Gateway
* High-confidence signals ($\ge 80\%$) trigger a WhatsApp proposal via OpenClaw:
  ```text
  PROPOSAL: BUY EURUSD
  Lot: 0.05 | Entry: 1.13150
  SL: 1.12980 | TP: 1.13490
  ATR: 0.00110 | R:R 1:2.0
  Risk: $2.21 | Confidence: 85%
  Reason: Strong downside momentum exhausted, bullish rejection candle on M5.
  Reply YES to execute or NO to abort.
  ```
* Waits on webhook `http://localhost:5055/webhook/approval`. If `YES` is received within 300s, the order is sent to MT5; otherwise it expires safely.

---

### 7. Supabase Cloud Database Telemetry
* All trade decisions, executions, and telemetry are synchronized to Supabase PostgreSQL:
  * **`forex_trade_decisions`**: Logs decision, confidence, sentiment, reasoning, and ATR trade geometry.
  * **`forex_executed_trades`**: Logs ticket ID, lots, open price, execution time, and PnL.
  * **`ai_agent_telemetry`**: Logs system health, 29-pair scans, and error audits.

---

## 🦾 Autonomous System & Coding Agent (`local_assistant.py`)

The project includes an **autonomous agentic loop** powered by your local Ollama model (`agent-brain:32k`). 

Unlike basic chatbots that only output text, this assistant has direct tools ("hands") to inspect and operate your system:
* `execute_command(command)`: Runs shell commands via PowerShell.
* `read_file(path, start_line, end_line)`: Inspects code and configs with line slicing.
* `write_file(path, content, mode)`: Autonomously creates or edits files.
* `list_directory(path, recursive)`: Explores folders and projects.
* `query_database(sql)`: Directly queries Supabase PostgreSQL.
* `get_system_status()`: Checks MT5 terminal connection, account equity, and disk space.
* `web_search(query)`: Performs real-time web search.

Launch it anytime with:
```powershell
.\assistant.bat
```

---

## 🛠️ System Requirements & Hardware Setup

| Component | Minimum | Recommended (Tested Configuration) |
|---|---|---|
| **Operating System** | Windows 10/11 64-bit | Windows 11 64-bit |
| **GPU** | NVIDIA GTX 1660 (6 GB) | NVIDIA RTX 4060 (8 GB VRAM) or higher |
| **RAM** | 16 GB | 32 GB DDR4 / DDR5 |
| **Python** | Python 3.10+ | Python 3.13 64-bit |
| **Broker Terminal** | MetaTrader 5 Build 4000+ | MetaTrader 5 Build 5.0.6231 (Tickmill-Demo) |

---

## 🚀 Installation & Quick Start Guide

### Step 1: Clone Repository
```powershell
cd "c:\Users\User\OneDrive\Desktop"
git clone https://github.com/syarief02/bubat-ai.git "bubat AI"
cd "bubat AI"
```

### Step 2: Install Python Dependencies
```powershell
cd forex_local_agent
python -m pip install -r requirements.txt
```

### Step 3: Set Up Ollama Local Brain
1. Download and install [Ollama for Windows](https://ollama.com).
2. Pull the coder base model and build the 32K context model:
   ```powershell
   ollama pull qwen2.5-coder:1.5b
   ollama create agent-brain:32k -f Modelfile
   ```
3. Test that the model is ready:
   ```powershell
   ollama list
   ```

### Step 4: Configure MetaTrader 5
1. Launch MetaTrader 5 and log into your broker account.
2. In MT5 top menu: **Tools** $\rightarrow$ **Options** $\rightarrow$ **Expert Advisors**:
   - Check **"Allow Algo Trading"**
   - Check **"Allow DLL imports"**
   - Click **OK**
3. Ensure the green **"Algo Trading"** button in the top toolbar is active.

### Step 5: Launch the Agent
Double-click **`run_agent.bat`** or run from PowerShell:
```powershell
.\run_agent.bat
```

---

## ⚙️ Configuration Guide (`config.json`)

The central configuration file is located at [`forex_local_agent/config.json`](file:///c:/Users/User/OneDrive/Desktop/bubat%20AI/forex_local_agent/config.json):

```json
{
  "active_model": "agent-brain:32k",
  "ollama_base_url": "http://localhost:11434",
  "trading": {
    "symbols": [
      "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
      "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURNZD", "EURCAD",
      "GBPJPY", "GBPCHF", "GBPAUD", "GBPNZD", "GBPCAD",
      "AUDJPY", "AUDCHF", "AUDNZD", "AUDCAD",
      "NZDJPY", "NZDCHF", "NZDCAD",
      "CADJPY", "CADCHF", "CHFJPY",
      "XAUUSD"
    ],
    "timeframe": "M5",
    "analysis_bars": 100,
    "max_news_tokens": 4000
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
  }
}
```

### Changing Timeframes:
To switch between timeframes, simply edit `"timeframe"`:
* `"M1"`: 1-Minute scalping cycle
* `"M5"`: 5-Minute high-frequency cycle *(Default)*
* `"M15"`: 15-Minute intraday cycle
* `"M30"`: 30-Minute swing cycle
* `"H1"`: 1-Hour standard institutional cycle
* `"H4"`: 4-Hour macro cycle
* `"D1"`: Daily position cycle

---

## ⚡ 1-Click Desktop Batch Runners

| Runner | Purpose | Description |
|---|---|---|
| **`run_agent.bat`** | **Forex Trading Loop** | Starts Ollama daemon and launches the live multi-pair autonomous trading loop (`main.py`). |
| **`chat.bat`** | **Market Intelligence Chat** | Interactive CLI chat with `agent-brain:32k` to ask questions and view live 29-pair rankings. |
| **`assistant.bat`** | **System & Coding Agent** | Launches the tool-enabled local autonomous assistant with PowerShell, DB, and MT5 access. |
| **`stop_all.bat`** | **Emergency Terminator** | Instantly kills all background Ollama, Python agent, and terminal processes. |

---

## ❓ Troubleshooting & Common Scenarios

#### 1. Terminal says: `⏳ Waiting Xs for next M5 candle close at HH:MM:00 UTC...` — Is this an error?
* **No, this is NOT an error.** It is the disciplined candle synchronizer. After analyzing all 29 pairs on startup, the bot waits until the current candle completes before re-evaluating the market, preventing repetitive spam.

#### 2. VS Code shows white dots (`•`) or "unsaved" on tabs:
* When an external script or Git updates code on disk while tabs are open in VS Code, the editor retains the memory buffer.
* **Do NOT click Save (`Ctrl+S`)** on those tabs, as this would overwrite new code with old memory buffers.
* Press `Ctrl + Shift + P` $\rightarrow$ type **`Developer: Reload Window`** $\rightarrow$ press Enter. All tabs will refresh cleanly from Git.

#### 3. MT5 says `Failed to get rates for [SYMBOL]`:
* Open MetaTrader 5, press `Ctrl + U` (Symbols window), find the symbol, and double-click it to turn it yellow (visible in Market Watch).

#### 4. Stopping the Agent:
* Press `Ctrl + C` in the running terminal, or double-click `stop_all.bat`.

---

## 📜 License & Disclaimer
This software is built for educational, quantitative research, and autonomous algorithmic trading purposes. Forex and CFD trading involve significant financial risk. Always thoroughly test on a demo account before committing live capital.
