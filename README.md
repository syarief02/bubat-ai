# 🤖 Bubat AI — Local Autonomous Institutional Forex AI Agent

> **A 100% local, self-hosted, private, and self-evolving institutional-grade Forex trading system featuring strict architectural separation between Qualitative LLM Reasoning and Quantitative Deterministic Python Math Execution.**

---

## 📌 Table of Contents
1. [Executive Architecture & Philosophy](#-executive-architecture--philosophy)
2. [Supported Instruments (All 28 Pairs + Gold)](#-supported-instruments-all-28-pairs--gold)
3. [Project Directory & File Structure](#-project-directory--file-structure)
4. [Subsystem Deep Dive](#-subsystem-deep-dive)
   - [1. Deterministic Math & Execution Engine (`core/mt5_engine.py`)](#1-deterministic-math--execution-engine-coremt5_enginepy)
   - [2. Institutional Risk Walls & Defense-in-Depth](#2-institutional-risk-walls--defense-in-depth)
   - [3. Economic Calendar & News Blackout Filter (`learning/skills/`)](#3-economic-calendar--news-blackout-filter-learningskills)
   - [4. Resilient Qualitative LLM Reasoner (`core/agent_logic.py`)](#4-resilient-qualitative-llm-reasoner-coreagent_logicpy)
   - [5. Live WebSurfer Financial Pipeline (`core/web_surfer.py` & `sentiment_engine.py`)](#5-live-websurfer-financial-pipeline)
   - [6. Active Trailing Stop & Break-Even Manager](#6-active-trailing-stop--break-even-manager)
   - [7. Multi-Pair Market Scanner (`core/market_scanner.py`)](#7-multi-pair-market-scanner-coremarket_scannerpy)
   - [8. Continuous Learning & Reflexion Engine (`learning/`)](#8-continuous-learning--reflexion-engine-learning)
   - [9. Supabase Cloud Database Telemetry (`core/supabase_manager.py`)](#9-supabase-cloud-database-telemetry-coresupabase_managerpy)
   - [10. Human-In-The-Loop WhatsApp Gateway (`core/openclaw_bridge.py`)](#10-human-in-the-loop-whatsapp-gateway-coreopenclaw_bridgepy)
5. [Autonomous System & Coding Agent (`local_assistant.py`)](#-autonomous-system--coding-agent-local_assistantpy)
6. [Automated Regression Test Suite (`tests/test_mock_cycle.py`)](#-automated-regression-test-suite-teststest_mock_cyclepy)
7. [System Requirements & Hardware Setup](#-system-requirements--hardware-setup)
8. [Installation & Quick Start Guide](#-installation--quick-start-guide)
9. [Configuration Guide (`config.json`)](#-configuration-guide-configjson)
10. [1-Click Desktop Batch Runners](#-1-click-desktop-batch-runners)
11. [Troubleshooting & Best Practices](#-troubleshooting--best-practices)
12. [License & Disclaimer](#-license--disclaimer)

---

## 🏛️ Executive Architecture & Philosophy

Modern Large Language Models (LLMs) excel at qualitative reasoning, macroeconomic narrative analysis, and synthesizing cross-asset news headlines. However, **LLMs hallucinate floating-point arithmetic** and cannot be trusted to compute live pip distances, Stop Loss prices, or dynamic lot sizing.

Bubat AI solves this fundamental limitation with a **strict two-tier separation of concerns**:

```mermaid
flowchart TD
    subgraph Senses ["1. Senses & Intelligence Layer"]
        A1["MetaTrader 5 M5/H1 Candles"] --> B1["Multi-Pair Scanner"]
        A2["WebSurfer News & Headlines"] --> B2["Sentiment Engine"]
        A3["ForexFactory JSON API"] --> B3["Economic Calendar Skill"]
    end

    subgraph Reasoning ["2. Qualitative Reasoning Tier (Ollama AI Brain)"]
        B1 & B2 & B3 --> C1["Prompt Context Builder"]
        C1 --> C2["Qwen 2.5 Coder (agent-brain:32k)"]
        C2 --> C3["TradeDecision JSON (BUY / SELL / WAIT)"]
    end

    subgraph Execution ["3. Quantitative Tier (Deterministic Python Math)"]
        C3 --> D1["ATR Volatility Geometry Engine"]
        D1 --> D2["15-Pip SL Floor & 1:2 R:R Target"]
        D2 --> D3["Account Risk & Margin Sizing"]
    end

    subgraph RiskWalls ["4. Deterministic Risk Walls (Non-Negotiable)"]
        D3 --> E1{"High-Impact News Blackout?"}
        E1 -- Yes --> R1["REJECT: News Blackout Active"]
        E1 -- No --> E2{"H1 MTF Trend Conflict?"}
        E2 -- Yes --> R2["REJECT: Counter-Trend Trade"]
        E2 -- No --> E3{"Max Trades or Spread Exceeded?"}
        E3 -- Yes --> R3["REJECT: Capacity / Spread Wall"]
        E3 -- No --> F1["MetaTrader 5 Order Execution"]
    end

    subgraph Monitoring ["5. Active Trade Management & Memory"]
        F1 --> G1["Active Trailing Stop (+10p -> BE, +15p Trail)"]
        F1 --> G2["PostgreSQL / Supabase Cloud Sync"]
        F1 --> G3["Continuous Learner (learned_rules.md)"]
    end
```

---

## 🌐 Supported Instruments (All 28 Pairs + Gold)

The system scans, analyzes, and trades **all 28 major and cross currency pairs plus Gold (`XAUUSD`)** (total 29 instruments) with automatic tick-value conversion to the account's base currency:

| Category | Pairs | Description |
|---|---|---|
| **7 Major Pairs** | `EURUSD`, `GBPUSD`, `USDJPY`, `USDCHF`, `AUDUSD`, `NZDUSD`, `USDCAD` | Highest global liquidity, tightest broker spreads |
| **6 Euro Crosses** | `EURGBP`, `EURJPY`, `EURCHF`, `EURAUD`, `EURNZD`, `EURCAD` | European macro drivers & liquidity flows |
| **5 Pound Crosses** | `GBPJPY`, `GBPCHF`, `GBPAUD`, `GBPNZD`, `GBPCAD` | High-volatility session runners |
| **4 Aussie Crosses** | `AUDJPY`, `AUDCHF`, `AUDNZD`, `AUDCAD` | Asian / Pacific commodity proxies |
| **3 Kiwi Crosses** | `NZDJPY`, `NZDCHF`, `NZDCAD` | High-beta Pacific currency crosses |
| **3 Yen / Swiss Crosses**| `CADJPY`, `CADCHF`, `CHFJPY` | Carry trade & European safe-haven plays |
| **Commodity / Metal** | `XAUUSD` | Spot Gold vs US Dollar ($300 min balance guard) |

---

## 📁 Project Directory & File Structure

```
bubat AI/
├── .env                                  # Workspace environment variables (Supabase URL & API Keys)
├── .gitignore                            # Protection against committing logs, cache, or credentials
├── README.md                             # Comprehensive technical documentation & user guide
├── assistant.bat                         # 1-click launcher for Local Autonomous Assistant
├── run_agent.bat                         # 1-click launcher for 24/7 Forex Trading Loop
├── chat.bat                              # 1-click launcher for Interactive Market Intelligence Chat
├── stop_all.bat                          # 1-click bulletproof process terminator
├── local_assistant.py                    # Local Autonomous System & Coding Agent (Tool Calling)
└── forex_local_agent/                    # Core trading agent package
    ├── config.json                       # Central system configuration (timeframe, pairs, risk)
    ├── Modelfile                         # Custom Ollama model definition (32K context window)
    ├── requirements.txt                  # Python dependencies
    ├── main.py                           # Master orchestration loop & candle synchronizer
    ├── chat.py                           # Interactive market analysis chat with live MT5 scan
    │
    ├── core/
    │   ├── agent_logic.py                # Qualitative LLM reasoning (Ollama + multi-tier JSON parser)
    │   ├── mt5_engine.py                 # MT5 API connection + ATR Math + Trailing Stop Manager
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
    │       ├── __init__.py               # Skill export registry
    │       ├── economic_calendar_filter.py # Live ForexFactory economic calendar blackout parser
    │       └── calendar_cache.json       # Local TTL cache for macroeconomic events
    │
    ├── tests/
    │   └── test_mock_cycle.py            # Automated sandboxed end-to-end regression test suite
    │
    ├── maintenance/
    │   └── model_updater.py              # Automated weekly model discovery, benchmark & hot-swap
    │
    └── logs/
        ├── trades.log                    # Historical log of all proposed and executed trades
        ├── system_errors.log             # Exception traces and critical diagnostics
        ├── code_evolution.log            # Audit trail of autonomous patches & improvements
        └── model_auditions.log           # Audition logs from model upgrade benchmarks
```

---

## ⚙️ Subsystem Deep Dive

### 1. Deterministic Math & Execution Engine (`core/mt5_engine.py`)
Computes trade geometry mathematically without LLM hallucination:
* **BUY Orders**:
  $$\text{Entry Price} = \text{Ask}$$
  $$\text{Stop Loss} = \text{Entry} - \max(\text{ATR}_{14} \times \text{Multiplier}_{\text{SL}}, \text{Floor}_{\text{pips}})$$
  $$\text{Take Profit} = \text{Entry} + (\text{ATR}_{14} \times \text{Multiplier}_{\text{TP}})$$
* **SELL Orders**:
  $$\text{Entry Price} = \text{Bid}$$
  $$\text{Stop Loss} = \text{Entry} + \max(\text{ATR}_{14} \times \text{Multiplier}_{\text{SL}}, \text{Floor}_{\text{pips}})$$
  $$\text{Take Profit} = \text{Entry} - (\text{ATR}_{14} \times \text{Multiplier}_{\text{TP}})$$

* **Asset-Class Volatility Floors**:
  * **Forex Pairs**: Minimum **15.0 pips** Stop Loss floor. Stops tighter than 15.0 pips are automatically widened to 15.0 pips (and Take Profit expanded proportionally to maintain minimum 1:1.5 to 1:2.0 Risk/Reward).
  * **Gold (`XAUUSD`)**: Minimum **$3.50 (350 points)** Stop Loss floor to absorb normal commodity volatility.

* **Dynamic Lot Sizing (Risk Formula)**:
  $$\text{Monetary Risk} = \text{Account Balance} \times \left(\frac{\text{risk\_per\_trade\_pct}}{100}\right)$$
  $$\text{Loss Per Lot} = \left(\frac{|\text{Entry} - \text{SL}|}{\text{Tick Size}}\right) \times \text{Tick Value}$$
  $$\text{Calculated Lot} = \frac{\text{Monetary Risk}}{\text{Loss Per Lot}}$$

* **Broker Constraints Guard**:
  * Enforces broker `TRADE_STOPS_LEVEL`.
  * Clamps volume to `volume_min`, `volume_step`, `volume_max`, and `max_lot_size`.

---

### 2. Institutional Risk Walls & Defense-in-Depth

Before any order is dispatched to MetaTrader 5, it must pass through **6 deterministic risk walls**:

1. **Economic News Blackout Wall**: Rejects trade if high-impact news on relevant currencies is scheduled within 30 minutes or occurred within 15 minutes.
2. **H1 Multi-Timeframe Trend Wall**: Never enters a BUY if H1 trend is BEARISH; never enters a SELL if H1 trend is BULLISH.
3. **Capacity Wall**: Strict maximum of **10 open trades** account-wide (`max_open_trades`).
4. **Duplicate Position Wall**: Strictly **one active position per symbol** (eliminates stacking/churning).
5. **Spread Protection Wall**: Rejects orders if live spread exceeds `max_spread_pips` (default: 3.5 pips).
6. **Margin Gatekeeper**: Rejects orders if account free margin is negative or required margin exceeds free margin.
7. **Gold Balance Guard**: Forbids trading `XAUUSD` on accounts with balance under **$300 USD**.

---

### 3. Economic Calendar & News Blackout Filter (`learning/skills/`)
Located at [`forex_local_agent/learning/skills/economic_calendar_filter.py`](file:///c:/Users/User/OneDrive/Desktop/bubat%20AI/forex_local_agent/learning/skills/economic_calendar_filter.py):
* Connects directly to the **ForexFactory weekly JSON feed** (`https://nfs.faireconomy.media/ff_calendar_thisweek.json`).
* Maintains a local disk cache (`calendar_cache.json`) with a 2-hour TTL to prevent external rate limits.
* **Deterministic Blackout Window**:
  * **30 minutes prior** to High-Impact events (CPI, NFP, Fed/Central Bank Rate Decisions, GDP).
  * **15 minutes after** release (allowing broker spread blowout to normalize).
* **LLM Calendar Context**: Injects structured upcoming event schedules into LLM prompts so the agent is macro-aware.

---

### 4. Resilient Qualitative LLM Reasoner (`core/agent_logic.py`)
* Powered by local **Ollama** model (`agent-brain:32k`).
* **Temperature calibrated to `0.1`** for strict JSON format consistency.
* **Token limit raised to `800` (`num_predict: 800`)** to prevent mid-string truncations.
* **Multi-Tier Resilient Parser**:
  * **Tier 1:** Standard `json.loads`.
  * **Tier 2:** Auto-repair of unclosed reasoning strings and missing brackets.
  * **Tier 3:** Deterministic regex fallback extraction directly capturing `market_sentiment`, `decision`, and `confidence_score`.
* Parsing failure rate reduced from 8.1% to **0.0%**.

---

### 5. Live WebSurfer Financial Pipeline
Built into [`core/web_surfer.py`](file:///c:/Users/User/OneDrive/Desktop/bubat%20AI/forex_local_agent/core/web_surfer.py) and [`core/sentiment_engine.py`](file:///c:/Users/User/OneDrive/Desktop/bubat%20AI/forex_local_agent/core/sentiment_engine.py):
* **Zero SearXNG Delay**: Probes SearXNG on port 8080 once (0.6s). If offline, routes instantly to live WebSurfer (Google News RSS & Investing.com RSS).
* Fetches live news per pair in **~0.3 to 0.7 seconds**.
* Scrapes article context concurrently using `asyncio.gather` with strict timeouts.

---

### 6. Active Trailing Stop & Break-Even Manager
Managed natively within [`core/mt5_engine.py`](file:///c:/Users/User/OneDrive/Desktop/bubat%20AI/forex_local_agent/core/mt5_engine.py) on every candle cycle:
* **Break-Even Lock**: When a position moves **+10.0 pips** into profit, Stop Loss is automatically modified to Entry Price + 1.0 pip (locking in a risk-free trade).
* **Dynamic Trailing Stop**: When profit reaches **+15.0 pips**, Stop Loss trails behind market price with a 5.0-pip step.
* Fully configurable in `config.json`:
  ```json
  "trailing_stop_enabled": true,
  "trailing_breakeven_pips": 10.0,
  "trailing_step_pips": 5.0
  ```

---

### 7. Multi-Pair Market Scanner (`core/market_scanner.py`)
* Scans all 29 instruments simultaneously via MetaTrader 5 in **under 2 seconds**.
* Calculates RSI(14), ATR(14), 20/50 EMAs, 24-hour price change %, and live broker spread.
* Evaluates active trading sessions (London, New York, Tokyo, Sydney) and calculates an **Opportunity Score (0 to 100+)** to rank top setups.
* Prints a clean ASCII summary table at the end of every cycle:
  ```
  +----------+------------+------------+--------------+--------------------------------------------------+
  | SYMBOL   | DECISION   | CONFIDENCE | H1 TREND     | STATUS / ACTION                                  |
  +----------+------------+------------+--------------+--------------------------------------------------+
  | EURUSD   | BUY        | 82%        | BULLISH      | REJECTED: News blackout active for EURUSD        |
  | GBPUSD   | BUY        | 85%        | BULLISH      | ACTIVE #25482910 (BUY 0.01 lot) [BE +1.0p LOCKED]|
  | XAUUSD   | WAIT       | 50%        | BULLISH      | WAIT                                             |
  +----------+------------+------------+--------------+--------------------------------------------------+
  ```

---

### 8. Continuous Learning & Reflexion Engine (`learning/`)
* **Continuous Learner (`continuous_learner.py`)**: Automatically captures user instructions, communication preferences (e.g. Malaysian Malay format), and trading rules into `learning/learned_rules.md` and Supabase cloud.
* **Post-Mortem Loss Reflexion**: Closed losing trades trigger an automated LLM post-mortem that identifies the root cause and distills an imperative rule into `learned_rules.md`.
* **Supabase Cloud Sync**: Rules are saved to both local disk and the `forex_learned_rules` PostgreSQL table.

---

### 9. Supabase Cloud Database Telemetry (`core/supabase_manager.py`)
Provides full cloud persistence using both the Supabase PostgREST client and direct PostgreSQL (`psycopg2`):
* **`forex_trade_decisions`**: Stores LLM decision, confidence score, sentiment, reasoning, and ATR parameters.
* **`forex_executed_trades`**: Stores ticket IDs, order types, lot size, entry price, SL, TP, and outcomes.
* **`forex_learned_rules`**: Stores all learned operational rules with UUID primary keys and JSONB metadata.
* **`ai_agent_telemetry`**: Logs system health, 29-pair scans, cycle latencies, and error audits.

---

### 10. Human-In-The-Loop WhatsApp Gateway (`core/openclaw_bridge.py`)
* Optional semi-autonomous mode: High-confidence signals ($\ge 80\%$) trigger a WhatsApp proposal via OpenClaw:
  ```text
  PROPOSAL: BUY EURUSD
  Lot: 0.01 | Entry: 1.13570
  SL: 1.13420 | TP: 1.13810
  ATR: 0.00031 | R:R 1:2.0
  Risk: $1.50 | Confidence: 82%
  Reason: Strong bullish EMA breakout with RSI momentum.
  Reply YES to execute or NO to abort.
  ```
* Waits on webhook `http://localhost:5055/webhook/approval`. If `YES` is received within 300s, the order executes; otherwise it expires safely.
* Autonomous mode (`"auto_approve": true`) bypasses WhatsApp for 100% hands-free trading.

---

## 🦾 Autonomous System & Coding Agent (`local_assistant.py`)

The workspace includes a built-in **autonomous coding and systems agent** powered by local Ollama:
* `execute_command(command)`: Executes shell commands via PowerShell.
* `read_file(path, start_line, end_line)`: Inspects code and configs with line slicing.
* `write_file(path, content, mode)`: Autonomously creates or edits files.
* `list_directory(path, recursive)`: Explores folders and projects.
* `query_database(sql)`: Directly queries Supabase PostgreSQL.
* `get_system_status()`: Checks MT5 terminal connection, account equity, and disk space.
* `web_search(query)`: Performs real-time web search.

Launch anytime via:
```powershell
.\assistant.bat
```

---

## 🧪 Automated Regression Test Suite (`tests/test_mock_cycle.py`)

A sandboxed dry-run test suite validates the entire end-to-end trading pipeline without placing real orders:
```powershell
python forex_local_agent/tests/test_mock_cycle.py
```

### Validated Pipeline Steps:
1. **Engine Initialization**: Connects to MT5 terminal and loads account credentials.
2. **Bar Retrieval**: Pulls live bars, ATR(14), RSI(14), and H1 MTF trend confirmation.
3. **Sentiment & Calendar Scraper**: Fetches live news headlines and checks the live ForexFactory blackout schedule.
4. **Ollama Qualitative Reasoning**: Verifies JSON schema compliance and parsing on Attempt 1.
5. **Deterministic ATR Math**: Verifies Stop Loss floor (15.0 pips), 1:2 Risk/Reward calculation, and lot sizing.

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

### Step 3: Configure Environment Variables (`.env`)
Create or edit `.env` in the project root:
```ini
GEMINI_API_KEY=your_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres
```

### Step 4: Set Up Ollama Local Brain
1. Download and install [Ollama for Windows](https://ollama.com).
2. Pull base model and build 32K context model:
   ```powershell
   ollama pull qwen2.5-coder:1.5b
   ollama create agent-brain:32k -f Modelfile
   ```

### Step 5: Configure MetaTrader 5
1. Launch MT5 and log into your broker account.
2. Under **Tools** $\rightarrow$ **Options** $\rightarrow$ **Expert Advisors**:
   - Check **"Allow Algo Trading"**
   - Check **"Allow DLL imports"**
   - Click **OK**
3. Ensure the green **"Algo Trading"** button on the toolbar is active.

### Step 6: Launch the Agent
```powershell
.\run_agent.bat
```

---

## ⚙️ Configuration Guide (`config.json`)

Located at [`forex_local_agent/config.json`](file:///c:/Users/User/OneDrive/Desktop/bubat%20AI/forex_local_agent/config.json):

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
    "lot_mode": "fixed",
    "fixed_lot": 0.01,
    "max_lot_size": 0.1,
    "max_drawdown_pct": 2.0,
    "max_open_trades": 10,
    "confidence_threshold": 0.80,
    "auto_approve": true,
    "min_sl_pips": 15.0,
    "max_spread_pips": 3.5,
    "symbol_cooldown_minutes": 30,
    "news_blackout_pre_mins": 30,
    "news_blackout_post_mins": 15,
    "trailing_stop_enabled": true,
    "trailing_breakeven_pips": 10.0,
    "trailing_step_pips": 5.0
  }
}
```

---

## ⚡ 1-Click Desktop Batch Runners

| Runner | Target | Description |
|---|---|---|
| **`run_agent.bat`** | **Forex Trading Loop** | Starts Ollama daemon and runs the 24/7 autonomous trading loop (`main.py`). |
| **`chat.bat`** | **Market Intelligence Chat** | Interactive CLI chat with `agent-brain:32k` to query 29 pairs and live setups. |
| **`assistant.bat`** | **Autonomous Coding Agent** | Launches the tool-enabled coding assistant with PowerShell, DB, and MT5 access. |
| **`stop_all.bat`** | **Emergency Terminator** | Instantly kills all background Ollama, Python agent, and terminal processes. |

---

## ❓ Troubleshooting & Best Practices

#### 1. Terminal says: `⏳ Waiting Xs for next M5 candle close...`
* **Normal Behavior.** After analyzing all 29 pairs on startup, the bot waits for the next candle close before scanning again, preventing redundant CPU churn.

#### 2. VS Code shows unsaved dots (`•`) on tabs:
* If tabs were open when Git or scripts modified files on disk, do **NOT** click Save (`Ctrl+S`).
* Press `Ctrl + Shift + P` $\rightarrow$ type **`Developer: Reload Window`** $\rightarrow$ press Enter. All tabs will reload cleanly from disk.

#### 3. MT5 says `Symbol [SYMBOL] not found`:
* In MT5, press `Ctrl + U` (Symbols), find the pair, and double-click it to add it to Market Watch.

#### 4. Safely stopping the agent:
* Press `Ctrl + C` in the running terminal, or double-click `stop_all.bat`.

---

## 📜 License & Disclaimer
This software is developed for educational, quantitative research, and autonomous algorithmic trading purposes. Forex and CFD trading involve substantial financial risk. Always test thoroughly on a demo account before risking real capital.
