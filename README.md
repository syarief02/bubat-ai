# 🤖 Bubat AI — Local Autonomous Institutional Forex AI Agent

> **A 100% local, self-hosted, private, and self-evolving institutional-grade Forex trading system featuring strict architectural separation between Qualitative LLM Reasoning and Quantitative Deterministic Python Math Execution.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![MetaTrader 5](https://img.shields.io/badge/MetaTrader_5-API-green)](https://www.metatrader5.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple)](https://ollama.com)
[![Supabase](https://img.shields.io/badge/Supabase-Cloud_DB-orange)](https://supabase.com)

---

## 📌 Table of Contents
1. [Executive Architecture & Philosophy](#-executive-architecture--philosophy)
2. [Supported Instruments (All 28 Pairs + Gold)](#-supported-instruments-all-28-pairs--gold)
3. [Project Directory & File Structure](#-project-directory--file-structure)
4. [Subsystem Deep Dive](#-subsystem-deep-dive)
   - [1. Deterministic Math & Execution Engine (`core/mt5_engine.py`)](#1-deterministic-math--execution-engine-coremt5_enginepy)
   - [2. Institutional Risk Walls & Defense-in-Depth](#2-institutional-risk-walls--defense-in-depth)
   - [3. Economic Calendar & News Blackout Filter](#3-economic-calendar--news-blackout-filter-learningskills)
   - [4. Currency Correlation & Portfolio Exposure Filter](#4-currency-correlation--portfolio-exposure-filter-learningskills)
   - [5. Resilient Qualitative LLM Reasoner (`core/agent_logic.py`)](#5-resilient-qualitative-llm-reasoner-coreagent_logicpy)
   - [6. Live WebSurfer Financial Pipeline](#6-live-websurfer-financial-pipeline)
   - [7. Active Trailing Stop & Break-Even Manager](#7-active-trailing-stop--break-even-manager)
   - [8. Multi-Pair Market Scanner (`core/market_scanner.py`)](#8-multi-pair-market-scanner-coremarket_scannerpy)
   - [9. Continuous Learning & Reflexion Engine (`learning/`)](#9-continuous-learning--reflexion-engine-learning)
   - [10. Supabase Cloud Database Telemetry (`core/supabase_manager.py`)](#10-supabase-cloud-database-telemetry-coresupabase_managerpy)
   - [11. Human-In-The-Loop WhatsApp Gateway (`core/openclaw_bridge.py`)](#11-human-in-the-loop-whatsapp-gateway-coreopenclaw_bridgepy)
5. [Self-Evolving Code Architecture](#-self-evolving-code-architecture)
6. [Autonomous System & Coding Agent (`local_assistant.py`)](#-autonomous-system--coding-agent-local_assistantpy)
7. [Automated Regression Test Suite (`tests/test_mock_cycle.py`)](#-automated-regression-test-suite-teststest_mock_cyclepy)
8. [System Requirements & Hardware Setup](#-system-requirements--hardware-setup)
9. [Installation & Quick Start Guide](#-installation--quick-start-guide)
10. [Configuration Guide (`config.json`)](#-configuration-guide-configjson)
11. [1-Click Desktop Batch Runners](#-1-click-desktop-batch-runners)
12. [Troubleshooting & Best Practices](#-troubleshooting--best-practices)
13. [License & Disclaimer](#-license--disclaimer)

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

    subgraph RiskWalls ["4. Deterministic Risk Walls (8 Layers)"]
        D3 --> E0{"H1 MTF Trend Conflict?"}
        E0 -- Yes --> R0["REJECT: Counter-Trend"]
        E0 -- No --> E1{"High-Impact News Blackout?"}
        E1 -- Yes --> R1["REJECT: News Blackout Active"]
        E1 -- No --> E4{"Currency Exposure > 3?"}
        E4 -- Yes --> R4["REJECT: Correlation Risk"]
        E4 -- No --> E3{"Max Trades / Spread?"}
        E3 -- Yes --> R3["REJECT: Capacity / Spread"]
        E3 -- No --> F1["MetaTrader 5 Order Execution"]
    end

    subgraph Monitoring ["5. Active Trade Management & Memory"]
        F1 --> G1["Trailing Stop (+10p → BE, +15p Trail)"]
        F1 --> G2["Supabase Cloud Sync"]
        F1 --> G3["Reflexion Learner"]
        F1 --> G4["Circuit Breaker & Error Recovery"]
    end
```

### Key Design Principles
| Principle | Implementation |
|---|---|
| **LLM = Qualitative ONLY** | The LLM determines *direction* (BUY/SELL/WAIT) and sentiment — never prices, lot sizes, or SL/TP levels |
| **Math = Deterministic Python** | ATR geometry, pip distance, lot sizing, and risk % are all computed by pure Python functions |
| **Defense-in-Depth** | 8 independent risk walls must ALL pass before any order reaches the broker |
| **Fail-Safe by Default** | Any parsing failure, connectivity loss, or ambiguous signal defaults to WAIT (no trade) |
| **Self-Evolution** | Autonomous code patches, learned rules, and circuit breakers continuously improve the system |

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
| **Commodity / Metal** | `XAUUSD` | Spot Gold vs US Dollar (\$300 min balance guard) |

---

## 📁 Project Directory & File Structure

```
bubat AI/
├── .env                                  # Environment variables (Supabase, Gemini API Keys)
├── .gitignore                            # Protection against committing logs, cache, or credentials
├── README.md                             # This documentation
├── assistant.bat                         # 1-click launcher for Local Autonomous Assistant
├── run_agent.bat                         # 1-click launcher for 24/7 Forex Trading Loop
├── chat.bat                              # 1-click launcher for Interactive Market Intelligence Chat
├── stop_all.bat                          # 1-click bulletproof process terminator
└── forex_local_agent/                    # Core trading agent package
    ├── config.json                       # Central system configuration (risk, pairs, model)
    ├── Modelfile                         # Custom Ollama model definition (32K context window)
    ├── requirements.txt                  # Python dependencies
    ├── main.py                           # Master orchestration loop & candle synchronizer
    ├── chat.py                           # Interactive market analysis chat with live MT5 scan
    ├── local_assistant.py                # Local autonomous coding & systems agent
    │
    ├── core/
    │   ├── agent_logic.py                # Qualitative LLM reasoning (Ollama + multi-tier parser
    │   │                                 #   + synonym normalizer + circuit breaker)
    │   ├── mt5_engine.py                 # MT5 API + ATR math + trailing stop manager
    │   ├── market_scanner.py             # 29-pair live technical scanner & opportunity ranker
    │   ├── web_surfer.py                 # High-speed live financial news & web intelligence
    │   ├── sentiment_engine.py           # News aggregation with SearXNG + RSS fallback
    │   ├── openclaw_bridge.py            # WhatsApp approval gateway & email alerts
    │   └── supabase_manager.py           # Supabase REST & PostgreSQL telemetry sync
    │
    ├── learning/
    │   ├── continuous_learner.py         # Autonomous rule capture engine
    │   ├── memory_manager.py             # ChromaDB / JSON episodic memory & recall
    │   ├── skill_factory.py              # Dynamic trading skill code generator
    │   ├── learned_rules.md              # Curated trading rules (quality-filtered by loader)
    │   └── skills/
    │       ├── __init__.py               # Skill export registry
    │       ├── economic_calendar_filter.py # ForexFactory calendar blackout parser
    │       ├── currency_correlation_filter.py # Portfolio exposure & correlation risk monitor
    │       ├── template_skill.py         # Skill template for auto-generated tools
    │       ├── skills_index.json         # Skill metadata registry
    │       └── calendar_cache.json       # Local TTL cache for macroeconomic events
    │
    ├── tests/
    │   └── test_mock_cycle.py            # 8-step sandboxed end-to-end regression test suite
    │
    ├── maintenance/
    │   ├── model_updater.py              # Automated weekly model discovery & hot-swap
    │   └── daily_report.py               # 24h post-mortem metrics, counterfactuals & baselines (JSON -> reports/)
    │
    └── logs/
        ├── trades.log                    # All proposed and executed trades
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
  * **Forex Pairs**: Minimum **15.0 pips** Stop Loss floor. Stops tighter than 15.0 pips are automatically widened (and Take Profit expanded to maintain minimum 1:1.5 to 1:2.0 Risk/Reward).
  * **Gold (`XAUUSD`)**: Minimum **\$3.50 (350 points)** Stop Loss floor to absorb normal commodity volatility.

* **Dynamic Lot Sizing (Risk Formula)**:
  $$\text{Monetary Risk} = \text{Account Balance} \times \left(\frac{\text{risk\_per\_trade\_pct}}{100}\right)$$
  $$\text{Loss Per Lot} = \left(\frac{|\text{Entry} - \text{SL}|}{\text{Tick Size}}\right) \times \text{Tick Value}$$
  $$\text{Calculated Lot} = \frac{\text{Monetary Risk}}{\text{Loss Per Lot}}$$

* **Broker Constraints Guard**: Enforces broker `TRADE_STOPS_LEVEL`, clamps volume to `volume_min`, `volume_step`, `volume_max`, and `max_lot_size`.

---

### 2. Institutional Risk Walls & Defense-in-Depth

Before any order is dispatched to MetaTrader 5, it must pass through **9 deterministic risk walls** in sequence:

| Wall # | Name | Logic |
|---|---|---|
| 1 | **H1 Multi-Timeframe Trend Wall** | Never BUY if H1 is BEARISH; never SELL if H1 is BULLISH |
| 2 | **Daily Loss Stop** | Blocks all new entries when net realized losses (profit + swap + commission) since 00:00 **real UTC** exceed `daily_loss_limit_pct` (5%) of balance |
| 3 | **Capacity Wall** | Max **10 open trades** account-wide |
| 4 | **Duplicate Position Wall** | Only **one active position per symbol** |
| 5 | **Economic News Blackout Wall** | Rejects if high-impact news for either currency is ≤30 min ahead or ≤15 min past. Tier-1 USD releases (NFP, CPI, Fed rate decision, FOMC) black out **all** symbols (`news_blackout_global_tier1`) |
| 6 | **Currency Correlation Wall** | Max **3 positions per currency** to prevent correlated cascade stops |
| 7 | **Margin Gatekeeper** | Rejects if free margin is negative or insufficient |
| 8 | **Spread Protection Wall** | Rejects if live spread exceeds `max_spread_pips` (3.5 pips) |
| 9 | **Gold Balance Guard** | Forbids `XAUUSD` on accounts under **\$300 USD** |
| 10 | **Entry Window Wall** | Optional: new trades only inside `entry_hours_utc` (`null` = all sessions trade, the current setting) |
| 11 | **Confidence Calibration Cap** | Rejects LLM confidence above `max_confidence` (0.89); the highest buckets have been anti-predictive |
| 12 | **Spread Cost Wall** | Rejects if live spread exceeds `max_spread_sl_ratio` (6%) of the SL distance |
| 13 | **Session Profile Wall** | Per-session symbols, open-trade slots, spread cap and **daily loss budget** (`session_profiles`); budgets sum to `daily_loss_limit_pct`, so one bad session cannot lock out the others |

> **Note:** These walls are deterministic Python code — the LLM cannot override, bypass, or modify them.

> **Per-session execution:** every session trades, each with its own profile (`core/session_profiles.py`). Once per UTC day each session is graded from its last 5 days of trades (R = pips / SL pips, by entry hour): `NORMAL` (owner baseline), `REDUCED` (half the slots, USD majors only, spread <= 4% of SL) when n >= 20 and avg R < -0.15, `MINIMAL` (1 slot, USD majors, spread <= 3%) when avg R < -0.30; it steps back up one level per day after n >= 10 trades at avg R >= 0, and never above the baseline. Levels persist in `state/session_levels.json`; each review is logged as `SESSION_REVIEW`.

> **Asia shadow strategy:** in addition, Asian hours (00:00-06:00 UTC) run a separate deterministic **range-fade** strategy on 5 low-spread majors (`session_strategies.asia_range_fade`): fade M5 Bollinger(20, 2.0) extremes confirmed by RSI(14) < 30 / > 70, SL = max(1.5 x ATR, 8 pips), TP = 1R, force-exit at 07:00 UTC. A 180-day backtest showed about +0.05R/trade before costs but roughly break-even after a realistic 0.3-1.0 pip cost, so it runs in **shadow mode**: paper trades at the live bid/ask, resolved on M5 bars, logged to `state/asia_shadow_trades.jsonl`, and **no orders are sent**. Going live needs the owner's approval after 100+ paper trades averaging at least +0.05R (`execution_quality.py` reports progress).

> **Ranked execution:** each cycle first analyses every symbol, then executes the tradeable signals best-first (full H1 trend before bias-only, then lowest spread/SL) so the limited open-trade slots go to the best setups instead of whichever symbol comes first in the config (`rank_signals`).

> **Broker server time:** MT5 stamps deals, ticks and bars in broker server time (Tickmill: UTC+2/UTC+3). Every history query goes through `core/mt5_time.py`, which measures the offset from live ticks during market hours and persists it in `state/`. Querying MT5 with plain UTC datetimes silently misses the most recent hours of deals.


---

### 3. Economic Calendar & News Blackout Filter (`learning/skills/`)
* Connects to the **ForexFactory weekly JSON feed** (`nfs.faireconomy.media/ff_calendar_thisweek.json`).
* Maintains a local disk cache (`calendar_cache.json`) with a 2-hour TTL.
* **Deterministic Blackout Window**:
  * **30 minutes prior** to High-Impact events (CPI, NFP, Fed/Central Bank Rate Decisions, GDP).
  * **15 minutes after** release (allowing broker spread blowout to normalize).
* Injects structured upcoming event schedules into LLM prompts for macro awareness.

---

### 4. Currency Correlation & Portfolio Exposure Filter (`learning/skills/`)
* **Decomposes** each forex pair into base and quote currencies (e.g., `GBPJPY` → `GBP` + `JPY`).
* **Tracks gross exposure** per currency across all open positions.
* **Blocks new trades** when any single currency exceeds `max_currency_exposure` (default: **3 positions**).
  * Example: If you have 3 positions involving JPY (`USDJPY`, `EURJPY`, `GBPJPY`), a 4th JPY pair is rejected.
* Prevents **correlated cascade stop-outs** where one currency move wipes multiple positions simultaneously.
* Integrated directly into `mt5_engine.py` as Risk Wall #5.

---

### 5. Resilient Qualitative LLM Reasoner (`core/agent_logic.py`)
* Powered by local **Ollama** model (`agent-brain:32k` — Qwen 2.5 Coder with 32K context).
* **Temperature calibrated to `0.1`** for strict JSON format consistency.
* **Token limit: `800` tokens** (`num_predict: 800`) to prevent mid-string truncations.
* **Multi-Tier Resilient Parser**:
  * **Tier 1:** `json.loads(strict=False)` → `_normalize_trade_decision_dict()` — handles unescaped newlines and field synonyms.
  * **Tier 2:** Auto-repair of unclosed reasoning strings and missing brackets.
  * **Tier 3:** Deterministic regex fallback with `re.S` multiline support, extracting fields by alias (`decision/action/signal`, `sentiment/market_sentiment`, `confidence_score/confidence/score`).
* **Synonym Normalizer** (`_normalize_trade_decision_dict`): Maps LLM drift synonyms:
  * `HOLD`, `PASS`, `STAND ASIDE`, `FLAT` → `WAIT`
  * `SHORT`, `SELLING` → `SELL`
  * `LONG`, `BUYING` → `BUY`
  * `POSITIVE`, `UP` → `BULLISH`
  * `NEGATIVE`, `DOWN` → `BEARISH`
  * `"82%"` → `0.82`, `85` → `0.85`
* **Ollama Circuit Breaker**: After 5 consecutive connection failures, blocks all queries for 60 seconds to prevent resource hammering. Resets automatically on first successful query.
* **Quality-Filtered Rules Loader** (`learning/rules_loader.py`): Injects curated `### [CATEGORY]` rules as compact one-liners, capped at 2,500 chars on whole-rule boundaries (newest win). `REFLEXION` and `COMMUNICATION` rules and anything with `Source: loss_reflexion` are never injected into the trading prompt.

---

### 6. Live WebSurfer Financial Pipeline
Built into `core/web_surfer.py` and `core/sentiment_engine.py`:
* **Zero SearXNG Delay**: Probes SearXNG on port 8080 once (0.6s). If offline, routes instantly to live WebSurfer (Google News RSS & Investing.com RSS).
* Fetches live news per pair in **~0.3 to 0.7 seconds**.
* Scrapes article context concurrently using `asyncio.gather` with strict timeouts.
* Falls back gracefully if all external sources are unreachable — the system never blocks on news.

---

### 7. Active Trailing Stop & Break-Even Manager
Managed natively within `core/mt5_engine.py` on every candle cycle:
* **Break-Even Lock**: When a position moves **+15.0 pips** into profit, Stop Loss is automatically modified to Entry Price + 1.0 pip (locking in a risk-free trade).
* **Dynamic Trailing Stop**: When profit reaches **+20.0 pips**, Stop Loss trails **10.0 pips** behind market price, updated whenever it can improve by at least the **2.0-pip** step.
* Fully configurable in `config.json`:
  ```json
  "trailing_stop_enabled": true,
  "trailing_breakeven_pips": 15.0,
  "trailing_breakeven_lock_pips": 1.0,
  "trailing_start_pips": 20.0,
  "trailing_distance_pips": 10.0,
  "trailing_step_pips": 2.0
  ```

---

### 8. Multi-Pair Market Scanner (`core/market_scanner.py`)
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
  | USDJPY   | SELL       | 78%        | BEARISH      | REJECTED: Currency exposure > 3 for USD          |
  | XAUUSD   | WAIT       | 50%        | BULLISH      | WAIT (Balance < $300)                            |
  +----------+------------+------------+--------------+--------------------------------------------------+
  ```

---

### 9. Continuous Learning & Reflexion Engine (`learning/`)
* **Continuous Learner** (`continuous_learner.py`): Automatically captures user instructions, communication preferences, and trading rules into `learned_rules.md` and Supabase cloud.
* **Post-Mortem Loss Reflexion**: Closed losing trades (enriched with entry, original SL/TP, exit type and hold time) trigger an LLM post-mortem. Its proposed rule is validated and **quarantined** in `learning/reflexion_candidates.jsonl`; it is never written to `learned_rules.md` or injected into prompts. Curated rules are added only by the daily audit. Processed tickets persist in `state/processed_tickets.json`, so a restart never re-reflects old losses.
* **Post-Trade Cooldown**: `symbol_cooldown_minutes` runs from the trade's real close time.
* **Quality-Filtered Rules**: See `learning/rules_loader.py` (shared by the trading prompt and the chat/assistant prompts).
* **Supabase Cloud Sync**: Rules are saved to both local disk and the `forex_learned_rules` PostgreSQL table.
* **Episodic Memory**: ChromaDB vector database stores historical trade episodes for similarity recall on future decisions.

---

### 10. Supabase Cloud Database Telemetry (`core/supabase_manager.py`)
Provides full cloud persistence using the Supabase PostgREST client:

| Table | Contents |
|---|---|
| `forex_trade_decisions` | LLM decision, confidence, sentiment, reasoning, ATR parameters |
| `forex_executed_trades` | Ticket IDs, order types, lot size, entry price, SL, TP, execution results |
| `forex_learned_rules` | Curated operational rules with UUID primary keys and JSONB metadata |
| `ai_agent_telemetry` | System health, 29-pair scans, cycle latencies, error audits |

---

### 11. Human-In-The-Loop WhatsApp Gateway (`core/openclaw_bridge.py`)
* Optional semi-autonomous mode: High-confidence signals (≥80%) trigger a WhatsApp proposal via OpenClaw:
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
* **Autonomous mode** (`"auto_approve": true`) bypasses WhatsApp for 100% hands-free trading.

---

## 🧬 Self-Evolving Code Architecture

Bubat AI features a unique **autonomous self-improvement pipeline** where the system diagnoses its own bugs, writes patches, and deploys fixes without human intervention:

```mermaid
flowchart LR
    A["24h Post-Mortem Audit"] --> B["Root-Cause Diagnosis"]
    B --> C{"Bug Found?"}
    C -- Yes --> D["Auto-Patch Code"]
    C -- No --> E["Calibrate Parameters"]
    D --> F["Regression Tests"]
    E --> F
    F --> G{"All Passed?"}
    G -- Yes --> H["Git Commit & Push"]
    G -- No --> I["Rollback & Alert"]
    H --> J["Log to code_evolution.log"]
```

### Evolution Features
* **Circuit Breaker Pattern**: Prevents cascading failures when Ollama or external services go down. Opens after 5 consecutive failures, blocks for 60s, auto-resets on recovery.
* **Synonym Normalizer**: Learns from LLM output drift and maps non-standard responses back to valid enum values without crashing.
* **Autonomous Code Patching**: Daily maintenance cycles identify bugs from `system_errors.log`, write fixes, run regression tests, and commit to `origin/master`.
* **Evolution Audit Trail**: All patches are logged to `logs/code_evolution.log` with timestamps, metrics, and rationale.

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

### Validated Pipeline Steps (8 Assertions):
| Step | Test |
|---|---|
| 1 | **Engine Initialization**: Connects to MT5 terminal and loads account credentials |
| 2 | **Bar Retrieval**: Pulls live bars, ATR(14), RSI(14), and H1 MTF trend confirmation |
| 3 | **Sentiment & Calendar Scraper**: Fetches live headlines and checks ForexFactory blackout |
| 4 | **Ollama Qualitative Reasoning**: Verifies JSON schema compliance and parsing |
| 5 | **Deterministic ATR Math**: Verifies SL floor (15.0 pips), 1:2 R:R, and lot sizing |
| 6 | **Trailing Stop Manager**: Evaluates break-even / trailing logic on live positions with `order_send` intercepted (nothing reaches the broker) |
| 7 | **Risk Wall Defenses**: Tests H1 trend wall, capacity wall, and spread wall |
| 8 | **Currency Correlation Wall**: Validates portfolio exposure limits per currency |

The whole mock cycle runs with `mt5.order_send` patched, so no test can open, close or modify a real position.

### Offline Regression Tests
```powershell
python forex_local_agent/tests/test_cycle5_regressions.py   # rules loader, reflexion quarantine, MT5 server time, daily loss, tier-1 blackout, trailing config, chatbot
python forex_local_agent/tests/test_execution_upgrade.py    # ranked execution, entry window, confidence cap, spread cost wall, no-lookahead simulation
python forex_local_agent/tests/test_session_profiles.py     # per-session slots, spread cap, loss budget, daily grading
python forex_local_agent/tests/test_session_strategies.py   # Asia range-fade signal, paper trade fill/resolve, shadow runner never sends orders
python forex_local_agent/tests/test_daily_loss_stop.py
python forex_local_agent/tests/test_chat_logger.py
```

### Daily Post-Mortem Report
```powershell
python forex_local_agent/maintenance/daily_report.py --hours 24
```
Prints and saves (to `reports/`, gitignored) win rate, payoff, profit factor, expectancy, peak-to-trough drawdown, breakdowns by symbol / direction / session / hour / confidence / exit type / spread / H1 alignment, cost drag, exposure, LLM health, risk-wall counts, counterfactual R for rejected signals, and the LLM vs follow-H1-trend vs always-WAIT baselines. Simulated entries use the next M5 bar open (no lookahead).

```powershell
python forex_local_agent/maintenance/execution_quality.py --hours 24
```
Execution-quality breakdown: real fills vs a shadow entry at the next bar open, managed vs unmanaged R, R by spread/SL bucket, and quick same-symbol re-entries.

---

## 🛠️ System Requirements & Hardware Setup

| Component | Minimum | Recommended (Tested Configuration) |
|---|---|---|
| **Operating System** | Windows 10/11 64-bit | Windows 11 64-bit |
| **GPU** | NVIDIA GTX 1660 (6 GB) | NVIDIA RTX 4060 (8 GB VRAM) or higher |
| **RAM** | 16 GB | 32 GB DDR4 / DDR5 |
| **Python** | Python 3.10+ | Python 3.13 64-bit |
| **Broker Terminal** | MetaTrader 5 Build 4000+ | MetaTrader 5 Build 5.0+ (Tickmill-Demo) |

### Software Stack
| Component | Role | Local Port |
|---|---|---|
| **Ollama** | Local LLM inference server | `localhost:11434` |
| **MetaTrader 5** | Broker terminal & execution | IPC (shared memory) |
| **SearXNG** *(optional)* | Privacy-respecting news search | `localhost:8080` |
| **Supabase** | Cloud PostgreSQL telemetry | Remote (HTTPS) |
| **OpenClaw** *(optional)* | WhatsApp approval webhook | `localhost:5055` |

---

## 🚀 Installation & Quick Start Guide

### Step 1: Clone Repository
```powershell
git clone https://github.com/syarief02/bubat-ai.git "bubat AI"
cd "bubat AI"
```

### Step 2: Install Python Dependencies
```powershell
cd forex_local_agent
python -m pip install -r requirements.txt
```

### Step 3: Configure Environment Variables (`.env`)
Create or edit `.env` in the project root (`bubat AI/.env`):
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
   cd forex_local_agent
   ollama create agent-brain:32k -f Modelfile
   ```

### Step 5: Configure MetaTrader 5
1. Launch MT5 and log into your broker account.
2. Under **Tools** → **Options** → **Expert Advisors**:
   - Check **"Allow Algo Trading"**
   - Check **"Allow DLL imports"**
   - Click **OK**
3. Ensure the green **"Algo Trading"** button on the toolbar is active.
4. In **Market Watch** (`Ctrl + U`), ensure all 28 forex pairs + XAUUSD are visible.

### Step 6: Launch the Agent
```powershell
.\run_agent.bat
```

The agent will:
1. Connect to MT5 and verify account credentials
2. Scan all 29 instruments for technical data
3. Fetch live news for each pair
4. Query the local LLM for directional decisions
5. Execute trades that pass all 8 risk walls
6. Print a summary table and wait for the next M5 candle close

---

## ⚙️ Configuration Guide (`config.json`)

Located at `forex_local_agent/config.json`:

```json
{
  "active_model": "agent-brain:32k",
  "ollama_base_url": "http://localhost:11434",
  "searxng_url": "http://localhost:8080",
  "trading": {
    "symbols": ["EURUSD", "GBPUSD", "USDJPY", "...all 28 pairs...", "XAUUSD"],
    "timeframe": "M5",
    "analysis_bars": 100,
    "max_news_tokens": 4000,
    "order_comment": "Bubat AI"
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
    "news_blackout_global_tier1": true,
    "trailing_breakeven_pips": 15.0,
    "trailing_breakeven_lock_pips": 1.0,
    "trailing_start_pips": 20.0,
    "trailing_distance_pips": 10.0,
    "trailing_step_pips": 2.0,
    "max_currency_exposure": 3,
    "max_spread_sl_ratio": 0.06,
    "max_confidence": 0.89,
    "entry_hours_utc": null,
    "rank_signals": true,
    "daily_loss_limit_pct": 5.0
  },
  "model_upgrade": {
    "scan_interval_days": 7,
    "max_parameter_size_b": 35,
    "architecture_filter": "coder",
    "max_response_latency_seconds": 60
  },
  "memory": {
    "backend": "chromadb",
    "collection_name": "forex_episodes",
    "similarity_top_k": 5
  }
}
```

### Risk Parameters Reference

| Parameter | Default | Description |
|---|---|---|
| `fixed_lot` | `0.01` | Fixed lot size for all trades (micro-lot) |
| `max_lot_size` | `0.1` | Hard ceiling on calculated lot size |
| `max_drawdown_pct` | `2.0` | Per-trade cap: rejects a trade whose actual loss at SL (lot × SL distance) exceeds this % of balance. Not an account drawdown limit; see `daily_loss_limit_pct` |
| `max_open_trades` | `10` | Maximum simultaneous open positions |
| `confidence_threshold` | `0.80` | Minimum LLM confidence to trigger trade execution |
| `min_sl_pips` | `15.0` | Minimum Stop Loss distance (pips) — prevents spread noise hits |
| `max_spread_pips` | `3.5` | Maximum acceptable broker spread before rejection |
| `symbol_cooldown_minutes` | `30` | Cooldown after a trade closes on a symbol (prevents revenge trading) |
| `news_blackout_pre_mins` | `30` | Minutes before high-impact news to stop trading |
| `news_blackout_post_mins` | `15` | Minutes after high-impact news to resume trading |
| `news_blackout_global_tier1` | `true` | Tier-1 USD releases (NFP, CPI, Fed rate, FOMC) black out all symbols, not only USD pairs |
| `trailing_breakeven_pips` | `15.0` | Pips in profit before auto-moving SL to breakeven |
| `trailing_stop_enabled` | `true` | Master switch for the break-even / trailing manager |
| `trailing_breakeven_lock_pips` | `1.0` | SL is moved to entry + this many pips at break-even |
| `trailing_start_pips` | `20.0` | Profit (pips) at which trailing starts |
| `trailing_distance_pips` | `10.0` | Distance of the trailing SL behind price |
| `trailing_step_pips` | `2.0` | Minimum SL improvement before a trailing modification is sent |
| `max_currency_exposure` | `3` | Max positions containing any single currency |
| `session_profiles.<SESSION>` | see config | `max_open_trades`, `loss_budget_pct`, `max_spread_sl_ratio`, `symbols` per session (ASIA, LONDON, LONDON_NY_OVERLAP, NEW_YORK, ROLLOVER) |
| `session_profiles.grading` | see config | Daily grading thresholds (review window, R thresholds, minimum trades) |
| `max_spread_sl_ratio` | `0.06` | Spread Cost Wall: max live spread as a fraction of the SL distance |
| `max_confidence` | `0.89` | Confidence Calibration Cap: LLM confidence above this is rejected |
| `entry_hours_utc` | `null` | Entry Window Wall: new trades only in `[start, end)` UTC hours; `null` = all sessions |
| `rank_signals` | `true` | Rank each cycle's signals before execution instead of config order |
| `daily_loss_limit_pct` | `5.0` | Max cumulative realized loss (% of balance) before halting all new entries for the UTC day |

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
* Press `Ctrl + Shift + P` → type **`Developer: Reload Window`** → press Enter. All tabs will reload cleanly from disk.

#### 3. MT5 says `Symbol [SYMBOL] not found`:
* In MT5, press `Ctrl + U` (Symbols), find the pair, and double-click it to add it to Market Watch.

#### 4. Ollama circuit breaker activates:
* If you see `Ollama circuit breaker OPENED`, the local LLM server is down. Start Ollama (`ollama serve`) or check GPU memory.
* The circuit breaker auto-resets after 60 seconds — no manual intervention needed.

#### 5. Parse failures on specific pairs:
* The multi-tier parser with synonym normalizer handles 99%+ of LLM output drift.
* Persistent failures on specific pairs may indicate prompt length issues — check `logs/system_errors.log` for raw response previews.

#### 6. Safely stopping the agent:
* Press `Ctrl + C` in the running terminal, or double-click `stop_all.bat`.

#### 7. `learned_rules.md` file growing large:
* **Normal.** The file accumulates auto-generated post-mortem rules over time. Only curated rules (marked with `### [CATEGORY]` headers) are loaded into the LLM prompt. The rest is kept for audit trail only.

---

## 📜 License & Disclaimer
This software is developed for educational, quantitative research, and autonomous algorithmic trading purposes. Forex and CFD trading involve substantial financial risk. Past performance does not guarantee future results. Always test thoroughly on a demo account before risking real capital.

**© 2026 Bubat AI — Built by [syarief02](https://github.com/syarief02)**
