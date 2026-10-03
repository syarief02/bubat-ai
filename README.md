# 🤖 Bubat AI — Local Autonomous Forex AI Agent

> **A self-hosted, private forex trading agent for MetaTrader 5. A local LLM decides *direction* only (BUY / SELL / WAIT); every price, lot size, stop and risk check is deterministic Python. Each trading session runs under its own execution profile, re-graded daily from its own results.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![MetaTrader 5](https://img.shields.io/badge/MetaTrader_5-API-green)](https://www.metatrader5.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple)](https://ollama.com)
[![Supabase](https://img.shields.io/badge/Supabase-Cloud_DB-orange)](https://supabase.com)

---

## 📊 Project Status

**This is an active research project, not a proven profitable system.** Live results over 416 trades (2026-09-28 to 2026-10-02) averaged **−0.29R per trade** (win rate 36%, profit factor 0.49). The daily post-mortem cycle (see [Daily Post-Mortem & Self-Improvement Loop](#-daily-post-mortem--self-improvement-loop)) found where the R was lost: spread cost on wide-spread crosses, trade management, and weak sessions. The current version adds per-session execution profiles, a spread-cost wall, ranked execution and daily session grading in response. **Run it on a demo account.**

---

## 📌 Table of Contents
1. [Architecture & Philosophy](#-architecture--philosophy)
2. [One Trading Cycle, Step by Step](#-one-trading-cycle-step-by-step)
3. [Supported Instruments](#-supported-instruments-28-pairs--gold)
4. [Per-Session Execution](#-per-session-execution)
5. [Deterministic Risk Walls](#-deterministic-risk-walls-14-layers)
6. [Subsystem Deep Dive](#-subsystem-deep-dive)
7. [Daily Post-Mortem & Self-Improvement Loop](#-daily-post-mortem--self-improvement-loop)
8. [Project Structure](#-project-structure)
9. [System Requirements](#-system-requirements)
10. [Installation & Quick Start](#-installation--quick-start)
11. [Configuration Reference (`config.json`)](#-configuration-reference-configjson)
12. [Testing](#-testing)
13. [1-Click Batch Runners](#-1-click-batch-runners)
14. [Troubleshooting](#-troubleshooting)
15. [License & Disclaimer](#-license--disclaimer)

---

## 🏛️ Architecture & Philosophy

LLMs are useful for qualitative judgement: reading a technical picture, weighing news headlines, summarizing a market narrative. They are **not** reliable at arithmetic, so Bubat AI never lets the model produce a number that touches the broker.

```mermaid
flowchart TD
    subgraph Senses ["1. Market Data & Context"]
        A1["MT5 M5 bars + H1 trend label"]
        A2["Live news (SearXNG / RSS WebSurfer)"]
        A3["ForexFactory economic calendar"]
        A4["Episodic memory + curated rules"]
    end

    subgraph Reasoning ["2. Qualitative Tier (local Ollama LLM)"]
        A1 & A2 & A3 & A4 --> C1["Prompt builder"]
        C1 --> C2["agent-brain-8b:8k"]
        C2 --> C3["TradeDecision JSON: BUY / SELL / WAIT + confidence"]
    end

    subgraph Math ["3. Quantitative Tier (deterministic Python)"]
        C3 --> D1["ATR geometry: SL = max(1.5 ATR, 15 pips), TP = 3 ATR"]
        D1 --> D2["Lot sizing + broker constraints"]
    end

    subgraph Rank ["4. Ranked Execution"]
        D2 --> R1["Queue every tradeable signal of the cycle"]
        R1 --> R2["Sort: full H1 trend first, then lowest spread / SL"]
    end

    subgraph Walls ["5. 14 Deterministic Risk Walls"]
        R2 --> W1{"All walls pass? (trend, daily loss, capacity, news, correlation, spread cost, session profile, margin...)"}
        W1 -- No --> X["REJECTED (logged with reason)"]
        W1 -- Yes --> F1["MT5 order_send"]
    end

    subgraph After ["6. Management & Learning"]
        F1 --> G1["Break-even +15p / trailing from +20p (every 15 s)"]
        F1 --> G2["Supabase telemetry"]
        F1 --> G3["Loss reflexion (quarantined candidates)"]
        F1 --> G4["Daily session grading"]
    end
```

### Key Design Principles
| Principle | Implementation |
|---|---|
| **LLM = direction only** | The model outputs BUY / SELL / WAIT, sentiment, confidence and reasoning. It never outputs prices, lots or SL/TP |
| **Math = deterministic Python** | ATR geometry, pip distances, lot sizing and every risk check are plain Python functions |
| **Defense-in-depth** | 14 independent walls must all pass before an order reaches the broker |
| **Fail-safe** | Parse failures, connectivity loss or ambiguity default to WAIT. A wall that errors rejects the trade |
| **Evidence before risk** | New strategies run in shadow (paper) mode first. Anything that adds risk needs the owner's approval |
| **Broker-time correct** | All MT5 history is queried through `core/mt5_time.py` (MT5 stamps deals in broker server time, not UTC) |

---

## 🔁 One Trading Cycle, Step by Step

`main.py` runs one cycle at every **M5 candle close**:

1. **Analyse every symbol (29).** For each one, `main.py` does the following:
   - Skips the symbol if it already has an open position or is in its post-trade cooldown.
   - Pulls M5 technicals and the H1 trend label, live news and similar past episodes.
   - Asks the LLM for a decision.
   - For a BUY or SELL at confidence ≥ `confidence_threshold` (0.80), calculates the deterministic trade parameters and **queues** the signal. Nothing is executed during analysis.
2. **Rank and execute.** Queued signals are sorted best-first (full H1 trend before bias-only, then lowest spread relative to SL), and each goes through the 14 risk walls in that order. This way the limited slots go to the best setups, not to whichever symbol comes first in the config.
3. **Closed trades.** Each close starts the symbol cooldown from the trade's real close time. Each loss produces a reflexion candidate, which is quarantined and never injected into prompts.
4. **Asia shadow strategy.** The Asia range-fade strategy opens and resolves paper trades during 00:00–06:00 UTC. It sends no orders.
5. **Session review.** Once per UTC day, each session is graded from its last 5 days of trades.
6. **Telemetry.** `CYCLE_SUMMARY` (outcomes, parse health, session levels) goes to `logs/agent.log` and Supabase, and a summary table is printed.
7. **Wait for the next candle.** While waiting, the break-even / trailing manager checks open positions **every 15 seconds**.

---

## 🌐 Supported Instruments (28 Pairs + Gold)

| Category | Symbols |
|---|---|
| **7 USD majors** | `EURUSD`, `GBPUSD`, `USDJPY`, `USDCHF`, `AUDUSD`, `NZDUSD`, `USDCAD` |
| **6 EUR crosses** | `EURGBP`, `EURJPY`, `EURCHF`, `EURAUD`, `EURNZD`, `EURCAD` |
| **5 GBP crosses** | `GBPJPY`, `GBPCHF`, `GBPAUD`, `GBPNZD`, `GBPCAD` |
| **4 AUD crosses** | `AUDJPY`, `AUDCHF`, `AUDNZD`, `AUDCAD` |
| **3 NZD crosses** | `NZDJPY`, `NZDCHF`, `NZDCAD` |
| **3 CAD/CHF crosses** | `CADJPY`, `CADCHF`, `CHFJPY` |
| **Metal** | `XAUUSD` (blocked on accounts under \$300) |

Which symbols a session may trade can be narrowed by its session profile (see below).

---

## 🕒 Per-Session Execution

Every session trades, but each has its **own execution profile and its own daily loss budget**, and is **re-graded once per UTC day** from its own results (`core/session_profiles.py`). A trade belongs to the session in which it was **opened**.

| Session | Hours (UTC) | Slots | Loss budget | Spread cap (of SL) |
|---|---|---|---|---|
| `ASIA` | 22:00–07:00 | 4 | 1.0% | 4% |
| `LONDON` | 07:00–12:00 | 10 | 1.75% | 6% |
| `LONDON_NY_OVERLAP` | 12:00–16:00 | 8 | 1.0% | 6% |
| `NEW_YORK` | 16:00–21:00 | 6 | 1.0% | 5% |
| `ROLLOVER` | 21:00–22:00 | 2 | 0.25% | 3% |

These are the **owner baselines** in `config.json → session_profiles`. The loss budgets add up to `daily_loss_limit_pct` (5%), so **a bad Asian session can't use up London's budget**. The global daily loss stop still applies on top.

### Daily Session Grading
Once per UTC day, each session's trades from the last 5 days are scored in R (pips ÷ SL pips):

| Level | Trigger | Effect |
|---|---|---|
| `NORMAL` | Default / recovered | Owner baseline |
| `REDUCED` | ≥ 20 trades and avg R < −0.15 | Half the slots, USD majors only, spread cap ≤ 4% |
| `MINIMAL` | ≥ 20 trades and avg R < −0.30 | 1 slot, USD majors only, spread cap ≤ 3% |

A session moves **up one level per day** once it has ≥ 10 trades at avg R ≥ 0. Grading only ever tightens and restores; it **never goes above the owner baseline**. Levels persist in `state/session_levels.json`, and every review is logged as `SESSION_REVIEW` in `agent.log` and Supabase. Thresholds are in `session_profiles.grading`.

### Asia Range-Fade Strategy (Shadow Mode)
Asian hours are range-bound, so Asia also runs a separate deterministic strategy (`core/session_strategies.py`) that **paper-trades only**:

- **Symbols:** EURUSD, GBPUSD, AUDUSD, USDCAD, USDJPY, 00:00–06:00 UTC.
- **Entries:** BUY when the last M5 close is below the Bollinger(20, 2.0) lower band with RSI(14) < 30. SELL on the mirror setup.
- **Exits:** SL = max(1.5 × ATR, 8 pips), TP = 1R. Every trade is force-closed at 07:00 UTC.
- **Recording:** paper fills happen at the live bid/ask (real spread) and are resolved on M5 bars (SL wins intrabar ties). Results go to `state/asia_shadow_trades.jsonl`.
- **Feed guard:** no paper trades on weekends or when the price feed is stale.

A 155-day backtest found about +0.05R per trade **before costs**, but break-even or negative after a realistic 0.3–1.0 pip cost. It therefore stays in shadow mode until it has **100+ paper trades averaging ≥ +0.05R**, and then needs the owner's approval to go live. A config value of `mode: "live"` is deliberately forced back to shadow.

---

## 🛡️ Deterministic Risk Walls (14 Layers)

`MT5Engine.execute_trade()` checks these **in order**. The first failure rejects the trade with a logged reason. Every rejection is recorded, and the daily report later simulates what rejected signals would have earned.

| # | Wall | Rule |
|---|---|---|
| 1 | **Entry Window** | Optional: new trades only inside `entry_hours_utc` (`null` = all sessions, current setting) |
| 2 | **Confidence Calibration Cap** | Rejects LLM confidence above `max_confidence` (0.89); the top confidence buckets were anti-predictive |
| 3 | **H1 Trend** | Never BUY against a BEARISH H1; never SELL against a BULLISH H1 |
| 4 | **Daily Loss Stop** | Blocks new entries once net realized loss since 00:00 **real UTC** exceeds `daily_loss_limit_pct` (5%) of balance |
| 5 | **Capacity** | Max `max_open_trades` (10) positions account-wide |
| 6 | **Duplicate Position** | One open position per symbol |
| 7 | **News Blackout** | 30 min before to 15 min after high-impact news for either currency. Tier-1 USD releases (NFP, CPI, Fed rate, FOMC) black out **all** symbols |
| 8 | **Currency Correlation** | Max `max_currency_exposure` (3) open positions involving any single currency |
| 9 | **Gold Balance Guard** | No `XAUUSD` on accounts under \$300 |
| 10 | **Per-Trade Risk Cap** | Rejects if the actual loss at SL exceeds `max_drawdown_pct` (2%) of balance |
| 11 | **Spread (pips)** | Rejects if the live spread exceeds `max_spread_pips` (3.5) |
| 12 | **Spread Cost** | Rejects if the live spread exceeds `max_spread_sl_ratio` (6%) of the SL distance |
| 13 | **Session Profile** | Current session's symbol list, slot cap, spread cap and loss budget (see above) |
| 14 | **Margin** | Rejects if free margin is insufficient for the order |

> These walls are plain Python. The LLM cannot override, bypass or modify them.

---

## ⚙️ Subsystem Deep Dive

### 1. Deterministic Trade Geometry (`core/mt5_engine.py`)
* **BUY:** entry at Ask; $\text{SL} = \text{Entry} - \max(1.5 \times \text{ATR}_{14},\ \text{floor})$; $\text{TP} = \text{Entry} + 3.0 \times \text{ATR}_{14}$
* **SELL:** mirror image at Bid.
* **SL floors:** forex **15 pips**, gold **\$3.50**. When the floor widens the SL, the TP is widened to keep at least **1:1.5** R:R.
* **Lot sizing:** `lot_mode: "fixed"` (current) uses `fixed_lot` (0.01). In risk mode, the lot comes from the formula below and is clamped to the broker's `volume_min / volume_step / volume_max` and to `max_lot_size`:

  $$\text{Lot} = \frac{\text{Balance} \times \text{risk\_per\_trade\_pct}/100}{(|\text{Entry} - \text{SL}| / \text{tick size}) \times \text{tick value}}$$

* **Broker constraints:** `TRADE_STOPS_LEVEL` is respected, and a margin check runs before every order.

### 2. Break-Even & Trailing Manager
Runs every **15 seconds** while the agent waits for the next candle:
* **Break-even:** at **+15 pips** profit, the SL moves to entry + 1 pip.
* **Trailing:** from **+20 pips** profit, the SL trails **10 pips** behind price, updated when it can improve by ≥ **2 pips**.
* All values are configurable (`trailing_*` keys). Replaying 412 trades on M5 bars, these settings beat the earlier 10/15-pip settings, but **no management** still came out ahead overall. This is tracked daily in `execution_quality.py`.

### 3. Qualitative LLM Reasoner (`core/agent_logic.py`)
* Local **Ollama** model `agent-brain-8b:8k`, built from `qwen3:8b` with an 8K context (`Modelfile.8b`), sized to run fully on an 8 GB GPU. Temperature 0.1, `num_predict` 800, thinking off (`ollama_think: false`, ~2 s per decision). The chat and assistant read the same `active_model` so Ollama keeps one copy in VRAM.
* The previous model, `agent-brain:32k` (from `qwen2.5-coder:1.5b`, `Modelfile`), is kept for rollback: set `active_model` back to it.
* **Multi-tier JSON parser:**
  1. Strict JSON parse plus synonym normalization (`HOLD`/`FLAT` → `WAIT`, `LONG` → `BUY`, `"82%"` → `0.82`, …).
  2. Repair of unclosed strings and brackets.
  3. Regex field extraction.
  Each decision records its parse tier, number of attempts and fallback flag for telemetry.
* **Circuit breaker:** after 5 consecutive Ollama failures, the agent stops querying for 60 seconds, then retries automatically.
* **Curated rules only** (`learning/rules_loader.py`): `### [CATEGORY]` rules are injected as one-liners, capped at **2,500 chars** on whole-rule boundaries. `REFLEXION` / `COMMUNICATION` rules and anything sourced from loss reflexion are never injected into the trading prompt.

### 4. News & Web Intelligence (`core/sentiment_engine.py`, `core/web_surfer.py`)
* The agent probes SearXNG (`localhost:8080`) once. If it's offline, it falls back to Google News / Investing.com RSS.
* Live headlines per pair in about 0.3–0.7 s, with article context fetched concurrently under strict timeouts. News is never allowed to block a cycle.

### 5. Economic Calendar Filter (`learning/skills/economic_calendar_filter.py`)
* ForexFactory weekly JSON feed with a 2-hour local cache.
* High-impact blackout from 30 min before to 15 min after each event. **Tier-1 USD events** (NFP, CPI, Fed funds rate, FOMC) black out every symbol, not only USD pairs (`news_blackout_global_tier1`).
* Upcoming events are also included in the LLM prompt as context.

### 6. Currency Correlation Filter (`learning/skills/currency_correlation_filter.py`)
Splits each pair into its two currencies (e.g. `GBPJPY` → `GBP` + `JPY`) and blocks a new trade once any currency already appears in `max_currency_exposure` open positions. This stops one currency move from hitting several stops at once.

### 7. Broker Server Time (`core/mt5_time.py`)
MT5 stamps deals, ticks and bars in **broker server time** (e.g. UTC+3), and history queries compare against that clock. Querying with plain UTC silently misses the most recent hours of deals. `mt5_time.py` measures the offset from fresh ticks during market hours, rounds it to whole hours, persists it in `state/mt5_server_offset.json`, and provides the helpers that every history query uses (`history_deals_utc`, `server_epoch_to_utc`). The daily loss stop, cooldowns and reports all rely on it.

### 8. Learning & Memory (`learning/`)
* **Loss reflexion:** each losing trade (with entry, original SL/TP, exit type and hold time) gets an LLM post-mortem. The proposed rule is validated and **quarantined** in `learning/reflexion_candidates.jsonl`. It never goes into `learned_rules.md` or any prompt. Processed tickets persist in `state/processed_tickets.json`, so restarts don't re-reflect old trades.
* **Curated rules:** `learned_rules.md` holds rules added by the daily audit, synced to Supabase `forex_learned_rules`.
* **Episodic memory:** ChromaDB stores past episodes for similarity recall in the prompt.
* **Continuous learner:** captures owner instructions and preferences from chat sessions.

### 9. Supabase Telemetry (`core/supabase_manager.py`)
| Table | Contents |
|---|---|
| `forex_trade_decisions` | Decision, confidence, sentiment, reasoning, trade params, plus metadata: outcome / wall, H1 label, RSI, ATR, LLM parse health, approved / executed flags |
| `forex_executed_trades` | Tickets, order type, lot, entry, SL, TP, execution result (including rejection message) |
| `forex_learned_rules` | Curated rules |
| `ai_agent_telemetry` | `CYCLE_SUMMARY`, `SESSION_REVIEW`, errors and system health |

### 10. WhatsApp Approval Gateway (`core/openclaw_bridge.py`, optional)
With `"auto_approve": false` and a configured number, each trade proposal is sent over WhatsApp, and the trade executes only if `YES` arrives at `localhost:5055/webhook/approval` within `approval_timeout_seconds`. With `"auto_approve": true` (current), trades are approved automatically but still pass every risk wall. Alerts for executions and rejections are sent either way.

### 11. Market Intelligence Chat & Local Assistant
* `chat.py` (`chat.bat`): interactive chat. It can scan all 29 instruments live and report account status (balance, open positions, floating P&L, today's realized P&L). Sessions are logged to `forex_local_agent/logs/chat_sessions.log`, with secrets scrubbed.
  * **Knows the bot:** every prompt carries the latest report's results, the simulated baselines, the brain's last assessment and the open proposals (`brain/chat_support.py`).
  * **Two models:** quick questions use the fast trading model (`qwen3:8b`, about 20 s). "Why / should I / explain / kenapa / patut" questions go to the brain's `gpt-oss:20b` at `chat.deep_think` ("medium"). That takes about 1 minute on weekends (GPU) or a few minutes while trading (CPU). The code gathers the report digest, proposals, brain memory and live technicals for any pair named. Force a mode with `think <question>` or `fast <question>`.
  * **Language:** English by default. It replies in Malay (casual Malaysian) or another language only when you write in it or ask ("reply in Spanish"). The assistant follows the same rule.
  * **Facts computed in code, not read by the model:** `brain/digest.py` `key_facts()` turns the report into plain conclusions, for example "SPREAD wall BLOCKED 86 trades that were never taken: they would have lost money, so the wall avoided losses", or "ROLLOVER: too few trades to judge". `proposal_check()` tests each open proposal against the report. For P3 that reads: a 0.85 threshold skips 92% of trades, and the 31 it keeps still lose. The chat and the brain's reflection both use these checks instead of the brain's own evidence text, which had misread the tables.
  * **Streaming:** deep answers appear as they are written, with a live thinking counter (about 30–50 s on weekends, about 1.5–2 min while trading). News or event questions add the latest headlines as untrusted data.
  * **Owner decisions:** `proposals` lists them. `approve P5` and `reject P5 <reason>` (also `lulus` / `tolak`) ask for y/n confirmation, then run the same gated apply as `brain.bat`. These commands are handled in code: the model has no tool that can approve anything.
* `local_assistant.py` (`assistant.bat`): an autonomous engineer with **full control** of the machine: PowerShell, files, Supabase SQL, MT5 status, market scans, web search and pages, the bot's results and the brain's proposals, and the test suites. The repo-root `local_assistant.py` is a launcher shim for the copy in `forex_local_agent/`.
  * **Reasoning model with native tool calling:** `gpt-oss:20b` at `assistant.think` ("medium"). Like the brain, it runs on the CPU while the market is open and uses the GPU on weekends.
  * **Careful method, built into its prompt and tools:** it inspects before changing, edits exact text (`edit_file` refuses missing or ambiguous matches), syntax-checks every Python and JSON file it writes, re-reads its changes, runs the tests, and reports only what it verified. `run_tests` counts a missing test file or "Ran 0 tests" as a failure, never a pass.
  * **Safety net, not restrictions:** every changed file is backed up to `state/assistant_backups/`, and `undo` restores the last change (`/changes` lists them). Only irreversible actions ask first: recursive delete, force push, `git reset --hard`, stopping processes, `DROP`/`TRUNCATE`, or `DELETE`/`UPDATE` without `WHERE`. Set `assistant.confirm_irreversible: false` for no confirmations at all. Secret values in `.env` are masked when it reads the file.
  * It knows the project layout and the bot's live state, and keeps a long session inside its context window by trimming old tool output first.

---

## 🧬 Daily Post-Mortem & Self-Improvement Loop

The system is improved through a **daily, human-supervised audit**, not by unsupervised self-modification:

```mermaid
flowchart LR
    A["Collect last 24h: MT5 deals, logs, Supabase"] --> B["daily_report.py + execution_quality.py"]
    B --> C["Root causes labelled CONFIRMED / LIKELY / HYPOTHESIS"]
    C --> D{"Adds risk?"}
    D -- No --> E["Implement fix + regression test"]
    D -- Yes --> F["Listed under NEEDS APPROVAL for the owner"]
    E --> G["Tests pass -> restart main.py -> watch cycles"]
    G --> H["Commit, push, append logs/code_evolution.log"]
```

* **Evidence rules:** only CONFIRMED (≥ 30 trades, or repeated in ≥ 2 cycles) or LIKELY findings may change strategy parameters, with at most 2 parameter changes per cycle. Every bug fix gets a regression test.
* **Owner approval required** for anything that adds risk: bigger lots, more open trades, a lower confidence threshold, a smaller SL floor, a loosened wall, new symbols, or account / broker changes.
* **Audit trail:** `forex_local_agent/logs/code_evolution.log` (public, so it contains no account details) records every cycle's metrics, findings, commits and watch-list.

### Bubat Brain: daily research and gated proposals (`brain/`)

Once a day (default 21:15 UTC, and on weekends), the trading loop starts the brain as a separate process so it never delays a trading cycle:

```mermaid
flowchart LR
    A["Observe: 5-day report digest"] --> B["Reflect: qwen3 with thinking on, plus its journal and past proposals"]
    B --> C["Research: web search, pages summarised as untrusted text"]
    B --> D["Propose: config / remove symbol / rule / idea"]
    D --> E{"Code checks: whitelisted key, bounds, 30-trade minimum"}
    E -- refused --> J["Journal"]
    E -- ok --> F["Inbox: waits for the owner"]
    F -- "brain.bat approve P3" --> G["Apply, run 7 test suites"]
    G -- fail --> H["Roll back"]
    G -- pass --> I["Applied; judged later against its before-numbers"]
```

* **Its own reasoning model:** `gpt-oss:20b` (free, open weights) with `think: "high"`, run on the **CPU only** (`num_gpu: 0`) while the market is open, so the 14 GB model never pushes the trading model out of the 8 GB GPU. A run takes about 30–40 minutes. While the market is closed (Friday 17:00 to Sunday 17:00 New York time: 21:00 UTC during US daylight saving time and 22:00 UTC otherwise, which is Saturday to Monday 05:00 or 06:00 in Malaysia. The shift is automatic), `gpu_when_market_closed: true` lets it use the GPU as well, which is about 3x faster. This is decided for each call. It was chosen with `maintenance/brain_benchmark.py`, which scores a data-reading quiz with code-computed answers, a small-sample trap, and proposal validity. It was the only model to score 5/5 on the quiz, beating `qwen3:8b`, `qwen3:30b` and `gpt-oss:20b` at medium reasoning.
* **It never changes anything by itself.** Each proposal waits in `state/brain/inbox.md` until the owner runs `brain.bat approve <id>` or `brain.bat reject <id>`. Config changes take effect when the agent restarts.
* **Code, not the LLM, decides what is allowed.** Only a fixed list of risk settings can be tuned, each within hard bounds: confidence threshold, open-trade slots, spread caps, cooldown, news blackout, currency exposure, and per-session slots and loss budgets. Code labels each change SAFER or RISKIER. Lot size, SL/TP, the daily loss limit, auto-approve and adding symbols are never tunable.
* **No acting on noise.** Config, symbol and rule changes are refused when they rest on fewer than 30 trades.
* **It learns over time.** Every observation, reflection, research finding and owner decision goes to `state/brain/journal.jsonl`. The next reflection reads its recent memory, the open proposals, the rejected ones, and the applied ones with their before-numbers.
* **Web content is data, never instructions.** Search results are summarised with an "untrusted content" prompt. Sources are the URLs actually fetched, not what the model claims.
* `idea` proposals are code or strategy changes for the owner (or a coding agent) to implement. The brain cannot edit code.

### Reports
```powershell
python forex_local_agent/maintenance/daily_report.py --hours 24
```
Outputs win rate, payoff, profit factor, expectancy and peak-to-trough drawdown. It breaks results down by symbol, direction, session, hour, confidence, exit type, spread and H1 alignment, and also reports:
- cost drag, exposure and LLM health;
- risk-wall counts;
- **counterfactual R** for rejected signals, by wall;
- **LLM vs follow-H1-trend vs always-WAIT** baselines.

Simulated entries use the next M5 bar's open, so there is no lookahead.

```powershell
python forex_local_agent/maintenance/execution_quality.py --hours 24
```
Breaks down where R is lost between signal and result:
- real fills vs a shadow entry one bar later;
- managed vs unmanaged R;
- R by spread/SL bucket;
- quick same-symbol re-entries;
- a **per-session breakdown** with current grading levels;
- the Asia shadow strategy's progress toward its promotion bar.

Both scripts are read-only and save JSON to `forex_local_agent/reports/` (gitignored).

### Health Check (is the bot working?)
`health_check.bat` (or `python forex_local_agent/maintenance/health_check.py --hours 24`) checks whether the bot **works**, not whether it is profitable. Each check reports OK, WARN or FAULT:
- **agent:** the trading agent is running, and not silent while the market is open;
- **activity:** pairs were analysed while the market was open, with no gaps over 30 min, and nothing ran while it was closed;
- **errors:** ERROR lines in `system_errors.log` (failed orders count most);
- **ollama:** the server answers and the trading model is installed;
- **mt5:** the terminal is connected and Algo Trading is on;
- **positions:** every open bot position has SL and TP, lots are within limits, and no more trades are open than allowed;
- **loss_wall:** no trade was opened after the day's loss passed `daily_loss_limit_pct`;
- **brain:** the brain ran recently. **disk:** free space.

Windows Task Scheduler runs it daily at 08:00 and 20:00 (task "Bubat AI Health Check"). The window closes by itself when everything is OK and stays open on WARN/FAULT. The chat shows the latest result when it starts, and the result is also saved in `state/health/latest.json` and `logs/health.log`. Exit code: 0 OK, 1 WARN, 2 FAULT.

---

## 📁 Project Structure

```
bubat AI/
├── README.md                          # This document
├── run_agent.bat                      # Start the trading loop (main.py)
├── chat.bat                           # Market intelligence chat
├── assistant.bat                      # Local tool-using assistant
├── stop_all.bat                       # Stop Ollama + all agent processes
├── brain.bat                          # Brain inbox; approve / reject proposals
├── health_check.bat                   # Is the bot working? (scheduled 08:00 and 20:00)
├── local_assistant.py                 # Launcher shim -> forex_local_agent/local_assistant.py
├── .env                               # Secrets (gitignored)
└── forex_local_agent/
    ├── main.py                        # Orchestrator: analyse -> rank -> execute -> learn -> wait
    ├── config.json                    # All risk, session, strategy and model settings
    ├── Modelfile.8b                   # Ollama model definition: qwen3:8b, 8K context (active)
    ├── Modelfile                      # Previous model: qwen2.5-coder:1.5b, 32K context (rollback)
    ├── requirements.txt
    ├── chat.py / chat_logger.py       # Interactive chat + secret-scrubbed session log
    ├── local_assistant.py             # Tool-using local assistant
    ├── core/
    │   ├── mt5_engine.py              # MT5 API, trade geometry, 14 risk walls, trailing manager
    │   ├── mt5_time.py                # Broker server-time offset + UTC-correct history queries
    │   ├── trade_analytics.py         # Shared pure helpers: sessions, exits, ranking, wall labels, sims
    │   ├── session_profiles.py        # Per-session profiles, loss budgets, daily grading
    │   ├── session_strategies.py      # Asia range-fade strategy (shadow / paper only)
    │   ├── agent_logic.py             # LLM prompt, multi-tier parser, circuit breaker
    │   ├── market_scanner.py          # 29-instrument technical scanner
    │   ├── sentiment_engine.py        # News aggregation (SearXNG + RSS fallback)
    │   ├── web_surfer.py              # Fast live news / web fetcher
    │   ├── openclaw_bridge.py         # WhatsApp approvals, alerts, webhook :5055
    │   └── supabase_manager.py        # Supabase telemetry
    ├── brain/                         # Daily research + gated proposals (python -m brain)
    │   ├── agent.py                   # observe -> reflect -> research -> propose
    │   ├── proposals.py               # Whitelist, bounds, risk labels, apply + gates + rollback
    │   ├── digest.py / journal.py / llm.py
    ├── learning/
    │   ├── rules_loader.py            # Curated, capped rule injection
    │   ├── reflexion_store.py         # Reflexion validator + quarantine + processed tickets
    │   ├── continuous_learner.py      # Owner instruction / preference capture
    │   ├── memory_manager.py          # ChromaDB episodic memory
    │   ├── skill_factory.py           # Skill code generator
    │   ├── learned_rules.md           # Curated rules
    │   └── skills/
    │       ├── economic_calendar_filter.py
    │       ├── currency_correlation_filter.py
    │       └── template_skill.py
    ├── maintenance/
    │   ├── daily_report.py            # 24h post-mortem metrics, counterfactuals, baselines
    │   ├── execution_quality.py       # Entry timing, management, spread, per-session breakdown
    │   ├── health_check.py            # OK / WARN / FAULT checks that the bot works
    │   └── model_updater.py           # Weekly model discovery & hot-swap
    ├── tests/                         # Offline regression suites + sandboxed mock cycle
    ├── logs/                          # agent.log, trades.log, system_errors.log, chat_sessions.log (gitignored)
    │   └── code_evolution.log         # Public audit trail of every improvement cycle
    ├── state/                         # Runtime state (gitignored): server offset, processed tickets,
    │                                  #   session levels, Asia shadow trades, brain journal/proposals/inbox
    └── reports/                       # Report JSON output (gitignored)
```

---

## 🛠️ System Requirements

| Component | Minimum | Tested |
|---|---|---|
| **OS** | Windows 10/11 64-bit | Windows 11 64-bit |
| **GPU** | NVIDIA GTX 1660 (6 GB) | NVIDIA RTX 4060 (8 GB) |
| **RAM** | 16 GB | 32 GB |
| **Python** | 3.10+ | 3.13 64-bit |
| **Broker terminal** | MetaTrader 5 | MetaTrader 5 (Tickmill, server UTC+2/+3) |

| Service | Role | Address |
|---|---|---|
| **Ollama** | Local LLM inference | `localhost:11434` |
| **MetaTrader 5** | Market data & execution | Local terminal (IPC) |
| **SearXNG** *(optional)* | News search | `localhost:8080` |
| **Supabase** | Cloud telemetry | HTTPS |
| **OpenClaw** *(optional)* | WhatsApp approval webhook | `localhost:5055` |

---

## 🚀 Installation & Quick Start

**1. Clone**
```powershell
git clone https://github.com/syarief02/bubat-ai.git "bubat AI"
cd "bubat AI"
```

**2. Install dependencies**
```powershell
cd forex_local_agent
python -m pip install -r requirements.txt
```

**3. Create `.env`** in the repo root (it is gitignored, never commit it):
```ini
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres
GEMINI_API_KEY=your_key   # optional
```

**4. Build the local model**
```powershell
ollama pull qwen3:8b
cd forex_local_agent
ollama create agent-brain-8b:8k -f Modelfile.8b
```

**5. Prepare MetaTrader 5**
- Log in to your broker account (**use a demo account first**).
- Under **Tools → Options → Expert Advisors**, enable **Allow Algo Trading**. Make sure the **Algo Trading** toolbar button is on.
- In **Market Watch** (`Ctrl+U`), show all 28 pairs and `XAUUSD`.

**6. Run the tests, then start the agent**
```powershell
python forex_local_agent/tests/test_session_profiles.py
.\run_agent.bat
```
The agent connects to MT5 and runs its first cycle immediately, then one cycle per M5 candle close.

---

## ⚙️ Configuration Reference (`config.json`)

All settings live in `forex_local_agent/config.json`. The agent reads them at startup, so restart it after changes.

### Risk & Execution (`risk_parameters`)
| Key | Current | Meaning |
|---|---|---|
| `lot_mode` / `fixed_lot` | `fixed` / `0.01` | Fixed micro-lots; risk mode sizes lots from `risk_per_trade_pct` |
| `max_lot_size` | `0.1` | Hard ceiling on any lot |
| `max_drawdown_pct` | `2.0` | Per-trade cap: actual loss at SL ≤ this % of balance |
| `daily_loss_limit_pct` | `5.0` | Daily net realized loss stop (real UTC day) |
| `max_open_trades` | `10` | Account-wide open positions |
| `confidence_threshold` | `0.80` | Minimum LLM confidence to act |
| `max_confidence` | `0.89` | Confidence calibration cap |
| `auto_approve` | `true` | Skip WhatsApp approval (walls still apply) |
| `approval_timeout_seconds` | `300` | WhatsApp approval window |
| `min_sl_pips` | `15.0` | Forex SL floor |
| `max_spread_pips` | `3.5` | Absolute spread limit |
| `max_spread_sl_ratio` | `0.06` | Spread cost limit as a fraction of SL distance |
| `symbol_cooldown_minutes` | `30` | No re-entry on a symbol for this long after a close |
| `news_blackout_pre_mins` / `_post_mins` | `30` / `15` | News blackout window |
| `news_blackout_global_tier1` | `true` | Tier-1 USD events black out all symbols |
| `max_currency_exposure` | `3` | Max open positions per currency |
| `entry_hours_utc` | `null` | Optional global entry window `[start, end)`; `null` = all sessions |
| `rank_signals` | `true` | Rank each cycle's signals before executing |
| `trailing_stop_enabled` | `true` | Break-even / trailing manager on/off |
| `trailing_breakeven_pips` / `_lock_pips` | `15.0` / `1.0` | Break-even trigger and lock |
| `trailing_start_pips` / `_distance_pips` / `_step_pips` | `20.0` / `10.0` / `2.0` | Trailing start, distance, minimum step |

### Sessions (`session_profiles`)
One block per session (`ASIA`, `LONDON`, `LONDON_NY_OVERLAP`, `NEW_YORK`, `ROLLOVER`), each with:
- `max_open_trades`
- `loss_budget_pct`
- `max_spread_sl_ratio`
- `symbols` (`null` = all)

`grading` holds the daily grading thresholds (`review_days`, `reduce_below_r`, `minimal_below_r`, `min_trades_down`, `min_trades_up`, `restore_at_or_above_r`). `"enabled": false` turns per-session profiles off.

### Session Strategies (`session_strategies.asia_range_fade`)
`enabled`, `mode` (shadow only), `hours_utc`, `exit_hour_utc`, `symbols`, `bb_period`, `bb_dev`, `rsi_low`, `sl_atr`, `sl_floor_pips`, `tp_r`, `promote_after_trades`, `promote_min_avg_r`.

### Other Sections
`trading` (symbols, `timeframe: "M5"`, bars, order comment), `active_model` and `ollama_think` (the model used by the agent, chat and assistant; `false` skips qwen3's hidden reasoning), `model_upgrade` (weekly model scan; off unless `enabled: true`, since it swaps in any model passing a trivial audition), `assistant` (`model`, `think`, `max_steps`, `confirm_irreversible`), `brain` (`enabled`, `daily_run_utc`, `report_hours` window, `model`, `think` (true/false or low/medium/high), `num_predict`, `num_ctx`, `num_gpu` (0 = CPU only), `gpu_when_market_closed`, research/proposal limits), `memory` (ChromaDB collection, `similarity_top_k`), `alerts`, and service URLs (`ollama_base_url`, `searxng_url`).

---

## 🧪 Testing

**Offline regression suites.** MT5 is mocked and `order_send` is blocked, so they can't touch an account:
```powershell
python forex_local_agent/tests/test_session_profiles.py     # session slots, spread cap, loss budgets, daily grading
python forex_local_agent/tests/test_session_strategies.py   # Asia range-fade signal, paper fills, weekend/stale guard
python forex_local_agent/tests/test_execution_upgrade.py    # ranked execution, confidence cap, spread cost wall, no-lookahead sim
python forex_local_agent/tests/test_cycle5_regressions.py   # rules loader, reflexion quarantine, broker time, tier-1 blackout, chatbot
python forex_local_agent/tests/test_daily_loss_stop.py
python forex_local_agent/tests/test_chat_logger.py
python forex_local_agent/tests/test_market_hours.py       # pause while the FX market is closed
python forex_local_agent/tests/test_brain.py              # brain validation, gates/rollback, cycle, web parsing
python forex_local_agent/tests/test_chat_upgrade.py       # chat language, deep routing, bot status, owner decisions
python forex_local_agent/tests/test_assistant.py          # assistant edits, backups/undo, run_tests honesty, irreversible checks, loop
```

**Sandboxed end-to-end cycle.** `test_mock_cycle.py` uses the **live MT5 terminal** for data, with `order_send` patched so nothing reaches the broker:
```powershell
python forex_local_agent/tests/test_mock_cycle.py
```
It covers MT5 connection, bars and H1 trend, news and calendar, the LLM parse, ATR math, trailing logic, the risk walls and the correlation wall.

---

## ⚡ 1-Click Batch Runners

| Runner | What it does |
|---|---|
| `run_agent.bat` | Starts Ollama if needed, then runs the trading loop (`main.py`) |
| `chat.bat` | Chat about the market and your bot: live MT5 scans, deep reasoning for why/should questions, `approve` / `reject` proposals |
| `assistant.bat` | Autonomous engineer with full machine control (reasoning model, verify-every-change, `undo`) |
| `brain.bat` | Shows the brain's latest assessment and proposals; `brain.bat approve P3`, `brain.bat reject P3 reason`, `brain.bat run --force` |
| `stop_all.bat` | Stops Ollama and all agent processes |

Keep the `run_agent.bat` window open: **closing it stops the agent.**

---

## ❓ Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Waiting Xs for next M5 candle close...` | Normal. The agent runs one cycle per candle and manages trailing stops while it waits |
| Many `REJECTED: Session ...` alerts | The session is at `REDUCED` / `MINIMAL` level or has used its loss budget. Check the latest `SESSION_REVIEW` in `logs/agent.log` or the `sessions` section of `execution_quality.py` |
| `REJECTED: Daily loss limit reached` all day | The 5% daily loss stop has been hit. It resets at 00:00 UTC; open positions keep their SL/TP and trailing |
| `REJECTED: Spread cost ...` outside liquid hours | Expected: evening and rollover spreads are often 10–20% of a 15-pip SL |
| `Ollama circuit breaker OPENED` | The Ollama server is down. Start `ollama serve` or free GPU memory; the breaker retries after 60 s |
| `Symbol XXX not found` | Add the symbol in MT5 Market Watch (`Ctrl+U`) |
| Agent stopped without an error in the log | The console window was closed; restart with `run_agent.bat` |
| Daily numbers look shifted by a few hours | Broker server time: check `state/mt5_server_offset.json` (it should match the broker's UTC offset, e.g. 10800 = UTC+3) |

Stop the agent safely with `Ctrl+C` in its window, or with `stop_all.bat`.

---

## 📜 License & Disclaimer

This software is for education and quantitative research. Forex and CFD trading carry substantial risk of loss, and **live results so far have been negative** (see [Project Status](#-project-status)). Past performance does not guarantee future results. Test on a demo account before risking real capital; you are solely responsible for any trades it places.

**© 2026 Bubat AI — Built by [syarief02](https://github.com/syarief02)**
