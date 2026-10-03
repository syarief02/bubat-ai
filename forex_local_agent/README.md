# 🤖 `forex_local_agent` — Developer Guide

This folder is the Bubat AI trading agent package. **For the full system documentation, see the [root README](../README.md):**
- architecture and the per-cycle flow;
- per-session execution and daily grading;
- the 14 risk walls;
- the daily post-mortem loop;
- installation and the configuration reference.

This page covers working *inside* the package: what each module does, where runtime files go, and how to check components safely.

> ⚠️ **Live-account safety:** `main.py`, `ForexAgent.run_analysis_cycle()` and `MT5Engine.execute_trade()` can place **real orders** on whatever account the MT5 terminal is logged into. Use a demo account, and use only the read-only checks and offline tests below while developing.

---

## 📁 Module Map

| Path | Responsibility |
|---|---|
| `main.py` | `ForexAgent`: analyse all symbols → rank queued signals → execute through the risk walls → closed-trade handling → Asia shadow → daily session review → wait for the next M5 close (trailing checks every 15 s) |
| `config.json` | Every setting: risk parameters, `session_profiles`, `session_strategies`, symbols, model, services. Read at startup |
| `Modelfile.8b` | Active Ollama model `agent-brain-8b:8k` (from `qwen3:8b`, 8K context, fits an 8 GB GPU) |
| `Modelfile` | Previous model `agent-brain:32k` (from `qwen2.5-coder:1.5b`), kept for rollback |
| `core/mt5_engine.py` | MT5 connection, technicals + H1 trend label, ATR trade geometry, the 14 risk walls (`execute_trade`), daily loss stop, closed-trade enrichment, break-even / trailing manager |
| `core/mt5_time.py` | Broker server-time offset and UTC-correct history helpers. **Every MT5 history query must use these** |
| `core/trade_analytics.py` | Pure helpers shared by the engine and the reports: sessions, exit classification, spread/SL ratio, entry windows, H1 strength, signal ranking, wall labels, path simulation |
| `core/session_profiles.py` | `SessionManager`: per-session symbols, slots, spread cap, loss budget; daily `NORMAL / REDUCED / MINIMAL` grading |
| `core/session_strategies.py` | `AsiaRangeFadeShadow`: Asia range-fade paper trading (never sends orders) |
| `core/agent_logic.py` | LLM prompt, multi-tier JSON parser, synonym normalizer, circuit breaker, `last_decision_meta` |
| `core/market_scanner.py` | 29-instrument technical scan and opportunity ranking (used by chat) |
| `core/sentiment_engine.py`, `core/web_surfer.py` | Live news: SearXNG probe, RSS fallback, concurrent article fetch |
| `core/openclaw_bridge.py` | WhatsApp proposals and alerts, approval webhook on `:5055` |
| `core/supabase_manager.py` | Decisions, executions, rules and telemetry to Supabase |
| `brain/` | Daily self-improvement agent: report digest → reflection → web research → gated proposals. `python -m brain status/run/list/approve/reject/journal` |
| `learning/rules_loader.py` | Curated rule selection for prompts (2,500-char cap for trading) |
| `learning/reflexion_store.py` | Loss-reflexion validation, quarantine, persisted processed tickets |
| `learning/continuous_learner.py`, `memory_manager.py`, `skill_factory.py` | Instruction capture, ChromaDB episodic memory, skill generation |
| `learning/skills/` | Economic calendar blackout and currency correlation filters (used as risk walls) |
| `maintenance/daily_report.py` | 24h post-mortem: metrics, breakdowns, wall counterfactuals, LLM vs baselines |
| `maintenance/execution_quality.py` | Entry timing, management, spread cost, per-session breakdown, Asia shadow progress |
| `maintenance/model_updater.py` | Weekly model discovery and hot-swap |
| `chat.py`, `chat_logger.py` | Market chat with live scans and account status; secret-scrubbed session log |
| `local_assistant.py` | Tool-using local assistant (the repo-root `local_assistant.py` is a launcher shim for this file) |

---

## 🗂️ Runtime Files

| Location | Contents | In git? |
|---|---|---|
| `logs/agent.log` | Full INFO log incl. `Cycle stats` and `SESSION_REVIEW` | No |
| `logs/trades.log`, `logs/system_errors.log` | Trade results; ERROR traces | No |
| `logs/chat_sessions.log` | Chat / assistant transcripts (secrets scrubbed) | No |
| `logs/code_evolution.log` | Public audit trail of improvement cycles (contains no account details) | **Yes** |
| `state/mt5_server_offset.json` | Measured broker UTC offset (e.g. `10800` = UTC+3) | No |
| `state/processed_tickets.json` | Closed tickets already reflected on | No |
| `state/session_levels.json` | Current session grading levels + last review | No |
| `state/asia_shadow_open.json`, `state/asia_shadow_trades.jsonl` | Asia paper trades (open / resolved) | No |
| `learning/reflexion_candidates.jsonl` | Quarantined reflexion rules (never injected) | No |
| `reports/` | JSON from `daily_report.py` and `execution_quality.py` | No |
| `state/brain/` | Brain `journal.jsonl`, `proposals.json`, `inbox.md` | No |
| `logs/brain.log` | Output of the daily brain run started by `main.py` | No |

To reset the session grading, stop the agent and delete `state/session_levels.json`. All sessions start from `NORMAL` and are re-graded on the first cycle.

---

## 🚀 Running

From the repo root, use `run_agent.bat`, which starts Ollama if needed. Or run it directly:
```powershell
cd forex_local_agent
python main.py
```
On startup, the agent:
1. starts the approval webhook on port 5055;
2. connects to MT5;
3. runs one full cycle immediately;
4. then runs one cycle per **M5** candle close.

While the FX market is closed (Friday 17:00 to Sunday 17:00 New York time: 21:00 UTC during US daylight saving time and 22:00 UTC otherwise, which is Saturday to Monday 05:00 or 06:00 in Malaysia. The shift is automatic) it does not analyse or trade. It logs one `FX market closed` line, then re-checks every 15 minutes until the open.

Stop it with `Ctrl+C`. Closing the console window also stops it, without a shutdown log line.

---

## 🔍 Read-Only Component Checks

Run these from `forex_local_agent/`. None of them place orders:

```powershell
# MT5 connection + M5 technicals with the H1 trend label
python -c "from core.mt5_engine import MT5Engine; e = MT5Engine('config.json'); e.initialize(); print(e.get_technical_data('EURUSD', 'M5')); e.shutdown()"

# Broker server-time offset (seconds; 10800 = UTC+3)
python -c "import MetaTrader5 as mt5; mt5.initialize(); from core.mt5_time import get_server_utc_offset_seconds as g; print(g(force=True)); mt5.shutdown()"

# Live news for one pair (SearXNG or RSS fallback)
python -c "import asyncio; from core.sentiment_engine import SentimentEngine; print(asyncio.run(SentimentEngine('config.json').get_live_news('EURUSD')))"

# Ollama reachability
python -c "import asyncio; from core.agent_logic import AgentLogic; print(asyncio.run(AgentLogic('config.json').query_ollama('Ping test', 'You are an AI assistant.')))"

# Current session profiles and grading levels
python -c "import json; from core.session_profiles import SessionManager, SESSIONS; m = SessionManager(json.load(open('config.json'))); [print(s, m.profile(s)) for s in SESSIONS]"

# Reports (read-only, JSON saved to reports/)
python maintenance/daily_report.py --hours 24
python maintenance/execution_quality.py --hours 24
```

---

## 🧪 Tests

**Offline suites.** MT5 is mocked and `order_send` is blocked:
```powershell
python tests/test_session_profiles.py
python tests/test_session_strategies.py
python tests/test_execution_upgrade.py
python tests/test_cycle5_regressions.py
python tests/test_daily_loss_stop.py
python tests/test_chat_logger.py
python tests/test_market_hours.py
python tests/test_brain.py
python tests/test_chat_upgrade.py
```

**Sandboxed end-to-end cycle.** It uses the live terminal for data, with `order_send` patched:
```powershell
python tests/test_mock_cycle.py
```

---

## 🧭 Development Rules

These are the rules the daily post-mortem cycle follows. Please keep to them in manual changes too:
- **The LLM picks direction only.** Prices, lots, SL/TP and risk checks stay in deterministic Python.
- **Use `core/mt5_time.py` for MT5 history.** Never pass plain UTC datetimes to `history_deals_get` / `copy_ticks_range`.
- **Risk walls fail closed.** A check that raises must reject the trade.
- **Prove new strategies in shadow mode first.** Anything that adds risk (bigger lots, more open trades, a lower threshold, a smaller SL floor, a loosened wall, new symbols) needs the owner's approval.
- **Bug fixes need tests.** Every fix gets an offline regression test, and no test may send a real order.
- **No secrets in the repo.** No credentials or account details in tracked files or in `logs/code_evolution.log`. Secrets belong in the repo-root `.env`, which is gitignored.
