"""
Master Orchestration Loop - Local Autonomous Forex AI Agent
============================================================
Coordinates the complete operational cycle:
1. Monitors H1 candle closes on the MT5 terminal clock.
2. Runs fundamental + technical multi-agent analysis via Ollama.
3. Enforces deterministic risk wall before any trade execution.
4. Handles crash resilience, reflexion learning, and model upgrades.
"""

import asyncio
import json
import signal
import sys
import threading
import schedule
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
from pathlib import Path
from loguru import logger

# Ensure agent directory is in sys.path
agent_root = str(Path(__file__).resolve().parent)
if agent_root not in sys.path:
    sys.path.insert(0, agent_root)

from core.mt5_engine import MT5Engine
from core.sentiment_engine import SentimentEngine
from core.agent_logic import AgentLogic, TradeDecision
from core.openclaw_bridge import OpenClawBridge, start_webhook_server
from learning.memory_manager import MemoryManager
from learning.skill_factory import SkillFactory
from maintenance.model_updater import ModelUpdater
from core.supabase_manager import SupabaseManager
from learning.reflexion_store import ReflexionStore, validate_reflexion_rule
from core.trade_analytics import wall_from_message, rank_signals

# ── Logging Configuration ────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
logger.add("logs/system_errors.log", rotation="10 MB", level="ERROR", backtrace=True, diagnose=True)
logger.add("logs/agent.log", rotation="10 MB", retention=5, level="INFO", encoding="utf-8",
           filter=lambda record: "trade" not in record.get("extra", {}))
logger.add("logs/trades.log", rotation="10 MB", level="INFO", filter=lambda record: "trade" in record.get("extra", {}))


class ForexAgent:
    """Master orchestrator for the autonomous forex trading agent."""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.running = True

        # Load config
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            logger.critical(f"Failed to load config: {e}")
            raise SystemExit(1)

        # ── Initialize subsystems ─────────────────────────────────────────
        self.mt5_engine = MT5Engine(str(self.config_path))
        self.sentiment_engine = SentimentEngine(str(self.config_path))
        self.agent_logic = AgentLogic(str(self.config_path))
        self.openclaw = OpenClawBridge(str(self.config_path))
        self.memory = MemoryManager(str(self.config_path))
        self.skill_factory = SkillFactory(str(self.config_path))
        self.model_updater = ModelUpdater(str(self.config_path))
        self.supabase = SupabaseManager()

        # Trading parameters from config
        self.symbols = self.config.get("trading", {}).get("symbols", ["EURUSD"])
        self.timeframe = self.config.get("trading", {}).get("timeframe", "H1")
        self.confidence_threshold = self.config.get("risk_parameters", {}).get("confidence_threshold", 0.80)
        self.approval_timeout = self.config.get("risk_parameters", {}).get("approval_timeout_seconds", 300)
        self.max_open_trades = self.config.get("risk_parameters", {}).get("max_open_trades", 10)
        # Rank each cycle's tradeable signals before execution instead of first-come config order
        self.rank_signals = self.config.get("risk_parameters", {}).get("rank_signals", True)

        # Symbol cooldown management to prevent revenge-trading
        self.symbol_cooldowns: dict[str, datetime] = {}
        self.cooldown_minutes = self.config.get("risk_parameters", {}).get("symbol_cooldown_minutes", 30)
        # Persisted across restarts so a restart never re-runs reflexion on old losses
        self.reflexion_store = ReflexionStore(Path(__file__).resolve().parent)
        self.processed_closed_tickets: set[int] = self.reflexion_store.load_processed_tickets()
        self.cycle_stats: Dict[str, Any] = {}

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        logger.info(f"Signal {sig} received, initiating graceful shutdown...")
        self.running = False

    # ── Core Analysis Cycle ───────────────────────────────────────────────

    async def run_analysis_cycle(self, symbol: str) -> TradeDecision | None:
        """
        Execute the full H1 candle analysis cycle for a single symbol.

        Steps:
            a. Load learned_rules.md constraints
            b. Get technical data from MT5
            c. Get live news from sentiment engine
            d. Query memory for similar historical conditions
            e. Send combined payload to agent logic (Ollama)
            f. Validate response with Pydantic TradeDecision
            g. If decision != WAIT and confidence >= threshold:
               - Send WhatsApp proposal via OpenClaw
               - Wait for approval (5 min timeout)
               - If approved: enforce risk wall and execute via MT5
            h. Store episode in memory
            i. Return the decision
        """
        logger.info(f"═══ Starting analysis cycle for {symbol} ═══")
        try:
            # 1. Skip if position is already open on this symbol
            open_positions = self.mt5_engine.get_open_positions()
            open_pos = [p for p in open_positions if p.get("symbol") == symbol]
            if open_pos:
                p = open_pos[0]
                ptype = "BUY" if p.get("type") == 0 else "SELL"
                profit = p.get("profit", 0.0)
                profit_str = f"+${profit:.2f}" if profit >= 0 else f"-${abs(profit):.2f}"
                logger.info(f"[{symbol}] Active position already open in MT5 ({ptype} {p.get('volume')} lot, {profit_str}). Skipping cycle.")
                return {
                    "symbol": symbol,
                    "decision": ptype,
                    "confidence": None,
                    "sentiment": "OPEN",
                    "h1_trend": "-",
                    "status": f"ACTIVE #{p.get('ticket')} ({ptype} {p.get('volume')} lot, {profit_str})"
                }

            # 2. Skip if symbol is in post-trade cooldown period
            now_utc = datetime.now(timezone.utc)
            if symbol in self.symbol_cooldowns:
                if now_utc < self.symbol_cooldowns[symbol]:
                    remaining_mins = max(1, int((self.symbol_cooldowns[symbol] - now_utc).total_seconds() / 60))
                    logger.info(f"[{symbol}] In cooldown period ({remaining_mins}m remaining). Skipping to prevent overtrading.")
                    return {
                        "symbol": symbol,
                        "decision": "COOLDOWN",
                        "confidence": None,
                        "sentiment": "WAIT",
                        "h1_trend": "-",
                        "status": f"COOLDOWN ({remaining_mins}m remaining)"
                    }
                else:
                    del self.symbol_cooldowns[symbol]

            # a. Learned rules are loaded (curated + capped) inside AgentLogic.get_trade_decision

            # b. Get technical data from MT5
            logger.info(f"[{symbol}] Fetching technical data...")
            tech_data = self.mt5_engine.get_technical_data(symbol, timeframe=self.timeframe)
            if not tech_data:
                logger.warning(f"[{symbol}] No technical data available, skipping cycle.")
                return {
                    "symbol": symbol,
                    "decision": "ERROR",
                    "confidence": None,
                    "sentiment": "N/A",
                    "h1_trend": "-",
                    "status": "ERROR (No MT5 rates)"
                }

            # c. Get live news from sentiment engine
            logger.info(f"[{symbol}] Fetching live news...")
            news_data = await self.sentiment_engine.get_live_news(symbol)

            # d. Query memory for similar historical conditions
            logger.info(f"[{symbol}] Querying episodic memory...")
            similar_episodes = await self.memory.query_similar(
                {"symbol": symbol, "tech": tech_data, "news_headlines": news_data.get("headlines", [])},
                top_k=self.config.get("memory", {}).get("similarity_top_k", 5),
            )
            memory_strings = self.memory._format_episode_for_prompt(similar_episodes)

            # e+f. Send to Ollama and validate with Pydantic (Qualitative only)
            logger.info(f"[{symbol}] Requesting qualitative decision from LLM...")
            decision: TradeDecision = await self.agent_logic.get_trade_decision(
                technical_data=tech_data,
                news_data=news_data,
                memories=memory_strings,
            )
            logger.info(
                f"[{symbol}] Decision: {decision.decision} | "
                f"Confidence: {decision.confidence_score:.0%} | "
                f"Sentiment: {decision.market_sentiment}"
            )

            # g. Actionable signal -> deterministic trade parameters
            trade_params = None
            if decision.decision != "WAIT" and decision.confidence_score >= self.confidence_threshold:
                logger.info(f"[{symbol}] Signal meets threshold. Calculating deterministic trade parameters...")
                trade_params = self.mt5_engine.calculate_trade_parameters(
                    symbol=symbol,
                    decision=decision.decision
                )
                if trade_params.get("status") == "error":
                    logger.error(f"[{symbol}] Parameter calculation error: {trade_params.get('message')}")
                    await self.openclaw.send_alert(
                        f"Parameter calculation failed for {symbol}: {trade_params.get('message')}", level="WARNING"
                    )
                else:
                    trade_params["confidence"] = decision.confidence_score

            ctx = {"symbol": symbol, "decision": decision, "tech_data": tech_data,
                   "news_data": news_data, "trade_params": trade_params}
            if self.rank_signals and trade_params and trade_params.get("status") != "error":
                # Defer: the cycle ranks every tradeable signal and fills free slots best-first
                ctx["h1_trend"] = tech_data.get("higher_timeframe_h1", "")
                ctx["spread_sl_ratio"] = self.mt5_engine.current_spread_sl_ratio(
                    symbol, trade_params.get("sl_distance") or abs(trade_params["entry"] - trade_params["sl"])
                )
                ratio = ctx["spread_sl_ratio"]
                logger.info(f"[{symbol}] Signal queued for ranked execution "
                            f"(H1: {ctx['h1_trend'] or '-'}, spread/SL: {f'{ratio:.1%}' if ratio is not None else 'n/a'})")
                return {"symbol": symbol, "pending": ctx}
            return await self._approve_execute_and_record(ctx)

        except Exception as e:
            logger.error(f"[{symbol}] Error in analysis cycle: {e}", exc_info=True)
            self.supabase.log_telemetry("ForexAgent", "ANALYSIS_CYCLE_ERROR", {"symbol": symbol, "error": str(e)}, status="ERROR")
            await self.openclaw.send_alert(f"Analysis cycle error for {symbol}: {e}", level="CRITICAL")
            return {
                "symbol": symbol,
                "decision": "ERROR",
                "confidence": None,
                "sentiment": "ERROR",
                "h1_trend": "-",
                "status": f"ERROR: {str(e)[:40]}"
            }

    async def _approve_execute_and_record(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Approval + risk-wall execution for one analysed symbol, then Supabase/memory logging."""
        symbol = ctx["symbol"]
        decision = ctx["decision"]
        tech_data = ctx["tech_data"]
        news_data = ctx["news_data"]
        trade_params = ctx["trade_params"]
        try:
            approved = False
            result = None
            if trade_params and trade_params.get("status") != "error":
                # Format proposal for WhatsApp with exact numbers
                proposal = {
                    "action": trade_params["action"],
                    "symbol": symbol,
                    "lot": trade_params["lot"],
                    "entry": trade_params["entry"],
                    "sl": trade_params["sl"],
                    "tp": trade_params["tp"],
                    "atr": trade_params["atr"],
                    "risk_amount": trade_params["risk_amount"],
                    "risk_reward": trade_params["risk_reward_ratio"],
                    "confidence": int(decision.confidence_score * 100),
                    "reasoning": decision.reasoning,
                }
                auto_approve = self.config.get("risk_parameters", {}).get("auto_approve", True)
                whatsapp_number = self.config.get("alerts", {}).get("whatsapp_number")

                if auto_approve or not whatsapp_number:
                    logger.info(f"[{symbol}] Autonomous Execution Mode: Trade AUTO-APPROVED through deterministic Risk Wall.")
                    approved = True
                else:
                    sent = await self.openclaw.send_trade_proposal(proposal)
                    if sent:
                        approved = await self.openclaw.wait_for_approval(timeout_seconds=self.approval_timeout)
                    else:
                        logger.warning(f"[{symbol}] WhatsApp proposal could not be sent. Trade NOT approved.")
                        approved = False

                if approved:
                    logger.info(f"[{symbol}] Trade APPROVED — executing through risk wall...")
                    result = self.mt5_engine.execute_trade(trade_params)
                    logger.bind(trade=True).info(f"[{symbol}] Trade result: {json.dumps(result)}")

                    if result.get("status") == "rejected":
                        await self.openclaw.send_alert(
                            f"RISK WALL REJECTED trade on {symbol}: {result.get('message')}", level="WARNING"
                        )
                    elif result.get("status") == "success":
                        await self.openclaw.send_alert(
                            f"EXECUTED {trade_params['action']} {symbol} | Ticket: {result.get('ticket')} | Lot: {result.get('lot')}",
                            level="INFO"
                        )

                    # Sync executed trade to Supabase
                    self.supabase.log_trade_execution(
                        symbol=symbol,
                        order_type=trade_params["action"],
                        volume=result.get("lot", trade_params["lot"]),
                        open_price=trade_params["entry"],
                        stop_loss=trade_params["sl"],
                        take_profit=trade_params["tp"],
                        ticket_id=result.get("ticket"),
                        status="OPEN" if result.get("status") == "success" else "REJECTED",
                        execution_result=result,
                    )
                else:
                    logger.info(f"[{symbol}] Trade NOT approved (rejected or timed out).")
            elif trade_params is None:
                logger.info(f"[{symbol}] Decision is WAIT or confidence below threshold. No action.")

            # Sync decision to Supabase (metadata feeds the daily post-mortem report)
            outcome = self._classify_outcome(decision, trade_params, approved, result)
            self._record_cycle_outcome(outcome, decision)
            self.supabase.log_decision(
                symbol=symbol,
                decision=decision.decision,
                confidence=decision.confidence_score,
                market_sentiment=decision.market_sentiment,
                reasoning=decision.reasoning,
                trade_params=trade_params if (decision.decision != "WAIT" and trade_params and trade_params.get("status") != "error") else None,
                metadata={
                    "outcome": outcome,
                    "h1_trend": tech_data.get("higher_timeframe_h1"),
                    "m5_bias": tech_data.get("technical_bias"),
                    "rsi": tech_data.get("rsi"),
                    "atr": tech_data.get("atr"),
                    "price": tech_data.get("current_price"),
                    "llm": dict(getattr(self.agent_logic, "last_decision_meta", {}) or {}),
                },
                approved=bool(approved),
                executed=bool(result and result.get("status") == "success"),
            )

            # h. Store episode in memory
            episode = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "technical_data": tech_data,
                "news_headlines": news_data.get("headlines", []),
                "decision": decision.model_dump(),
            }
            await self.memory.store_episode(episode)

            h1_trend = tech_data.get("higher_timeframe_h1", "NEUTRAL") if tech_data else "NEUTRAL"
            status_desc = "WAIT (No action)"
            if decision.decision == "WAIT":
                status_desc = "WAIT (Neutral / Rangebound)"
            elif decision.confidence_score < self.confidence_threshold:
                status_desc = f"WAIT (Confidence {decision.confidence_score:.0%} < {self.confidence_threshold:.0%})"
            else:
                if trade_params and trade_params.get("status") == "error":
                    status_desc = f"ERROR: {trade_params.get('message')}"
                elif approved and result:
                    if result.get("status") == "success":
                        status_desc = f"EXECUTED #{result.get('ticket')} ({trade_params['action']} {result.get('lot')} lot)"
                    elif result.get("status") == "rejected":
                        status_desc = f"REJECTED: {result.get('message', 'Risk Wall')}"
                    else:
                        status_desc = f"FAILED: {result.get('message', 'Unknown')}"
                else:
                    status_desc = "REJECTED (Not approved)"

            return {
                "symbol": symbol,
                "decision": decision.decision,
                "confidence": decision.confidence_score,
                "sentiment": decision.market_sentiment,
                "h1_trend": h1_trend,
                "status": status_desc,
                "decision_obj": decision,
            }

        except Exception as e:
            logger.error(f"[{symbol}] Error in analysis cycle: {e}", exc_info=True)
            self.supabase.log_telemetry("ForexAgent", "ANALYSIS_CYCLE_ERROR", {"symbol": symbol, "error": str(e)}, status="ERROR")
            await self.openclaw.send_alert(f"Analysis cycle error for {symbol}: {e}", level="CRITICAL")
            return {
                "symbol": symbol,
                "decision": "ERROR",
                "confidence": None,
                "sentiment": "ERROR",
                "h1_trend": "-",
                "status": f"ERROR: {str(e)[:40]}"
            }

    # ── Cycle Telemetry ───────────────────────────────────────────────────

    @staticmethod
    def _classify_outcome(decision, trade_params, approved, result) -> str:
        if decision.decision == "WAIT":
            return "WAIT"
        if trade_params is None:
            return "BELOW_THRESHOLD"
        if trade_params.get("status") == "error":
            return "PARAM_ERROR"
        if not approved:
            return "NOT_APPROVED"
        if result and result.get("status") == "success":
            return "EXECUTED"
        if result and result.get("status") == "rejected":
            return "REJECTED:" + wall_from_message(result.get("message", ""))
        return "ORDER_FAILED"

    def _record_cycle_outcome(self, outcome: str, decision):
        st = self.cycle_stats
        st.setdefault("outcomes", {})
        st["outcomes"][outcome] = st["outcomes"].get(outcome, 0) + 1
        st.setdefault("mix", {})
        st["mix"][decision.decision] = st["mix"].get(decision.decision, 0) + 1
        meta = getattr(self.agent_logic, "last_decision_meta", {}) or {}
        st["parse_failures"] = st.get("parse_failures", 0) + int(meta.get("parse_failures", 0))
        st["defensive_waits"] = st.get("defensive_waits", 0) + int(bool(meta.get("fallback")))
        tier = meta.get("parse_tier")
        if tier:
            st.setdefault("parse_tiers", {})
            st["parse_tiers"][str(tier)] = st["parse_tiers"].get(str(tier), 0) + 1

    # ── Cycle Summary Table ───────────────────────────────────────────────

    def _print_cycle_summary_table(
        self,
        cycle_results: list[dict],
        next_close: Optional[datetime] = None,
        sleep_seconds: Optional[float] = None
    ):
        """Print a clean ASCII summary table of all symbols analyzed in this cycle."""
        if not cycle_results:
            return

        try:
            account = self.mt5_engine.get_account_info() or {}
            balance = account.get("balance", 0.0)
            equity = account.get("equity", 0.0)
            free_margin = account.get("free_margin", 0.0)
            open_positions = self.mt5_engine.get_open_positions()
            open_count = len(open_positions)
            max_trades = getattr(self, "max_open_trades", 10)

            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            tf = self.timeframe.upper()

            buys = sum(1 for r in cycle_results if r.get("decision") == "BUY")
            sells = sum(1 for r in cycle_results if r.get("decision") == "SELL")
            waits = sum(1 for r in cycle_results if r.get("decision") in ("WAIT", "COOLDOWN"))
            active = sum(1 for r in cycle_results if "ACTIVE" in r.get("status", ""))
            executed = sum(1 for r in cycle_results if "EXECUTED" in r.get("status", ""))
            rejected = sum(1 for r in cycle_results if "REJECTED" in r.get("status", ""))

            sep = "+" + "-" * 10 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 14 + "+" + "-" * 50 + "+"
            header_title = f"BUBAT AI - {tf} MARKET CYCLE ANALYSIS SUMMARY"
            time_title = f"Time: {now_str}"

            lines = [
                "",
                sep,
                "| " + header_title.center(102) + " |",
                "| " + time_title.center(102) + " |",
                sep,
                "| " + "SYMBOL".center(8) + " | " + "DECISION".center(10) + " | " + "CONFIDENCE".center(10) + " | " + "H1 TREND".center(12) + " | " + "STATUS / ACTION".ljust(48) + " |",
                sep,
            ]

            for r in cycle_results:
                sym = str(r.get("symbol", "")).ljust(8)
                dec = str(r.get("decision", "-")).center(10)
                conf_val = r.get("confidence")
                conf = f"{int(conf_val * 100)}%".center(10) if conf_val is not None else "-".center(10)
                h1 = str(r.get("h1_trend", "-"))
                if "BULLISH" in h1:
                    h1_short = "BULLISH".center(12)
                elif "BEARISH" in h1:
                    h1_short = "BEARISH".center(12)
                elif h1 == "-":
                    h1_short = "-".center(12)
                else:
                    h1_short = "NEUTRAL".center(12)

                status = str(r.get("status", "-"))
                if len(status) > 48:
                    status = status[:45] + "..."

                lines.append(f"| {sym} | {dec} | {conf} | {h1_short} | {status.ljust(48)} |")

            acc_str = f"ACCOUNT: Balance: ${balance:.2f} | Equity: ${equity:.2f} | Free Margin: ${free_margin:.2f} | Open Positions: {open_count}/{max_trades}"
            stat_str = f"SIGNALS: {buys} BUY | {sells} SELL | {waits} WAIT/CD | {active} Active | {executed} Executed | {rejected} Rejected"

            lines.extend([
                sep,
                "| " + acc_str.ljust(102) + " |",
                "| " + stat_str.ljust(102) + " |",
            ])

            if next_close and sleep_seconds is not None:
                next_str = f"NEXT CANDLE: Waiting for next {tf} candle close at {next_close.strftime('%H:%M:%S')} UTC ({int(sleep_seconds)}s remaining)..."
                lines.append("| " + next_str.ljust(102) + " |")

            lines.extend([
                sep,
                "",
            ])

            table_output = "\n".join(lines)
            sys.stderr.write(table_output + "\n")
            sys.stderr.flush()
        except Exception as e:
            logger.error(f"Error generating cycle summary table: {e}")

    # ── Reflexion: Learn from Closed Trades ───────────────────────────────

    async def check_closed_trades(self):
        """Apply post-close cooldowns and run reflexion on newly closed losses.

        Cooldowns run from the real close time (MT5 server-time offset handled in
        mt5_engine). Reflexion output is quarantined in learning/reflexion_candidates.jsonl
        and never injected into the trading prompt; curated rules come only from audits.
        """
        try:
            now = datetime.now(timezone.utc)
            since = now - timedelta(hours=2)
            closed_trades = self.mt5_engine.check_closed_trades(since)

            for trade in closed_trades:
                sym = trade.get("symbol")
                close_ts = trade.get("close_ts")
                if sym and close_ts:
                    until = datetime.fromtimestamp(close_ts, tz=timezone.utc) + timedelta(minutes=self.cooldown_minutes)
                    if until > now and (sym not in self.symbol_cooldowns or self.symbol_cooldowns[sym] < until):
                        self.symbol_cooldowns[sym] = until
                        logger.info(f"[{sym}] Cooldown until {until.strftime('%H:%M')} UTC ({self.cooldown_minutes}m after close).")

                ticket = trade.get("ticket")
                if not ticket or ticket in self.processed_closed_tickets:
                    continue
                self.processed_closed_tickets.add(ticket)
                self.reflexion_store.mark_processed(ticket, close_ts)

                profit = trade.get("net_profit", trade.get("profit", 0))
                if profit < 0:
                    logger.info(f"Loss detected on trade {ticket} ({trade.get('exit_type', '?')}), triggering reflexion...")
                    facts = {k: trade.get(k) for k in (
                        "symbol", "direction", "entry_price", "price", "orig_sl", "orig_tp", "sl_pips", "pips",
                        "exit_type", "hold_minutes", "entry_hour_utc", "entry_time_utc", "close_time_utc", "net_profit")}
                    facts["exit_price"] = facts.pop("price")
                    post_mortem = await self.agent_logic.generate_post_mortem(facts)
                    valid, why = validate_reflexion_rule(post_mortem.new_rule)
                    self.reflexion_store.add_candidate(facts, post_mortem.model_dump(), valid, why)
                    logger.info(f"Reflexion candidate stored (valid={valid}, {why}): {post_mortem.new_rule[:120]}")
                    await self.memory.store_trade_outcome(trade, post_mortem.outcome, profit)
                else:
                    # Also store winning trades for balanced memory
                    await self.memory.store_trade_outcome(trade, "WIN", profit)

            self.reflexion_store.save_processed()
        except Exception as e:
            logger.error(f"Error checking closed trades: {e}", exc_info=True)

    # ── Dynamic Candle Monitor ───────────────────────────────────────────

    def get_next_candle_info(self) -> tuple[datetime, float]:
        """Calculate next candle close time and remaining seconds for the configured timeframe."""
        now = datetime.now(timezone.utc)
        tf = self.timeframe.upper()

        if tf == "M1":
            next_close = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        elif tf == "M5":
            next_minute = ((now.minute // 5) + 1) * 5
            if next_minute >= 60:
                next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_close = now.replace(minute=next_minute, second=0, microsecond=0)
        elif tf == "M15":
            next_minute = ((now.minute // 15) + 1) * 15
            if next_minute >= 60:
                next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_close = now.replace(minute=next_minute, second=0, microsecond=0)
        elif tf == "M30":
            next_minute = 30 if now.minute < 30 else 60
            if next_minute >= 60:
                next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_close = now.replace(minute=30, second=0, microsecond=0)
        elif tf == "H1":
            next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif tf == "H4":
            next_hour = ((now.hour // 4) + 1) * 4
            if next_hour >= 24:
                next_close = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_close = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
        elif tf == "D1":
            next_close = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        sleep_seconds = max((next_close - now).total_seconds(), 5)
        return next_close, sleep_seconds

    async def monitor_candle_close(self, sleep_seconds: Optional[float] = None):
        """Wait cleanly until the next candle close while actively managing trailing stops and breakeven."""
        if sleep_seconds is None:
            _, sleep_seconds = self.get_next_candle_info()

        # Sleep in 15-second chunks so we actively manage trailing stops in real-time
        while sleep_seconds > 0 and self.running:
            try:
                mods = self.mt5_engine.manage_trailing_stops()
                self.cycle_stats["trailing_checks"] = self.cycle_stats.get("trailing_checks", 0) + 1
                self.cycle_stats["trailing_mods"] = self.cycle_stats.get("trailing_mods", 0) + len(mods)
                for m in mods:
                    self.supabase.log_telemetry("MT5Engine", "TRAILING_STOP_UPDATE", m, status="SUCCESS")
            except Exception as e:
                logger.debug(f"Error in trailing stop check: {e}")

            chunk = min(sleep_seconds, 15)
            await asyncio.sleep(chunk)
            sleep_seconds -= chunk

        return self.running

    # ── Master Loop ───────────────────────────────────────────────────────

    async def main_loop(self):
        """
        The master orchestration loop.
        
        Crash resilient:
        - Global try/except envelope
        - Automatic MT5 reconnection (3 attempts)
        - Emergency alerts on critical failures
        - The loop NEVER silently dies
        """
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries and self.running:
            try:
                # Initialize MT5
                connected = self.mt5_engine.initialize()
                if not connected:
                    raise ConnectionError("Failed to initialize MT5 terminal")

                account = self.mt5_engine.get_account_info()
                logger.info(
                    f"✅ MT5 connected | Balance: {account.get('balance', 'N/A')} | "
                    f"Equity: {account.get('equity', 'N/A')} | "
                    f"Free Margin: {account.get('free_margin', 'N/A')}"
                )
                await self.openclaw.send_alert("ForexAgent started successfully. Monitoring markets.", level="INFO")

                retry_count = 0  # Reset on successful connection

                first_run = True
                # ── Main trading loop ─────────────────────────────────────
                while self.running:
                    # Run any scheduled tasks (e.g., weekly model check)
                    schedule.run_pending()

                    if first_run:
                        first_run = False
                        logger.info("═══════════════════════════════════════════════")
                        logger.info("🚀 Executing initial market analysis cycle on startup...")
                        logger.info("═══════════════════════════════════════════════")
                    else:
                        logger.info("═══════════════════════════════════════════════")
                        logger.info(f"🕐 {self.timeframe.upper()} candle closed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
                        logger.info("═══════════════════════════════════════════════")

                    # 1. Analyze each configured symbol
                    cycle_started = time.time()
                    prev_trailing = {k: self.cycle_stats.get(k, 0) for k in ("trailing_checks", "trailing_mods")}
                    self.cycle_stats = {}
                    cycle_results = []
                    pending = []
                    for symbol in self.symbols:
                        if not self.running:
                            break
                        res = await self.run_analysis_cycle(symbol)
                        if res and res.get("pending"):
                            pending.append(res["pending"])
                        elif res:
                            cycle_results.append(res)

                    # 1b. Execute queued signals best-first (full H1 trend, then cheapest spread/SL)
                    if pending:
                        ranked = rank_signals(pending)
                        logger.info("Ranked execution order: " + ", ".join(c["symbol"] for c in ranked))
                        for ctx in ranked:
                            cycle_results.append(await self._approve_execute_and_record(ctx))
                        order = {sym: i for i, sym in enumerate(self.symbols)}
                        cycle_results.sort(key=lambda r: order.get(r.get("symbol"), len(order)))

                    # 2. Check for recently closed trades and learn from losses FIRST
                    await self.check_closed_trades()

                    # Cycle telemetry (trailing counts cover the previous sleep window)
                    self.cycle_stats["cycle_seconds"] = round(time.time() - cycle_started, 1)
                    self.cycle_stats["symbols"] = len(cycle_results)
                    self.cycle_stats["skipped_open"] = sum(1 for r in cycle_results if str(r.get("status", "")).startswith("ACTIVE"))
                    self.cycle_stats["skipped_cooldown"] = sum(1 for r in cycle_results if r.get("decision") == "COOLDOWN")
                    self.cycle_stats["errors"] = sum(1 for r in cycle_results if r.get("decision") == "ERROR")
                    self.cycle_stats["breaker_openings_total"] = AgentLogic.breaker_openings
                    self.cycle_stats.update({f"prev_{k}": v for k, v in prev_trailing.items()})
                    logger.info(f"Cycle stats: {json.dumps(self.cycle_stats, default=str)}")
                    self.supabase.log_telemetry("ForexAgent", "CYCLE_SUMMARY", dict(self.cycle_stats), status="SUCCESS")
                    self.cycle_stats = {}

                    # 3. Calculate next candle timing
                    next_close, sleep_seconds = self.get_next_candle_info()

                    # 4. Print end-of-cycle summary table for all symbols AS THE FINAL PROMINENT BLOCK
                    self._print_cycle_summary_table(cycle_results, next_close, sleep_seconds)

                    # 5. Sleep cleanly until next candle close without cluttering terminal
                    candle_ready = await self.monitor_candle_close(sleep_seconds)
                    if not candle_ready:
                        break

            except Exception as e:
                logger.critical(f"💥 Critical error in main loop: {e}", exc_info=True)

                # Send emergency notifications
                try:
                    await self.openclaw.send_alert(
                        f"CRITICAL CRASH: {e}\nRetry {retry_count + 1}/{max_retries}",
                        level="CRITICAL",
                    )
                except Exception:
                    logger.error("Failed to send emergency alert")

                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Attempting reconnection {retry_count}/{max_retries} in 60s...")
                    # Try to reconnect MT5
                    self.mt5_engine.reconnect()
                    await asyncio.sleep(60)

        # ── Exit ──────────────────────────────────────────────────────────
        if not self.running:
            logger.info("Shutdown requested — exiting main loop.")
        else:
            logger.critical(f"Maximum retries ({max_retries}) reached. Agent terminated.")
            try:
                await self.openclaw.send_alert(
                    "AGENT TERMINATED: Maximum retries reached. Manual intervention required.",
                    level="CRITICAL",
                )
            except Exception:
                pass

        await self.shutdown()

    # ── Weekly Model Upgrade ──────────────────────────────────────────────

    def setup_weekly_model_check(self):
        """Schedule the autonomous model upgrade cycle to run weekly."""
        scan_day = "sunday"
        schedule.every().sunday.at("02:00").do(
            lambda: asyncio.ensure_future(self._run_model_upgrade())
        )
        logger.info(f"📅 Weekly model upgrade check scheduled for {scan_day} at 02:00 UTC.")

    async def _run_model_upgrade(self):
        """Wrapper for the model upgrade cycle with notification."""
        try:
            logger.info("Starting weekly model upgrade cycle...")
            await self.model_updater.run_upgrade_cycle()
            logger.info("Model upgrade cycle complete.")
        except Exception as e:
            logger.error(f"Model upgrade cycle failed: {e}")

    # ── Shutdown ──────────────────────────────────────────────────────────

    async def shutdown(self):
        """Graceful shutdown of all subsystems."""
        logger.info("🛑 Initiating graceful shutdown...")
        self.running = False

        try:
            self.mt5_engine.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down MT5: {e}")

        logger.info("Shutdown complete. Goodbye.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Entry point for the Forex Agent."""
    logger.info("╔═══════════════════════════════════════════════════╗")
    logger.info("║   LOCAL AUTONOMOUS FOREX AI AGENT                ║")
    logger.info("║   100% Local • Zero External API Dependencies    ║")
    logger.info("╚═══════════════════════════════════════════════════╝")

    agent = ForexAgent()
    agent.setup_weekly_model_check()

    # Start webhook server in a background thread
    webhook_thread = threading.Thread(
        target=start_webhook_server,
        args=(str(agent.config_path),),
        daemon=True,
    )
    webhook_thread.start()
    logger.info(f"Webhook server started on port {agent.openclaw.webhook_port}")

    try:
        asyncio.run(agent.main_loop())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Agent process exiting.")


if __name__ == "__main__":
    main()
