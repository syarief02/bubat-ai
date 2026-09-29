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

# ── Logging Configuration ────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)
logger.add("logs/system_errors.log", rotation="10 MB", level="ERROR", backtrace=True, diagnose=True)
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
            # a. Load learned rules
            rules_path = Path("learning/learned_rules.md")
            learned_rules = rules_path.read_text(encoding="utf-8") if rules_path.exists() else ""

            # b. Get technical data from MT5
            logger.info(f"[{symbol}] Fetching technical data...")
            tech_data = self.mt5_engine.get_technical_data(symbol, timeframe=self.timeframe)
            if not tech_data:
                logger.warning(f"[{symbol}] No technical data available, skipping cycle.")
                return None

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

            # g. If actionable signal, calculate deterministic trade parameters & propose
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
            else:
                logger.info(f"[{symbol}] Decision is WAIT or confidence below threshold. No action.")

            # Sync decision to Supabase
            self.supabase.log_decision(
                symbol=symbol,
                decision=decision.decision,
                confidence=decision.confidence_score,
                market_sentiment=decision.market_sentiment,
                reasoning=decision.reasoning,
                trade_params=trade_params if (decision.decision != "WAIT" and trade_params and trade_params.get("status") != "error") else None,
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

            return decision

        except Exception as e:
            logger.error(f"[{symbol}] Error in analysis cycle: {e}", exc_info=True)
            self.supabase.log_telemetry("ForexAgent", "ANALYSIS_CYCLE_ERROR", {"symbol": symbol, "error": str(e)}, status="ERROR")
            await self.openclaw.send_alert(f"Analysis cycle error for {symbol}: {e}", level="CRITICAL")
            return None

    # ── Reflexion: Learn from Closed Trades ───────────────────────────────

    async def check_closed_trades(self):
        """Check recently closed trades and trigger reflexion on losses."""
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=2)
            closed_trades = self.mt5_engine.check_closed_trades(since)

            for trade in closed_trades:
                profit = trade.get("profit", 0)
                if profit < 0:
                    logger.info(f"Loss detected on trade {trade.get('ticket')}, triggering reflexion...")

                    # Generate post-mortem via LLM
                    post_mortem = await self.agent_logic.generate_post_mortem(trade)

                    # Append new rule to learned_rules.md
                    rules_path = Path("learning/learned_rules.md")
                    new_rule = f"\n\nRULE #{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')} [{post_mortem.trade_symbol}]: {post_mortem.new_rule}\n"
                    new_rule += f"  Root cause: {post_mortem.root_cause}\n"
                    new_rule += f"  Lesson: {post_mortem.lesson_learned}\n"

                    with open(rules_path, "a", encoding="utf-8") as f:
                        f.write(new_rule)
                    logger.info(f"Reflexion rule added: {post_mortem.new_rule}")

                    # Store outcome in memory
                    await self.memory.store_trade_outcome(trade, post_mortem.outcome, profit)
                else:
                    # Also store winning trades for balanced memory
                    await self.memory.store_trade_outcome(trade, "WIN", profit)

        except Exception as e:
            logger.error(f"Error checking closed trades: {e}", exc_info=True)

    # ── Dynamic Candle Monitor ───────────────────────────────────────────

    async def monitor_candle_close(self):
        """Wait until the next candle close for the configured timeframe (M1, M5, M15, M30, H1, H4, D1)."""
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
        logger.info(f"⏳ Waiting {sleep_seconds:.0f}s for next {tf} candle close at {next_close.strftime('%H:%M:%S')} UTC...")

        # Sleep in chunks so we can check the running flag
        while sleep_seconds > 0 and self.running:
            chunk = min(sleep_seconds, 10)
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

                    if not first_run:
                        # Wait for candle close (M5, H1, etc.)
                        candle_ready = await self.monitor_candle_close()
                        if not candle_ready:
                            break

                        logger.info("═══════════════════════════════════════════════")
                        logger.info(f"🕐 {self.timeframe.upper()} candle closed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
                        logger.info("═══════════════════════════════════════════════")
                    else:
                        first_run = False
                        logger.info("═══════════════════════════════════════════════")
                        logger.info("🚀 Executing initial market analysis cycle on startup...")
                        logger.info("═══════════════════════════════════════════════")

                    # Analyze each configured symbol
                    for symbol in self.symbols:
                        if not self.running:
                            break
                        await self.run_analysis_cycle(symbol)

                    # Check for recently closed trades and learn from losses
                    await self.check_closed_trades()

                    logger.info("✅ Analysis cycle complete for all symbols.\n")

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
