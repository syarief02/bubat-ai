"""
Agent Logic — Qualitative Sentiment Engine
============================================
The LLM is a QUALITATIVE reasoning engine ONLY.
It reads live news + technical structure to determine market DIRECTION (BUY/SELL/WAIT).
It does NOT calculate prices, stop losses, take profits, or lot sizes.
All quantitative math is handled deterministically by mt5_engine.py.
"""

import httpx
import json
import asyncio
import re
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List
from loguru import logger
from pathlib import Path
from datetime import datetime

try:
    from learning.rules_loader import load_prompt_rules, TRADING_PROMPT_CAP
except ImportError:
    from forex_local_agent.learning.rules_loader import load_prompt_rules, TRADING_PROMPT_CAP


class TradeDecision(BaseModel):
    """Simplified qualitative output — no floating-point arithmetic allowed."""
    market_sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    decision: Literal["BUY", "SELL", "WAIT"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


def _normalize_trade_decision_dict(data: Dict[str, Any]) -> Optional[TradeDecision]:
    """Deterministically normalizes and sanitizes raw dictionary into a valid TradeDecision."""
    if not isinstance(data, dict):
        return None

    # 1. Decision mapping & normalization
    raw_dec = data.get("decision") or data.get("action") or data.get("signal") or data.get("order") or ""
    dec_str = str(raw_dec).strip().upper()
    if dec_str in ["BUY", "LONG", "BUYING"]:
        decision = "BUY"
    elif dec_str in ["SELL", "SHORT", "SELLING"]:
        decision = "SELL"
    elif dec_str in ["WAIT", "HOLD", "PASS", "STAND_ASIDE", "STAND ASIDE", "NEUTRAL", "NONE", "NO_TRADE", "NO TRADE", "FLAT"]:
        decision = "WAIT"
    else:
        # If no recognizable decision, return None
        return None

    # 2. Sentiment mapping & normalization
    raw_sent = data.get("market_sentiment") or data.get("sentiment") or ""
    sent_str = str(raw_sent).strip().upper()
    if "BULL" in sent_str or sent_str in ["BUY", "POSITIVE", "UP"]:
        sentiment = "BULLISH"
    elif "BEAR" in sent_str or sent_str in ["SELL", "NEGATIVE", "DOWN"]:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    # 3. Confidence mapping & normalization
    raw_conf = data.get("confidence_score")
    if raw_conf is None:
        raw_conf = data.get("confidence") or data.get("score") or data.get("confidence_level")

    try:
        if isinstance(raw_conf, str):
            raw_conf = raw_conf.replace("%", "").strip()
        confidence = float(raw_conf)
        if confidence > 1.0:
            confidence = confidence / 100.0
        confidence = min(1.0, max(0.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.50 if decision != "WAIT" else 0.0

    # 4. Reasoning normalization
    raw_reasoning = data.get("reasoning") or data.get("rationale") or data.get("explanation") or ""
    reasoning = str(raw_reasoning).strip()[:500]
    if not reasoning:
        reasoning = f"Deterministic decision based on technical and momentum alignment: {decision} ({sentiment})."

    return TradeDecision(
        market_sentiment=sentiment,
        decision=decision,
        confidence_score=round(confidence, 2),
        reasoning=reasoning
    )


class PostMortem(BaseModel):
    trade_symbol: str
    outcome: Literal["WIN", "LOSS"]
    root_cause: str
    lesson_learned: str
    new_rule: str


class AgentLogic:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)

        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            self.config = {}

        self.ollama_url = self.config.get("ollama_base_url", "http://localhost:11434")
        self.default_model = self.config.get("active_model", "agent-brain:32k")
        self.rules_path = self.config_path.parent / "learning" / "learned_rules.md"
        self.last_decision_meta: Dict[str, Any] = {}
        self._last_parse_tier: Optional[int] = None

    def _load_learned_rules(self) -> str:
        """Curated ### [CATEGORY] rules only (no REFLEXION / COMMUNICATION), capped at whole rules.

        See learning/rules_loader.py.
        """
        if self.rules_path.exists():
            try:
                return load_prompt_rules(self.rules_path, cap=TRADING_PROMPT_CAP)
            except Exception as e:
                logger.error(f"Failed to read learned rules: {e}")
        return ""

    def _build_system_prompt(self, learned_rules: str, memories: List[str]) -> str:
        prompt = (
            "You are an elite institutional forex QUALITATIVE TRADING ENGINE.\n\n"
            "YOUR ROLE:\n"
            "Analyze market trend, technical indicator structure (RSI, MACD, EMAs), and live news "
            "for the SPECIFIC SYMBOL requested to determine decisive MARKET DIRECTION (BUY, SELL, or WAIT) "
            "and realistic confidence.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. DO NOT CALCULATE NUMBERS: Entry prices, SL, TP, and lot sizes are handled deterministically "
            "by the Python ATR engine. Do not calculate prices.\n"
            "2. ANTI-HALLUCINATION POLICY FOR NEWS:\n"
            "   - Only cite news events (such as central bank intervention, interest rate decisions, inflation data) "
            "if they are EXPLICITLY present in the Live News Headlines provided for this symbol.\n"
            "   - NEVER copy news narratives from past episodes or unrelated pairs (e.g. do not cite 'intervention fears' on Gold, CAD, or AUD unless the headline actually says so).\n"
            "   - If no specific news is available or headlines are general market noise, state: 'No high-impact pair-specific news; decision guided by technical structure.'\n"
            "3. REALISTIC CONFIDENCE SPECTRUM (DO NOT DEFAULT TO 0.95 OR 0.50):\n"
            "   - 0.82 - 0.90: Strong technical trend alignment (EMAs + RSI momentum) WITH a directly matching news catalyst.\n"
            "   - 0.68 - 0.80: Clear technical trend structure (EMAs + RSI), but pair-specific news is neutral or absent.\n"
            "   - 0.50 - 0.65: Moderate or emerging trend with mixed/conflicting indicators.\n"
            "   - 0.30 - 0.49: Consolidation, ranging, or flat market (Decision: WAIT).\n"
            "4. DECISION DIRECTION:\n"
            "   - Downtrend below 20 & 50 EMA with bearish momentum -> SELL.\n"
            "   - Uptrend above 20 & 50 EMA with bullish momentum -> BUY.\n"
            "   - Conflicting or sideways -> WAIT.\n"
            "5. MULTI-TIMEFRAME CONFIRMATION (CRITICAL TO AVOID LOSSES):\n"
            "   - Inspect 'higher_timeframe_h1' in the Technical Analysis.\n"
            "   - Never BUY if H1 is BEARISH (avoid buying into a dominant downtrend).\n"
            "   - Never SELL if H1 is BULLISH (avoid selling into a dominant uptrend).\n"
            "   - If M5 and H1 are in direct conflict, output WAIT with NEUTRAL sentiment.\n"
            "6. HIGH-IMPACT MACROECONOMIC NEWS DISCIPLINE:\n"
            "   - If an upcoming High-Impact news release (CPI, NFP, FOMC / Central Bank Rates, GDP) is scheduled within 30 minutes, you MUST output WAIT with NEUTRAL sentiment.\n"
            "   - Never gamble through high-impact economic releases.\n\n"
            "YOUR OUTPUT FORMAT (Valid JSON only):\n"
            "{\n"
            '  "market_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            '  "decision": "BUY" | "SELL" | "WAIT",\n'
            '  "confidence_score": <float between 0.30 and 0.90>,\n'
            '  "reasoning": "<1-2 concise sentences maximum explaining technical momentum and catalyst. Do not repeat calendar tables or list events>"\n'
            "}\n\n"
        )

        if learned_rules:
            prompt += f"--- LEARNED RULES (ABIDE BY THESE) ---\n{learned_rules}\n\n"

        if memories and len(memories) > 0:
            prompt += "--- HISTORICAL EPISODES ---\n"
            for mem in memories:
                prompt += f"- {mem}\n"
            prompt += "\n"

        return prompt

    def _extract_json(self, text: str) -> str:
        text = text.strip()
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx : end_idx + 1]
        return text

    def _parse_or_recover_trade_decision(self, raw_text: str) -> Optional[TradeDecision]:
        """Multi-tier resilient parser: Standard JSON -> Auto-Repair -> Regex extraction."""
        if not raw_text or not isinstance(raw_text, str):
            return None

        # Tier 1: Standard JSON parse (allowing unescaped control chars like newlines)
        json_str = self._extract_json(raw_text)
        try:
            data = json.loads(json_str, strict=False)
            res = _normalize_trade_decision_dict(data)
            if res:
                self._last_parse_tier = 1
                return res
        except Exception:
            pass

        # Tier 2: Auto-repair unclosed reasoning string and braces
        try:
            cleaned = json_str.strip()
            if '"reasoning":' in cleaned:
                idx = cleaned.find('"reasoning":')
                q_start = cleaned.find('"', idx + 12)
                if q_start != -1:
                    content_after = cleaned[q_start + 1:]
                    # If unclosed quote or missing closing brace
                    if not content_after.endswith('"}') and not content_after.endswith('"\n}'):
                        last_space = content_after.rfind(' ')
                        salvaged = content_after[:last_space] if last_space != -1 else content_after[:150]
                        salvaged = salvaged.replace('"', "'")
                        repaired = f'{cleaned[:q_start + 1]}{salvaged}"}}'
                        data = json.loads(repaired, strict=False)
                        res = _normalize_trade_decision_dict(data)
                        if res:
                            self._last_parse_tier = 2
                            return res
        except Exception:
            pass

        # Tier 3: Deterministic Regex Fallback Extraction (handles multiline with re.S)
        try:
            sent_m = re.search(r'"(?:market_sentiment|sentiment)"\s*:\s*"([^"]+)"', raw_text, re.I)
            dec_m = re.search(r'"(?:decision|action|signal|order)"\s*:\s*"([^"]+)"', raw_text, re.I)
            conf_m = re.search(r'"(?:confidence_score|confidence|score)"\s*:\s*([0-9.]+)', raw_text, re.I)
            reas_m = re.search(r'"(?:reasoning|rationale|explanation)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', raw_text, re.S)

            extracted_data = {}
            if dec_m:
                extracted_data["decision"] = dec_m.group(1)
            if sent_m:
                extracted_data["market_sentiment"] = sent_m.group(1)
            if conf_m:
                extracted_data["confidence_score"] = conf_m.group(1)
            if reas_m:
                extracted_data["reasoning"] = reas_m.group(1).replace("\n", " ").strip()

            if extracted_data.get("decision"):
                recovered = _normalize_trade_decision_dict(extracted_data)
                if recovered:
                    self._last_parse_tier = 3
                    return recovered
        except Exception:
            pass

        return None

    # Ollama circuit breaker state
    _ollama_failures: int = 0
    _ollama_circuit_open_until: float = 0.0
    breaker_openings: int = 0

    async def query_ollama(self, prompt: str, system_prompt: str, model: str = None) -> str:
        """Query Ollama with circuit breaker to prevent hammering when Ollama is down."""
        import time as _time
        
        # Circuit breaker: if Ollama has failed repeatedly, skip queries for a cooldown period
        now = _time.time()
        if AgentLogic._ollama_circuit_open_until > now:
            remaining = int(AgentLogic._ollama_circuit_open_until - now)
            raise ConnectionError(f"Ollama circuit breaker OPEN ({remaining}s remaining). Skipping query.")
        
        model_to_use = model or self.default_model
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": 800,
                "temperature": 0.1,
            },
        }

        for q_attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(f"{self.ollama_url}/api/generate", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    # Reset circuit breaker on success
                    AgentLogic._ollama_failures = 0
                    AgentLogic._ollama_circuit_open_until = 0.0
                    return data.get("response", "")
            except Exception as e:
                if q_attempt == 0:
                    logger.warning(f"Ollama query attempt 1 encountered error: {e}. Retrying in 1.5s...")
                    await asyncio.sleep(1.5)
                else:
                    AgentLogic._ollama_failures += 1
                    # After 5 consecutive failures, open circuit for 60 seconds
                    if AgentLogic._ollama_failures >= 5:
                        AgentLogic._ollama_circuit_open_until = _time.time() + 60.0
                        AgentLogic.breaker_openings += 1
                        logger.warning(f"Ollama circuit breaker OPENED after {AgentLogic._ollama_failures} failures. Cooling down 60s.")
                        AgentLogic._ollama_failures = 0
                    logger.warning(f"Ollama query failed: {e}")
                    raise

    async def get_trade_decision(
        self, technical_data: Dict, news_data: Dict, memories: List[str] = None
    ) -> TradeDecision:
        memories = memories or []
        learned_rules = self._load_learned_rules()
        system_prompt = self._build_system_prompt(learned_rules, memories)

        sym = technical_data.get("symbol", "UNKNOWN")
        tf = technical_data.get("timeframe", "M5")
        headlines = news_data.get("headlines", []) if isinstance(news_data, dict) else news_data
        cal_summary = news_data.get("economic_calendar_summary", "") if isinstance(news_data, dict) else ""

        user_prompt = (
            f"Symbol: {sym} (Timeframe: {tf})\n\n"
            f"Technical Analysis:\n{json.dumps(technical_data, indent=2, default=str)}\n\n"
            f"Live News Headlines:\n{json.dumps(headlines, indent=2, default=str)}\n\n"
        )
        if cal_summary:
            user_prompt += f"Macroeconomic Calendar & Event Schedule:\n{cal_summary}\n\n"

        user_prompt += "Analyze the market direction and output your TradeDecision as JSON."

        import time as _time
        started = _time.time()
        # Parse-health telemetry for the caller (logged to Supabase decision metadata)
        self.last_decision_meta = {"attempts": 0, "parse_tier": None, "parse_failures": 0, "fallback": False,
                                   "rules_chars": len(learned_rules)}
        max_retries = 3
        for attempt in range(max_retries):
            raw_response = ""
            self.last_decision_meta["attempts"] = attempt + 1
            try:
                self._last_parse_tier = None
                raw_response = await self.query_ollama(user_prompt, system_prompt)
                decision = self._parse_or_recover_trade_decision(raw_response)
                if decision:
                    logger.info(f"[{sym}] Successfully parsed TradeDecision on attempt {attempt + 1}")
                    self.last_decision_meta["parse_tier"] = self._last_parse_tier
                    self.last_decision_meta["latency_s"] = round(_time.time() - started, 2)
                    return decision
                raise ValueError(f"Could not parse or recover TradeDecision from response (len={len(raw_response)}).")
            except Exception as e:
                preview = repr(raw_response)[:180] if raw_response else "EMPTY"
                logger.warning(f"[{sym}] Failed to parse TradeDecision on attempt {attempt + 1}: {e} | Raw: {preview}")
                self.last_decision_meta["parse_failures"] += 1
                self.last_decision_meta["last_error"] = type(e).__name__
                if attempt == max_retries - 1:
                    self.last_decision_meta["fallback"] = True
                    self.last_decision_meta["latency_s"] = round(_time.time() - started, 2)
                    logger.error(f"[{sym}] All {max_retries} attempts failed to obtain TradeDecision: {e}. Emitting defensive WAIT.")
                    return TradeDecision(
                        market_sentiment="NEUTRAL",
                        decision="WAIT",
                        confidence_score=0.0,
                        reasoning=f"LLM query or parsing failed after {max_retries} attempts ({e}); safe defensive WAIT."
                    )
                await asyncio.sleep(1)

    async def generate_post_mortem(self, trade_data: Dict) -> PostMortem:
        system_prompt = (
            "Analyze this closed trade and generate a post-mortem report.\n"
            "Output ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "trade_symbol": "string",\n'
            '  "outcome": "WIN" | "LOSS",\n'
            '  "root_cause": "string",\n'
            '  "lesson_learned": "string",\n'
            '  "new_rule": "string"\n'
            "}\n"
            "Trade fields: direction, entry_price, exit_price, orig_sl/orig_tp (stops at entry), sl_pips, "
            "pips (signed result), exit_type (SL_FULL = full stop hit, BREAK_EVEN, TRAILING, TP), hold_minutes, "
            "entry_hour_utc, net_profit (account currency, NOT pips).\n"
            "Important: Stop-loss, take-profit, trailing stops and lot size are computed by deterministic code and "
            "are NOT yours to change. Never propose price levels or stop distances.\n"
            "The `new_rule` must be about WHEN not to enter (symbol + session hour, trend context, news timing).\n"
            "The `new_rule` must start with an action verb and reference the specific symbol.\n"
            "Bad example: 'Set a stop-loss level of 208.320' (price levels are rejected automatically).\n"
            "Good example: 'Avoid SELL entries on GBPJPY during Tokyo session open (22:00-01:00 UTC) when RSI > 60'."
        )
        user_prompt = f"Trade Data: {json.dumps(trade_data, indent=2, default=str)}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw = await self.query_ollama(user_prompt, system_prompt)
                clean_json = self._extract_json(raw)
                data = json.loads(clean_json, strict=False)
                # Normalize outcome
                raw_outcome = str(data.get("outcome", "LOSS")).strip().upper()
                data["outcome"] = "WIN" if "WIN" in raw_outcome or "PROFIT" in raw_outcome else "LOSS"
                return PostMortem(**data)
            except Exception as e:
                logger.warning(f"PostMortem parsing attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    sym = trade_data.get("symbol", "UNKNOWN")
                    return PostMortem(
                        trade_symbol=sym,
                        outcome="LOSS" if trade_data.get("profit", 0) < 0 else "WIN",
                        root_cause=f"Automated post-mortem: Trade closed with profit {trade_data.get('profit')}",
                        lesson_learned="Maintain disciplined risk management and adherence to technical momentum.",
                        new_rule=f"RULE #{datetime.now().strftime('%Y%m%d%H%M')} [{sym}]: Verify H1 trend and spread before execution."
                    )
                await asyncio.sleep(1)

