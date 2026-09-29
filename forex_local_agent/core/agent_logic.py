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
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List
from loguru import logger
from pathlib import Path
from datetime import datetime


class TradeDecision(BaseModel):
    """Simplified qualitative output — no floating-point arithmetic allowed."""
    market_sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    decision: Literal["BUY", "SELL", "WAIT"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


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

    def _load_learned_rules(self) -> str:
        if self.rules_path.exists():
            try:
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read learned rules: {e}")
        return ""

    def _build_system_prompt(self, learned_rules: str, memories: List[str]) -> str:
        prompt = (
            "You are an elite institutional forex QUALITATIVE SENTIMENT ENGINE.\n\n"
            "YOUR ROLE:\n"
            "Analyze fundamental news headlines alongside technical indicator structure "
            "(RSI, MACD, ATR values) to determine the MARKET DIRECTION for the given "
            "currency pair. You assess whether conditions favour a BUY, SELL, or WAIT.\n\n"
            "CRITICAL INSTRUCTION — DO NOT CALCULATE NUMBERS:\n"
            "Do NOT attempt to calculate entry prices, stop-loss distances, take-profit distances, "
            "or lot sizes. All pricing, risk sizing, and order math is handled by a separate "
            "deterministic Python engine using the Average True Range (ATR).\n\n"
            "YOUR OUTPUT FORMAT:\n"
            "Return ONLY valid JSON matching this exact schema:\n"
            "{\n"
            '  "market_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            '  "decision": "BUY" | "SELL" | "WAIT",\n'
            '  "confidence_score": <float between 0.0 and 1.0>,\n'
            '  "reasoning": "<concise explanation of signals>"\n'
            "}\n\n"
            "GUIDELINES:\n"
            "- Output BUY only when fundamentals and technicals strongly align bullish.\n"
            "- Output SELL only when fundamentals and technicals strongly align bearish.\n"
            "- Output WAIT if signals conflict or data is ambiguous.\n"
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

    async def query_ollama(self, prompt: str, system_prompt: str, model: str = None) -> str:
        model_to_use = model or self.default_model
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(f"{self.ollama_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            raise

    async def get_trade_decision(
        self, technical_data: Dict, news_data: Dict, memories: List[str] = None
    ) -> TradeDecision:
        memories = memories or []
        learned_rules = self._load_learned_rules()
        system_prompt = self._build_system_prompt(learned_rules, memories)

        user_prompt = (
            f"Technical Indicators:\n{json.dumps(technical_data, indent=2, default=str)}\n\n"
            f"Live News Headlines:\n{json.dumps(news_data, indent=2, default=str)}\n\n"
            "Analyze the market direction and output your TradeDecision as JSON."
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw_response = await self.query_ollama(user_prompt, system_prompt)
                json_str = self._extract_json(raw_response)
                decision_dict = json.loads(json_str)
                decision = TradeDecision(**decision_dict)
                logger.info(f"Successfully parsed TradeDecision on attempt {attempt + 1}")
                return decision
            except Exception as e:
                logger.warning(f"Failed to parse TradeDecision on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to get valid TradeDecision after {max_retries} attempts.")
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
            "}"
        )
        user_prompt = f"Trade Data: {json.dumps(trade_data, indent=2, default=str)}"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw = await self.query_ollama(user_prompt, system_prompt)
                clean_json = self._extract_json(raw)
                return PostMortem(**json.loads(clean_json))
            except Exception as e:
                logger.warning(f"PostMortem parsing attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise ValueError("Failed to generate PostMortem")
                await asyncio.sleep(1)
