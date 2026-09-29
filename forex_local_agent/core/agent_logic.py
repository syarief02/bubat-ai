import httpx
import json
import asyncio
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List
from loguru import logger
from pathlib import Path
from datetime import datetime

class TradeDecision(BaseModel):
    market_sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    technical_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    decision: Literal["BUY", "SELL", "WAIT"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    stop_loss_pips: float = Field(gt=0)
    take_profit_pips: float = Field(gt=0)
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
        
        # Load config to get ollama url and model
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}, using defaults. Error: {e}")
            self.config = {}

        # All services bind to localhost only
        self.ollama_url = self.config.get("ollama_base_url", "http://localhost:11434")
        self.default_model = self.config.get("active_model", "agent-brain:32k")
        self.rules_path = self.config_path.parent / "learning" / "learned_rules.md"

    def _load_learned_rules(self) -> str:
        """Reads learning/learned_rules.md and returns contents as string."""
        if self.rules_path.exists():
            try:
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read learned rules: {e}")
        return ""

    def _build_system_prompt(self, learned_rules: str, memories: List[str]) -> str:
        """Builds a comprehensive system prompt defining the AI as an institutional risk manager."""
        prompt = (
            "You are an elite institutional forex risk manager. "
            "Your job is to analyze technical indicators and news data, and provide a strictly deterministic and logically sound trading decision.\n\n"
            "CRITICAL: Your output MUST be ONLY valid JSON matching the TradeDecision schema. Do not output any other text or markdown formatting.\n\n"
            "JSON Schema:\n"
            "{\n"
            '  "market_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            '  "technical_bias": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            '  "decision": "BUY" | "SELL" | "WAIT",\n'
            '  "confidence_score": float (0.0 to 1.0),\n'
            '  "stop_loss_pips": float (>0),\n'
            '  "take_profit_pips": float (>0),\n'
            '  "reasoning": "Detailed string explaining the decision."\n'
            "}\n\n"
        )
        if learned_rules:
            prompt += f"--- LEARNED RULES (ABIDE BY THESE STRICTLY) ---\n{learned_rules}\n\n"
        
        if memories and len(memories) > 0:
            prompt += f"--- HISTORICAL MEMORIES ---\n"
            for mem in memories:
                prompt += f"- {mem}\n"
            prompt += "\n"
        
        return prompt

    def _extract_json(self, text: str) -> str:
        """Strips markdown code fences and finds the first { to last }."""
        text = text.strip()
        
        # Try to find JSON bounds
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx+1]
            return text
            
        return text

    async def query_ollama(self, prompt: str, system_prompt: str, model: str = None) -> str:
        """Queries the Ollama API."""
        model_to_use = model or self.default_model
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json" # Force JSON output from supported models
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

    async def get_trade_decision(self, technical_data: Dict, news_data: Dict, memories: List[str] = None) -> TradeDecision:
        """Analyzes data and returns a TradeDecision."""
        memories = memories or []
        learned_rules = self._load_learned_rules()
        system_prompt = self._build_system_prompt(learned_rules, memories)
        
        user_prompt = (
            f"Technical Data: {json.dumps(technical_data, indent=2)}\n"
            f"News Data: {json.dumps(news_data, indent=2)}\n\n"
            "Analyze the above data and provide your TradeDecision in JSON format."
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
                logger.warning(f"Failed to parse TradeDecision on attempt {attempt + 1}: {e}\nRaw output: {raw_response if 'raw_response' in locals() else 'None'}")
                if attempt == max_retries - 1:
                    logger.error("Max retries reached for getting trade decision.")
                    raise ValueError(f"Failed to get valid TradeDecision from LLM after {max_retries} attempts.")
                await asyncio.sleep(1)

    async def generate_post_mortem(self, trade_data: Dict) -> PostMortem:
        """Analyzes a closed trade and generates a PostMortem."""
        system_prompt = (
            "You are an elite quantitative analyst. Analyze this closed trade to determine the root cause of its outcome, "
            "and generate a post-mortem report.\n\n"
            "CRITICAL: Your output MUST be ONLY valid JSON matching the PostMortem schema.\n"
            "Schema:\n"
            "{\n"
            '  "trade_symbol": "string",\n'
            '  "outcome": "WIN" | "LOSS",\n'
            '  "root_cause": "string",\n'
            '  "lesson_learned": "string",\n'
            '  "new_rule": "string (A new trading rule to add to the knowledge base)"\n'
            "}\n"
        )
        
        user_prompt = f"Trade Data: {json.dumps(trade_data, indent=2)}\n\nGenerate the PostMortem JSON."
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw_response = await self.query_ollama(user_prompt, system_prompt)
                json_str = self._extract_json(raw_response)
                pm_dict = json.loads(json_str)
                post_mortem = PostMortem(**pm_dict)
                return post_mortem
            except Exception as e:
                logger.warning(f"Failed to parse PostMortem on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid PostMortem after {max_retries} attempts.")
                await asyncio.sleep(1)
