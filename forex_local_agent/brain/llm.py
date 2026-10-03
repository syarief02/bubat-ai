"""Minimal Ollama JSON client for the brain (thinking on, larger output budget than trading calls)."""
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from core.mt5_time import market_open


class BrainLLM:
    def __init__(self, config: Dict[str, Any]):
        brain_cfg = config.get("brain", {})
        self.url = config.get("ollama_base_url", "http://localhost:11434")
        self.model = brain_cfg.get("model") or config.get("active_model", "agent-brain:32k")
        # true/false, or a level ("low" | "medium" | "high") for models such as gpt-oss
        self.think = brain_cfg.get("think", True)
        self.num_predict = brain_cfg.get("num_predict", 3072)
        self.num_ctx = brain_cfg.get("num_ctx")
        self.num_gpu = brain_cfg.get("num_gpu")
        # While the FX market is closed the trading model is idle, so the brain may use the GPU
        self.gpu_when_market_closed = brain_cfg.get("gpu_when_market_closed", False)
        self.timeout = brain_cfg.get("timeout_seconds", 600)
        self.last_stats: Dict[str, Any] = {}

    def gpu_layers(self, now_utc: datetime = None) -> Optional[int]:
        """num_gpu for this call: None lets Ollama place layers on the GPU, 0 keeps the model on CPU/RAM."""
        if self.gpu_when_market_closed and not market_open(now_utc or datetime.now(timezone.utc)):
            return None
        return self.num_gpu

    def ask_json(self, system: str, prompt: str) -> Optional[Dict[str, Any]]:
        """One JSON-mode completion; returns the parsed object or None."""
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": self.think,
            "options": {"temperature": 0.3, "num_predict": self.num_predict},
        }
        # Leave num_ctx unset when the brain shares the trading model: a different value would load a second copy
        if self.num_ctx:
            payload["options"]["num_ctx"] = self.num_ctx
        # 0 = run entirely on CPU/RAM so a big brain model never evicts the trading model from VRAM.
        # Decided per call: a run that crosses the Sunday open moves back to CPU for its remaining calls.
        num_gpu = self.gpu_layers()
        if num_gpu is not None:
            payload["options"]["num_gpu"] = num_gpu
        try:
            resp = httpx.post(f"{self.url}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "")
            self.last_stats = {"seconds": round(data.get("total_duration", 0) / 1e9, 1),
                               "prompt_tokens": data.get("prompt_eval_count"),
                               "output_tokens": data.get("eval_count"),
                               "thinking_chars": len(data.get("thinking") or ""),
                               "done_reason": data.get("done_reason")}
            if data.get("done_reason") == "length":
                logger.warning(f"[Brain] {self.model} hit num_predict={self.num_predict} before finishing "
                               f"(thinking used the budget); raise brain.num_predict or lower brain.think")
        except Exception as e:
            logger.error(f"[Brain] Ollama request failed: {e}")
            return None
        return parse_json_object(text)


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        pass
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except ValueError:
            return None
    return None
