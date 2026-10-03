"""Minimal Ollama JSON client for the brain (thinking on, larger output budget than trading calls)."""
import json
import re
from typing import Any, Dict, Optional

import httpx
from loguru import logger


class BrainLLM:
    def __init__(self, config: Dict[str, Any]):
        brain_cfg = config.get("brain", {})
        self.url = config.get("ollama_base_url", "http://localhost:11434")
        self.model = brain_cfg.get("model") or config.get("active_model", "agent-brain:32k")
        self.think = brain_cfg.get("think", True)
        self.num_predict = brain_cfg.get("num_predict", 3072)
        self.timeout = brain_cfg.get("timeout_seconds", 600)

    def ask_json(self, system: str, prompt: str) -> Optional[Dict[str, Any]]:
        """One JSON-mode completion; returns the parsed object or None."""
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": self.think,
            # num_ctx comes from the Modelfile; a different value would load a second copy of the model
            "options": {"temperature": 0.3, "num_predict": self.num_predict},
        }
        try:
            resp = httpx.post(f"{self.url}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            text = resp.json().get("response", "")
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
