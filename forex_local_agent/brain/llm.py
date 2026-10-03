"""Minimal Ollama JSON client for the brain (thinking on, larger output budget than trading calls)."""
import json
import re
import time
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
        self.think_fallback = list(brain_cfg.get("think_fallback", []))
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
        """JSON-mode completion; returns the parsed object or None.

        Tries `think`, then each `think_fallback` level: high reasoning can think past any budget on
        open-ended prompts (2026-10-04: 16,000 tokens, 49 min on CPU, no answer), so a cut-off or
        unparseable answer is retried at the next, cheaper level.
        """
        for level in [self.think] + [lv for lv in self.think_fallback if lv != self.think]:
            text, data = self._generate(system, prompt, level)
            parsed = parse_json_object(text) if data is not None else None
            if data is not None:
                self.last_stats["think_used"] = level
            if parsed is not None and data.get("done_reason") != "length":
                return parsed
            if data is None:
                return None          # Ollama unreachable: a retry at another level would fail too
            logger.warning(f"[Brain] no usable answer at think={level!r} "
                           f"(done_reason={data.get('done_reason')}); trying the next level")
        return None

    def _match_placement(self, num_gpu: Optional[int]):
        """Unload the brain model if it is loaded on the wrong device.

        Ollama reuses an already-loaded copy, so a CPU-only copy (num_gpu 0) left from a weekday
        run would keep a weekend run on the CPU, and a GPU copy would keep VRAM after the open.
        """
        try:
            loaded = httpx.get(f"{self.url}/api/ps", timeout=10).json().get("models", [])
            entry = next((m for m in loaded if m.get("name") == self.model or m.get("model") == self.model), None)
            if not entry:
                return
            on_gpu = (entry.get("size_vram") or 0) > 0
            if on_gpu == (num_gpu != 0):
                return
            logger.info(f"[Brain] reloading {self.model} on {'CPU' if num_gpu == 0 else 'GPU'}")
            httpx.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=60)
            for _ in range(30):
                loaded = httpx.get(f"{self.url}/api/ps", timeout=10).json().get("models", [])
                if not any(m.get("name") == self.model for m in loaded):
                    return
                time.sleep(1)
        except Exception as e:
            logger.debug(f"[Brain] placement check skipped: {e}")

    def _generate(self, system: str, prompt: str, think: Any):
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": think,
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
        self._match_placement(num_gpu)
        try:
            resp = httpx.post(f"{self.url}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"[Brain] Ollama request failed: {e}")
            return "", None
        self.last_stats = {"seconds": round(data.get("total_duration", 0) / 1e9, 1),
                           "prompt_tokens": data.get("prompt_eval_count"),
                           "output_tokens": data.get("eval_count"),
                           "thinking_chars": len(data.get("thinking") or ""),
                           "done_reason": data.get("done_reason")}
        if data.get("done_reason") == "length":
            logger.warning(f"[Brain] {self.model} hit num_predict={self.num_predict} at think={think!r} "
                           f"before finishing (thinking used the budget)")
        return data.get("response", ""), data


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
