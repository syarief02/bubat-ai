import httpx
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Literal
from loguru import logger
from pathlib import Path
from pydantic import BaseModel, Field

class AuditionResult(BaseModel):
    model_name: str
    schema_test_passed: bool
    coding_test_passed: bool  
    latency_seconds: float
    overall_passed: bool
    notes: str

class ModelUpdater:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.ollama_url = "http://localhost:11434"
        self.searxng_url = "http://localhost:8080"
        self._load_config()
        self.ollama_url = self.config.get("ollama_base_url", self.ollama_url)
        self.searxng_url = self.config.get("searxng_url", self.searxng_url)
        self.max_latency = self.config.get("model_upgrade", {}).get("max_response_latency_seconds", 60)
        self.max_params = self.config.get("model_upgrade", {}).get("max_parameter_size_b", 35)
        
        # Setup audition logging
        self.log_file = self.config_path.parent / "logs" / "model_auditions.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {}
            logger.warning(f"Config file not found at {self.config_path}, using defaults.")

    def _save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)

    async def discover_models(self) -> List[Dict]:
        logger.info("Discovering potential new models...")
        candidates = []
        
        # Query local models
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                if resp.status_code == 200:
                    local_models = resp.json().get("models", [])
                    for m in local_models:
                        candidates.append({
                            "name": m.get("name"),
                            "size_bytes": m.get("size", 0),
                            "source": "local"
                        })
        except Exception as e:
            logger.error(f"Failed to fetch local models: {e}")

        # Query SearXNG for latest ollama models
        try:
            async with httpx.AsyncClient() as client:
                params = {"q": "ollama latest coder models github", "format": "json"}
                resp = await client.get(f"{self.searxng_url}/search", params=params, timeout=10.0)
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    for r in results:
                        title = r.get("title", "").lower()
                        if "llama" in title or "coder" in title or "qwen" in title:
                            # Implementation to parse real models would go here
                            pass
        except Exception as e:
            logger.warning(f"SearXNG discovery failed: {e}")

        return candidates

    async def pull_model(self, model_name: str) -> bool:
        logger.info(f"Pulling model {model_name} from Ollama registry. This will not block trading.")
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                request_data = {"model": model_name, "stream": True}
                async with client.stream("POST", f"{self.ollama_url}/api/pull", json=request_data) as response:
                    if response.status_code != 200:
                        logger.error(f"Failed to pull model: {response.status_code}")
                        return False
                    
                    async for chunk in response.aiter_lines():
                        if chunk:
                            try:
                                data = json.loads(chunk)
                                status = data.get("status", "")
                                if "downloading" in status:
                                    completed = data.get("completed", 0)
                                    total = data.get("total", 1)
                                    pct = (completed / total) * 100
                                    logger.debug(f"Pulling {model_name}: {pct:.1f}%")
                            except json.JSONDecodeError:
                                pass
                
                logger.info(f"Successfully pulled model {model_name}")
                return True
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    async def audition_model(self, model_name: str) -> AuditionResult:
        logger.info(f"Auditioning model {model_name}")
        schema_passed = False
        coding_passed = False
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.max_latency + 10) as client:
                # Test 1: Schema Compliance
                schema_prompt = '''Output a JSON object for a trade decision with keys: "action" (BUY/SELL/HOLD), "symbol", "reason", "confidence" (float 0-1).
Scenario: EURUSD is showing strong bullish momentum, RSI is 40.
'''
                payload = {
                    "model": model_name,
                    "prompt": schema_prompt,
                    "stream": False,
                    "format": "json"
                }
                resp1 = await client.post(f"{self.ollama_url}/api/generate", json=payload)
                if resp1.status_code == 200:
                    resp_json = resp1.json()
                    out_text = resp_json.get("response", "{}")
                    try:
                        parsed = json.loads(out_text)
                        if all(k in parsed for k in ["action", "symbol", "reason", "confidence"]):
                            schema_passed = True
                    except:
                        pass
                
                # Test 2: Coding Test
                code_prompt = "Write a Python function named 'add' that takes two numbers and returns their sum. Return ONLY valid Python code."
                payload2 = {
                    "model": model_name,
                    "prompt": code_prompt,
                    "stream": False
                }
                resp2 = await client.post(f"{self.ollama_url}/api/generate", json=payload2)
                if resp2.status_code == 200:
                    out_code = resp2.json().get("response", "")
                    if "def add(" in out_code and "return" in out_code:
                        coding_passed = True

        except Exception as e:
            logger.error(f"Audition failed for {model_name} due to error: {e}")

        latency = time.time() - start_time
        latency_passed = latency <= self.max_latency
        overall = schema_passed and coding_passed and latency_passed
        
        result = AuditionResult(
            model_name=model_name,
            schema_test_passed=schema_passed,
            coding_test_passed=coding_passed,
            latency_seconds=latency,
            overall_passed=overall,
            notes="Audition completed."
        )
        self._log_audition(result)
        return result

    async def hot_swap(self, new_model: str) -> bool:
        logger.info(f"Initiating hot-swap to {new_model}")
        try:
            self.config["active_model"] = new_model
            self._save_config()
            
            modelfile_content = f"FROM {new_model}\nPARAMETER num_ctx 32768\n"
            
            payload = {
                "name": "agent-brain:32k",
                "modelfile": modelfile_content,
                "stream": False
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{self.ollama_url}/api/create", json=payload, timeout=120.0)
                if resp.status_code == 200:
                    logger.success(f"Successfully hot-swapped to {new_model}")
                    return True
                else:
                    logger.error(f"Failed to create agent-brain:32k: {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"Hot-swap failed: {e}")
            return False

    def _log_audition(self, result: AuditionResult):
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "model": result.model_name,
                "schema_passed": result.schema_test_passed,
                "coding_passed": result.coding_test_passed,
                "latency": result.latency_seconds,
                "overall_passed": result.overall_passed,
                "notes": result.notes
            }
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audition log: {e}")

    async def run_upgrade_cycle(self):
        logger.info("Starting model upgrade cycle...")
        candidates = await self.discover_models()
        
        if not candidates:
            logger.info("No candidates found, skipping upgrade.")
            return
            
        # Example logic, take the first valid candidate
        best_candidate = candidates[0].get("name")
        if not best_candidate:
            return
            
        pulled = await self.pull_model(best_candidate)
        if pulled:
            result = await self.audition_model(best_candidate)
            if result.overall_passed:
                await self.hot_swap(best_candidate)
