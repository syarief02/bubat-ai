import os
import json
import logging
from typing import Dict, List, Any, Optional

try:
    from mem0 import Memory
except ImportError:
    Memory = None

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Episodic memory using Mem0 + ChromaDB.
    Falls back to simple JSON file storage if Mem0/ChromaDB is unavailable.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.fallback_file = os.path.join(os.path.dirname(config_path), "memory_fallback.json")
        self.use_fallback = False
        self.mem0 = None

        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = {}

        if Memory:
            try:
                # Initialize mem0 with local storage
                # This assumes appropriate config format for Mem0 local setup
                mem0_config = {
                    "vector_store": {
                        "provider": "chroma",
                        "config": {
                            "collection_name": "forex_agent_memory",
                            "path": os.path.join(os.path.dirname(config_path), "chroma_db")
                        }
                    }
                }
                self.mem0 = Memory.from_config(mem0_config)
                logger.info("Mem0 memory initialized with ChromaDB.")
            except Exception as e:
                logger.warning(f"Failed to initialize Mem0/ChromaDB, falling back to JSON: {e}")
                self.use_fallback = True
        else:
            logger.warning("Mem0 not installed, falling back to JSON storage.")
            self.use_fallback = True

        if self.use_fallback:
            self._init_fallback()

    def _init_fallback(self):
        if not os.path.exists(self.fallback_file):
            with open(self.fallback_file, "w") as f:
                json.dump({"episodes": [], "preferences": {}}, f)

    def _read_fallback(self) -> Dict:
        try:
            with open(self.fallback_file, "r") as f:
                return json.load(f)
        except Exception:
            return {"episodes": [], "preferences": {}}

    def _write_fallback(self, data: Dict):
        with open(self.fallback_file, "w") as f:
            json.dump(data, f, indent=4)

    async def store_episode(self, episode: Dict):
        """Store a trade episode with metadata."""
        if self.use_fallback:
            data = self._read_fallback()
            data["episodes"].append(episode)
            self._write_fallback(data)
        else:
            try:
                text = f"Trade episode: {json.dumps(episode)}"
                self.mem0.add(text, user_id="agent", metadata=episode)
            except Exception as e:
                logger.error(f"Error storing episode in Mem0: {e}")

    async def query_similar(self, current_conditions: Dict, top_k: int = 5) -> List[Dict]:
        """Find similar historical episodes for the current symbol."""
        symbol = current_conditions.get("symbol")
        if self.use_fallback:
            data = self._read_fallback()
            episodes = data.get("episodes", [])
            # Filter strictly by matching symbol to avoid cross-pair hallucination contamination
            if symbol:
                episodes = [ep for ep in episodes if ep.get("symbol") == symbol]
            return episodes[-top_k:]
        else:
            try:
                query_text = f"Find trade conditions for symbol {symbol}: {json.dumps(current_conditions)}"
                results = self.mem0.search(query_text, user_id="agent", limit=top_k)
                episodes = [r.get('metadata', r) for r in results]
                if symbol:
                    episodes = [ep for ep in episodes if ep.get("symbol") == symbol]
                return episodes
            except Exception as e:
                logger.error(f"Error querying Mem0: {e}")
                return []

    async def store_user_preference(self, key: str, value: str):
        """Store user preferences."""
        if self.use_fallback:
            data = self._read_fallback()
            data["preferences"][key] = value
            self._write_fallback(data)
        else:
            try:
                text = f"User preference {key}: {value}"
                self.mem0.add(text, user_id="user", metadata={"type": "preference", "key": key})
            except Exception as e:
                logger.error(f"Error storing user preference in Mem0: {e}")

    async def get_user_preferences(self) -> Dict:
        """Retrieve all user preferences."""
        if self.use_fallback:
            return self._read_fallback().get("preferences", {})
        else:
            try:
                results = self.mem0.search("User preferences", user_id="user")
                prefs = {}
                for r in results:
                    meta = r.get("metadata", {})
                    if meta.get("type") == "preference":
                        # Simplistic extraction
                        prefs[meta.get("key", "unknown")] = r.get("text", "")
                return prefs
            except Exception as e:
                logger.error(f"Error retrieving user preferences: {e}")
                return {}

    async def store_trade_outcome(self, trade: Dict, outcome: str, pnl: float):
        """Store trade result for future reference."""
        record = {
            "trade": trade,
            "outcome": outcome,
            "pnl": pnl
        }
        await self.store_episode(record)

    def _format_episode_for_prompt(self, episodes: List[Dict]) -> List[str]:
        """Format past trade episodes concisely to prevent prompt parrot copying."""
        formatted = []
        for i, ep in enumerate(episodes):
            sym = ep.get("symbol", "N/A")
            dec = ep.get("decision", {})
            decision_val = dec.get("decision", "N/A") if isinstance(dec, dict) else "N/A"
            conf = dec.get("confidence_score", "N/A") if isinstance(dec, dict) else "N/A"
            tech = ep.get("technical_data", {})
            bias = tech.get("technical_bias", "N/A") if isinstance(tech, dict) else "N/A"
            outcome = ep.get("outcome", "PENDING")
            formatted.append(
                f"Historical Episode {i+1} [{sym}]: Bias={bias}, Decision={decision_val}, Confidence={conf}, Outcome={outcome}"
            )
        return formatted
