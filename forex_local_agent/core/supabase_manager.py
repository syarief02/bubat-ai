"""
Supabase Database Manager for Bubat AI
Provides cloud persistence for trade decisions, executions, and agent telemetry.
Supports both Supabase PostgREST Client and Direct PostgreSQL (psycopg2) with graceful fallback.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from loguru import logger
from dotenv import load_dotenv

# Search for .env in current dir and parent dirs
current_dir = Path(__file__).resolve().parent
for parent in [current_dir, current_dir.parent, current_dir.parent.parent]:
    env_file = parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        break


class SupabaseManager:
    """Manages cloud database persistence for Forex AI & Autonomous System Agent."""

    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        self.db_url = os.getenv("DATABASE_URL")
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initializes the Supabase client if credentials exist."""
        if not self.url or not self.key:
            logger.warning("[SupabaseManager] Credentials not found in .env. Cloud logging disabled.")
            return

        try:
            from supabase import create_client
            self.client = create_client(self.url, self.key)
            logger.info("[SupabaseManager] Supabase PostgREST client connected successfully.")
        except Exception as e:
            logger.warning(f"[SupabaseManager] Failed to init supabase client ({e}). Will use psycopg2 fallback if needed.")

    def log_decision(
        self,
        symbol: str,
        decision: str,
        confidence: float,
        market_sentiment: str,
        reasoning: str,
        trade_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Log an LLM qualitative decision and ATR parameters into forex_trade_decisions table.
        """
        data = {
            "symbol": symbol,
            "decision": decision,
            "confidence": float(confidence * 100) if confidence <= 1.0 else float(confidence),
            "market_sentiment": market_sentiment,
            "reasoning": reasoning,
            "approved": False,
            "executed": False,
            "metadata": metadata or {},
        }

        if trade_params and trade_params.get("status") != "error":
            data.update({
                "entry_price": trade_params.get("entry"),
                "stop_loss": trade_params.get("sl"),
                "take_profit": trade_params.get("tp"),
                "lot_size": trade_params.get("lot"),
                "atr": trade_params.get("atr"),
                "risk_reward_ratio": trade_params.get("risk_reward_ratio"),
            })

        return self._insert_record("forex_trade_decisions", data)

    def log_trade_execution(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        open_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        ticket_id: Optional[int] = None,
        status: str = "OPEN",
        execution_result: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Log an executed trade into forex_executed_trades table.
        """
        data = {
            "symbol": symbol,
            "order_type": order_type,
            "volume": float(volume),
            "open_price": float(open_price),
            "stop_loss": float(stop_loss) if stop_loss else None,
            "take_profit": float(take_profit) if take_profit else None,
            "ticket_id": ticket_id,
            "status": status,
            "execution_result": execution_result or {},
        }
        return self._insert_record("forex_executed_trades", data)

    def log_telemetry(
        self,
        agent_name: str,
        action_type: str,
        details: Optional[Dict[str, Any]] = None,
        status: str = "SUCCESS",
    ) -> Optional[str]:
        """
        Log agent telemetry, system audits, or errors into ai_agent_telemetry table.
        """
        data = {
            "agent_name": agent_name,
            "action_type": action_type,
            "details": details or {},
            "status": status,
        }
        return self._insert_record("ai_agent_telemetry", data)

    def _insert_record(self, table: str, data: Dict[str, Any]) -> Optional[str]:
        """Insert a record via PostgREST client or direct psycopg2 fallback."""
        # 1. Try Supabase PostgREST client
        if self.client:
            try:
                res = self.client.table(table).insert(data).execute()
                if res.data and len(res.data) > 0:
                    record_id = res.data[0].get("id")
                    logger.debug(f"[SupabaseManager] Inserted record into {table} (ID: {record_id})")
                    return str(record_id)
            except Exception as e:
                logger.warning(f"[SupabaseManager] PostgREST insert error on {table}: {e}. Trying direct Postgres...")

        # 2. Try Direct PostgreSQL (psycopg2) fallback
        if self.db_url:
            try:
                import psycopg2
                conn = psycopg2.connect(self.db_url, connect_timeout=5)
                cur = conn.cursor()
                
                columns = list(data.keys())
                values = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in data.values()]
                placeholders = ["%s"] * len(columns)

                sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)}) RETURNING id;"
                cur.execute(sql, values)
                rec_id = cur.fetchone()[0]
                conn.commit()
                conn.close()
                logger.debug(f"[SupabaseManager] psycopg2 fallback inserted into {table} (ID: {rec_id})")
                return str(rec_id)
            except Exception as e:
                logger.error(f"[SupabaseManager] Direct Postgres insert failed on {table}: {e}")

        return None

    def execute_query(self, sql: str) -> Dict[str, Any]:
        """
        Directly execute a SQL query on PostgreSQL (SELECT, INSERT, UPDATE, etc.)
        Used by the Autonomous System Agent for natural language DB queries.
        """
        if not self.db_url:
            return {"error": "DATABASE_URL is not configured in .env"}

        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url, connect_timeout=8)
            cur = conn.cursor()
            cur.execute(sql)

            if cur.description:
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                # Convert results to list of dicts
                results = [dict(zip(columns, row)) for row in rows]
                conn.commit()
                conn.close()
                return {"status": "success", "count": len(results), "data": results}
            else:
                conn.commit()
                rowcount = cur.rowcount
                conn.close()
                return {"status": "success", "rows_affected": rowcount}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_recent_decisions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent trade decisions."""
        if self.client:
            try:
                res = self.client.table("forex_trade_decisions").select("*").order("created_at", desc=True).limit(limit).execute()
                return res.data or []
            except Exception:
                pass
        
        # Fallback query
        res = self.execute_query(f"SELECT * FROM forex_trade_decisions ORDER BY created_at DESC LIMIT {limit};")
        return res.get("data", [])
