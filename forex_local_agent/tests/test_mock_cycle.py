"""
Sandboxed Dry-Run Regression Test
Simulates a complete analysis cycle:
1. Fetch 10 bars from MT5
2. Run sentiment scraper + economic calendar filter
3. Query Ollama with refined temperature (0.1) & parse JSON schema
4. Calculate ATR stops & evaluate risk walls
"""
import sys
import asyncio
from pathlib import Path
from loguru import logger

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from core.mt5_engine import MT5Engine
from core.sentiment_engine import SentimentEngine
from core.agent_logic import AgentLogic
from learning.skills.economic_calendar_filter import is_trade_permitted_by_calendar, get_economic_events_summary

async def run_sandboxed_dry_run():
    config_path = root_dir / "config.json"
    print("=" * 60)
    print("STARTING SANDBOXED DRY-RUN REGRESSION TEST")
    print("=" * 60)
    
    # 1. Initialize Engines
    print("\n[STEP 1] Initializing Engines...")
    mt5_eng = MT5Engine(str(config_path))
    assert mt5_eng.initialize(), "MT5Engine initialize() failed"
    sentiment_eng = SentimentEngine(str(config_path))
    agent_logic = AgentLogic(str(config_path))
    print("All engines initialized and MT5 connected successfully.")
    
    # Test Symbol: EURUSD
    symbol = "EURUSD"
    
    # 2. Fetch Bars from MT5
    print(f"\n[STEP 2] Fetching bars from MT5 for {symbol}...")
    tech_data = mt5_eng.get_technical_data(symbol, "M5", bars=60)
    assert tech_data is not None, "Failed to get technical data from MT5"
    assert "current_price" in tech_data, "Technical data missing current_price"
    print(f"MT5 Data retrieved: Price={tech_data.get('current_price')}, ATR={tech_data.get('atr')}, RSI={tech_data.get('rsi')}, H1 Trend={tech_data.get('higher_timeframe_h1')}")
    
    # 3. Run Sentiment & Calendar Scraper
    print(f"\n[STEP 3] Running Sentiment & Calendar Scraper for {symbol}...")
    news_data = await sentiment_eng.get_live_news(symbol)
    assert news_data is not None, "Failed to get news data"
    headlines = news_data.get("headlines", [])
    cal_summary = news_data.get("economic_calendar_summary", "")
    print(f"Scraper returned {len(headlines)} headlines.")
    print(f"Calendar filter summary present: {bool(cal_summary)}")
    if cal_summary:
        print(f"Calendar snippet:\n{cal_summary[:200]}...")
        
    # Check Calendar Permission Wall
    permitted, reason = is_trade_permitted_by_calendar(symbol, pre_buffer_mins=30, post_buffer_mins=15)
    print(f"Calendar Trade Permission: Permitted={permitted}, Reason={reason}")
    
    # 4. Query Ollama & Parse JSON Schema
    print(f"\n[STEP 4] Querying Ollama for {symbol} with JSON format...")
    decision = await agent_logic.get_trade_decision(tech_data, news_data)
    assert decision is not None, "Failed to obtain TradeDecision"
    print(f"TradeDecision Parsed Successfully:")
    print(f"  Market Sentiment: {decision.market_sentiment}")
    print(f"  Decision:         {decision.decision}")
    print(f"  Confidence:       {decision.confidence_score}")
    print(f"  Reasoning:        {decision.reasoning[:120]}...")
    
    # 5. Deterministic ATR Trade Parameters Math
    print(f"\n[STEP 5] Testing ATR Trade Parameters Math...")
    test_action = decision.decision if decision.decision in ["BUY", "SELL"] else "BUY"
    params = mt5_eng.calculate_trade_parameters(symbol, test_action)
    assert params.get("status") == "calculated", f"Trade params calculation failed: {params}"
    print(f"ATR Parameters calculated successfully:")
    print(f"  Action:   {params.get('action')}")
    print(f"  Entry:    {params.get('entry')}")
    print(f"  SL:       {params.get('sl')} (SL dist: {params.get('sl_distance')})")
    print(f"  TP:       {params.get('tp')} (TP dist: {params.get('tp_distance')})")
    print(f"  Lot:      {params.get('lot')}")
    print(f"  Risk/Rwd: {params.get('risk_reward_ratio')}")
    
    print("\n" + "=" * 60)
    print("DRY-RUN REGRESSION TEST COMPLETED: ALL ASSERTIONS PASSED (100% OK)")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_sandboxed_dry_run())
