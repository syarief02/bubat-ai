"""
Bubat Brain: the daily self-improvement agent that sits above the trading bot.

observe (daily report digest) -> reflect (LLM) -> research (web) -> propose (gated changes).
It never trades and never changes anything live on its own: every proposal waits for the
owner's approval, and an approved change is rolled back if the offline test suites fail.

Run: python -m brain --help   (from forex_local_agent/)
"""
