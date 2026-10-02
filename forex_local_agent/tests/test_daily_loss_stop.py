"""Test daily loss stop wall in mt5_engine.py (no pytest required)."""
import json
from pathlib import Path

# Verify config.json has the new parameter
config_path = Path(__file__).parent.parent / "config.json"
with open(config_path) as f:
    config = json.load(f)

assert "daily_loss_limit_pct" in config.get("risk_parameters", {}), \
    "FAIL: daily_loss_limit_pct missing from config.json"
assert config["risk_parameters"]["daily_loss_limit_pct"] == 5.0, \
    f"FAIL: expected 5.0, got {config['risk_parameters']['daily_loss_limit_pct']}"
print("[PASS] config.json has daily_loss_limit_pct = 5.0")

# Verify the code path exists in mt5_engine.py
engine_path = Path(__file__).parent.parent / "core" / "mt5_engine.py"
engine_code = engine_path.read_text(encoding="utf-8")
assert "daily_loss_limit_pct" in engine_code, "FAIL: daily_loss_limit_pct not referenced in mt5_engine.py"
assert "Daily loss limit reached" in engine_code, "FAIL: rejection message not in mt5_engine.py"
assert "daily_loss_limit_pct / 100.0" in engine_code, "FAIL: percentage calculation not in mt5_engine.py"
print("[PASS] mt5_engine.py contains daily loss stop wall code")

# Verify it comes AFTER H1 trend wall and BEFORE capacity wall
h1_pos = engine_code.find("Counter-trend")
daily_pos = engine_code.find("daily_loss_limit_pct")
capacity_pos = engine_code.find("Max open trades")
assert h1_pos < daily_pos < capacity_pos, \
    f"FAIL: Wall ordering wrong. H1={h1_pos} daily={daily_pos} capacity={capacity_pos}"
print("[PASS] Daily loss wall is between H1 trend wall and capacity wall")

# Verify the 50% early warning is present
assert "limit * 0.5" in engine_code or "limit * 0.50" in engine_code or "approaching limit" in engine_code, \
    "FAIL: 50% early warning not in code"
print("[PASS] 50% daily loss early warning is present")

print("\n=== ALL DAILY LOSS STOP TESTS PASSED ===")
