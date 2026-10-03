"""
Proposals: the only way the brain can change the trading bot.

Every proposal is validated in code (whitelisted keys, hard bounds), classified as SAFER or
RISKIER by code (never by the LLM), and waits for the owner's approval. Applying one backs up
the file it changes, runs the offline test suites, and restores the backup if any suite fails.

Not tunable here, on purpose: lot size, SL/TP geometry, daily loss limit, auto_approve,
adding symbols. Those stay owner-only edits.
"""
import copy
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from learning.reflexion_store import validate_reflexion_rule
from brain.digest import MIN_SAMPLE

# key -> (min, max, safer_when): safer_when +1 = a higher value is safer, -1 = a lower value is safer
TUNABLES: Dict[str, Tuple[float, float, int]] = {
    "risk_parameters.confidence_threshold": (0.70, 0.95, +1),
    "risk_parameters.max_open_trades": (1, 10, -1),
    "risk_parameters.max_spread_pips": (0.5, 3.5, -1),
    "risk_parameters.max_spread_sl_ratio": (0.02, 0.08, -1),
    "risk_parameters.symbol_cooldown_minutes": (15, 240, +1),
    "risk_parameters.news_blackout_pre_mins": (15, 90, +1),
    "risk_parameters.news_blackout_post_mins": (10, 60, +1),
    "risk_parameters.max_currency_exposure": (1, 5, -1),
}
SESSIONS = ("ASIA", "LONDON", "LONDON_NY_OVERLAP", "NEW_YORK", "ROLLOVER")
SESSION_TUNABLES = {"max_open_trades": (0, 10, -1), "loss_budget_pct": (0.1, 2.0, -1)}
INT_KEYS = {"max_open_trades", "symbol_cooldown_minutes", "news_blackout_pre_mins",
            "news_blackout_post_mins", "max_currency_exposure"}
RULE_CATEGORIES = {"STRATEGY", "RISK_MANAGEMENT", "NEWS", "SESSION"}
KINDS = {"config", "remove_symbol", "rule", "idea"}
MAX_OPEN = 6

DEFAULT_GATES = [
    "tests/test_session_profiles.py", "tests/test_session_strategies.py", "tests/test_execution_upgrade.py",
    "tests/test_cycle5_regressions.py", "tests/test_daily_loss_stop.py", "tests/test_market_hours.py",
    "tests/test_brain.py", "tests/test_chat_upgrade.py",
]


def tunable_bounds(key: str) -> Optional[Tuple[float, float, int]]:
    if key in TUNABLES:
        return TUNABLES[key]
    m = re.fullmatch(r"session_profiles\.([A-Z_]+)\.([a-z_]+)", key)
    if m and m.group(1) in SESSIONS and m.group(2) in SESSION_TUNABLES:
        return SESSION_TUNABLES[m.group(2)]
    return None


def get_path(cfg: Dict, key: str) -> Any:
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def set_path(cfg: Dict, key: str, value: Any):
    parts = key.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def validate(p: Dict[str, Any], cfg: Dict[str, Any], samples: Dict[str, Any] = None
             ) -> Tuple[bool, str, Dict[str, Any]]:
    """Check a raw LLM proposal against the current config; returns (ok, reason, normalized).

    `samples` (digest.sample_sizes) enforces the minimum sample at proposal time; it is omitted
    when re-validating at approval, where the sample was already checked.
    """
    kind = str(p.get("kind", "")).strip().lower()
    if kind not in KINDS:
        return False, f"unknown kind {kind!r}", {}
    if samples is not None and kind != "idea":
        if samples.get("total", 0) < MIN_SAMPLE:
            return False, f"only {samples.get('total', 0)} trades in window (minimum {MIN_SAMPLE})", {}
        if kind == "remove_symbol":
            sym = str(p.get("symbol", "")).strip().upper()
            n = samples.get("by_symbol", {}).get(sym, 0)
            if n < MIN_SAMPLE:
                return False, f"only {n} {sym} trades in window (minimum {MIN_SAMPLE})", {}
        m = re.fullmatch(r"session_profiles\.([A-Z_]+)\.[a-z_]+", str(p.get("key", "")).strip())
        if kind == "config" and m:
            n = samples.get("by_session", {}).get(m.group(1), 0)
            if n < MIN_SAMPLE:
                return False, f"only {n} {m.group(1)} trades in window (minimum {MIN_SAMPLE})", {}
    rationale = str(p.get("rationale", "")).strip()
    evidence = str(p.get("evidence", "")).strip()
    if len(rationale) < 20:
        return False, "rationale too short", {}
    if kind != "idea" and not re.search(r"\d", evidence):
        return False, "evidence must cite numbers from the report", {}
    out = {"kind": kind, "rationale": rationale[:600], "evidence": evidence[:400]}

    if kind == "config":
        key = str(p.get("key", "")).strip()
        bounds = tunable_bounds(key)
        if not bounds:
            return False, f"key {key!r} is not tunable", {}
        current = get_path(cfg, key)
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            return False, f"key {key!r} has no numeric current value", {}
        try:
            value = float(p.get("value"))
        except (TypeError, ValueError):
            return False, "value must be a number", {}
        lo, hi, safer_when = bounds
        if not lo <= value <= hi:
            return False, f"value {value} outside allowed range [{lo}, {hi}]", {}
        if key.split(".")[-1] in INT_KEYS:
            value = int(round(value))
        else:
            value = round(value, 4)
        if value == current:
            return False, "value equals current setting", {}
        risk = "SAFER" if (value - current) * safer_when > 0 else "RISKIER"
        out.update(key=key, value=value, current=current, risk=risk, target=key)
    elif kind == "remove_symbol":
        sym = str(p.get("symbol", "")).strip().upper()
        if sym not in cfg.get("trading", {}).get("symbols", []):
            return False, f"symbol {sym!r} is not traded", {}
        if len(cfg["trading"]["symbols"]) <= 5:
            return False, "keeping at least 5 symbols", {}
        out.update(symbol=sym, risk="SAFER", target=f"symbol:{sym}")
    elif kind == "rule":
        text = str(p.get("text", "")).strip()
        ok, why = validate_reflexion_rule(text)
        if not ok:
            return False, f"rule rejected: {why}", {}
        cat = str(p.get("category", "STRATEGY")).strip().upper()
        out.update(text=text[:400], category=cat if cat in RULE_CATEGORIES else "STRATEGY",
                   risk="REVIEW", target=f"rule:{text[:60].lower()}")
    else:  # idea: a code/strategy change only the owner (or a coding agent) can implement
        text = str(p.get("text", "") or rationale).strip()
        out.update(text=text[:800], risk="N/A", target=f"idea:{text[:60].lower()}")
    return True, "ok", out


def run_gates(gates: List[str] = None, timeout: int = 600) -> Tuple[bool, str]:
    for t in gates or DEFAULT_GATES:
        try:
            r = subprocess.run([sys.executable, t], cwd=AGENT_DIR, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"{t}: timeout"
        if r.returncode != 0:
            return False, f"{t}: exit {r.returncode}: {(r.stdout + r.stderr)[-300:]}"
    return True, f"{len(gates or DEFAULT_GATES)} suites passed"


class ProposalStore:
    def __init__(self, path: Path, config_path: Path, rules_path: Path = None):
        self.path = Path(path)
        self.config_path = Path(config_path)
        self.rules_path = Path(rules_path or AGENT_DIR / "learning" / "learned_rules.md")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _save(self, items: List[Dict[str, Any]]):
        self.path.write_text(json.dumps(items, indent=2, default=str), encoding="utf-8")

    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        return next((p for p in self.load() if p["id"].upper() == pid.upper()), None)

    def open_items(self) -> List[Dict[str, Any]]:
        return [p for p in self.load() if p["status"] == "proposed"]

    def add(self, raw: Dict[str, Any], before: Dict[str, Any], samples: Dict[str, Any] = None
            ) -> Tuple[Optional[Dict], str]:
        cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        ok, why, norm = validate(raw, cfg, samples)
        if not ok:
            return None, why
        items = self.load()
        live = [p for p in items if p["status"] in ("proposed", "noted")]
        if any(p.get("target") == norm["target"] for p in live):
            return None, "duplicate of an open proposal"
        if len([p for p in items if p["status"] == "proposed"]) >= MAX_OPEN:
            return None, "too many open proposals"
        norm.update(id=f"P{len(items) + 1}", status="noted" if norm["kind"] == "idea" else "proposed",
                    created=_now(), before=before)
        items.append(norm)
        self._save(items)
        return norm, "ok"

    def reject(self, pid: str, note: str = "") -> Optional[Dict]:
        return self._update(pid, status="rejected", decided=_now(), note=note)

    def approve(self, pid: str, before: Dict = None, gate_runner: Callable[[], Tuple[bool, str]] = run_gates
                ) -> Tuple[bool, str]:
        """Owner approval: re-validate, apply with backup, run gates, roll back on failure."""
        p = self.get(pid)
        if not p:
            return False, f"no proposal {pid}"
        if p["status"] != "proposed":
            return False, f"{pid} is {p['status']}, not proposed"
        cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        ok, why, norm = validate(p, cfg)
        if not ok:
            self._update(pid, status="stale", decided=_now(), note=why)
            return False, f"no longer valid: {why}"

        target = self.rules_path if p["kind"] == "rule" else self.config_path
        backup = target.read_text(encoding="utf-8")
        try:
            if p["kind"] == "rule":
                _append_rule(self.rules_path, p, pid)
            else:
                new_cfg = copy.deepcopy(cfg)
                if p["kind"] == "config":
                    set_path(new_cfg, p["key"], norm["value"])
                else:
                    new_cfg["trading"]["symbols"] = [s for s in new_cfg["trading"]["symbols"] if s != p["symbol"]]
                self.config_path.write_text(json.dumps(new_cfg, indent=2) + "\n", encoding="utf-8")
            passed, detail = gate_runner()
        except Exception as e:
            passed, detail = False, f"apply error: {e}"
        if not passed:
            target.write_text(backup, encoding="utf-8")
            self._update(pid, status="rolled_back", decided=_now(), note=detail)
            return False, f"gates failed, change rolled back: {detail}"
        # current/risk are re-derived at approval: the config may have changed since the proposal
        self._update(pid, status="applied", decided=_now(), applied=_now(), note=detail,
                     before=before or p.get("before"), current=norm.get("current"), risk=norm.get("risk"))
        return True, detail

    def _update(self, pid: str, **fields) -> Optional[Dict]:
        items = self.load()
        for p in items:
            if p["id"].upper() == pid.upper():
                p.update(fields)
                self._save(items)
                return p
        return None


def _append_rule(rules_path: Path, p: Dict, pid: str):
    entry = (f"\n### [{p['category']}] Rule #brain-{pid}\n"
             f"- **Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
             f"- **Source**: brain_proposal (owner-approved)\n"
             f"- **Directive**: {p['text']}\n")
    with open(rules_path, "a", encoding="utf-8") as f:
        f.write(entry)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
