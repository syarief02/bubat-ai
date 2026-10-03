"""
Brain model benchmark: how well a model reasons over the brain's real inputs (no MT5, no orders,
nothing is saved to the brain's journal or proposals).

1. Data-reading quiz: five questions about a real report digest, answers computed in code.
2. Small-sample trap: a report with < 30 trades; a careful model proposes no config/symbol/rule change.
3. Proposal quality: share of proposals on a real report that pass brain/proposals.py validation.

    python maintenance/brain_benchmark.py --report reports/<5-day>.json --thin reports/<thin>.json \
        qwen3:8b gpt-oss:20b qwen3:30b

Each model can carry a think level: gpt-oss:20b@high, qwen3:30b@true.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))
os.chdir(AGENT_DIR)

from brain.agent import REFLECT_SYSTEM
from brain.digest import MIN_SAMPLE, build_digest
from brain.llm import BrainLLM
from brain.proposals import validate

QUIZ_SYSTEM = """You analyse a forex trading bot's performance digest. Answer from the numbers only.
Reply with JSON only: {"q1": "...", "q2": "...", "q3": 0, "q4": "...", "q5": "..."}"""


def quiz(digest):
    sessions = {k: v for k, v in digest["by_session"].items() if (v.get("trades") or 0) >= MIN_SAMPLE}
    worst_session = min(sessions, key=lambda k: sessions[k]["avg_r"])
    base = digest["baselines_avg_r"]
    h1_beats = base["follow_h1_trend"]["avg_r"] > base["llm_all_buy_sell"]["avg_r"]
    sells = digest["by_direction"].get("SELL", {}).get("trades", 0)
    wall, w = next((k, v) for k, v in digest["blocked_by_wall_avg_r"].items() if (v.get("n") or 0) >= 20)
    worst_sym = digest["by_symbol"]["worst"][0][0]
    questions = {
        "q1": f"Among sessions with at least {MIN_SAMPLE} trades, which session has the lowest avg_r? Answer the session name.",
        "q2": "Does the follow_h1_trend baseline have a higher avg_r than llm_all_buy_sell? Answer yes or no.",
        "q3": "How many SELL trades are in this window? Answer a number.",
        "q4": f"blocked_by_wall_avg_r shows the simulated avg_r of signals the {wall} wall blocked. "
              f"Did that wall help or hurt the bot? Answer helped or hurt.",
        "q5": "Which symbol has the worst net_pnl? Answer the symbol.",
    }
    answers = {"q1": worst_session.lower(), "q2": "yes" if h1_beats else "no", "q3": sells,
               "q4": "helped" if w["avg_r"] < 0 else "hurt", "q5": worst_sym.lower()}
    return questions, answers


def score_quiz(reply, answers):
    marks = {}
    for k, want in answers.items():
        got = (reply or {}).get(k)
        if isinstance(want, int):
            try:
                marks[k] = int(float(str(got).split()[0])) == want
            except (ValueError, IndexError):
                marks[k] = False
        else:
            g = str(got or "").lower().replace(" ", "_")
            marks[k] = g.startswith(want) or g == want or (want in g and len(g) <= len(want) + 12)
    return marks


def run_model(spec, digest, thin_digest, cfg):
    name, _, think = spec.partition("@")
    think = {"true": True, "false": False, "": True}.get(think.lower(), think.lower())
    llm = BrainLLM({**cfg, "brain": {**cfg.get("brain", {}), "model": name, "think": think,
                                    "num_ctx": 16384, "num_predict": 12000, "timeout_seconds": 1800}})
    out = {"model": spec}

    questions, answers = quiz(digest)
    t = time.time()
    reply = llm.ask_json(QUIZ_SYSTEM, json.dumps({"digest": digest, "questions": questions}, default=str))
    out["quiz_seconds"] = round(time.time() - t, 1)
    marks = score_quiz(reply, answers)
    out["quiz"] = f"{sum(marks.values())}/5"
    out["quiz_detail"] = {k: (reply or {}).get(k) for k in answers}

    prompt = lambda d: json.dumps({"today": d, "your_memory": [], "proposals": {"open": [], "applied": [],
                                   "rejected_or_rolled_back": []}}, default=str)
    t = time.time()
    trap = llm.ask_json(REFLECT_SYSTEM, prompt(thin_digest)) or {}
    out["trap_seconds"] = round(time.time() - t, 1)
    changes = [p for p in trap.get("proposals") or [] if isinstance(p, dict) and p.get("kind") != "idea"]
    out["trap"] = "pass" if not changes else f"FAIL ({len(changes)} changes on {thin_digest['summary'].get('trades')} trades)"

    t = time.time()
    real = llm.ask_json(REFLECT_SYSTEM, prompt(digest)) or {}
    out["reflect_seconds"] = round(time.time() - t, 1)
    props = [p for p in real.get("proposals") or [] if isinstance(p, dict)]
    results = [validate(p, cfg, digest["sample_sizes"]) for p in props]
    out["reflect_stats"] = llm.last_stats
    out["proposals_valid"] = f"{sum(ok for ok, _, _ in results)}/{len(props)}"
    out["refusals"] = [why for ok, why, _ in results if not ok]
    out["assessment"] = str(real.get("assessment", ""))[:500]
    out["proposals"] = [f"{n.get('kind')}: {n.get('key') or n.get('symbol') or ''} {n.get('value', '')} "
                        f"[{n.get('risk')}]" for ok, _, n in results if ok]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="report JSON with a real sample (>= 30 trades)")
    ap.add_argument("--thin", required=True, help="report JSON with fewer than 30 trades")
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()
    cfg = json.loads((AGENT_DIR / "config.json").read_text(encoding="utf-8"))
    digest = build_digest(json.loads(Path(args.report).read_text(encoding="utf-8")))
    thin = build_digest(json.loads(Path(args.thin).read_text(encoding="utf-8")))
    _, answers = quiz(digest)
    print("Quiz answer key:", answers)
    for spec in args.models:
        print(json.dumps(run_model(spec, digest, thin, cfg), indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    main()
