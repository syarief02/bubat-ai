"""
Bubat Brain command line (run from forex_local_agent/):

    python -m brain run [--force] [--no-research]   daily cycle (skips if it already ran today)
    python -m brain status                          last assessment + proposals waiting for you
    python -m brain list [--all]                    proposals (open only, or every one)
    python -m brain approve P3                      apply P3, run the test suites, roll back on failure
    python -m brain reject P3 [reason]              reject P3 (the brain will not propose it again)
    python -m brain journal [N]                     last N journal entries
"""
import argparse
import json
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from brain.agent import Brain, _describe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m brain", description="Bubat Brain: daily research + gated proposals")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--force", action="store_true", help="run even if it already ran today")
    r.add_argument("--no-research", action="store_true", help="skip the web research step")
    sub.add_parser("status")
    ls = sub.add_parser("list")
    ls.add_argument("--all", action="store_true")
    a = sub.add_parser("approve")
    a.add_argument("id")
    rj = sub.add_parser("reject")
    rj.add_argument("id")
    rj.add_argument("reason", nargs="*")
    j = sub.add_parser("journal")
    j.add_argument("n", nargs="?", type=int, default=10)
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console defaults to cp1252
    except AttributeError:
        pass

    brain = Brain()
    if args.cmd == "run":
        if not brain.cfg.get("enabled", True):
            print("Brain disabled (config brain.enabled = false).")
            return 0
        if brain.already_ran_today() and not args.force:
            print("Brain already ran today (use --force to run again).")
            return 0
        res = brain.run(research=not args.no_research)
        if res["status"] != "ok":
            print(f"Brain run ended: {res['status']}")
            return 1
        print((brain.state_dir / "inbox.md").read_text(encoding="utf-8"))
        return 0

    if args.cmd == "status":
        inbox = brain.state_dir / "inbox.md"
        print(inbox.read_text(encoding="utf-8") if inbox.exists() else "No brain run yet.")
        return 0

    if args.cmd == "list":
        items = brain.proposals.load()
        shown = items if args.all else [p for p in items if p["status"] in ("proposed", "noted")]
        if not shown:
            print("No proposals.")
        for p in shown:
            print(f"{p['id']:>4}  {p['status']:<11} [{p['risk']}] {_describe(p)}")
        return 0

    if args.cmd == "approve":
        p = brain.proposals.get(args.id)
        if p:
            print(f"Applying {p['id']}: {_describe(p)} [{p['risk']}] — running test suites...")
        last = brain.journal.recent(1, kinds=("observation",))
        ok, detail = brain.proposals.approve(args.id, before=last[-1]["headline"] if last else None)
        brain.journal.add("owner_decision", id=args.id, decision="approve", ok=ok, detail=detail)
        print(("Applied. " if ok else "Not applied. ") + detail)
        if ok and p and p["kind"] != "rule":
            print("Restart the trading agent (run_agent.bat) for the config change to take effect.")
        return 0 if ok else 1

    if args.cmd == "reject":
        p = brain.proposals.reject(args.id, " ".join(args.reason))
        if not p:
            print(f"No proposal {args.id}")
            return 1
        brain.journal.add("owner_decision", id=args.id, decision="reject", note=" ".join(args.reason))
        print(f"Rejected {args.id}: {_describe(p)}")
        return 0

    if args.cmd == "journal":
        for e in brain.journal.recent(args.n):
            e = dict(e)
            e.pop("digest", None)
            print(json.dumps(e, ensure_ascii=False, default=str)[:400])
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
