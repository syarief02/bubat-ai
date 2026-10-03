"""Append-only JSONL journal: the brain's long-term memory of what it saw, learned and did."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, kind: str, **fields: Any) -> Dict[str, Any]:
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def recent(self, n: int = 20, kinds: tuple = ()) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not kinds or e.get("kind") in kinds:
                out.append(e)
        return out[-n:]
