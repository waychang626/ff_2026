"""Append-only audit log: (state_hash, output, timestamp) for every call.

Two jobs. During the draft it is the record of what the engine actually said,
so a disagreement afterwards is settled by reading rather than remembering.
Before the draft it is the determinism proof: the replay harness re-runs a
recorded sequence and asserts the same state hash produces the same output,
byte for byte.

It is also the dataset the brief asks for - every pick, in order, with a
timestamp. Free to collect live, expensive to reconstruct later.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def state_hash(
    league_fingerprint: str,
    board_fingerprint: str,
    drafted: list[str],
    my_seat: int | None,
    pick_number: int,
) -> str:
    """Canonical hash of everything that determines a recommendation."""
    payload = {
        "league": league_fingerprint,
        "board": board_fingerprint,
        "drafted": list(drafted),
        "my_seat": my_seat,
        "pick_number": pick_number,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, float):
        # Round on the way in so a log comparison is not defeated by the last
        # bit of a float that no human would call a difference.
        return round(obj, 8)
    if hasattr(obj, "item"):
        try:
            return _plain(obj.item())
        except Exception:
            return str(obj)
    return obj


@dataclass
class AuditEntry:
    timestamp: str
    state_hash: str
    league_id: str
    pick_number: int
    kind: str
    payload: dict[str, Any]

    def output_digest(self) -> str:
        blob = json.dumps(self.payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class AuditLog:
    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path(os.environ.get("FFDRAFT_LOG_DIR", "logs")) / "audit.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        state_hash: str,
        league_id: str,
        pick_number: int,
        kind: str,
        payload: Any,
        timestamp: str | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            state_hash=state_hash,
            league_id=league_id,
            pick_number=pick_number,
            kind=kind,
            payload=_plain(payload),
        )
        with self.path.open("a") as handle:
            handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[AuditEntry]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                out.append(AuditEntry(**json.loads(line)))
        return out
