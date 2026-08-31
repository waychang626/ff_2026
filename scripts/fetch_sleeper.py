#!/usr/bin/env python3
"""Fetch canonical player IDs and bye weeks from the Sleeper API.

Free, read-only, no auth. Stay under 1000 calls/min; the player list is ~5MB,
so it is cached to disk and refreshed at most once a day.

NOT EXERCISED AGAINST THE LIVE API in the container this was written in -
outbound access there is allowlisted and does not include api.sleeper.app. The
request shapes follow Sleeper's documented v1 endpoints. Run it once locally
before draft day.

Sleeper does not publish a season-long ADP endpoint; ADP in this project comes
from the ffanalytics pull (add_adp) or a FantasyPros export. What Sleeper is
genuinely good for is a canonical player list with byes and team changes, which
is what this writes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("pip install requests")

PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
STATE_URL = "https://api.sleeper.app/v1/state/nfl"
CACHE_MAX_AGE_S = 24 * 3600
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}


def fetch_players(cache: Path, force: bool = False) -> dict:
    if cache.exists() and not force:
        age = time.time() - cache.stat().st_mtime
        if age < CACHE_MAX_AGE_S:
            print(f"using cached player list ({age / 3600:.1f}h old)")
            return json.loads(cache.read_text())
    print("fetching ~5MB player list from Sleeper ...")
    response = requests.get(PLAYERS_URL, timeout=90)
    response.raise_for_status()
    data = response.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/raw/sleeper_players.json")
    ap.add_argument("--out", default="data/sleeper_ids.csv")
    ap.add_argument("--force", action="store_true", help="ignore the daily cache")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ffdraft.ids import make_player_id

    players = fetch_players(Path(args.cache), force=args.force)

    import csv

    rows = []
    for sleeper_id, player in players.items():
        pos = (player.get("position") or "").upper()
        if pos not in FANTASY_POSITIONS:
            continue
        if player.get("status") in ("Inactive",) and pos != "DEF":
            continue
        name = player.get("full_name") or player.get("last_name") or ""
        if pos == "DEF":
            name = f"{player.get('team') or sleeper_id} DST"
        if not name.strip():
            continue
        try:
            pid = make_player_id(name, pos)
        except ValueError:
            continue
        rows.append(
            {
                "player_id": pid,
                "sleeper_id": sleeper_id,
                "player": name,
                "pos": "DST" if pos == "DEF" else pos,
                "team": player.get("team") or "",
                "bye": player.get("bye_week") or 0,
                "years_exp": player.get("years_exp") or 0,
            }
        )

    rows.sort(key=lambda r: r["player_id"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["player_id", "sleeper_id", "player", "pos", "team", "bye", "years_exp"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}: {len(rows)} players")


if __name__ == "__main__":
    main()
