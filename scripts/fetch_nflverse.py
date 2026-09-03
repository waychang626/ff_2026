#!/usr/bin/env python3
"""Pull observed weekly usage from nflverse. Free, no auth, updated nightly.

The projection scrape in `R/pull_projections.R` returns what each site
*forecasts*. This returns what actually happened: snap share, target share,
carries, routes. They answer different questions, and the second one is the
only way to see a role change before the consensus has finished pricing it.

Two nflverse release assets, both plain CSV over HTTPS:

  player_stats  per player-week: carries, targets, target_share, air_yards_share
  snap_counts   per player-game: offense_snaps and offense_pct

Written as one tidy file with a `fetched_at` stamp, so `ffdraft lineup` can
apply the same freshness rules it applies to projections.

  python scripts/fetch_nflverse.py --season 2025 --out data/usage_2025.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffdraft.ids import POSITIONS, make_player_id  # noqa: E402

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
STATS_URL = BASE + "/player_stats/player_stats_{season}.csv"
SNAPS_URL = BASE + "/snap_counts/snap_counts_{season}.csv"


def _get(url: str, timeout: float) -> list[dict]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise SystemExit(f"{url} returned HTTP {response.status}")
        text = response.read().decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def _f(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", help="default data/usage_<season>.csv")
    ap.add_argument("--through-week", type=int,
                    help="drop weeks after this one (for reproducing an old view)")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    out = Path(args.out or f"data/usage_{args.season}.csv")
    stamp = datetime.now(timezone.utc).isoformat()

    print(f"fetching player_stats {args.season} ...", flush=True)
    stats = _get(STATS_URL.format(season=args.season), args.timeout)
    print(f"  {len(stats):,} player-weeks", flush=True)
    print(f"fetching snap_counts {args.season} ...", flush=True)
    snaps = _get(SNAPS_URL.format(season=args.season), args.timeout)
    print(f"  {len(snaps):,} player-games", flush=True)

    # snap_counts keys on a name and a PFR id; join on (name slug, position,
    # week), which is what the rest of this project already keys on.
    snap_pct: dict[tuple[str, int], float] = {}
    for row in snaps:
        pos = (row.get("position") or "").upper()
        if pos not in POSITIONS:
            continue
        try:
            pid = make_player_id(row.get("player", ""), pos)
        except ValueError:
            continue
        week = int(_f(row.get("week"), -1))
        if week < 0:
            continue
        snap_pct[(pid, week)] = _f(row.get("offense_pct"))

    rows = []
    seen: set[tuple[str, int]] = set()
    for row in stats:
        if (row.get("season_type") or "REG") not in ("REG", "POST"):
            continue
        pos = (row.get("position") or "").upper()
        if pos not in POSITIONS:
            continue
        try:
            pid = make_player_id(
                row.get("player_display_name") or row.get("player_name", ""), pos
            )
        except ValueError:
            continue
        week = int(_f(row.get("week"), -1))
        if week < 0 or (args.through_week and week > args.through_week):
            continue
        if (pid, week) in seen:
            continue
        seen.add((pid, week))
        rows.append({
            "fetched_at": stamp,
            "season": args.season,
            "week": week,
            "player_id": pid,
            "player": row.get("player_display_name") or row.get("player_name", ""),
            "pos": pos,
            "team": row.get("recent_team", ""),
            "snap_pct": f"{snap_pct.get((pid, week), 0.0):.4f}",
            "carries": f"{_f(row.get('carries')):.0f}",
            "targets": f"{_f(row.get('targets')):.0f}",
            "target_share": f"{_f(row.get('target_share')):.4f}",
            "air_yards_share": f"{_f(row.get('air_yards_share')):.4f}",
            "fantasy_points_ppr": f"{_f(row.get('fantasy_points_ppr')):.2f}",
        })

    if not rows:
        raise SystemExit(f"no usable rows for season {args.season}")
    rows.sort(key=lambda r: (r["player_id"], r["week"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    weeks = sorted({int(r["week"]) for r in rows})
    covered = sum(1 for r in rows if float(r["snap_pct"]) > 0)
    print(f"wrote {out}: {len(rows):,} player-weeks, "
          f"weeks {weeks[0]}-{weeks[-1]}, "
          f"{covered:,} with snap share ({covered / len(rows):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
