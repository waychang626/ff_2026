#!/usr/bin/env python3
"""Generate a SYNTHETIC projection set for tests, replay and benchmarking.

This is not real data and must never be treated as such. It exists so the
engine, the replay harness and the backtester can be exercised end to end
without a live scrape - the container this was built in has no R and no reach
to the projection sources.

The shape is deliberately realistic: declining production curves per position,
a starter cliff, per-source disagreement that widens down the board, and an ADP
that mostly-but-not-exactly tracks projected points. Deterministic given
--seed, so a fixture regenerated on another machine is identical.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

SOURCES = ["CBS", "ESPN", "FantasyPros", "FFToday", "NumberFire", "NFL"]

NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
    "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE",
    "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]

TEAM_NICKNAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
    "LAC": "Chargers", "LAR": "Rams", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
    "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders",
}

FIRST = [
    "Marcus", "Devon", "Trey", "Jalen", "Cooper", "Amari", "Tyreek", "Bijan",
    "Garrett", "Chase", "Rome", "Malik", "Xavier", "Drake", "Kyren", "Brock",
    "Jayden", "Rashee", "Tank", "Zay", "Puka", "Jaxon", "Ladd", "Quentin",
    "Isaiah", "Deven", "Roman", "Caleb", "Bo", "Tucker", "Emeka", "Blake",
    "Keon", "Xzavier", "Marvin", "Braelon", "Jonathon", "Ricky", "Audric",
    "Kimani", "Dylan", "Troy", "Javon", "Ainias", "Elic", "Grant", "Theo",
]
LAST = [
    "Robinson", "Jefferson", "Hill", "Adams", "Brown", "Wilson", "Carter",
    "Nabers", "Odunze", "Harrison", "Bowers", "Worthy", "Franklin", "Legette",
    "Coleman", "Mitchell", "McConkey", "Thomas", "Corum", "Wright", "Allen",
    "Daniels", "Maye", "Penix", "Nix", "Rice", "Bigsby", "Spears", "Tracy",
    "Shipley", "Benson", "Guerendo", "Estime", "Irving", "Brooks", "Hall",
    "Walker", "Gibbs", "Etienne", "Pollard", "Swift", "Dowdle", "Mason",
]


def _name(rng: random.Random, used: set[str]) -> str:
    for _ in range(500):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if name not in used:
            used.add(name)
            return name
    n = len(used)
    name = f"Player {n}"
    used.add(name)
    return name


def decay(rank: int, half_life: float, cliff: int, cliff_drop: float) -> float:
    """Declining production curve with a starter cliff."""
    value = math.exp(-math.log(2) * (rank - 1) / half_life)
    if rank > cliff:
        value *= cliff_drop ** ((rank - cliff) / 8.0)
    return value


def base_stats(pos: str, rank: int) -> dict[str, float]:
    if pos == "QB":
        d = decay(rank, 22, 20, 0.55)
        rushing = 1.0 if rank % 3 == 0 else 0.25   # a third of QBs run
        return {
            "pass_yds": 4700 * d, "pass_tds": 33 * d, "pass_int": 9 * (0.6 + 0.4 * d),
            "rush_yds": 620 * d * rushing, "rush_tds": 6 * d * rushing,
            "fumbles_lost": 2.2 * (0.6 + 0.4 * d), "two_pts": 0.5 * d,
        }
    if pos == "RB":
        d = decay(rank, 16, 30, 0.6)
        return {
            "rush_yds": 1350 * d, "rush_tds": 11 * d, "rec": 52 * d,
            "rec_yds": 430 * d, "rec_tds": 2.4 * d, "fumbles_lost": 1.4 * (0.5 + 0.5 * d),
        }
    if pos == "WR":
        d = decay(rank, 22, 45, 0.6)
        return {
            "rec": 98 * d, "rec_yds": 1370 * d, "rec_tds": 8.6 * d,
            "rush_yds": 40 * d, "fumbles_lost": 0.8 * (0.5 + 0.5 * d),
        }
    if pos == "TE":
        d = decay(rank, 12, 16, 0.5)
        return {"rec": 84 * d, "rec_yds": 940 * d, "rec_tds": 7.0 * d}
    if pos == "K":
        d = decay(rank, 40, 30, 0.7)
        return {
            "xp": 36 * d, "fg_0019": 0.3 * d, "fg_2029": 6 * d, "fg_3039": 9 * d,
            "fg_4049": 7 * d, "fg_50": 3.4 * d, "fg_miss": 4.5 * (1.4 - 0.4 * d),
        }
    d = decay(rank, 34, 26, 0.7)
    return {
        "dst_sacks": 44 * d, "dst_int": 13 * d, "dst_fum_rec": 9 * d,
        "dst_td": 3.1 * d, "dst_safety": 0.5 * d, "dst_blk": 0.6 * d,
        "dst_forced_fumble": 12 * d, "dst_pts_allowed": 19.5 / max(d, 0.45),
    }


POSITION_COUNTS = {"QB": 40, "RB": 72, "WR": 92, "TE": 40, "K": 32, "DST": 32}

# Where the market drafts each position relative to raw projected value.
ADP_POSITION_SHIFT = {"QB": 22.0, "RB": -6.0, "WR": -2.0, "TE": 6.0, "K": 150.0, "DST": 140.0}


def build(seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    used: set[str] = set()
    roster: list[tuple[str, str, str, int]] = []   # name, pos, team, bye

    byes = {team: 5 + (i % 10) for i, team in enumerate(NFL_TEAMS)}

    for pos, count in POSITION_COUNTS.items():
        for rank in range(1, count + 1):
            if pos == "DST":
                team = NFL_TEAMS[rank - 1]
                name = f"{TEAM_NICKNAMES[team]} DST"
            else:
                team = NFL_TEAMS[rng.randrange(len(NFL_TEAMS))]
                name = _name(rng, used)
            roster.append((name, pos, team, byes[team]))

    stat_rows: list[dict] = []
    consensus: dict[tuple[str, str], float] = {}
    per_pos_rank: dict[str, int] = {}

    for name, pos, team, bye in roster:
        per_pos_rank[pos] = per_pos_rank.get(pos, 0) + 1
        rank = per_pos_rank[pos]
        stats = base_stats(pos, rank)
        # Disagreement widens down the board: sources agree about the obvious.
        spread = 0.05 + 0.10 * min(rank / 40.0, 1.0)
        for source in SOURCES:
            row = {"source": source, "player": name, "pos": pos, "team": team, "bye": bye}
            for key, value in stats.items():
                row[key] = round(max(0.0, value * rng.gauss(1.0, spread)), 3)
            stat_rows.append(row)
        consensus[(name, pos)] = sum(stats.values())

    # ADP: rank by a crude scoring proxy, then shift by position and jitter.
    scored = []
    per_pos_rank = {}
    for name, pos, team, bye in roster:
        per_pos_rank[pos] = per_pos_rank.get(pos, 0) + 1
        rank = per_pos_rank[pos]
        stats = base_stats(pos, rank)
        points = (
            stats.get("pass_yds", 0) * 0.04 + stats.get("pass_tds", 0) * 4
            - stats.get("pass_int", 0) + stats.get("rush_yds", 0) * 0.1
            + stats.get("rush_tds", 0) * 6 + stats.get("rec", 0) * 0.5
            + stats.get("rec_yds", 0) * 0.1 + stats.get("rec_tds", 0) * 6
            + stats.get("xp", 0) + stats.get("fg_3039", 0) * 3
            + stats.get("fg_4049", 0) * 4 + stats.get("fg_50", 0) * 5
            + stats.get("dst_sacks", 0) + stats.get("dst_int", 0) * 2
            + stats.get("dst_td", 0) * 6
        )
        scored.append((points, name, pos))
    scored.sort(reverse=True)

    market_rows = []
    ranked = []
    for value_rank, (points, name, pos) in enumerate(scored, start=1):
        adp = value_rank + ADP_POSITION_SHIFT.get(pos, 0.0) + rng.gauss(0, 6)
        ranked.append((max(1.0, adp), name, pos))
    ranked.sort()
    for adp_rank, (_, name, pos) in enumerate(ranked, start=1):
        market_rows.append(
            {
                "player": name, "pos": pos, "adp": adp_rank,
                "adp_sd": round(6 + 0.09 * adp_rank, 2), "ecr": adp_rank,
            }
        )
    return stat_rows, market_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default="data/samples")
    args = ap.parse_args()

    stat_rows, market_rows = build(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stat_cols = ["source", "player", "pos", "team", "bye"]
    for row in stat_rows:
        for key in row:
            if key not in stat_cols:
                stat_cols.append(key)

    proj_path = out / "projections_synthetic.csv"
    with proj_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stat_cols, restval=0)
        writer.writeheader()
        writer.writerows(stat_rows)

    market_path = out / "market_synthetic.csv"
    with market_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["player", "pos", "adp", "adp_sd", "ecr"])
        writer.writeheader()
        writer.writerows(market_rows)

    print(f"wrote {proj_path} ({len(stat_rows)} rows)")
    print(f"wrote {market_path} ({len(market_rows)} rows)")


if __name__ == "__main__":
    main()
