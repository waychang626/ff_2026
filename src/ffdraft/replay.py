"""Replay harness and backtester.

Section 7 puts this first in the build order, and for a good reason: it defines
the interfaces, and without it "the engine is deterministic" is a claim rather
than a fact. Two jobs, one mechanism:

  replay      feed a recorded pick sequence back through the engine and assert
              that identical state produces identical output. An unseeded or
              order-dependent bug shows up here in seconds instead of costing
              you an evening mid-draft chasing a bug that isn't one.

  backtest    the same replay, but the engine drafts for one seat instead of
              reading what that seat actually did. Score the resulting roster
              against real results and you have a measurement rather than an
              opinion.

The draft log format is deliberately boring - one JSON object per pick, in
order, with a timestamp. It is what the CLI writes live during a real draft,
which makes every draft you run a test case for the next one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .audit import AuditLog
from .board import Board
from .config import LeagueConfig
from .draft import DraftState, pick_owner
from .engine import PickAdvice, _recommend
from .lineup import best_lineup_points, lineup_slots


@dataclass
class LoggedPick:
    pick: int
    player_id: str
    seat: int
    timestamp: str = ""
    note: str = ""


@dataclass
class DraftLog:
    league_id: str
    picks: list[LoggedPick] = field(default_factory=list)
    my_seat: int | None = None

    @property
    def player_ids(self) -> list[str]:
        return [p.player_id for p in self.picks]

    def append(self, player_id: str, seat: int, note: str = "") -> LoggedPick:
        entry = LoggedPick(
            pick=len(self.picks) + 1,
            player_id=player_id,
            seat=seat,
            timestamp=datetime.now(timezone.utc).isoformat(),
            note=note,
        )
        self.picks.append(entry)
        return entry

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {"league_id": self.league_id, "my_seat": self.my_seat}
        with path.open("w") as handle:
            handle.write(json.dumps({"header": header}) + "\n")
            for entry in self.picks:
                handle.write(json.dumps(asdict(entry)) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "DraftLog":
        lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
        if not lines:
            raise ValueError(f"{path}: empty draft log")
        head = json.loads(lines[0])
        if "header" not in head:
            raise ValueError(f"{path}: first line must be a header object")
        log = cls(
            league_id=head["header"].get("league_id", ""),
            my_seat=head["header"].get("my_seat"),
        )
        for line in lines[1:]:
            log.picks.append(LoggedPick(**json.loads(line)))
        # A log whose pick numbers are not 1..n in order is a log that was
        # edited by hand, and every conclusion drawn from it would be suspect.
        expected = list(range(1, len(log.picks) + 1))
        actual = [p.pick for p in log.picks]
        if actual != expected:
            raise ValueError(
                f"{path}: pick numbers are {actual[:5]}... but must be 1..n in order"
            )
        return log


@dataclass
class ReplayResult:
    advices: list[PickAdvice]
    picks_evaluated: int

    @property
    def digests(self) -> list[tuple[int, str, tuple[str, ...]]]:
        """(pick_number, state_hash, ranked player ids) - the comparison key."""
        return [
            (a.pick_number, a.state_hash, tuple(r.player_id for r in a.recommendations))
            for a in self.advices
        ]


def replay(
    config: LeagueConfig,
    board: Board,
    log: DraftLog,
    seat: int | None = None,
    audit: AuditLog | None = None,
    limit: int | None = None,
) -> ReplayResult:
    """Re-run the engine at every pick the given seat owned."""
    seat = seat or log.my_seat or config.my_seat
    if seat is None:
        raise ValueError("replay needs a seat: pass one, or set it in the log header")

    advices: list[PickAdvice] = []
    drafted: list[str] = []
    for entry in log.picks:
        owner = pick_owner(entry.pick, config.teams, config.draft_type)
        if owner == seat:
            state = DraftState(config=config, drafted=list(drafted), my_seat=seat)
            advices.append(
                _recommend(
                    config, board, list(drafted), state.my_roster, entry.pick, audit=audit
                )
            )
            if limit is not None and len(advices) >= limit:
                break
        drafted.append(entry.player_id)
    return ReplayResult(advices=advices, picks_evaluated=len(advices))


def assert_deterministic(
    config: LeagueConfig,
    board: Board,
    log: DraftLog,
    seat: int | None = None,
    runs: int = 2,
    limit: int | None = None,
) -> None:
    """Replay the same log `runs` times and require identical output.

    This is the guarantee the whole design rests on. If it fails, something is
    reading an unseeded generator, iterating a set, or depending on dict order,
    and no number the engine produces can be trusted until it is found.
    """
    baseline = replay(config, board, log, seat, audit=_NullLog(), limit=limit).digests
    for run in range(2, runs + 1):
        again = replay(config, board, log, seat, audit=_NullLog(), limit=limit).digests
        if again != baseline:
            for (p1, h1, r1), (p2, h2, r2) in zip(baseline, again):
                if (p1, h1, r1) != (p2, h2, r2):
                    raise AssertionError(
                        f"run {run} diverged at pick {p1}: state {h1} vs {h2}; "
                        f"ranking {r1[:3]} vs {r2[:3]}"
                    )
            raise AssertionError(f"run {run} produced a different number of picks")


class _NullLog(AuditLog):
    """Audit sink for replays, so a determinism check does not spam the log."""

    def __init__(self) -> None:  # noqa: D107
        pass

    def record(self, **kwargs):  # type: ignore[override]
        return None


# --- backtesting -------------------------------------------------------------
@dataclass
class BacktestResult:
    seat: int
    engine_roster: list[str]
    actual_roster: list[str]
    engine_points: float
    actual_points: float
    engine_lineup: dict[str, str | None]
    drafted: list[str] = field(default_factory=list)   # the full re-simulated draft
    missing_actuals: list[str] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.engine_points - self.actual_points


def load_actuals(path: str | Path) -> dict[str, float]:
    """Read realised season totals: a CSV of player_id (or player,pos) and points."""
    import csv

    from .ids import make_player_id

    out: dict[str, float] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            pid = row.get("player_id")
            if not pid:
                name = (row.get("player") or row.get("name") or "").strip()
                pos = (row.get("pos") or row.get("position") or "").strip()
                if not name or not pos:
                    continue
                pid = make_player_id(name, pos)
            value = row.get("points") or row.get("actual") or row.get("fantasy_points")
            if value:
                out[pid] = float(value)
    return out


def backtest(
    config: LeagueConfig,
    board: Board,
    log: DraftLog,
    actuals: dict[str, float],
    seat: int | None = None,
) -> BacktestResult:
    """Let the engine draft one seat against a recorded draft, then score it.

    Opponents keep their recorded picks. When the engine takes a player an
    opponent went on to take, that opponent falls through to their next
    recorded pick that is still available, and to best-available-by-ADP if they
    run out. That keeps the rest of the room behaving as it really did instead
    of handing the engine a league of simulated pushovers.
    """
    seat = seat or log.my_seat or config.my_seat
    if seat is None:
        raise ValueError("backtest needs a seat")

    recorded = log.player_ids
    remaining_recorded = list(recorded)
    drafted: list[str] = []
    engine_roster: list[str] = []

    for pick in range(1, min(len(recorded), config.total_drafted) + 1):
        owner = pick_owner(pick, config.teams, config.draft_type)
        if owner == seat:
            advice = _recommend(
                config, board, list(drafted), list(engine_roster), pick, audit=_NullLog()
            )
            if not advice.recommendations:
                break
            choice = advice.recommendations[0].player_id
            engine_roster.append(choice)
        else:
            choice = _next_available(remaining_recorded, drafted, board)
            if choice is None:
                break
        drafted.append(choice)
        if choice in remaining_recorded:
            remaining_recorded.remove(choice)

    actual_roster = [
        pid for n, pid in enumerate(recorded, start=1)
        if pick_owner(n, config.teams, config.draft_type) == seat
    ]

    missing = sorted(
        {p for p in engine_roster + actual_roster if p not in actuals}
    )
    engine_pts, engine_lineup = _score_roster(board, engine_roster, actuals, config)
    actual_pts, _ = _score_roster(board, actual_roster, actuals, config)

    return BacktestResult(
        seat=seat,
        engine_roster=engine_roster,
        actual_roster=actual_roster,
        engine_points=engine_pts,
        actual_points=actual_pts,
        engine_lineup=engine_lineup,
        drafted=drafted,
        missing_actuals=missing,
    )


def _next_available(pool: list[str], drafted: list[str], board: Board) -> str | None:
    taken = set(drafted)
    for pid in pool:
        if pid not in taken:
            return pid
    live = [p.player_id for p in board.players if p.player_id not in taken]
    return live[0] if live else None


def _score_roster(
    board: Board, roster: list[str], actuals: dict[str, float], config: LeagueConfig
) -> tuple[float, dict[str, str | None]]:
    """Best-lineup season total using realised points.

    A season-total lineup is an upper bound on what the roster really scored,
    because it sets a lineup with hindsight. It is a fair comparison only
    because both rosters get the same treatment; for a weekly-accurate score,
    feed weekly actuals through the simulator instead.
    """
    positions = {pid: board.player(pid).pos for pid in roster if pid in board.index}
    scores = {pid: actuals.get(pid, 0.0) for pid in positions}
    total = best_lineup_points(scores, positions, config.roster)
    return total, lineup_slots(scores, positions, config.roster)


def synthetic_log(
    config: LeagueConfig, board: Board, seed: int = 11, seat: int = 1
) -> DraftLog:
    """A plausible mock draft, for tests and for exercising the harness.

    Uses the opponent model for every seat, so the resulting log looks like a
    real draft (runs and all) rather than a straight walk down ADP.
    """
    from .opponents import DraftSimulator

    rng = np.random.default_rng(seed)
    simulator = DraftSimulator(board, config, my_seat=seat)
    rankings = simulator.base_rankings(rng, 1)
    order = np.argsort(board.adp, kind="stable").astype(float)
    value = -order
    result = simulator.rollout([], value, rankings)

    log = DraftLog(league_id=config.league_id, my_seat=seat)
    rosters = result.rosters[0]
    fill = {team: 0 for team in range(config.teams)}
    for pick in range(1, config.total_drafted + 1):
        team = pick_owner(pick, config.teams, config.draft_type) - 1
        idx = int(rosters[team, fill[team]])
        fill[team] += 1
        log.append(board.players[idx].player_id, seat=team + 1)
    return log
