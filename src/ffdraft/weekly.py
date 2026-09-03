"""Weekly lineup: who to start this week, against this opponent.

The draft engine answers a season-long question from season-long projections.
This answers a seven-day question, and almost nothing about it survives from
the draft board. A season projection says Player A outscores Player B over
seventeen weeks; it does not know that A is on a bye, that A tore something on
Thursday, or that B draws the worst run defence in the league. Starting the
draft board in week 9 is starting a number that was true in August.

So the input is a *weekly* file, and the first thing this module does is refuse
to use one it cannot vouch for.

STALENESS IS A REFUSAL, NOT A WARNING
-------------------------------------
A warning printed above a lineup gets read after the lineup, which is to say
never. The failure this guards against is silent and total: last week's file
parses perfectly, produces a confident lineup, and starts a player who is
already ruled out. There is no output that says so, because every number in it
is internally consistent. It is just seven days old.

`check_freshness` therefore raises. Three rules, escalating:

  1. The file's week must be the week you asked for. A week-3 file used in
     week 4 is the worst case and the cheapest to catch.
  2. It must be younger than `max_age_hours`. Default 24: NFL practice reports
     land Wednesday through Friday and a Wednesday file is wrong by Sunday.
  3. If anyone you would start is QUESTIONABLE or DOUBTFUL, the bar tightens to
     `active_max_age_hours` (default 3). Those tags resolve in the ninety
     minutes before kickoff, and that resolution is the single highest-value
     piece of information in the whole week.

`--allow-stale` exists because a person who understands the risk should not be
blocked by a tool. It prints what it is overriding, every time.

WHAT "OPTIMAL" MEANS HERE
-------------------------
Two different questions, and the tool answers whichever it has the data for.

Without an opponent it maximises expected points, which the greedy fill in
`lineup.py` already solves exactly for nested slots.

With one, it maximises P(you win), and those are not the same lineup. Facing a
team projected forty points above you, the high-floor start is the one that
loses by less; you want the volatile one that might spike. Facing a team you
outclass, variance is the enemy and the boring lineup is correct. This is the
same endogenous risk appetite the draft engine gets from maximising P(title),
one week wide.

WHY EQUAL WEIGHTS ACROSS SOURCES
--------------------------------
The same rule `projections.py` uses, and for a reason with a name. The
"forecast combination puzzle" (Stock and Watson 2004; Smith and Wallis 2009) is
the durable finding that a simple average of forecasts beats combinations using
weights estimated from past accuracy. The explanation is that estimating the
weights adds variance that swamps the bias it removes, and it bites hardest
when the history is short - which a fantasy week is. Weighting sources by how
they did last week is exactly the mistake the literature describes.

So sources are averaged flat, and their *disagreement* is kept rather than
discarded: `_weekly_sd` adds the cross-source spread in quadrature with the
league's weekly CV. Five sources clustered in two points is a different bet
from five spread over fifteen, and it is the second case where the start/sit
call is actually hard.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .board import Board
from .config import SLOT_ELIGIBILITY, LeagueConfig, ScoringRules
from .draft import DraftState
from .projections import _META_COLUMNS, _NULLISH, _f
from .scoring import score_stats
from .lineup import slot_fill_order
from .simulate import lognormal_params

# Statuses that mean the player cannot score this week, whatever the projection.
OUT_STATUSES = frozenset({"OUT", "IR", "BYE", "SUSPENDED", "DNP", "PUP"})
# Statuses that resolve shortly before kickoff, and so demand fresher data.
DOUBTFUL_STATUSES = frozenset({"QUESTIONABLE", "DOUBTFUL", "GTD"})
ACTIVE_STATUSES = frozenset({"ACTIVE", "OK", "PROBABLE", "", "PLAYING"})

# Weekly-only identifying columns, on top of the draft file's meta columns.
_WEEKLY_META = {"week", "fetched_at", "status", "opponent", "sd",
                "points", "projection", "data_src"}


class StaleDataError(RuntimeError):
    """The weekly file cannot be vouched for. Raised, never printed and ignored."""


class WeeklyDataError(ValueError):
    """The weekly file is malformed."""


@dataclass
class SourceMeta:
    """One source's contribution, and when it was pulled."""

    name: str
    fetched_at: datetime
    n_players: int
    from_mtime: bool = False
    dropped: str = ""

    def age_hours(self, now: datetime) -> float:
        return (now - self.fetched_at).total_seconds() / 3600.0


@dataclass
class WeeklyData:
    """One week, aggregated across sources the way the draft board is.

    `projections.py` averages sources equal-weight because source accuracy does
    not persist year to year. The same holds week to week, and more sharply: a
    site that nails Thursday night is not thereby better on Sunday. So the
    weekly path uses the identical rule, and the *spread* between sources
    becomes the per-player variance that P(win) is computed against. Five
    sources that agree is a different bet from five that do not, and averaging
    them into a single number would throw that away.
    """

    week: int
    points: dict[str, float] = field(default_factory=dict)
    source_sd: dict[str, float] = field(default_factory=dict)
    n_sources: dict[str, int] = field(default_factory=dict)
    per_source: dict[str, dict[str, float]] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)
    opponent: dict[str, str] = field(default_factory=dict)
    sources: list[SourceMeta] = field(default_factory=list)
    path: str = ""

    @property
    def kept(self) -> list[SourceMeta]:
        return [s for s in self.sources if not s.dropped]

    @property
    def fetched_at(self) -> datetime:
        """The oldest surviving source. An average is only as fresh as its
        stalest input, so this is the number every check is made against."""
        kept = self.kept
        if not kept:
            raise StaleDataError("every source was dropped; nothing to stand on")
        return min(s.fetched_at for s in kept)

    @property
    def from_mtime(self) -> bool:
        return any(s.from_mtime for s in self.kept)

    def age(self, now: datetime | None = None) -> timedelta:
        return (now or datetime.now(timezone.utc)) - self.fetched_at

    def is_out(self, player_id: str) -> bool:
        return self.status.get(player_id, "").upper() in OUT_STATUSES

    def is_doubtful(self, player_id: str) -> bool:
        return self.status.get(player_id, "").upper() in DOUBTFUL_STATUSES


def _parse_time(raw: str) -> datetime:
    text = raw.strip().replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise WeeklyDataError(
            f"cannot read fetched_at {raw!r}; expected ISO 8601, "
            f"e.g. 2026-09-14T11:30:00+00:00"
        ) from exc
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


# Most severe wins when sources disagree about a status. A player one site has
# as ACTIVE and another has as OUT is not someone to start on the optimistic
# reading.
_SEVERITY = {"": 0, "ACTIVE": 0, "OK": 0, "PLAYING": 0, "PROBABLE": 1,
             "GTD": 2, "QUESTIONABLE": 3, "DOUBTFUL": 4,
             "OUT": 5, "DNP": 5, "PUP": 5, "SUSPENDED": 6, "BYE": 7, "IR": 8}


def load_weekly(
    path: str | Path,
    week: int | None = None,
    *,
    rules: ScoringRules | None = None,
    now: datetime | None = None,
    max_age_hours: float = 24.0,
    min_sources: int = 1,
) -> WeeklyData:
    """Read a weekly projection file and aggregate it equal-weight.

    Accepts either shape. A `points` column is used directly; a stat-line file
    (what `R/pull_projections.R --week N` writes) is scored with `rules` first,
    per source, exactly as the draft board is. Identify players with
    `player_id`, or `player` plus `pos`. Optional: `source`, `fetched_at`,
    `week`, `status`, `sd`, `opponent`.

    `source` and `fetched_at` are what make this multi-source. Each source
    carries its own stamp, and one that is past `max_age_hours` is **dropped**
    rather than averaged in: a stale source does not announce itself in a mean,
    it just quietly drags it. What survives is reported, and if fewer than
    `min_sources` survive the caller is told rather than handed a thinner
    consensus than it asked for.
    """
    from .ids import make_player_id

    path = Path(path)
    if not path.exists():
        raise WeeklyDataError(f"no weekly file at {path}")
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise WeeklyDataError(f"{path}: no rows")

    now = now or datetime.now(timezone.utc)
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    by_source: dict[str, dict[str, float]] = defaultdict(dict)
    stamps: dict[str, datetime] = {}
    from_mtime: dict[str, bool] = {}
    status: dict[str, str] = {}
    opponent: dict[str, str] = {}
    explicit_sd: dict[str, list[float]] = defaultdict(list)
    weeks: set[int] = set()

    for n, row in enumerate(rows, start=2):
        pid = (row.get("player_id") or "").strip()
        if not pid:
            name = (row.get("player") or row.get("name") or "").strip()
            pos = (row.get("pos") or row.get("position") or "").strip()
            if not name or not pos:
                raise WeeklyDataError(
                    f"{path} line {n}: need player_id, or player and pos"
                )
            pid = make_player_id(name, pos)

        raw = (row.get("points") or row.get("projection") or "").strip()
        if raw:
            if raw.upper() in _NULLISH:
                continue
            try:
                value = float(raw)
            except ValueError as exc:
                raise WeeklyDataError(
                    f"{path} line {n}: points {raw!r} for {pid} is not a number. "
                    f"A projection that parsed as zero is a benched starter."
                ) from exc
        elif rules is not None:
            # A stat-line file, straight from `R/pull_projections.R --week N`.
            # Score it here, per source, before averaging - the same order
            # `projections.py` uses, and the reason is the same: the DST
            # points-allowed bracket is a step function, so averaging stat
            # lines and scoring once is not the same number.
            stats = {
                k: _f(v) for k, v in row.items()
                if k and k.lower() not in _META_COLUMNS
                and k.lower() not in _WEEKLY_META
            }
            value = score_stats(stats, rules)
        else:
            raise WeeklyDataError(
                f"{path} line {n}: no points column for {pid}, and no scoring "
                f"rules were passed to score the stat line with. A weekly pull "
                f"from R holds stat lines; run this through `ffdraft lineup`, "
                f"which supplies the league's rules."
            )

        source = (row.get("source") or row.get("data_src") or "single").strip() or "single"
        by_source[source][pid] = value
        if source not in stamps:
            stamp = row.get("fetched_at", "").strip()
            stamps[source] = _parse_time(stamp) if stamp else mtime
            from_mtime[source] = not stamp

        if row.get("status"):
            new_status = row["status"].strip().upper()
            if _SEVERITY.get(new_status, 0) >= _SEVERITY.get(status.get(pid, ""), 0):
                status[pid] = new_status
        if row.get("opponent"):
            opponent[pid] = row["opponent"].strip()
        if row.get("sd"):
            try:
                explicit_sd[pid].append(float(row["sd"]))
            except ValueError:
                pass
        if row.get("week"):
            weeks.add(int(float(row["week"])))

    if len(weeks) > 1:
        raise WeeklyDataError(f"{path}: rows span several weeks {sorted(weeks)}")
    file_week = weeks.pop() if weeks else week
    if file_week is None:
        raise WeeklyDataError(
            f"{path}: no week column and no --week given; a weekly file whose "
            f"week is unknown cannot be checked against the week you are setting"
        )

    metas = [
        SourceMeta(
            name=name, fetched_at=stamps[name], n_players=len(players),
            from_mtime=from_mtime.get(name, False),
        )
        for name, players in sorted(by_source.items())
    ]
    for meta in metas:
        hours = meta.age_hours(now)
        if hours > max_age_hours:
            meta.dropped = f"{hours:.1f}h old, past the {max_age_hours:g}h limit"

    kept = [m for m in metas if not m.dropped]
    if not kept:
        raise StaleDataError(
            f"{path}: every source is past the {max_age_hours:g}h limit "
            f"({', '.join(f'{m.name} {m.age_hours(now):.1f}h' for m in metas)}). "
            f"Pull again - there is nothing here to set a lineup from."
        )
    if len(kept) < min_sources:
        raise StaleDataError(
            f"{path}: only {len(kept)} fresh source(s) "
            f"({', '.join(m.name for m in kept)}) but --min-sources is "
            f"{min_sources}. Dropped: "
            f"{'; '.join(f'{m.name} ({m.dropped})' for m in metas if m.dropped)}"
        )

    # Equal weight across the surviving sources - the same rule, and for the
    # same reason, as the draft board's aggregation.
    points: dict[str, float] = {}
    spread: dict[str, float] = {}
    counts: dict[str, int] = {}
    everyone = {pid for m in kept for pid in by_source[m.name]}
    for pid in everyone:
        values = [by_source[m.name][pid] for m in kept if pid in by_source[m.name]]
        points[pid] = statistics.fmean(values)
        counts[pid] = len(values)
        if pid in explicit_sd and explicit_sd[pid]:
            spread[pid] = statistics.fmean(explicit_sd[pid])
        elif len(values) > 1:
            spread[pid] = statistics.stdev(values)

    return WeeklyData(
        week=file_week, points=points, source_sd=spread, n_sources=counts,
        per_source={m.name: by_source[m.name] for m in kept},
        status=status, opponent=opponent, sources=metas, path=str(path),
    )


def format_sources(data: WeeklyData, now: datetime | None = None) -> str:
    """Per-source coverage and age - `ffdraft sources`, for the week."""
    now = now or datetime.now(timezone.utc)
    lines = [f"  {'source':<20}{'players':>8}{'age':>10}   state"]
    for meta in data.sources:
        state = f"DROPPED - {meta.dropped}" if meta.dropped else "used"
        if meta.from_mtime and not meta.dropped:
            state += " (age from file mtime)"
        lines.append(
            f"  {meta.name[:19]:<20}{meta.n_players:>8}"
            f"{meta.age_hours(now):>9.1f}h   {state}"
        )
    multi = sum(1 for v in data.n_sources.values() if v > 1)
    lines.append("")
    lines.append(
        f"  {len(data.kept)} source(s) used; {multi}/{len(data.points)} players "
        f"have more than one opinion behind them"
    )
    return "\n".join(lines)


def check_freshness(
    data: WeeklyData,
    week: int,
    roster: list[str],
    *,
    now: datetime | None = None,
    max_age_hours: float = 24.0,
    active_max_age_hours: float = 3.0,
    allow_stale: bool = False,
) -> list[str]:
    """Refuse to set a lineup from data that cannot be vouched for.

    Returns the warnings that did not rise to a refusal. Raises
    `StaleDataError` otherwise - see the module docstring for why this is not a
    printed warning.
    """
    now = now or datetime.now(timezone.utc)
    notes: list[str] = []
    problems: list[str] = []

    if data.week != week:
        problems.append(
            f"the file is week {data.week} and you asked for week {week}. "
            f"Nothing else about it matters."
        )

    age = data.age(now)
    hours = age.total_seconds() / 3600.0
    if hours < 0:
        notes.append(
            f"fetched_at is {-hours:.1f}h in the future; check the clock on "
            f"whatever produced this file"
        )
    doubtful = sorted(p for p in roster if data.is_doubtful(p))
    limit = active_max_age_hours if doubtful else max_age_hours
    if hours > limit:
        why = (
            f"{len(doubtful)} player(s) you could start are questionable or "
            f"doubtful, so the bar is {limit:g}h, not {max_age_hours:g}h"
            if doubtful else f"the limit is {limit:g}h"
        )
        problems.append(
            f"the data is {hours:.1f}h old and {why}. Those tags resolve in the "
            f"90 minutes before kickoff; pull again."
        )

    missing = [p for p in roster if p not in data.points]
    if missing:
        notes.append(
            f"{len(missing)} rostered player(s) are absent from the file and "
            f"will be treated as unstartable, not as zero-point starters"
        )
    if data.from_mtime:
        notes.append(
            "a source carries no fetched_at; its age came from the file's "
            "modification time, which a copy or a sync would reset"
        )
    for meta in data.sources:
        if meta.dropped:
            notes.append(f"source {meta.name} dropped: {meta.dropped}")
    thin = [p for p in roster if data.n_sources.get(p, 0) == 1]
    if thin and len(data.kept) > 1:
        notes.append(
            f"{len(thin)} of your players appear in only one source; their "
            f"projections carry no cross-source spread and the simulation "
            f"leans on the league's weekly CV alone for them"
        )

    if problems:
        if not allow_stale:
            raise StaleDataError(
                "refusing to set a lineup from this file:\n  - "
                + "\n  - ".join(problems)
                + "\n\nRe-pull, or pass --allow-stale if you know why this is fine."
            )
        notes.extend(f"OVERRIDDEN by --allow-stale: {p}" for p in problems)
    return notes


# --- lineup construction -----------------------------------------------------
@dataclass
class LineupSlot:
    slot: str
    player_id: str | None
    points: float
    status: str = ""


@dataclass
class LineupPlan:
    week: int
    slots: list[LineupSlot]
    bench: list[tuple[str, float, str]]
    expected_points: float
    p_win: float | None = None
    opponent_seat: int | None = None
    opponent_points: float | None = None
    benched_by_status: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    swaps_considered: int = 0

    @property
    def starters(self) -> list[str]:
        return [s.player_id for s in self.slots if s.player_id]


def _startable(data: WeeklyData, roster: list[str]) -> dict[str, float]:
    """Projected points for everyone who can actually play.

    A player who is OUT is removed rather than set to zero. Zero would let the
    greedy fill hand him a slot when the bench is thin, which is how a lineup
    ends up with an inactive in it.
    """
    return {
        pid: data.points[pid]
        for pid in roster
        if pid in data.points and not data.is_out(pid)
    }


def _fill(
    scores: dict[str, float], positions: dict[str, str], config: LeagueConfig
) -> list[LineupSlot]:
    used: set[str] = set()
    out: list[LineupSlot] = []
    counter: dict[str, int] = {}
    for slot in slot_fill_order(config.roster.starters):
        counter[slot] = counter.get(slot, 0) + 1
        label = (
            slot if config.roster.starters.count(slot) == 1
            else f"{slot}{counter[slot]}"
        )
        eligible = SLOT_ELIGIBILITY[slot]
        best, best_score = None, None
        for pid, pos in positions.items():
            if pid in used or pos not in eligible:
                continue
            score = scores.get(pid, 0.0)
            if best_score is None or score > best_score:
                best, best_score = pid, score
        if best is not None:
            used.add(best)
        out.append(LineupSlot(slot=label, player_id=best, points=best_score or 0.0))
    return out


def _display_order(config: LeagueConfig, slots: list[LineupSlot]) -> list[LineupSlot]:
    """Read down the lineup card in the order the league lists it."""
    order, counts = [], {}
    for slot in config.roster.starters:
        counts[slot] = counts.get(slot, 0) + 1
        label = (
            slot if config.roster.starters.count(slot) == 1
            else f"{slot}{counts[slot]}"
        )
        match = next((s for s in slots if s.slot == label), None)
        if match:
            order.append(match)
    return order + [s for s in slots if s not in order]


def _weekly_sd(board: Board, data: WeeklyData, config: LeagueConfig, pid: str) -> float:
    """Spread for one player-week.

    Two components, added in quadrature:

      Outcome risk - the league's weekly coefficient of variation, the same
      number `simulate.py` uses to spread a season total across weeks. A
      receiver is volatile even when every source agrees on his mean.

      Disagreement between sources - the standard deviation of their
      projections. This is what aggregating buys: five sources clustered
      inside two points is a different bet from five spread over fifteen, and
      collapsing them to a mean throws that distinction away. It matters most
      in exactly the situation the user faces on Sunday morning, where the
      sources disagree because nobody knows the snap count yet.
    """
    pos = board.player(pid).pos if pid in board.index else "WR"
    cv = config.sim.weekly_cv.get(pos, 0.5)
    outcome = data.points.get(pid, 0.0) * cv
    disagreement = data.source_sd.get(pid, 0.0)
    return max((outcome ** 2 + disagreement ** 2) ** 0.5, 1e-6)


def _simulate_totals(
    board: Board,
    data: WeeklyData,
    config: LeagueConfig,
    starters: list[str],
    rng: np.random.Generator,
    n_sims: int,
) -> np.ndarray:
    if not starters:
        return np.zeros(n_sims)
    mean = np.array([data.points.get(p, 0.0) for p in starters], dtype=float)
    sd = np.array([_weekly_sd(board, data, config, p) for p in starters], dtype=float)
    mu, sigma = lognormal_params(mean, sd)
    draws = np.exp(rng.normal(mu[None, :], sigma[None, :], size=(n_sims, len(starters))))
    return draws.sum(axis=1)


def _candidates(
    base: list[LineupSlot],
    scores: dict[str, float],
    positions: dict[str, str],
) -> list[list[LineupSlot]]:
    """The base lineup plus every single legal swap.

    Not every legal lineup: the decision a person actually faces on Sunday
    morning is "start A or B in this one slot", and enumerating the full
    lattice would spend the simulation budget distinguishing lineups nobody was
    choosing between.
    """
    started = {s.player_id for s in base if s.player_id}
    out = [base]
    for i, slot in enumerate(base):
        eligible = SLOT_ELIGIBILITY[slot.slot.rstrip("0123456789")]
        for pid, pos in positions.items():
            if pid in started or pos not in eligible:
                continue
            variant = [
                LineupSlot(s.slot, s.player_id, s.points) for s in base
            ]
            variant[i] = LineupSlot(slot.slot, pid, scores.get(pid, 0.0))
            out.append(variant)
    return out


def best_lineup(
    config: LeagueConfig,
    board: Board,
    data: WeeklyData,
    roster: list[str],
    *,
    opponent_roster: list[str] | None = None,
    opponent_seat: int | None = None,
    n_sims: int = 20000,
    seed: int = 0,
    notes: list[str] | None = None,
) -> LineupPlan:
    """The lineup to set. Expected points, or P(win) when the opponent is known."""
    scores = _startable(data, roster)
    positions = {
        pid: board.player(pid).pos for pid in scores if pid in board.index
    }
    scores = {pid: v for pid, v in scores.items() if pid in positions}

    benched = [
        (pid, data.status.get(pid, "OUT"))
        for pid in roster
        if pid in data.points and data.is_out(pid)
    ]
    base = _fill(scores, positions, config)
    plan_notes = list(notes or [])

    chosen, p_win, opp_mean, considered = base, None, None, 0
    if opponent_roster:
        opp_scores = _startable(data, opponent_roster)
        opp_positions = {
            p: board.player(p).pos for p in opp_scores if p in board.index
        }
        opp_lineup = _fill(
            {p: v for p, v in opp_scores.items() if p in opp_positions},
            opp_positions, config,
        )
        rng = np.random.default_rng(seed)
        opp_totals = _simulate_totals(
            board, data, config, [s.player_id for s in opp_lineup if s.player_id],
            rng, n_sims,
        )
        opp_mean = float(opp_totals.mean())

        variants = _candidates(base, scores, positions)
        considered = len(variants)
        best_p, best_variant = -1.0, base
        for variant in variants:
            starters = [s.player_id for s in variant if s.player_id]
            # Common random numbers: one generator, re-seeded per variant, so
            # two lineups are compared on the same weather.
            mine = _simulate_totals(
                board, data, config, starters,
                np.random.default_rng(seed + 1), n_sims,
            )
            win = float((mine > opp_totals).mean() + 0.5 * (mine == opp_totals).mean())
            key = (round(win, 6), round(sum(s.points for s in variant), 4))
            if key > (round(best_p, 6), round(sum(s.points for s in best_variant), 4)):
                best_p, best_variant = win, variant
        chosen, p_win = best_variant, best_p
        if chosen is not base:
            plan_notes.append(
                "the P(win) lineup differs from the highest-projected one: "
                "against this opponent the variance is worth more than the points"
            )

    started = {s.player_id for s in chosen if s.player_id}
    bench = sorted(
        (
            (pid, scores.get(pid, 0.0), data.status.get(pid, ""))
            for pid in roster
            if pid not in started and pid in positions
        ),
        key=lambda row: -row[1],
    )
    return LineupPlan(
        week=data.week,
        slots=_display_order(config, chosen),
        bench=bench,
        expected_points=sum(s.points for s in chosen),
        p_win=p_win,
        opponent_seat=opponent_seat,
        opponent_points=opp_mean,
        benched_by_status=benched,
        notes=plan_notes,
        swaps_considered=considered,
    )


def roster_for(config: LeagueConfig, drafted: list[str], seat: int) -> list[str]:
    return DraftState(config=config, drafted=list(drafted)).roster_of(seat)


def format_plan(plan: LineupPlan, board: Board, data: WeeklyData) -> str:
    """The lineup card, in the shape the draft console uses."""
    lines = [
        f"WEEK {plan.week} lineup - projected {plan.expected_points:.1f} pts "
        f"(data pulled {data.fetched_at:%Y-%m-%d %H:%M %Z})"
    ]
    if plan.p_win is not None:
        lines.append(
            f"  vs seat {plan.opponent_seat}: they project "
            f"{plan.opponent_points:.1f} - you win {plan.p_win:.0%} of simulations"
        )
    lines.append("")
    for slot in plan.slots:
        if slot.player_id is None:
            lines.append(f"  {slot.slot:<8}{'-- EMPTY --':<34}")
            continue
        name = board.player(slot.player_id).display
        tag = data.status.get(slot.player_id, "")
        mark = f"  [{tag}]" if tag and tag not in ACTIVE_STATUSES else ""
        lines.append(f"  {slot.slot:<8}{name[:33]:<34}{slot.points:6.1f}{mark}")
    if plan.bench:
        lines.append("")
        lines.append("  bench")
        for pid, pts, tag in plan.bench[:8]:
            mark = f"  [{tag}]" if tag and tag not in ACTIVE_STATUSES else ""
            lines.append(f"          {board.player(pid).display[:33]:<34}{pts:6.1f}{mark}")
    if plan.benched_by_status:
        lines.append("")
        for pid, tag in plan.benched_by_status:
            lines.append(f"  OUT     {board.player(pid).display} [{tag}] - not startable")
    for note in plan.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)
