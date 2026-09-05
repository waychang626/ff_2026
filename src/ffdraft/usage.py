"""Observed usage, and the role changes a consensus has not finished pricing.

`weekly.py` consumes what the projection sites *forecast*. This consumes what
actually happened - snap share, target share, carries - from nflverse, and its
whole purpose is one case the forecast handles badly.

When a starter is ruled out, every site moves the backup's projection within
hours; that news is public and they are paid to react to it. What they are slow
on is a role changing without an announcement. A third-string back takes 22% of
snaps in week 4, 44% in week 5, 61% in week 6, and nobody has said anything.
The consensus anchors on the depth chart it published in August and re-rates
over weeks, because each site is averaging its own history too. That lag is the
only edge in here.

The metric is snap and target share rather than fantasy points, and that is the
settled finding rather than a preference: usage is *stickier* week to week than
scoring is. A back who posts RB1 numbers on 30% of snaps had a good afternoon;
a back on 70% of snaps has a job. Points regress, roles persist.

TWO THINGS THIS DELIBERATELY DOES NOT DO
----------------------------------------
It does not replace the projection. The sites see snap counts too, and a naive
trend model that overrides a professional consensus is how you get worse, not
better. The default output is a *flag* - the number, next to the projection, for
a human to act on.

When an adjustment is asked for, it is damped and capped. Damped because the
consensus has already priced part of any trend and applying it in full
double-counts; capped because a multiplier with no ceiling turns one weird week
into a lineup decision. Every adjustment is written to the audit log with the
usage that caused it, the same way a manual `out` or `bump` is.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .audit import AuditLog
from .weekly import _parse_time, WeeklyData, WeeklyDataError


@dataclass
class UsageWeek:
    week: int
    snap_pct: float
    target_share: float
    carries: float
    targets: float
    points: float


@dataclass
class UsageData:
    season: int
    fetched_at: datetime
    weeks: dict[str, list[UsageWeek]] = field(default_factory=dict)
    from_mtime: bool = False
    path: str = ""

    @property
    def last_week(self) -> int:
        return max((w.week for rows in self.weeks.values() for w in rows), default=0)

    def age_hours(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (now - self.fetched_at).total_seconds() / 3600.0


@dataclass
class RoleTrend:
    """One player's recent usage against his own earlier baseline."""

    player_id: str
    recent_snap: float
    baseline_snap: float
    recent_target_share: float
    baseline_target_share: float
    recent_weeks: int
    baseline_weeks: int

    @property
    def ratio(self) -> float:
        """Recent snap share over baseline. 1.0 is an unchanged role."""
        if self.baseline_snap <= 0.01:
            # No baseline to speak of: a player who did not play cannot have a
            # ratio, and inventing a large one from a near-zero denominator is
            # how a third-stringer's first appearance becomes a start.
            return 1.0 if self.recent_snap <= 0.01 else 2.0
        return self.recent_snap / self.baseline_snap

    @property
    def direction(self) -> str:
        if self.ratio >= 1.0:
            return "rising"
        return "falling"


def load_usage(
    path: str | Path,
    *,
    now: datetime | None = None,
    max_age_hours: float = 72.0,
) -> UsageData:
    """Read the file `scripts/fetch_nflverse.py` writes.

    The freshness bar is looser than the projection one (72h, not 24h) and that
    is not an oversight: nflverse publishes overnight, so the most recent game's
    usage is a day old the moment it exists. What matters is that it covers the
    weeks already played, which `check_usage` tests directly.
    """
    path = Path(path)
    if not path.exists():
        raise WeeklyDataError(f"no usage file at {path}")
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise WeeklyDataError(f"{path}: no rows")

    now = now or datetime.now(timezone.utc)
    stamps = {r["fetched_at"].strip() for r in rows if r.get("fetched_at")}
    if len(stamps) > 1:
        raise WeeklyDataError(
            f"{path}: rows carry different fetched_at values; its age is not "
            f"one number"
        )
    from_mtime = not stamps
    fetched = (
        _parse_time(stamps.pop()) if stamps
        else datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    )

    def _f(value, default=0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    weeks: dict[str, list[UsageWeek]] = {}
    seasons: set[int] = set()
    for row in rows:
        pid = (row.get("player_id") or "").strip()
        if not pid:
            continue
        seasons.add(int(_f(row.get("season"), 0)))
        weeks.setdefault(pid, []).append(UsageWeek(
            week=int(_f(row.get("week"), 0)),
            snap_pct=_f(row.get("snap_pct")),
            target_share=_f(row.get("target_share")),
            carries=_f(row.get("carries")),
            targets=_f(row.get("targets")),
            points=_f(row.get("fantasy_points_ppr")),
        ))
    for rows_for in weeks.values():
        rows_for.sort(key=lambda w: w.week)

    data = UsageData(
        season=max(seasons) if seasons else 0, fetched_at=fetched,
        weeks=weeks, from_mtime=from_mtime, path=str(path),
    )
    if data.age_hours(now) > max_age_hours:
        raise WeeklyDataError(
            f"{path}: usage data is {data.age_hours(now):.0f}h old, past the "
            f"{max_age_hours:g}h limit. Re-run scripts/fetch_nflverse.py - "
            f"stale usage is worse than none, because it looks like evidence."
        )
    return data


def check_usage(data: UsageData, week: int) -> list[str]:
    """Warn when the usage file cannot see the weeks it would need to."""
    notes: list[str] = []
    if data.last_week < week - 1:
        notes.append(
            f"usage data stops at week {data.last_week} but you are setting "
            f"week {week}; it cannot see the most recent game(s), which is "
            f"exactly where a role change would show"
        )
    if data.from_mtime:
        notes.append(
            "usage file has no fetched_at; its age came from the file's "
            "modification time"
        )
    return notes


def role_trend(
    data: UsageData,
    player_id: str,
    week: int,
    *,
    recent_weeks: int = 2,
    min_baseline_weeks: int = 2,
) -> RoleTrend | None:
    """Recent usage against this player's own earlier weeks, this season.

    Returns None when there is not enough history to say anything, which is the
    honest answer for a player with three games and a bye.
    """
    rows = [w for w in data.weeks.get(player_id, []) if w.week < week]
    if len(rows) < recent_weeks + min_baseline_weeks:
        return None
    recent = rows[-recent_weeks:]
    baseline = rows[:-recent_weeks]
    return RoleTrend(
        player_id=player_id,
        recent_snap=statistics.fmean(w.snap_pct for w in recent),
        baseline_snap=statistics.fmean(w.snap_pct for w in baseline),
        recent_target_share=statistics.fmean(w.target_share for w in recent),
        baseline_target_share=statistics.fmean(w.target_share for w in baseline),
        recent_weeks=len(recent),
        baseline_weeks=len(baseline),
    )


def notable_trends(
    data: UsageData,
    roster: list[str],
    week: int,
    *,
    recent_weeks: int = 2,
    rise_ratio: float = 1.35,
    fall_ratio: float = 0.70,
    min_recent_snap: float = 0.35,
) -> list[RoleTrend]:
    """Role changes worth a human's attention, both directions.

    A rise needs the player to be *actually* playing now (`min_recent_snap`), so
    a fourth-stringer going from 4% to 9% of snaps does not read as a breakout.
    A fall has no such floor: a starter losing his job matters at any level, and
    that is the half of this people forget to look at.
    """
    out = []
    for pid in roster:
        trend = role_trend(data, pid, week, recent_weeks=recent_weeks)
        if trend is None:
            continue
        rising = trend.ratio >= rise_ratio and trend.recent_snap >= min_recent_snap
        falling = trend.ratio <= fall_ratio
        if rising or falling:
            out.append(trend)
    out.sort(key=lambda t: -abs(t.ratio - 1.0))
    return out


def describe(trend: RoleTrend) -> str:
    parts = [
        f"snaps {trend.baseline_snap:.0%} -> {trend.recent_snap:.0%} "
        f"over the last {trend.recent_weeks}"
    ]
    if trend.recent_target_share > 0.01 or trend.baseline_target_share > 0.01:
        parts.append(
            f"targets {trend.baseline_target_share:.0%} -> "
            f"{trend.recent_target_share:.0%}"
        )
    return ", ".join(parts)


def apply_usage_adjustment(
    weekly: WeeklyData,
    data: UsageData,
    roster: list[str],
    week: int,
    *,
    damping: float = 0.5,
    cap: float = 0.25,
    recent_weeks: int = 2,
    rise_ratio: float = 1.35,
    fall_ratio: float = 0.70,
    min_recent_snap: float = 0.35,
    league_id: str = "",
    audit: AuditLog | None = None,
) -> list[str]:
    """Nudge projections toward observed usage. Opt-in, damped, capped, logged.

    `damping` exists because the consensus can see snap counts too - applying a
    trend in full assumes nobody else noticed, which is not a bet worth making.
    `cap` exists because one anomalous week should not be able to move a
    projection by more than a quarter, however extreme the ratio.

    Mutates `weekly.points` and returns a note per adjustment.
    """
    notes: list[str] = []
    trends = notable_trends(
        data, roster, week, recent_weeks=recent_weeks, rise_ratio=rise_ratio,
        fall_ratio=fall_ratio, min_recent_snap=min_recent_snap,
    )
    for trend in trends:
        if trend.player_id not in weekly.points:
            continue
        # Scaling a zero is a zero. A player who is out or on a bye has no
        # projection to nudge, and reporting the no-op as an adjustment buries
        # the real ones.
        if weekly.is_out(trend.player_id) or weekly.points[trend.player_id] <= 0.0:
            continue
        raw = 1.0 + damping * (trend.ratio - 1.0)
        multiplier = max(1.0 - cap, min(1.0 + cap, raw))
        before = weekly.points[trend.player_id]
        weekly.points[trend.player_id] = before * multiplier
        notes.append(
            f"usage adjust {trend.player_id}: x{multiplier:.2f} "
            f"({before:.1f} -> {weekly.points[trend.player_id]:.1f}); "
            f"{describe(trend)}"
        )
        (audit or AuditLog()).record(
            state_hash="", league_id=league_id, pick_number=week,
            kind="usage_adjustment",
            payload={
                "player_id": trend.player_id,
                "multiplier": round(multiplier, 4),
                "points_before": round(before, 3),
                "points_after": round(weekly.points[trend.player_id], 3),
                "recent_snap": round(trend.recent_snap, 4),
                "baseline_snap": round(trend.baseline_snap, 4),
                "recent_weeks": trend.recent_weeks,
                "damping": damping,
                "cap": cap,
            },
        )
    return notes


def format_trends(trends: list[RoleTrend], board, week: int) -> str:
    if not trends:
        return f"  no notable role changes on your roster going into week {week}"
    lines = [f"  role changes going into week {week}"]
    for trend in trends:
        name = (
            board.player(trend.player_id).display
            if trend.player_id in board.index else trend.player_id
        )
        arrow = "UP  " if trend.direction == "rising" else "DOWN"
        lines.append(f"   {arrow} {name[:30]:<32}{describe(trend)}")
    return "\n".join(lines)
