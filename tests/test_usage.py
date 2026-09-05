"""Observed usage, and the role changes a consensus has not finished pricing.

The case this exists for: a third-string back quietly takes over a backfield.
Nobody announces it, so the projection sites re-rate over weeks while the snap
share says it already happened.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pytest

from ffdraft.audit import AuditLog
from ffdraft.usage import (
    apply_usage_adjustment,
    check_usage,
    describe,
    load_usage,
    notable_trends,
    role_trend,
)
from ffdraft.weekly import WeeklyDataError, load_weekly

NOW = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)
COLS = ["fetched_at", "season", "week", "player_id", "player", "pos", "team",
        "snap_pct", "carries", "targets", "target_share", "air_yards_share",
        "fantasy_points_ppr"]


def _stamp(hours=1):
    return (NOW - timedelta(hours=hours)).isoformat()


def _usage(path, series, hours=1):
    """series: {player_id: [snap_pct per week, starting at week 1]}"""
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for pid, snaps in series.items():
            for week, snap in enumerate(snaps, start=1):
                w.writerow({
                    "fetched_at": _stamp(hours), "season": 2026, "week": week,
                    "player_id": pid, "player": pid.split("|")[0], "pos": pid.split("|")[1],
                    "team": "XX", "snap_pct": f"{snap:.3f}", "carries": "8",
                    "targets": "4", "target_share": f"{snap * 0.3:.3f}",
                    "air_yards_share": "0.1", "fantasy_points_ppr": "11.0",
                })
    return path


# --- trend detection ---------------------------------------------------------
def test_a_takeover_is_detected(tmp_path):
    """22% for four weeks, then 61%, and nobody said anything."""
    path = _usage(tmp_path / "u.csv", {"rb3|RB": [0.22, 0.20, 0.24, 0.22, 0.58, 0.61]})
    data = load_usage(path, now=NOW)
    trend = role_trend(data, "rb3|RB", week=7)
    assert trend.ratio > 2.0
    assert trend.direction == "rising"
    assert "20%" in describe(trend) or "22%" in describe(trend)


def test_a_lost_job_is_detected_too(tmp_path):
    """The half people forget to look at."""
    path = _usage(tmp_path / "u.csv", {"rb1|RB": [0.80, 0.78, 0.75, 0.79, 0.20, 0.14]})
    trend = role_trend(load_usage(path, now=NOW), "rb1|RB", week=7)
    assert trend.ratio < 0.5
    assert trend.direction == "falling"


def test_a_steady_role_is_not_flagged(tmp_path):
    path = _usage(tmp_path / "u.csv", {"rb1|RB": [0.70, 0.68, 0.72, 0.69, 0.71, 0.70]})
    data = load_usage(path, now=NOW)
    assert notable_trends(data, ["rb1|RB"], week=7) == []


def test_a_deep_reserve_ticking_up_is_not_a_breakout(tmp_path):
    """4% to 9% doubles the ratio and means nothing."""
    path = _usage(tmp_path / "u.csv", {"rb5|RB": [0.04, 0.03, 0.05, 0.04, 0.09, 0.09]})
    data = load_usage(path, now=NOW)
    assert notable_trends(data, ["rb5|RB"], week=7) == []


def test_too_little_history_returns_nothing(tmp_path):
    path = _usage(tmp_path / "u.csv", {"rb|RB": [0.5, 0.6]})
    assert role_trend(load_usage(path, now=NOW), "rb|RB", week=3) is None


def test_a_near_zero_baseline_does_not_invent_a_huge_ratio(tmp_path):
    """A first appearance is not an infinite increase."""
    path = _usage(tmp_path / "u.csv", {"rb|RB": [0.0, 0.0, 0.0, 0.0, 0.30, 0.35]})
    trend = role_trend(load_usage(path, now=NOW), "rb|RB", week=7)
    assert trend.ratio == 2.0


def test_only_weeks_before_the_target_are_used(tmp_path):
    """Week 7's own usage cannot inform the week 7 lineup."""
    path = _usage(tmp_path / "u.csv", {"rb|RB": [0.2, 0.2, 0.2, 0.2, 0.2, 0.9, 0.9]})
    trend = role_trend(load_usage(path, now=NOW), "rb|RB", week=6)
    assert trend.recent_snap == pytest.approx(0.2)


def test_both_sides_of_one_backfield_surface_together(tmp_path):
    path = _usage(tmp_path / "u.csv", {
        "up|RB": [0.25, 0.22, 0.24, 0.26, 0.62, 0.65],
        "down|RB": [0.70, 0.72, 0.68, 0.71, 0.25, 0.20],
    })
    trends = notable_trends(load_usage(path, now=NOW), ["up|RB", "down|RB"], week=7)
    assert {t.player_id for t in trends} == {"up|RB", "down|RB"}
    assert {t.direction for t in trends} == {"rising", "falling"}


# --- freshness ---------------------------------------------------------------
def test_stale_usage_is_refused(tmp_path):
    path = _usage(tmp_path / "u.csv", {"rb|RB": [0.5] * 6}, hours=200)
    with pytest.raises(WeeklyDataError, match="past the"):
        load_usage(path, now=NOW, max_age_hours=72)


def test_usage_that_cannot_see_recent_weeks_is_flagged(tmp_path):
    path = _usage(tmp_path / "u.csv", {"rb|RB": [0.5] * 4})
    notes = check_usage(load_usage(path, now=NOW), week=9)
    assert any("stops at week 4" in n for n in notes)


def test_a_file_stitched_from_two_pulls_is_refused(tmp_path):
    path = tmp_path / "u.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for hours in (1, 5):
            w.writerow({c: "" for c in COLS} | {
                "fetched_at": _stamp(hours), "season": 2026, "week": 1,
                "player_id": "rb|RB", "snap_pct": "0.5",
            })
    with pytest.raises(WeeklyDataError, match="different fetched_at"):
        load_usage(path, now=NOW)


# --- the adjustment ----------------------------------------------------------
@pytest.fixture
def weekly(tmp_path):
    path = tmp_path / "w.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "fetched_at", "week", "player_id", "points", "status"])
        for pid in ("up|RB", "down|RB", "steady|RB", "hurt|RB"):
            w.writerow(["A", _stamp(1), 7, pid, "12.0",
                        "OUT" if pid == "hurt|RB" else "ACTIVE"])
    return load_weekly(path, now=NOW)


@pytest.fixture
def usage(tmp_path):
    return load_usage(_usage(tmp_path / "u.csv", {
        "up|RB": [0.25, 0.22, 0.24, 0.26, 0.62, 0.65],
        "down|RB": [0.70, 0.72, 0.68, 0.71, 0.25, 0.20],
        "steady|RB": [0.60] * 6,
        "hurt|RB": [0.25, 0.22, 0.24, 0.26, 0.62, 0.65],
    }), now=NOW)


ROSTER = ["up|RB", "down|RB", "steady|RB", "hurt|RB"]


def test_the_adjustment_is_capped(weekly, usage):
    apply_usage_adjustment(weekly, usage, ROSTER, 7, cap=0.25, damping=1.0)
    assert weekly.points["up|RB"] == pytest.approx(12.0 * 1.25)
    assert weekly.points["down|RB"] == pytest.approx(12.0 * 0.75)


def test_damping_shrinks_the_move(weekly, usage):
    apply_usage_adjustment(weekly, usage, ROSTER, 7, cap=1.0, damping=0.5)
    # ratio ~2.5 -> raw 1.75 at full damping; halved it is ~1.75, not 2.5.
    assert 1.0 < weekly.points["up|RB"] / 12.0 < 2.5


def test_a_steady_role_is_left_alone(weekly, usage):
    apply_usage_adjustment(weekly, usage, ROSTER, 7)
    assert weekly.points["steady|RB"] == pytest.approx(12.0)


def test_an_out_player_is_not_adjusted(weekly, usage):
    """Scaling a zero is a zero; reporting it buries the real adjustments."""
    notes = apply_usage_adjustment(weekly, usage, ROSTER, 7)
    assert not any("hurt|RB" in n for n in notes)


def test_every_adjustment_reaches_the_audit_log(weekly, usage, tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    notes = apply_usage_adjustment(weekly, usage, ROSTER, 7, audit=audit,
                                   league_id="cuomo")
    entries = AuditLog(tmp_path / "audit.jsonl").entries()
    kinds = [e for e in entries if e.kind == "usage_adjustment"]
    assert len(kinds) == len(notes) == 2
    assert all("recent_snap" in e.payload for e in kinds)


def test_flagging_is_the_default_and_changes_nothing(weekly, usage):
    """Reporting a trend must not move a projection on its own."""
    before = dict(weekly.points)
    notable_trends(usage, ROSTER, 7)
    assert weekly.points == before
