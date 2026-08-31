"""League configuration: load, validate, and expose as typed objects.

Every opinion the engine holds lives in one of these files, written before the
draft. Nothing here is a parameter of `recommend_pick` - that is the whole
point of the tool contract (brief section 2). If you find yourself wanting to
pass something at call time, it belongs in this file instead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .ids import POSITIONS

# Which positions may legally fill each roster slot.
SLOT_ELIGIBILITY: dict[str, tuple[str, ...]] = {
    "QB": ("QB",),
    "RB": ("RB",),
    "WR": ("WR",),
    "TE": ("TE",),
    "K": ("K",),
    "DEF": ("DST",),
    "DST": ("DST",),
    "RT": ("RB", "TE"),
    "WT": ("WR", "TE"),
    "WR_RB": ("WR", "RB"),
    "WRT": ("WR", "RB", "TE"),      # W/R/T flex
    "QWRT": ("QB", "WR", "RB", "TE"),  # Q/W/R/T superflex
}

FLEX_SLOTS = {s for s, elig in SLOT_ELIGIBILITY.items() if len(elig) > 1}


class ConfigError(ValueError):
    """The config file is malformed."""


class MissingLeagueInput(ConfigError):
    """A required league input was never supplied.

    Distinct from ConfigError so callers can tell "you typo'd the YAML" from
    "nobody has told us this league's roster slots yet".
    """


@dataclass(frozen=True)
class ScoringRules:
    """Stat-line multipliers plus the points-allowed bracket for DST."""

    offense: dict[str, float] = field(default_factory=dict)
    kicking: dict[str, float] = field(default_factory=dict)
    dst: dict[str, float] = field(default_factory=dict)
    pts_bracket: tuple[tuple[float, float], ...] = ()

    @property
    def multipliers(self) -> dict[str, float]:
        merged: dict[str, float] = {}
        merged.update(self.offense)
        merged.update(self.kicking)
        merged.update(self.dst)
        return merged


@dataclass(frozen=True)
class RosterSpec:
    starters: tuple[str, ...]
    bench: int
    ir: int = 0

    @property
    def total_picks(self) -> int:
        """Draftable slots per team. IR is not drafted into."""
        return len(self.starters) + self.bench

    def slot_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for slot in self.starters:
            counts[slot] = counts.get(slot, 0) + 1
        return counts

    def fixed_slots_per_team(self) -> dict[str, int]:
        """Starter slots, per team, that admit exactly one position."""
        demand = {p: 0 for p in POSITIONS}
        for slot, n in self.slot_counts().items():
            elig = SLOT_ELIGIBILITY[slot]
            if len(elig) == 1:
                demand[elig[0]] += n
        return demand


@dataclass(frozen=True)
class CalibrationParams:
    """Section 3.3 corrections. Every number here is a published figure."""

    slopes: dict[str, float]
    optimism: dict[str, float]
    r_squared: dict[str, float]
    anchor_multiple: float = 1.5
    aggregate: str = "average"  # never "weighted"; see brief 3.3

    def __post_init__(self) -> None:
        if self.aggregate != "average":
            raise ConfigError(
                "calibration.aggregate must be 'average'. Source accuracy does not "
                "persist year to year (brief 3.3); weighting by past accuracy lost "
                "64% of head-to-heads against the simple mean."
            )


@dataclass(frozen=True)
class OpponentParams:
    """Section 3.4 priors for how the other seats behave."""

    adp_noise_picks: float = 8.0
    herding_multiplier: float = 2.5
    herding_qb_pick_limit: int = 20
    herding_positions: tuple[str, ...] = ("QB", "K", "DST")
    always_herd_positions: tuple[str, ...] = ("K", "DST")
    reach_temperature: float = 1.0


@dataclass(frozen=True)
class SimParams:
    n_sims: int = 1000
    seed: int = 20260903
    weekly_cv: dict[str, float] = field(default_factory=dict)
    games: int = 17
    candidate_pool: int = 12
    injury_game_loss: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftPolicy:
    """Hard guards derived from research, not from taste."""

    # Brief 3.5: common builds over-spend on K/DST. Never spend a pick on one
    # before this round unless the remaining picks are exactly the mandatory
    # slots left to fill.
    min_round_k: int = 15
    min_round_dst: int = 14


@dataclass(frozen=True)
class LeagueConfig:
    league_id: str
    name: str
    teams: int
    roster: RosterSpec
    scoring: ScoringRules
    vor_baseline: dict[str, int]
    calibration: CalibrationParams
    opponents: OpponentParams
    sim: SimParams
    policy: DraftPolicy
    playoff_teams: int
    regular_season_weeks: int
    playoff_weeks: tuple[int, ...]
    draft_type: str = "snake"
    my_seat: int | None = None
    flex_shares: dict[str, dict[str, float]] = field(default_factory=dict)
    bench_shares: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    source_path: str = ""

    @property
    def rounds(self) -> int:
        return self.roster.total_picks

    @property
    def total_drafted(self) -> int:
        return self.teams * self.rounds

    def fingerprint(self) -> str:
        """Stable hash of everything the engine reads from this config.

        Goes into the audit log so a replay can prove it ran against the same
        settings, not just the same picks.
        """
        payload = {
            "league_id": self.league_id,
            "teams": self.teams,
            "starters": list(self.roster.starters),
            "bench": self.roster.bench,
            "scoring": {
                "offense": self.scoring.offense,
                "kicking": self.scoring.kicking,
                "dst": self.scoring.dst,
                "pts_bracket": [list(b) for b in self.scoring.pts_bracket],
            },
            "vor_baseline": self.vor_baseline,
            "calibration": {
                "slopes": self.calibration.slopes,
                "optimism": self.calibration.optimism,
                "r_squared": self.calibration.r_squared,
                "anchor_multiple": self.calibration.anchor_multiple,
            },
            "opponents": vars(self.opponents),
            "sim": {
                "n_sims": self.sim.n_sims,
                "seed": self.sim.seed,
                "weekly_cv": self.sim.weekly_cv,
                "games": self.sim.games,
                "candidate_pool": self.sim.candidate_pool,
                "injury_game_loss": self.sim.injury_game_loss,
            },
            "policy": vars(self.policy),
            "playoff_teams": self.playoff_teams,
            "regular_season_weeks": self.regular_season_weeks,
            "playoff_weeks": list(self.playoff_weeks),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _require(data: dict[str, Any], key: str, path: str) -> Any:
    if key not in data or data[key] is None:
        raise MissingLeagueInput(
            f"{path}: required field '{key}' is missing or null. "
            f"The engine cannot proceed without it."
        )
    return data[key]


def load_league(path: str | Path) -> LeagueConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"no league config at {path}")
    data = yaml.safe_load(path.read_text()) or {}
    return parse_league(data, source_path=str(path))


def parse_league(data: dict[str, Any], source_path: str = "") -> LeagueConfig:
    where = source_path or "<config>"

    roster_raw = data.get("roster") or {}
    starters = roster_raw.get("starters")
    if not starters:
        raise MissingLeagueInput(
            f"{where}: roster.starters is not set.\n"
            "Replacement level - and therefore every VOR number the engine "
            "produces - is a direct function of the starting lineup. There is no "
            "sensible default: an 8-team superflex and a 12-team single-QB league "
            "value the same player completely differently.\n"
            "Supply the starting slots (e.g. [QB, RB, RB, WR, WR, TE, WRT, K, DEF]) "
            "and the bench size, then re-run."
        )
    unknown = [s for s in starters if s not in SLOT_ELIGIBILITY]
    if unknown:
        raise ConfigError(
            f"{where}: unknown roster slot(s) {unknown}. "
            f"Known slots: {sorted(SLOT_ELIGIBILITY)}"
        )
    roster = RosterSpec(
        starters=tuple(starters),
        bench=int(_require(roster_raw, "bench", where)),
        ir=int(roster_raw.get("ir", 0)),
    )

    scoring_raw = data.get("scoring") or {}
    bracket = tuple(
        (float(b["threshold"]), float(b["points"]))
        for b in scoring_raw.get("pts_bracket", [])
    )
    if bracket != tuple(sorted(bracket)):
        raise ConfigError(f"{where}: scoring.pts_bracket must be sorted by threshold")
    scoring = ScoringRules(
        offense={k: float(v) for k, v in (scoring_raw.get("offense") or {}).items()},
        kicking={k: float(v) for k, v in (scoring_raw.get("kicking") or {}).items()},
        dst={k: float(v) for k, v in (scoring_raw.get("dst") or {}).items()},
        pts_bracket=bracket,
    )

    cal_raw = data.get("calibration") or {}
    calibration = CalibrationParams(
        slopes={k.upper(): float(v) for k, v in (cal_raw.get("slopes") or {}).items()},
        optimism={k.upper(): float(v) for k, v in (cal_raw.get("optimism") or {}).items()},
        r_squared={k.upper(): float(v) for k, v in (cal_raw.get("r_squared") or {}).items()},
        anchor_multiple=float(cal_raw.get("anchor_multiple", 1.5)),
        aggregate=str(cal_raw.get("aggregate", "average")),
    )

    opp_raw = data.get("opponents") or {}
    opponents = OpponentParams(
        adp_noise_picks=float(opp_raw.get("adp_noise_picks", 8.0)),
        herding_multiplier=float(opp_raw.get("herding_multiplier", 2.5)),
        herding_qb_pick_limit=int(opp_raw.get("herding_qb_pick_limit", 20)),
        herding_positions=tuple(opp_raw.get("herding_positions", ("QB", "K", "DST"))),
        always_herd_positions=tuple(opp_raw.get("always_herd_positions", ("K", "DST"))),
        reach_temperature=float(opp_raw.get("reach_temperature", 1.0)),
    )

    sim_raw = data.get("sim") or {}
    sim = SimParams(
        n_sims=int(sim_raw.get("n_sims", 1000)),
        seed=int(sim_raw.get("seed", 20260903)),
        weekly_cv={k.upper(): float(v) for k, v in (sim_raw.get("weekly_cv") or {}).items()},
        games=int(sim_raw.get("games", 17)),
        candidate_pool=int(sim_raw.get("candidate_pool", 12)),
        injury_game_loss={
            k.upper(): float(v) for k, v in (sim_raw.get("injury_game_loss") or {}).items()
        },
    )

    pol_raw = data.get("policy") or {}
    policy = DraftPolicy(
        min_round_k=int(pol_raw.get("min_round_k", 15)),
        min_round_dst=int(pol_raw.get("min_round_dst", 14)),
    )

    baselines_raw = data.get("baselines") or {}
    explicit = baselines_raw.get("explicit") or {}
    if not explicit:
        raise MissingLeagueInput(
            f"{where}: baselines.explicit is not set. ffanalytics' default "
            "vor_baseline assumes a 12-team league and is wrong for any other "
            "shape (brief 5). Run `ffdraft baselines --explain` to derive them."
        )
    vor_baseline = {k.upper(): int(v) for k, v in explicit.items()}
    missing_pos = [p for p in POSITIONS if p not in vor_baseline]
    if missing_pos:
        raise ConfigError(f"{where}: baselines.explicit missing positions {missing_pos}")

    sched = data.get("schedule") or {}
    playoff_weeks = tuple(int(w) for w in sched.get("playoff_weeks", ()))

    cfg = LeagueConfig(
        league_id=str(_require(data, "league_id", where)),
        name=str(data.get("name", data.get("league_id", ""))),
        teams=int(_require(data, "teams", where)),
        roster=roster,
        scoring=scoring,
        vor_baseline=vor_baseline,
        calibration=calibration,
        opponents=opponents,
        sim=sim,
        policy=policy,
        playoff_teams=int(_require(sched, "playoff_teams", where + ".schedule")),
        regular_season_weeks=int(sched.get("regular_season_weeks", 14)),
        playoff_weeks=playoff_weeks,
        draft_type=str((data.get("draft") or {}).get("type", "snake")),
        my_seat=(data.get("draft") or {}).get("my_seat"),
        flex_shares={
            k: {kk.upper(): float(vv) for kk, vv in v.items()}
            for k, v in (baselines_raw.get("flex_shares") or {}).items()
        },
        bench_shares={
            k.upper(): float(v) for k, v in (baselines_raw.get("bench_shares") or {}).items()
        },
        notes=tuple(data.get("notes", ())),
        source_path=source_path,
    )
    _validate(cfg, where)
    return cfg


def _validate(cfg: LeagueConfig, where: str) -> None:
    if cfg.teams < 2:
        raise ConfigError(f"{where}: teams must be >= 2")
    if cfg.playoff_teams > cfg.teams:
        raise ConfigError(f"{where}: playoff_teams exceeds teams")
    if cfg.draft_type not in ("snake", "linear"):
        raise ConfigError(f"{where}: draft.type must be 'snake' or 'linear'")
    if cfg.my_seat is not None and not (1 <= cfg.my_seat <= cfg.teams):
        raise ConfigError(f"{where}: draft.my_seat must be in 1..{cfg.teams}")
    for pos in POSITIONS:
        if pos not in cfg.calibration.slopes:
            raise ConfigError(f"{where}: calibration.slopes missing {pos}")
        if pos not in cfg.sim.weekly_cv:
            raise ConfigError(f"{where}: sim.weekly_cv missing {pos}")
    for slot in set(cfg.roster.starters) & FLEX_SLOTS:
        if slot not in cfg.flex_shares:
            raise ConfigError(
                f"{where}: roster uses flex slot {slot!r} but baselines.flex_shares "
                f"has no share for it. The split across eligible positions is an "
                f"assumption and must be stated explicitly."
            )


def league_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "leagues"


def load_by_id(league_id: str) -> LeagueConfig:
    """Resolve a league_id to its config file."""
    path = league_dir() / f"{league_id}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in league_dir().glob("*.yaml"))
        raise ConfigError(f"unknown league_id {league_id!r}; available: {available}")
    return load_league(path)
