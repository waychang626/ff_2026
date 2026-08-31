"""Replacement level: how many of a position are gone before you can get one free.

ffanalytics' default `vor_baseline` (QB 13, RB 35, WR 36, TE 13, K 8, DST 3)
assumes a 12-team league. It is badly wrong for an 8-team superflex, so the
baseline is always passed explicitly (brief section 5).

There are two defensible definitions of "replacement", and the brief's League 1
table uses both:

  starter_demand  the worst *starter* in the league. fixed + flex slots + 1.
                  Answers "what do I give up by not starting this guy?"
  drafted         the best *free agent* after the draft. fixed + flex + bench + 1.
                  Answers "what can I replace this guy with for nothing?"

For a punted position they coincide (nobody benches a second kicker). For TE in
a league with no TE slot they diverge sharply: starter demand says ~3, roster
count says ~10. `explain()` prints both against the configured values so the
gap is visible rather than buried.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import FLEX_SLOTS, SLOT_ELIGIBILITY, LeagueConfig
from .ids import POSITIONS


@dataclass(frozen=True)
class BaselineDerivation:
    fixed: dict[str, float]
    flex: dict[str, float]
    bench: dict[str, float]
    starter_demand: dict[str, int]
    drafted: dict[str, int]
    configured: dict[str, int]

    def gap(self) -> dict[str, tuple[int, int]]:
        """configured - starter_demand, and configured - drafted, per position."""
        return {
            pos: (
                self.configured[pos] - self.starter_demand[pos],
                self.configured[pos] - self.drafted[pos],
            )
            for pos in POSITIONS
        }


def _flex_allocation(cfg: LeagueConfig) -> dict[str, float]:
    """Spread each flex slot across eligible positions by the configured share."""
    alloc = {p: 0.0 for p in POSITIONS}
    counts = cfg.roster.slot_counts()
    for slot, per_team in counts.items():
        if slot not in FLEX_SLOTS:
            continue
        shares = cfg.flex_shares.get(slot, {})
        total_share = sum(shares.values())
        if total_share <= 0:
            raise ValueError(f"flex_shares[{slot}] sums to {total_share}; must be > 0")
        eligible = set(SLOT_ELIGIBILITY[slot])
        stray = set(shares) - eligible
        if stray:
            raise ValueError(
                f"flex_shares[{slot}] names position(s) {sorted(stray)} that cannot "
                f"fill that slot (eligible: {sorted(eligible)})"
            )
        league_slots = per_team * cfg.teams
        for pos, share in shares.items():
            alloc[pos] += league_slots * share / total_share
    return alloc


def _bench_allocation(cfg: LeagueConfig) -> dict[str, float]:
    alloc = {p: 0.0 for p in POSITIONS}
    shares = cfg.bench_shares
    total = sum(shares.values())
    if total <= 0:
        return alloc
    bench_slots = cfg.roster.bench * cfg.teams
    for pos, share in shares.items():
        alloc[pos] += bench_slots * share / total
    return alloc


def derive(cfg: LeagueConfig) -> BaselineDerivation:
    fixed = {p: v * cfg.teams for p, v in cfg.roster.fixed_slots_per_team().items()}
    flex = _flex_allocation(cfg)
    bench = _bench_allocation(cfg)

    starter_demand = {p: int(round(fixed[p] + flex[p])) + 1 for p in POSITIONS}
    drafted = {p: int(round(fixed[p] + flex[p] + bench[p])) + 1 for p in POSITIONS}
    return BaselineDerivation(
        fixed=fixed,
        flex=flex,
        bench=bench,
        starter_demand=starter_demand,
        drafted=drafted,
        configured=dict(cfg.vor_baseline),
    )


def explain(cfg: LeagueConfig) -> str:
    d = derive(cfg)
    lines = [
        f"Replacement level - {cfg.name} ({cfg.teams} teams, {cfg.rounds} rounds, "
        f"{cfg.total_drafted} players drafted)",
        "",
        f"  starters: {' '.join(cfg.roster.starters)}   bench: {cfg.roster.bench}",
        "",
        f"  {'pos':<5}{'fixed':>7}{'flex':>7}{'bench':>7}"
        f"{'starter':>9}{'drafted':>9}{'config':>8}{'gap':>8}",
    ]
    for pos in POSITIONS:
        g_start, g_draft = d.gap()[pos]
        flag = ""
        if abs(g_start) > 1 and abs(g_draft) > 1:
            flag = "  <-- matches neither derivation"
        lines.append(
            f"  {pos:<5}{d.fixed[pos]:>7.0f}{d.flex[pos]:>7.1f}{d.bench[pos]:>7.1f}"
            f"{d.starter_demand[pos]:>9}{d.drafted[pos]:>9}"
            f"{d.configured[pos]:>8}{g_start:>+8}{flag}"
        )
    lines += [
        "",
        "  starter = fixed + flex + 1   (value over the worst starter)",
        "  drafted = fixed + flex + bench + 1   (value over the best free agent)",
        "  gap     = configured - starter",
        "",
        "  The engine uses the `config` column. Edit baselines.explicit to change it.",
    ]
    return "\n".join(lines)
