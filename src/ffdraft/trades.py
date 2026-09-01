"""Post-draft trade search: what to offer, to whom, and why they would accept.

A fantasy trade creates value out of *positional imbalance*, not out of one
side being fooled. You start two RBs; if you rostered five startable ones, the
fifth scores you nothing on Sunday. An owner who started the draft WR-heavy has
the mirror-image problem. Moving your RB4 for their WR3 can raise both starting
lineups at once, and nobody had to lose.

So the search is a surplus search, and it is run in two stages for the same
reason `engine.py` shortlists before simulating.

  Stage 1 - the filter. Recompute both teams' optimal starting lineups on
  point estimates, for every candidate trade. This is the `lineup.py` primitive
  the draft engine already uses, vectorised over candidates, and it costs
  microseconds each. Keep only the trades that raise *both* lineups.

  Stage 2 - the number that matters. Run the season Monte Carlo on the
  survivors and report the change in *your* P(title). Stage 1 cannot see bye
  weeks, week-to-week variance or the value of depth as injury cover; stage 2
  can, and it sometimes reverses stage 1's ranking. Depth that looks like dead
  weight in a point estimate is worth something across seventeen weeks.

Byes, specifically, because a trade is the easiest way to wreck a week 9 by
accident. Stage 1 is bye-blind: it compares season totals, so two receivers who
are both off in week 11 look identical to two who are not. Stage 2 is not -
`draw_season` zeroes a player's bye week outright, and the weekly lineup
optimiser then has to cover the hole from your bench or eat the zero, so a
collision shows up as lower P(title) without anything special being done about
it. That is the correct place for it to be priced, but a number moving by half
a point does not *tell* you what went wrong, so every finalist is also profiled
week by week and any trade that deepens your worst week is flagged in words.

The objective is your title odds. The other owner's gain is a *constraint*, not
something being maximised - this tool finds trades they have a reason to take,
which is not the same as being fair to them.

What this cannot know, and does not pretend to: the other owner does not run a
lineup optimiser. They value players by name, by draft position, and by who
they watched on Sunday. Every idea here carries the market-value gap alongside
the projection gap, because a trade that is correct and unsellable is worth
nothing on a Tuesday night.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .audit import AuditLog, state_hash
from .board import Board
from .config import SLOT_ELIGIBILITY, LeagueConfig
from .draft import DraftState
from .lineup import VectorLineup, lineup_slots
from .simulate import SeasonSimulator, draw_season


@dataclass(frozen=True)
class TradeIdea:
    """One proposal, from your side of the table."""

    partner_seat: int
    give: tuple[str, ...]          # player_ids you send
    get: tuple[str, ...]           # player_ids you receive
    my_lineup_gain: float          # points added to your optimal starting lineup
    their_lineup_gain: float       # points added to theirs
    my_p_title: float
    my_delta_p_title: float        # versus not trading at all
    delta_se: float                # standard error on that delta
    adp_sent: float                # mean market ADP of what you send
    adp_received: float            # ...and of what you get
    my_drops: tuple[str, ...] = ()      # forced by roster limits
    their_drops: tuple[str, ...] = ()
    bye_week: int = 0                   # your worst week after the trade
    bye_drop_before: float = 0.0        # points below a typical week, as drafted
    bye_drop_after: float = 0.0         # ...and after
    flags: tuple[str, ...] = ()

    @property
    def bye_worsening(self) -> float:
        return self.bye_drop_after - self.bye_drop_before

    @property
    def joint_gain(self) -> float:
        return self.my_lineup_gain + self.their_lineup_gain

    @property
    def market_gap(self) -> float:
        """Positive means you are receiving the later-drafted, lesser names.

        That is the direction a trade is easy to sell in. Negative means you
        are asking for the bigger name and should expect to hear no.
        """
        return self.adp_received - self.adp_sent


@dataclass
class TradeReport:
    league_id: str
    my_seat: int
    state_hash: str
    seed: int
    n_sims: int
    baseline_p_title: float
    ideas: list[TradeIdea] = field(default_factory=list)
    surplus: dict[int, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    considered: int = 0
    simulated: int = 0


# --- roster shape ------------------------------------------------------------
def starter_demand(config: LeagueConfig) -> dict[str, float]:
    """Roughly how many of each position a team starts.

    Flex slots are split evenly across the positions they admit. Approximate on
    purpose - this feeds the human-readable "you are RB-heavy" line, never a
    number the recommendation depends on.
    """
    demand = {pos: 0.0 for pos in ("QB", "RB", "WR", "TE", "K", "DST")}
    for slot in config.roster.starters:
        eligible = SLOT_ELIGIBILITY[slot]
        for pos in eligible:
            if pos in demand:
                demand[pos] += 1.0 / len(eligible)
    return demand


def positional_surplus(
    config: LeagueConfig, board: Board, roster: list[str]
) -> dict[str, float]:
    """Startable bodies minus starting slots, per position.

    "Startable" is anyone projected above the team's own median at the
    position - a crude cut, but the point is only to say which way a roster
    leans, and it does not enter the arithmetic.
    """
    demand = starter_demand(config)
    counts: dict[str, float] = {}
    for pid in roster:
        pos = board.player(pid).pos
        counts[pos] = counts.get(pos, 0.0) + 1.0
    return {pos: counts.get(pos, 0.0) - need for pos, need in demand.items()}


def _rosters_by_seat(config: LeagueConfig, drafted: list[str]) -> dict[int, list[str]]:
    state = DraftState(config=config, drafted=list(drafted))
    return {seat: state.roster_of(seat) for seat in range(1, config.teams + 1)}


# --- stage 1: lineup arithmetic, vectorised ----------------------------------
def _lineup_totals(
    vector_lineup: VectorLineup, board: Board, rosters: np.ndarray
) -> np.ndarray:
    """Optimal starting-lineup points for each row of `rosters`.

    `rosters` is (n, depth) of board indices, -1 for an empty spot. Same
    gather-then-mask shape `SeasonSimulator.team_weekly_totals` uses, so the
    filter and the simulation agree on what a lineup is worth.
    """
    filled = rosters >= 0
    safe = np.where(filled, rosters, 0)
    scores = np.where(filled, board.points[safe], 0.0)
    codes = np.where(filled, board.pos_code[safe], -1)
    return vector_lineup.total(scores, codes)


def _pad(rows: list[list[int]], width: int) -> np.ndarray:
    out = np.full((len(rows), width), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def _trim(idx: list[int], board: Board, limit: int) -> tuple[list[int], list[int]]:
    """Drop the lowest-projected players until the roster is legal again.

    A 1-for-2 that leaves you over the roster limit is not a free upgrade: the
    league makes you cut someone, and pretending otherwise makes every
    lopsided trade look like a win.
    """
    if len(idx) <= limit:
        return idx, []
    order = sorted(idx, key=lambda i: (-float(board.points[i]), board.players[i].player_id))
    return order[:limit], order[limit:]


def bye_week_profile(
    config: LeagueConfig,
    board: Board,
    vector_lineup: VectorLineup,
    roster_idx: list[int],
) -> np.ndarray:
    """Optimal starting-lineup points in each week, on point estimates.

    Deterministic - no weekly noise, no injuries, just each player's season
    total spread over the weeks they actually play, with their bye zeroed. It
    answers one question: which week does this roster fall apart in, and how
    far. The Monte Carlo already prices that; this names it.
    """
    n_weeks = config.regular_season_weeks + len(config.playoff_weeks)
    if not roster_idx:
        return np.zeros(n_weeks)
    idx = np.asarray(roster_idx, dtype=np.int64)
    bye = board.bye[idx]
    has_bye = (bye >= 1) & (bye <= n_weeks)
    active = np.where(has_bye, n_weeks - 1, n_weeks).astype(float)
    rate = board.points[idx] / active

    weeks = np.arange(1, n_weeks + 1)[:, None]
    on_bye = (weeks == bye[None, :]) & has_bye[None, :]
    scores = np.where(on_bye, 0.0, rate[None, :])
    codes = np.broadcast_to(board.pos_code[idx][None, :], scores.shape)
    return vector_lineup.total(scores, codes)


def worst_bye_week(profile: np.ndarray) -> tuple[int, float]:
    """(week number, points below a typical week) for the roster's worst week.

    Measured against the median week rather than the best one, so a single
    stacked bye reads as the outlier it is.
    """
    if profile.size == 0:
        return 0, 0.0
    week = int(np.argmin(profile)) + 1
    return week, float(np.median(profile) - profile.min())


# --- the search --------------------------------------------------------------
def find_trades(
    config: LeagueConfig,
    board: Board,
    drafted: list[str],
    my_seat: int,
    *,
    partners: list[int] | None = None,
    max_give: int = 2,
    max_get: int = 2,
    min_their_gain: float = 1.0,
    min_my_gain: float = 1.0,
    top: int = 5,
    per_partner: int = 8,
    max_per_partner_shown: int = 2,
    audit: AuditLog | None = None,
) -> TradeReport:
    """Trades that raise your title odds and that the other owner would take.

    `min_their_gain` is the constraint that makes this a trade search rather
    than a wish list: an offer that only helps you is not an offer.
    """
    unknown = [p for p in drafted if p not in board.index]
    if unknown:
        raise ValueError(
            f"not on the board: {unknown}. A trade computed against a roster "
            f"that does not match the league is worse than no answer."
        )
    if not 1 <= my_seat <= config.teams:
        raise ValueError(f"seat {my_seat} does not exist; this league has {config.teams}")

    rosters = _rosters_by_seat(config, drafted)
    mine = rosters[my_seat]
    if not mine:
        raise ValueError(f"seat {my_seat} has no players; nothing to trade")

    sh = state_hash(config.fingerprint(), board.fingerprint(), list(drafted), my_seat, 0)
    seed = (config.sim.seed ^ int(sh[:8], 16)) % (2**32)
    rng = np.random.default_rng(seed)

    limit = config.roster.total_picks
    vector_lineup = VectorLineup(config.roster)
    notes: list[str] = []

    my_idx = [board.idx(p) for p in mine]
    depth = max(len(r) for r in rosters.values()) + max_get
    my_base = float(_lineup_totals(vector_lineup, board, _pad([my_idx], depth))[0])
    base_bye_week, base_bye_drop = worst_bye_week(
        bye_week_profile(config, board, vector_lineup, my_idx)
    )

    candidates: list[dict] = []
    considered = 0
    seats = partners if partners is not None else [
        s for s in range(1, config.teams + 1) if s != my_seat
    ]
    for seat in seats:
        if seat == my_seat or not rosters[seat]:
            continue
        their_idx = [board.idx(p) for p in rosters[seat]]
        their_base = float(_lineup_totals(vector_lineup, board, _pad([their_idx], depth))[0])

        gives = [c for k in range(1, max_give + 1) for c in combinations(my_idx, k)]
        gets = [c for k in range(1, max_get + 1) for c in combinations(their_idx, k)]

        rows_mine: list[list[int]] = []
        rows_theirs: list[list[int]] = []
        meta: list[tuple] = []
        for give in gives:
            my_kept = [i for i in my_idx if i not in give]
            for get in gets:
                their_kept = [i for i in their_idx if i not in get]
                my_new, my_drop = _trim(my_kept + list(get), board, limit)
                their_new, their_drop = _trim(their_kept + list(give), board, limit)
                rows_mine.append(my_new)
                rows_theirs.append(their_new)
                meta.append((give, get, tuple(my_drop), tuple(their_drop)))
        if not meta:
            continue
        considered += len(meta)

        my_after = _lineup_totals(vector_lineup, board, _pad(rows_mine, depth))
        their_after = _lineup_totals(vector_lineup, board, _pad(rows_theirs, depth))
        my_gain = my_after - my_base
        their_gain = their_after - their_base

        keep = np.flatnonzero((my_gain >= min_my_gain) & (their_gain >= min_their_gain))
        found = []
        for i in keep:
            give, get, my_drop, their_drop = meta[i]
            found.append({
                "seat": seat,
                "give": give,
                "get": get,
                "my_gain": float(my_gain[i]),
                "their_gain": float(their_gain[i]),
                "my_new": rows_mine[i],
                "their_new": rows_theirs[i],
                "my_drop": my_drop,
                "their_drop": their_drop,
            })
        # Shortlist per partner rather than globally. One opponent whose roster
        # happens to be the mirror of yours will otherwise fill every slot with
        # near-identical variants of the same package, and the other six teams
        # never get simulated at all.
        found.sort(key=lambda c: (-(c["my_gain"] + c["their_gain"]), -c["my_gain"]))
        candidates.extend(_spread(found, per_partner))

    if not candidates:
        notes.append(
            "no trade raises both starting lineups. That is a normal result: it "
            "means your roster shape already matches the league's."
        )
        return TradeReport(
            league_id=config.league_id, my_seat=my_seat, state_hash=sh, seed=seed,
            n_sims=config.sim.n_sims, baseline_p_title=float("nan"), ideas=[],
            surplus={s: positional_surplus(config, board, r) for s, r in rosters.items()},
            notes=notes, considered=considered, simulated=0,
        )

    finalists = candidates

    # --- stage 2: the season Monte Carlo, on common random numbers ----------
    n_sims = config.sim.n_sims
    draws = draw_season(board, config, n_sims, rng)
    season = SeasonSimulator(board, config)

    base_matrix = _pad([[board.idx(p) for p in rosters[s]]
                        for s in range(1, config.teams + 1)], depth)
    baseline = season.run(
        np.broadcast_to(base_matrix[None, :, :], (n_sims, config.teams, depth)),
        draws, my_seat,
    )

    ideas: list[TradeIdea] = []
    for cand in finalists:
        matrix = base_matrix.copy()
        matrix[my_seat - 1] = _pad([cand["my_new"]], depth)[0]
        matrix[cand["seat"] - 1] = _pad([cand["their_new"]], depth)[0]
        result = season.run(
            np.broadcast_to(matrix[None, :, :], (n_sims, config.teams, depth)),
            draws, my_seat,
        )
        paired = result.title_indicator.astype(float) - baseline.title_indicator.astype(float)
        delta = result.p_title - baseline.p_title
        delta_se = float(paired.std(ddof=1) / np.sqrt(n_sims)) if n_sims > 1 else 0.0

        give_ids = tuple(board.players[i].player_id for i in cand["give"])
        get_ids = tuple(board.players[i].player_id for i in cand["get"])
        adp_sent = float(np.mean([board.adp[i] for i in cand["give"]]))
        adp_received = float(np.mean([board.adp[i] for i in cand["get"]]))

        after_week, after_drop = worst_bye_week(
            bye_week_profile(config, board, vector_lineup, cand["my_new"])
        )

        flags: list[str] = []
        if after_drop > base_bye_drop + 5.0:
            flags.append(
                f"bye collision: week {after_week} drops "
                f"{after_drop:.0f} pts below a normal week, up from "
                f"{base_bye_drop:.0f} - the title odds already count this"
            )
        if abs(delta) < 2 * delta_se:
            flags.append("inside simulation noise; the lineup gain is the firmer number")
        if adp_received < adp_sent - 12:
            flags.append("hard sell: you are asking for the earlier-drafted name")
        if any(board.adp_imputed[i] for i in cand["give"] + cand["get"]):
            flags.append("a player here has no market ADP; the sell-difficulty read is a guess")
        if cand["their_drop"]:
            flags.append("they must cut a player to fit this")
        if cand["my_drop"]:
            flags.append("you must cut a player to fit this")

        ideas.append(TradeIdea(
            partner_seat=cand["seat"],
            give=give_ids,
            get=get_ids,
            my_lineup_gain=cand["my_gain"],
            their_lineup_gain=cand["their_gain"],
            my_p_title=result.p_title,
            my_delta_p_title=delta,
            delta_se=delta_se,
            adp_sent=adp_sent,
            adp_received=adp_received,
            my_drops=tuple(board.players[i].player_id for i in cand["my_drop"]),
            their_drops=tuple(board.players[i].player_id for i in cand["their_drop"]),
            bye_week=after_week,
            bye_drop_before=base_bye_drop,
            bye_drop_after=after_drop,
            flags=tuple(flags),
        ))

    ideas.sort(key=lambda t: (
        -round(t.my_delta_p_title, 9), -round(t.my_lineup_gain, 6), t.give, t.get
    ))
    ideas = _distinct(ideas, max_per_partner_shown)
    if ideas and ideas[0].my_delta_p_title <= 0:
        notes.append(
            "every candidate is flat or negative on title odds once the season "
            "is simulated: the lineup upgrades are being paid for in depth."
        )

    report = TradeReport(
        league_id=config.league_id, my_seat=my_seat, state_hash=sh, seed=seed,
        n_sims=n_sims, baseline_p_title=baseline.p_title, ideas=ideas[:top],
        surplus={s: positional_surplus(config, board, r) for s, r in rosters.items()},
        notes=notes, considered=considered, simulated=len(finalists),
    )
    (audit or AuditLog()).record(
        state_hash=sh, league_id=config.league_id, pick_number=0, kind="find_trades",
        payload={
            "seed": seed, "n_sims": n_sims, "my_seat": my_seat,
            "considered": considered, "simulated": len(finalists),
            "baseline_p_title": round(baseline.p_title, 6),
            "ideas": [
                {
                    "partner_seat": t.partner_seat, "give": list(t.give), "get": list(t.get),
                    "my_lineup_gain": round(t.my_lineup_gain, 3),
                    "their_lineup_gain": round(t.their_lineup_gain, 3),
                    "delta_p_title": round(t.my_delta_p_title, 6),
                }
                for t in report.ideas
            ],
        },
    )
    return report


def _spread(found: list[dict], limit: int) -> list[dict]:
    """Top `limit` candidates for one partner, no two sending the same package.

    Variants that differ only in which of two near-identical players comes back
    are one idea, not several, and simulating all of them spends the budget on
    a distinction the user cannot act on.
    """
    out, seen = [], set()
    for cand in found:
        if cand["give"] in seen:
            continue
        seen.add(cand["give"])
        out.append(cand)
        if len(out) >= limit:
            break
    return out


def _distinct(ideas: list[TradeIdea], per_partner: int) -> list[TradeIdea]:
    """Drop repeats so the list reads as options rather than as one idea, thrice."""
    out: list[TradeIdea] = []
    counts: dict[int, int] = {}
    seen: set[tuple] = set()
    for idea in ideas:
        key = (idea.partner_seat, idea.give)
        if key in seen or counts.get(idea.partner_seat, 0) >= per_partner:
            continue
        seen.add(key)
        counts[idea.partner_seat] = counts.get(idea.partner_seat, 0) + 1
        out.append(idea)
    return out


# --- presentation ------------------------------------------------------------
def format_report(report: TradeReport, board: Board, config: LeagueConfig) -> str:
    """The four-line-per-idea card, in the shape the draft console uses."""
    out: list[str] = []
    if report.baseline_p_title == report.baseline_p_title:
        out.append(
            f"seat {report.my_seat}: P(title) {report.baseline_p_title:.1%} as drafted "
            f"({report.considered:,} trades checked, {report.simulated} simulated, "
            f"{report.n_sims} sims each)"
        )
    for note in report.notes:
        out.append(f"  note: {note}")
    if not report.ideas:
        return "\n".join(out)

    out.append("")
    for rank, idea in enumerate(report.ideas, start=1):
        give = ", ".join(board.player(p).display for p in idea.give)
        get = ", ".join(board.player(p).display for p in idea.get)
        out.append(f"{rank}. WITH seat {idea.partner_seat}")
        out.append(f"   SEND  {give}")
        out.append(f"   GET   {get}")
        out.append(
            f"   YOU   +{idea.my_lineup_gain:.1f} lineup pts, "
            f"P(title) {idea.my_p_title:.1%} ({idea.my_delta_p_title:+.2%})"
        )
        out.append(
            f"   THEM  +{idea.their_lineup_gain:.1f} lineup pts  "
            f"-> why they say yes"
        )
        out.append(
            f"   BYE   worst week {idea.bye_week}: -{idea.bye_drop_after:.0f} pts "
            f"vs a normal week ({idea.bye_worsening:+.0f} from this trade)"
        )
        sell = ("easy sell" if idea.market_gap > 12 else
                "hard sell" if idea.market_gap < -12 else "roughly even on name value")
        out.append(
            f"   SELL  ADP {idea.adp_sent:.0f} out / {idea.adp_received:.0f} in "
            f"- {sell}"
        )
        for drop in idea.my_drops:
            out.append(f"   CUT   you drop {board.player(drop).display}")
        for drop in idea.their_drops:
            out.append(f"   CUT   they drop {board.player(drop).display}")
        for flag in idea.flags:
            out.append(f"   FLAG  {flag}")
        out.append("")
    return "\n".join(out)


def format_surplus(report: TradeReport, config: LeagueConfig) -> str:
    """Who is long and short at each position - the map the trades come from."""
    positions = ["QB", "RB", "WR", "TE", "K", "DST"]
    lines = ["  seat  " + "".join(f"{p:>6}" for p in positions)]
    for seat in sorted(report.surplus):
        row = report.surplus[seat]
        marker = " <- you" if seat == report.my_seat else ""
        cells = "".join(f"{row.get(p, 0.0):>+6.1f}" for p in positions)
        lines.append(f"  {seat:<6}{cells}{marker}")
    lines.append("")
    lines.append("  Startable bodies minus starting slots. Positive is surplus to "
                 "trade from,")
    lines.append("  negative is a hole to trade into.")
    return "\n".join(lines)


def explain_lineup(board: Board, config: LeagueConfig, roster: list[str]) -> dict[str, str | None]:
    """The optimal lineup for a roster, for showing what a trade actually changed."""
    scores = {p: float(board.points[board.idx(p)]) for p in roster}
    positions = {p: board.player(p).pos for p in roster}
    return lineup_slots(scores, positions, config.roster)
