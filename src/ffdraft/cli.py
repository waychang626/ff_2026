"""Command line interface, including the live draft console.

The console is the thing you actually sit in front of on draft day. It does
the two jobs the LLM would otherwise do - resolve messy names, refuse
ambiguous ones - without needing an API key or a network, and it writes the
pick log the brief asks for as a side effect of being used.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

from .audit import AuditLog
from .baselines import explain as explain_baselines
from .board import Board, ProjectionUpdate
from .config import ConfigError, LeagueConfig, load_by_id
from .data import DEFAULT_POOL, build_board_for, default_paths
from .draft import DraftState, DraftStateError, pick_owner
from .engine import _recommend
from .ids import POSITIONS
from .replay import DraftLog, assert_deterministic, backtest, load_actuals, replay
from .scoring import to_r_scoring_rules
from .vor import replacement_points, vor_array


def _load(args) -> tuple[LeagueConfig, Board]:
    config = load_by_id(args.league)
    if getattr(args, "seat", None):
        config = dataclasses.replace(config, my_seat=args.seat)
    if getattr(args, "sims", None):
        config = dataclasses.replace(
            config, sim=dataclasses.replace(config.sim, n_sims=args.sims)
        )
    proj, market = default_paths(getattr(args, "season", 2026))
    if getattr(args, "projections", None):
        proj = Path(args.projections)
    if getattr(args, "market", None):
        market = Path(args.market)
    if not Path(proj).exists():
        raise SystemExit(
            f"no projections at {proj}\n"
            f"  run the pull:  Rscript R/pull_projections.R --season {getattr(args,'season',2026)}\n"
            f"  or point at a file:  --projections <path>\n"
            f"  or try the synthetic board:  --projections data/samples/projections_synthetic.csv"
        )
    board = build_board_for(config, proj, market, pool_size=getattr(args, "pool", DEFAULT_POOL))
    return config, board


# --- commands ----------------------------------------------------------------
def cmd_baselines(args) -> int:
    print(explain_baselines(load_by_id(args.league)))
    return 0


def cmd_export_r(args) -> int:
    config = load_by_id(args.league)
    code = to_r_scoring_rules(config.scoring, f"{config.league_id}_scoring")
    header = (
        f"# GENERATED from configs/leagues/{config.league_id}.yaml by "
        f"`ffdraft export-r --league {config.league_id}`.\n"
        f"# Do not edit by hand - edit the YAML and regenerate, so the Python\n"
        f"# engine and the ffanalytics pull cannot drift apart.\n"
    )
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + code)
        print(f"wrote {path}")
    else:
        print(header + code, end="")
    return 0


def cmd_board(args) -> int:
    config, board = _load(args)
    available = np.ones(len(board), dtype=bool)
    vor = vor_array(board, available, config.vor_baseline)
    repl = replacement_points(board, available, config.vor_baseline)

    idx = np.flatnonzero(
        board.pos_mask(args.pos.upper()) if args.pos else np.ones(len(board), bool)
    )
    idx = idx[np.argsort(-vor[idx], kind="stable")][: args.top]

    print(f"{config.name} - top {len(idx)} by VOR"
          + (f" ({args.pos.upper()})" if args.pos else ""))
    print("  replacement: " + "  ".join(f"{p}={repl[p]:.0f}" for p in POSITIONS))
    print(f"\n  {'#':<4}{'player':<28}{'pos':<5}{'tm':<5}{'proj':>7}{'sd':>7}{'VOR':>8}{'ADP':>7}")
    for rank, i in enumerate(idx, start=1):
        player = board.players[i]
        adp = f"{board.adp[i]:.0f}" + ("*" if board.adp_imputed[i] else "")
        print(f"  {rank:<4}{player.name[:27]:<28}{player.pos:<5}{player.team:<5}"
              f"{board.points[i]:7.1f}{board.sd[i]:7.1f}{vor[i]:8.1f}{adp:>7}")
    if board.adp_imputed[idx].any():
        print("\n  * ADP imputed from projection rank; survival for these is an estimate")
    return 0


def cmd_check(args) -> int:
    """Pre-draft preflight. Run this the morning of, not five minutes before."""
    config, board = _load(args)
    problems, warnings = [], []

    if config.my_seat is None:
        problems.append("draft.my_seat is unset - set it in the config or pass --seat")
    imputed = int(board.adp_imputed.sum())
    coverage = 1.0 - (imputed / max(len(board), 1))
    if coverage < 0.5:
        warnings.append(
            f"ADP MISSING for {imputed}/{len(board)} players "
            f"({coverage:.0%} covered).\n"
            f"        Draft order is falling back to VOR rank. The board and the\n"
            f"        recommendations are still sound, but SURVIVAL probabilities and\n"
            f"        the numbered pick list are approximations.\n"
            f"        Usual cause: the market CSV has an empty or 'NA' adp column.\n"
            f"        Check with:  head -3 data/market_2026.csv"
        )
    for pos in POSITIONS:
        n = int(board.pos_mask(pos).sum())
        need = config.vor_baseline[pos]
        if n < need:
            problems.append(f"only {n} {pos} on the board but replacement rank is {need}")
    thin = [p for p in POSITIONS if int(board.pos_mask(p).sum()) < 3]
    if thin:
        warnings.append(f"very few players at {thin}")

    print(f"league        {config.name} ({config.league_id})")
    print(f"shape         {config.teams} teams, {config.rounds} rounds, "
          f"{config.total_drafted} picks, {config.draft_type}")
    print(f"seat          {config.my_seat}")
    print(f"starters      {' '.join(config.roster.starters)}")
    print(f"playoffs      {config.playoff_teams} of {config.teams}, weeks {list(config.playoff_weeks)}")
    print(f"board         {len(board)} players  "
          + " ".join(f"{p}:{int(board.pos_mask(p).sum())}" for p in POSITIONS))
    print(f"market ADP    {len(board) - imputed}/{len(board)} real "
          f"({coverage:.0%}), {imputed} imputed from VOR rank")
    print(f"baselines     {config.vor_baseline}")
    print(f"sims          {config.sim.n_sims}, seed {config.sim.seed}")
    print(f"fingerprints  league={config.fingerprint()} board={board.fingerprint()}")
    for w in warnings:
        print(f"  WARN  {w}")
    for p in problems:
        print(f"  FAIL  {p}")
    print("\n" + ("READY" if not problems else "NOT READY"))
    return 1 if problems else 0


def cmd_replay(args) -> int:
    config, board = _load(args)
    log = DraftLog.load(args.log)
    seat = args.seat or log.my_seat or config.my_seat
    if args.check:
        assert_deterministic(config, board, log, seat=seat, runs=args.runs, limit=args.limit)
        print(f"deterministic across {args.runs} runs "
              f"({args.limit or 'all'} picks for seat {seat})")
        return 0
    result = replay(config, board, log, seat=seat, limit=args.limit)
    for advice in result.advices:
        actual = log.picks[advice.pick_number - 1].player_id
        top = advice.recommendations[0] if advice.recommendations else None
        match = "==" if top and top.player_id == actual else "!="
        print(f"pick {advice.pick_number:>4}  engine: {top.player_id if top else '-':<28}"
              f" {match} actual: {actual}")
    return 0


def cmd_backtest(args) -> int:
    config, board = _load(args)
    log = DraftLog.load(args.log)
    actuals = load_actuals(args.actuals)
    result = backtest(config, board, log, actuals, seat=args.seat or log.my_seat)
    print(f"seat {result.seat}")
    print(f"  engine roster : {result.engine_points:8.1f} pts")
    print(f"  actual roster : {result.actual_points:8.1f} pts")
    print(f"  delta         : {result.delta:+8.1f} pts")
    if result.missing_actuals:
        print(f"  WARN {len(result.missing_actuals)} drafted players have no actuals "
              f"and scored 0: {result.missing_actuals[:5]}")
    print("\n  engine lineup:")
    for slot, pid in result.engine_lineup.items():
        name = board.player(pid).name if pid and pid in board.index else "-"
        print(f"    {slot:<8}{name:<28}{actuals.get(pid, 0.0):8.1f}")
    return 0


def _run_stamp() -> str:
    """A sortable per-run stamp for log filenames."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _unique(path: Path) -> Path:
    """The first free filename at or after `path`.

    A timestamp alone is not enough: two mocks started inside the same second
    produce the same name, and the second silently overwrites the first - which
    is the bug this was meant to fix, just narrower.
    """
    if not path.exists():
        return path
    for n in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise SystemExit(f"cannot find a free filename near {path}")


def _preserve(path: Path) -> Path | None:
    """Move an existing non-empty log aside instead of truncating it.

    Reopening the console on the same league pointed at the same default path,
    which silently replaced the log of a draft that had already happened. A
    pick log is the only record of the room and cannot be reconstructed
    afterwards, so the cost of losing one is not symmetric with the cost of an
    extra file.
    """
    # Not a byte count: `DraftLog.save` always writes a header line, so a log
    # from a console that was opened and closed again is small but never zero.
    # Backing that up would leave a trail of files holding nothing.
    if not path.exists():
        return None
    if len([l for l in path.read_text().splitlines() if l.strip()]) <= 1:
        return None
    backup = _unique(path.with_name(f"{path.stem}.{_run_stamp()}{path.suffix}"))
    path.rename(backup)
    return backup


# --- the live console --------------------------------------------------------
HELP = """
  <number>          take that numbered player - the engine's 1/2/3 on your
                    pick, or the likely-pick list on someone else's
  <name>            record a pick by anyone   (e.g. "bijan gone", "lions d")
  me <name>         record a pick as yours
  go                get a recommendation for the pick on the clock
  undo              take back the last pick
  log [n]           show the last n picks with their numbers
  fix <n> <name>    correct pick n (use when a name resolved to the wrong guy)
  insert <n> <name> add a pick you missed at n; everything after shifts down
  drop <n>          remove a pick that never happened; the rest shift up
  roster [seat]     show a roster
  board [pos]       show the best available
  out <name> : <r>  rule a player out for the season, with a reason
  bump <name> <x> : <r>   multiply a player's projection by x
  save              write the pick log
  help / quit
"""


def cmd_draft(args) -> int:
    config, board = _load(args)
    if config.my_seat is None:
        raise SystemExit("set draft.my_seat in the config or pass --seat")

    log = DraftLog(league_id=config.league_id, my_seat=config.my_seat)
    audit = AuditLog(args.audit) if args.audit else AuditLog()
    log_path = Path(args.out or f"logs/draft_{config.league_id}.jsonl")
    moved = _preserve(log_path)
    state = DraftState(config=config, drafted=[], my_seat=config.my_seat)

    from .opponents import DraftSimulator

    simulator = DraftSimulator(board, config, config.my_seat)
    suggest_n = 0 if args.no_suggest else args.suggest
    suggestions: list[int] = []
    # The pick number `suggestions` was built for. Without this the list
    # outlives its pick: on your own turn the variable still held the previous
    # opponent's shortlist, so a bare number would have silently recorded a
    # player from a list that was no longer on screen.
    suggestions_for: int | None = None
    shown_for: int | None = None
    recommended_for: tuple | None = None

    print(f"{config.name} - seat {config.my_seat} of {config.teams}, "
          f"{config.rounds} rounds, {config.sim.n_sims} sims/pick")
    imputed = int(board.adp_imputed.sum())
    if imputed > len(board) * 0.5:
        print(f"  WARNING: {imputed}/{len(board)} players have no market ADP; "
              f"draft order is VOR rank.\n"
              f"           Survival % and the numbered list are approximate. "
              f"Run `ffdraft check` for the fix.")
    if moved:
        print(f"  kept the previous log as {moved}")
    print(f"pick log -> {log_path}")
    print(HELP)

    while not state.is_complete:
        if (
            suggest_n
            and state.on_the_clock != config.my_seat
            and shown_for != state.pick_number
        ):
            suggestions = _show_suggestions(
                simulator, board, config, state, suggest_n
            )
            suggestions_for = state.pick_number
            shown_for = state.pick_number

        # Recommend from the loop rather than after recording a pick. Doing it
        # on record missed the case where it is already your turn when the
        # console opens - seat 1 at pick 1 got no recommendation at all - and
        # also failed to refresh after `undo`, `fix`, or a projection update.
        # Keying on the board fingerprint as well as the pick number means
        # ruling a player out re-runs it, which is the whole point of doing so.
        if state.on_the_clock == config.my_seat:
            key = (state.pick_number, board.fingerprint())
            if recommended_for != key:
                print()
                suggestions = _recommend_now(
                    config, board, state, audit, show=args.show
                )
                suggestions_for = state.pick_number
                recommended_for = key

        prompt = (
            f"[R{state.current_round} p{state.pick_number} "
            f"{'YOU' if state.on_the_clock == config.my_seat else f'seat {state.on_the_clock}'}]> "
        )
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        try:
            live = suggestions if suggestions_for == state.pick_number else []
            state, board, done, refresh = _handle(
                line, state, board, config, log, log_path, audit, live
            )
            if done:
                break
            if refresh:
                # `undo` and `fix` can land back on a state we have already
                # recommended for, byte for byte. The recommendation is not
                # news, but the user just rewound deliberately and is looking
                # at the console to find out where they now are.
                recommended_for = None
        except (DraftStateError, ConfigError, KeyError, ValueError) as exc:
            print(f"  ! {exc}")

    log.save(log_path)
    print(f"\nsaved {len(log.picks)} picks to {log_path}")
    return 0


def _word(lower: str, word: str) -> bool:
    """`lower` is exactly `word`, or `word` followed by an argument.

    Not `startswith`: the board has a Logan and a Boarders on it, and
    `lower.startswith("log")` quietly turned a pick into a log listing.
    """
    return lower == word or lower.startswith(word + " ")


def _shared_command(line, state, board, config, log, log_path, audit,
                    help_text=HELP):
    """Commands that mean the same thing in the live console and in `mock`.

    Both consoles route input through here first, so a command learned in
    practice behaves identically under a clock. `mock` used to accept a strict
    subset - no `log`, no `undo`, no `fix`, no `roster <seat>` - which is the
    one thing a rehearsal must not do.

    Returns (state, board, handled, refresh). `handled` is False when the line
    is not one of these and the caller should read it itself. `refresh` means
    the state or the board moved, so any recommendation on screen is stale.
    """
    lower = line.lower()
    if lower in ("help", "?", "h"):
        print(help_text)
        return state, board, True, False
    if lower == "save":
        log.save(log_path)
        print(f"  saved {len(log.picks)} picks to {log_path}")
        return state, board, True, False
    if lower == "undo":
        if not log.picks:
            print("  nothing to undo")
            return state, board, True, False
        removed = log.picks.pop()
        state = state.undo()
        print(f"  undid pick {removed.pick}: {removed.player_id}")
        return state, board, True, True
    if _word(lower, "log"):
        parts = line.split()
        _show_log(board, state, config, _count_arg(parts, 12))
        return state, board, True, False
    if _word(lower, "fix"):
        state = _fix_pick(line, state, board, log, log_path)
        return state, board, True, True
    if _word(lower, "insert"):
        state = _insert_pick(line, state, board, config, log, log_path)
        return state, board, True, True
    if _word(lower, "drop"):
        state = _drop_pick(line, state, board, config, log, log_path)
        return state, board, True, True
    if _word(lower, "roster"):
        parts = line.split()
        seat = _seat_arg(parts, config)
        if seat is not None:
            _show_roster(board, state, seat, config)
        return state, board, True, False
    if _word(lower, "board"):
        parts = line.split()
        _show_available(board, state, config, parts[1].upper() if len(parts) > 1 else None)
        return state, board, True, False
    if _word(lower, "out") or _word(lower, "bump"):
        board = _apply_update(line, board, state, config, audit)
        return state, board, True, True
    return state, board, False, False


def _count_arg(parts, default: int) -> int:
    if len(parts) < 2:
        return default
    try:
        return int(parts[1])
    except ValueError:
        print(f"  ! {parts[1]!r} is not a number; showing {default}")
        return default


def _seat_arg(parts, config) -> int | None:
    """The seat for `roster [seat]`. None means the argument was unusable."""
    if len(parts) < 2:
        return config.my_seat
    try:
        seat = int(parts[1])
    except ValueError:
        print(f"  ! {parts[1]!r} is not a seat number (1-{config.teams})")
        return None
    if not 1 <= seat <= config.teams:
        print(f"  ! seat {seat} does not exist; this league has "
              f"{config.teams} seats")
        return None
    return seat


def _select(text, board, state, shortlist=()):
    """Turn one typed line into a player: a bare number, or a name.

    A bare number picks off the numbered list. That is the whole point of the
    list: entering someone else's pick is the slowest thing you do in a live
    draft and the one place a typo silently corrupts the log. Shared with
    `mock` so `2` means the same thing in practice as it does on the day.

    Returns the player, or None having already printed why not.
    """
    text = text.strip()
    if text.isdigit() and shortlist:
        choice = int(text)
        if not 1 <= choice <= len(shortlist):
            print(f"  ! pick a number from 1 to {len(shortlist)}, or type a name")
            return None
        player = board.players[shortlist[choice - 1]]
    else:
        resolution = board.resolver.resolve(text)
        if not resolution.found:
            print(f"  ! {resolution.note}")
            return None
        if resolution.ambiguous:
            print(f"  ! {resolution.note}")
            print("    say which one - add the position (e.g. 'josh QB') "
                  "or type the full name")
            return None
        player = resolution.best
    if player.player_id in state.drafted:
        print(f"  ! {player.display} was already taken at pick "
              f"{state.drafted.index(player.player_id) + 1}")
        return None
    return player


def _handle(line, state, board, config, log, log_path, audit, suggestions=()):
    """Returns (state, board, done, refresh).

    `refresh` forces the caller to re-issue a recommendation even if the state
    key is unchanged.
    """
    lower = line.lower()
    if lower in ("quit", "exit", "q"):
        return state, board, True, False

    state, board, handled, refresh = _shared_command(
        line, state, board, config, log, log_path, audit
    )
    if handled:
        return state, board, False, refresh

    if lower in ("go", "rec", "."):
        _recommend_now(config, board, state, audit)
        return state, board, False, False

    # Otherwise: a pick.
    mine = False
    text = line
    if lower.startswith("me "):
        mine, text = True, line[3:]

    player = _select(text, board, state, suggestions)
    if player is None:
        return state, board, False, False

    seat = state.on_the_clock
    if mine and seat != config.my_seat:
        print(f"  ! seat {seat} is on the clock, not you (seat {config.my_seat})")
        return state, board, False, False

    pick_no = state.pick_number
    state = state.record(player.player_id)
    log.append(player.player_id, seat=seat)
    print(f"  {pick_no}. seat {seat}: {player.display}")
    return state, board, False, False


def _print_advice(board, advice, show: int = 3) -> list[int]:
    """Print the card and the ranked candidates. Returns their board indices.

    The indices come back so a bare number can select from this list the same
    way it selects from the opponent shortlist - on your own pick, `2` should
    take the engine's second choice.

    Split out from `_recommend_now` so `mock` renders an identical screen from
    the advice it has already computed, rather than a second layout that can
    drift away from this one.
    """
    print(advice.format_card())
    print()
    print(f"  {'#':<3}{'player':<28}{'VOR':>7}{'surv':>7}{'P(title)':>10}{'delta':>8}")
    shown = advice.recommendations[:show]
    for rec in shown:
        surv = f"{rec.survival:.0%}" if rec.survival == rec.survival else "  -"
        print(f"  {rec.rank:<3}{rec.display[:27]:<28}{rec.vor:7.1f}{surv:>7}"
              f"{rec.p_title:10.2%}{rec.delta_p_title * 100:+8.2f}")
    if shown:
        print(f"\n  type 1-{len(shown)} to take one, or type a name")
    print()
    return [board.idx(rec.player_id) for rec in shown]


def _recommend_now(config, board, state, audit, show: int = 3) -> list[int]:
    if state.on_the_clock != config.my_seat:
        print(f"  seat {state.on_the_clock} is on the clock; "
              f"your next pick is {state.my_next_pick()}")
        return []
    advice = _recommend(
        config, board, list(state.drafted), state.my_roster, state.pick_number, audit=audit
    )
    return _print_advice(board, advice, show)


def _show_suggestions(simulator, board, config, state, n) -> list[int]:
    """Numbered shortlist of who the seat on the clock probably takes."""
    drafted_idx = [board.idx(p) for p in state.drafted]
    last_pos = (
        int(board.pos_code[drafted_idx[-1]]) if drafted_idx else None
    )
    rows = simulator.likely_next_picks(
        drafted_idx,
        seat=state.on_the_clock,
        n=n,
        # Seeded off the pick number so the same state always shows the same
        # list - you should not see the options reshuffle when you type `roster`.
        seed=(config.sim.seed + state.pick_number) % (2**32),
        last_pos_code=last_pos,
        pick_number=state.pick_number,
    )
    if not rows:
        return []
    covered = sum(p for _, p in rows)
    print(f"  likely for seat {state.on_the_clock} "
          f"(type the number; {covered:.0%} of the time it is one of these)")
    for k, (idx, prob) in enumerate(rows, start=1):
        player = board.players[idx]
        # A player who won none of 4,000 draws is not impossible, he is just
        # under the resolution of the simulation. Printing 0% next to a name
        # the console is actively offering reads as a bug.
        shown = f"{prob:.0%}" if prob >= 0.005 else "<1%"
        print(f"   {k:>2}  {player.name[:24]:<25}{player.pos:<4}{player.team:<4}"
              f"{shown:>6}")
    return [idx for idx, _ in rows]


def _show_log(board, state, config, count: int) -> None:
    """Recent picks with their numbers, so `fix` has something to aim at."""
    picks = list(enumerate(state.drafted, start=1))[-max(count, 1):]
    if not picks:
        print("  no picks yet")
        return
    for n, pid in picks:
        seat = pick_owner(n, config.teams, config.draft_type)
        mine = " <- you" if seat == config.my_seat else ""
        print(f"  {n:>4}. seat {seat:<3} {board.player(pid).display}{mine}")


def _fix_pick(line, state, board, log, log_path):
    """`fix <pick number> <name>` - correct one recorded pick in place.

    The failure this exists for: you type `josh`, it resolves to the wrong
    Josh, and you notice four picks later. Undoing back to it would throw away
    four correct picks under a clock.
    """
    parts = line.split(None, 2)
    if len(parts) < 3:
        print("  ! usage: fix <pick number> <name>   (try `log` to find the number)")
        return state
    try:
        number = int(parts[1])
    except ValueError:
        print(f"  ! {parts[1]!r} is not a pick number. Try `log` to find it.")
        return state

    resolution = board.resolver.resolve(parts[2])
    if not resolution.found:
        print(f"  ! {resolution.note}")
        return state
    if resolution.ambiguous:
        print(f"  ! {resolution.note}")
        return state

    was = state.drafted[number - 1] if 1 <= number <= len(state.drafted) else None
    try:
        state = state.replace(number, resolution.best.player_id)
    except DraftStateError as exc:
        print(f"  ! {exc}")
        return state

    log.picks[number - 1].player_id = resolution.best.player_id
    log.picks[number - 1].note = f"corrected from {was}"
    log.save(log_path)
    print(f"  pick {number}: {board.player(was).display} -> "
          f"{resolution.best.display}")
    return state


def _renumber(log, state, config) -> None:
    """Rewrite pick numbers and seats after an insert or a drop.

    The seat that owns a pick is derived from its number, so shifting the log
    re-attributes every pick after the shift. Recomputing the whole tail is
    cheaper to reason about than patching the entries individually, and it
    cannot drift out of step with `pick_owner`.
    """
    for n, pid in enumerate(state.drafted, start=1):
        if n - 1 < len(log.picks):
            entry = log.picks[n - 1]
            entry.pick = n
            entry.player_id = pid
            entry.seat = pick_owner(n, config.teams, config.draft_type)


def _insert_pick(line, state, board, config, log, log_path):
    """`insert <pick number> <name>` - a pick that was missed at the time.

    Everything from that number on shifts down one, which is the whole point:
    a missed pick leaves the entire tail of the log attributed to the wrong
    seats, and no amount of `fix` repairs that.
    """
    parts = line.split(None, 2)
    if len(parts) < 3:
        print("  ! usage: insert <pick number> <name>   (try `log` to find the number)")
        return state
    try:
        number = int(parts[1])
    except ValueError:
        print(f"  ! {parts[1]!r} is not a pick number. Try `log` to find it.")
        return state

    resolution = board.resolver.resolve(parts[2])
    if not resolution.found:
        print(f"  ! {resolution.note}")
        return state
    if resolution.ambiguous:
        print(f"  ! {resolution.note}")
        print("    say which one - add the position (e.g. 'josh QB') "
              "or type the full name")
        return state

    try:
        state = state.insert(number, resolution.best.player_id)
    except DraftStateError as exc:
        print(f"  ! {exc}")
        return state

    log.append(state.drafted[-1], seat=0)
    _renumber(log, state, config)
    log.picks[number - 1].note = "inserted; later picks shifted down"
    log.save(log_path)
    seat = pick_owner(number, config.teams, config.draft_type)
    print(f"  inserted at pick {number} (seat {seat}): {resolution.best.display}")
    print(f"  {len(state.drafted) - number} later pick(s) shifted down one; "
          f"pick {state.pick_number} is now on the clock")
    return state


def _drop_pick(line, state, board, config, log, log_path):
    """`drop <pick number>` - a pick that never happened. The mirror of insert."""
    parts = line.split()
    if len(parts) < 2:
        print("  ! usage: drop <pick number>   (try `log` to find the number)")
        return state
    try:
        number = int(parts[1])
    except ValueError:
        print(f"  ! {parts[1]!r} is not a pick number. Try `log` to find it.")
        return state

    was = state.drafted[number - 1] if 1 <= number <= len(state.drafted) else None
    try:
        state = state.remove(number)
    except DraftStateError as exc:
        print(f"  ! {exc}")
        return state

    log.picks.pop()
    _renumber(log, state, config)
    log.save(log_path)
    print(f"  removed pick {number}: {board.player(was).display}")
    print(f"  later pick(s) shifted up one; pick {state.pick_number} is now "
          f"on the clock")
    return state


def _show_roster(board, state, seat, config) -> None:
    from .lineup import lineup_slots

    roster = state.roster_of(seat)
    if not roster:
        print(f"  seat {seat}: empty")
        return
    scores = {p: float(board.points[board.idx(p)]) for p in roster}
    positions = {p: board.player(p).pos for p in roster}
    starters = lineup_slots(scores, positions, config.roster)
    started = {p for p in starters.values() if p}
    print(f"  seat {seat} ({len(roster)} players)")
    # Display in the order the league lists its starters, not the order the
    # optimiser fills them - under a pick clock you read down the lineup card.
    for slot in _display_order(config, starters):
        pid = starters[slot]
        name = board.player(pid).display if pid else "-"
        print(f"    {slot:<8}{name}")
    bench = [p for p in roster if p not in started]
    if bench:
        print(f"    bench   {', '.join(board.player(p).display for p in bench)}")


def _display_order(config, starters: dict) -> list[str]:
    order, counts = [], {}
    for slot in config.roster.starters:
        counts[slot] = counts.get(slot, 0) + 1
        label = slot if config.roster.starters.count(slot) == 1 else f"{slot}{counts[slot]}"
        if label in starters:
            order.append(label)
    return order + [s for s in starters if s not in order]


def _show_available(board, state, config, pos=None) -> None:
    available = np.ones(len(board), dtype=bool)
    for pid in state.drafted:
        available[board.idx(pid)] = False
    vor = vor_array(board, available, config.vor_baseline)
    idx = np.flatnonzero(available & (board.pos_mask(pos) if pos else True))
    idx = idx[np.argsort(-vor[idx], kind="stable")][:12]
    for rank, i in enumerate(idx, start=1):
        print(f"  {rank:<3}{board.players[i].display[:34]:<36}"
              f"VOR {vor[i]:7.1f}   ADP {board.adp[i]:5.0f}")


def _apply_update(line, board, state, config, audit) -> Board:
    """Brief job #3: new information enters through an explicit, logged tool."""
    body, _, reason = line.partition(":")
    parts = body.split()
    kind = parts[0].lower()
    multiplier = 1.0
    if kind == "bump":
        try:
            multiplier = float(parts[-1])
            name = " ".join(parts[1:-1])
        except ValueError:
            print("  ! usage: bump <name> <multiplier> : <reason>")
            return board
    else:
        name = " ".join(parts[1:])

    resolution = board.resolver.resolve(name)
    if not resolution.found or resolution.ambiguous:
        print(f"  ! {resolution.note or 'no match'}")
        return board
    reason = reason.strip()
    if not reason:
        print("  ! a reason is required - this goes in the audit log")
        return board

    update = ProjectionUpdate(
        player_id=resolution.best.player_id,
        out_for_season=(kind == "out"),
        points_multiplier=multiplier,
        reason=reason,
        source="console",
    )
    board = board.apply_update(update)
    audit.record(
        state_hash=board.fingerprint(), league_id=config.league_id,
        pick_number=state.pick_number, kind="projection_update",
        payload={"update": update.describe()},
    )
    print(f"  applied: {update.describe()}")
    return board


def cmd_llm(args) -> int:
    from .llm.orchestrator import run_console
    from .llm.session import DraftSession

    config, board = _load(args)
    if config.my_seat is None:
        raise SystemExit("set draft.my_seat in the config or pass --seat")
    session = DraftSession(config, board, log_path=args.out)
    return run_console(session, model=args.model)



# Deliberately the live console's list plus `auto`. Practice is worthless if
# the command you reach for on draft day was not in the thing you practised on.
MOCK_HELP = """
  <enter>           take the engine's #1 pick
  <number>          take that numbered candidate instead
  <name>            take someone else entirely
  go                reprint the recommendation
  undo              take back the last pick
  log [n]           show the last n picks with their numbers
  fix <n> <name>    correct pick n (use when a name resolved to the wrong guy)
  insert <n> <name> add a pick you missed at n; everything after shifts down
  drop <n>          remove a pick that never happened; the rest shift up
  roster [seat]     show a roster - yours, or any opponent's
  board [pos]       show the best available
  out <name> : <r>  rule a player out for the season, with a reason
  bump <name> <x> : <r>   multiply a player's projection by x
  save              write the pick log
  auto              let the engine finish the draft for you
  help / quit
"""


def cmd_mock(args) -> int:
    """Practice draft. The opponent model fills every other seat.

    The point is not the resulting roster - it is getting the console into your
    fingers before a clock is running. Everything behaves exactly as it does in
    a real draft except that you do not have to type the other 143 picks.
    """
    import numpy as np

    from .opponents import DraftSimulator

    if args.sims is None:
        # Practice should move quickly: 13 recommendations at the real-draft
        # setting is several minutes of waiting to rehearse typing.
        args.sims = 800
    config, board = _load(args)
    if config.my_seat is None:
        raise SystemExit("set draft.my_seat in the config or pass --seat")

    seat = config.my_seat
    rng = np.random.default_rng(args.seed)
    simulator = DraftSimulator(board, config, seat)
    rankings = simulator.base_rankings(rng, 1)

    audit = AuditLog(args.audit) if args.audit else AuditLog()
    log = DraftLog(league_id=config.league_id, my_seat=seat)
    # One file per run. A mock is something you do repeatedly to practise, so
    # a fixed name meant every rehearsal overwrote the last one and there was
    # nothing to compare against.
    log_path = (
        Path(args.out) if args.out
        else _unique(Path(f"logs/mock_{config.league_id}_{_run_stamp()}.jsonl"))
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = DraftState(config=config, drafted=[], my_seat=seat)
    auto = args.auto

    print(f"MOCK - {config.name}, seat {seat} of {config.teams}, "
          f"{config.rounds} rounds, {config.sim.n_sims} sims/pick")
    print("Opponents are simulated. Nothing here touches your real draft log.")
    if not auto:
        print(MOCK_HELP)

    while not state.is_complete:
        drafted_idx = [board.idx(p) for p in state.drafted]
        if state.on_the_clock != seat:
            # Value used by the model's stand-in for future picks by any seat.
            value = vor_array(board, _available(board, state), config.vor_baseline)
            idx = simulator.next_selection(drafted_idx, value, rankings)
            pid = board.players[idx].player_id
            picking_seat = state.on_the_clock
            print(f"  {state.pick_number:>4}. seat {picking_seat}: "
                  f"{board.player(pid).display}")
            state = state.record(pid)
            log.append(pid, seat=picking_seat)
            continue

        advice = _recommend(
            config, board, list(state.drafted), state.my_roster,
            state.pick_number, audit=audit,
        )
        print()
        print(f"  --- YOUR PICK (round {state.current_round}, overall "
              f"{state.pick_number}) ---")
        shortlist = _print_advice(board, advice, show=args.show)

        choice = None
        # `undo`, `fix` and a projection update all move the state out from
        # under the advice on screen. Drop back to the outer loop and re-run
        # the engine rather than letting a stale shortlist record a pick.
        replan = False
        while choice is None and not replan:
            if auto:
                choice = advice.recommendations[0].player_id
                break
            try:
                line = input("  pick> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                line = "quit"
            low = line.lower()
            if low in ("quit", "exit", "q"):
                _finish(board, state, config, log, log_path, stopped=True)
                return 0
            if low == "auto":
                auto = True
                choice = advice.recommendations[0].player_id
                break
            if not line:
                choice = advice.recommendations[0].player_id
                break
            if low in ("go", "rec", "."):
                # Reprint what we already have. Re-running the engine here
                # would cost another full simulation to redraw the same rows.
                shortlist = _print_advice(board, advice, show=args.show)
                continue
            try:
                state, board, handled, refresh = _shared_command(
                    line, state, board, config, log, log_path, audit,
                    help_text=MOCK_HELP,
                )
                if handled:
                    replan = refresh
                    continue
                player = _select(line, board, state, shortlist)
            except (DraftStateError, ConfigError, KeyError, ValueError) as exc:
                print(f"  ! {exc}")
                continue
            if player is not None:
                choice = player.player_id

        if replan:
            continue

        print(f"  -> you take {board.player(choice).display}\n")
        state = state.record(choice)
        log.append(choice, seat=seat)

    _finish(board, state, config, log, log_path, stopped=False)
    return 0


def _available(board, state):
    import numpy as np

    mask = np.ones(len(board), dtype=bool)
    for pid in state.drafted:
        mask[board.idx(pid)] = False
    return mask


def _finish(board, state, config, log, log_path, stopped: bool) -> None:
    from .lineup import best_lineup_points

    log.save(log_path)
    roster = state.roster_of(config.my_seat)
    print("\n" + "=" * 58)
    print("MOCK COMPLETE" if not stopped else "MOCK STOPPED EARLY")
    print("=" * 58)
    if not roster:
        print("  no picks made")
        return
    _show_roster(board, state, config.my_seat, config)
    scores = {p: float(board.points[board.idx(p)]) for p in roster}
    positions = {p: board.player(p).pos for p in roster}
    total = best_lineup_points(scores, positions, config.roster)
    print(f"\n  projected starting-lineup points: {total:.0f}")
    print(f"  pick log: {log_path}")
    print("\n  This is practice. The projections are a point estimate and the")
    print("  season is mostly noise - do not read the total as a prediction.")



def cmd_lineup(args) -> int:
    """Set this week's lineup from fresh, multi-source weekly data."""
    from .replay import DraftLog
    from .weekly import (
        StaleDataError, best_lineup, check_freshness, format_plan,
        format_sources, load_weekly, roster_for,
    )

    config, board = _load(args)
    log = DraftLog.load(args.log)
    seat = args.seat or config.my_seat or log.my_seat
    if seat is None:
        raise SystemExit(
            "cannot tell which seat is yours: pass --seat, or set draft.my_seat"
        )

    try:
        data = load_weekly(
            args.weekly, week=args.week, rules=config.scoring,
            max_age_hours=args.max_age_hours,
            min_sources=args.min_sources,
        )
    except StaleDataError as exc:
        raise SystemExit(f"stale data: {exc}") from exc

    roster = roster_for(config, log.player_ids, seat)
    if args.sources:
        print(format_sources(data))
        print()
    try:
        notes = check_freshness(
            data, args.week or data.week, roster,
            max_age_hours=args.max_age_hours,
            active_max_age_hours=args.active_max_age_hours,
            allow_stale=args.allow_stale,
        )
    except StaleDataError as exc:
        raise SystemExit(f"stale data: {exc}") from exc

    if args.usage:
        from .usage import (
            apply_usage_adjustment, check_usage, format_trends, load_usage,
            notable_trends,
        )

        usage = load_usage(args.usage, max_age_hours=args.usage_max_age_hours)
        notes.extend(check_usage(usage, args.week or data.week))
        trends = notable_trends(usage, roster, args.week or data.week)
        print(format_trends(trends, board, args.week or data.week))
        print()
        if args.usage_adjust:
            audit = AuditLog(args.audit) if args.audit else AuditLog()
            notes.extend(apply_usage_adjustment(
                data, usage, roster, args.week or data.week,
                damping=args.usage_damping, cap=args.usage_cap,
                league_id=config.league_id, audit=audit,
            ))

    opponent_roster = None
    if args.opponent:
        if not 1 <= args.opponent <= config.teams:
            raise SystemExit(
                f"seat {args.opponent} does not exist; this league has "
                f"{config.teams} seats"
            )
        opponent_roster = roster_for(config, log.player_ids, args.opponent)

    plan = best_lineup(
        config, board, data, roster,
        opponent_roster=opponent_roster,
        opponent_seat=args.opponent,
        n_sims=args.sims or 20000,
        notes=notes,
    )
    print(format_plan(plan, board, data))
    return 0


def cmd_trades(args) -> int:
    """Post-draft trade search. The draft is over; this is the next lever."""
    from .replay import DraftLog
    from .trades import find_trades, format_report, format_surplus

    config, board = _load(args)
    log = DraftLog.load(args.log)
    seat = args.seat or config.my_seat or log.my_seat
    if seat is None:
        raise SystemExit(
            "cannot tell which seat is yours: pass --seat, or set draft.my_seat"
        )
    audit = AuditLog(args.audit) if args.audit else AuditLog()

    partners = [args.partner] if args.partner else None
    report = find_trades(
        config, board, log.player_ids, seat,
        partners=partners,
        max_give=args.max_give,
        max_get=args.max_get,
        min_their_gain=args.min_their_gain,
        top=args.top,
        audit=audit,
    )
    if args.surplus:
        print(format_surplus(report, config))
        print()
    print(format_report(report, board, config))
    return 0


# The sources R/pull_projections.R asks for - keep the two lists in step. A
# source that errors during a scrape is skipped with a warning and simply never
# appears in the CSV, so the only way to notice is to compare what arrived
# against what was requested.
#
# FantasyData, FleaFlicker, NumberFire and NFL are deliberately absent: none
# returns season-long data (paywall, no draft data, and a parser broken by a
# page-structure change respectively). See the note in R/pull_projections.R.
EXPECTED_SOURCES = ("CBS", "ESPN", "FantasyPros", "FFToday", "RTSports")


def cmd_sources(args) -> int:
    """Per-source, per-position coverage of the projection file.

    Row counts alone hide the failure that matters: a source that returns
    quarterbacks and silently no defenses still looks present. Since the engine
    equal-weights whatever it finds, a source covering half the positions
    quietly changes the weighting at the other half.
    """
    from collections import defaultdict

    from .projections import load_stat_lines

    config = load_by_id(args.league) if args.league else None
    proj, _ = default_paths(args.season)
    if args.projections:
        proj = Path(args.projections)
    if not Path(proj).exists():
        raise SystemExit(f"no projections at {proj}")

    rows = load_stat_lines(proj)
    per_source: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        per_source[row.source][row.pos].add(row.player_id)
        totals[row.source] += 1

    found = sorted(per_source)
    print(f"{proj}  -  {len(rows)} rows, {len(found)} sources\n")
    header = f"  {'source':<16}{'rows':>7}" + "".join(f"{p:>6}" for p in POSITIONS)
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Median coverage per position, to spot a source that is thin rather than absent.
    median: dict[str, float] = {}
    for pos in POSITIONS:
        counts = sorted(len(per_source[src].get(pos, ())) for src in found)
        median[pos] = counts[len(counts) // 2] if counts else 0

    # A position no source covers is a missing position, not nine thin sources.
    absent_positions = [p for p in POSITIONS if median[p] == 0]

    thin: list[tuple[float, str]] = []
    for src in found:
        cells = ""
        for pos in POSITIONS:
            n = len(per_source[src].get(pos, ()))
            is_thin = median[pos] > 0 and n < median[pos] * 0.5
            cells += f"{str(n) + ('!' if is_thin else ''):>6}"
            if is_thin:
                thin.append((n / median[pos], f"{src}/{pos}"))
        print(f"  {src:<16}{totals[src]:>7}{cells}")

    missing = [s for s in EXPECTED_SOURCES if s not in found]
    extra = [s for s in found if s not in EXPECTED_SOURCES]

    print()
    if missing:
        print(f"  MISSING ({len(missing)}): {', '.join(missing)}")
        print("           These were requested but returned nothing. Usually the")
        print("           source changed its page layout or was unreachable.")
        print("           Re-run the pull; if one keeps failing, drop it from the")
        print("           `sources` vector in R/pull_projections.R and re-run.")
    if extra:
        print(f"  UNEXPECTED: {', '.join(extra)}")
    if absent_positions:
        print(f"  NO DATA AT ALL for: {', '.join(absent_positions)}")
        print("           No source returned these positions. The engine cannot")
        print("           compute replacement level for them.")
    if thin:
        # Worst first - a source at 5% of its peers matters more than one at 45%.
        worst = [name for _, name in sorted(thin)][:8]
        print(f"  THIN (marked !): {', '.join(worst)}")
        print("           Present but covering far fewer players than its peers.")
        print("           The engine equal-weights whatever it finds, so a source")
        print("           that covers half a position shifts the average there.")
    if not missing and not thin and not absent_positions:
        print(f"  all {len(found)} sources present with comparable coverage")

    if config is not None:
        print()
        for pos in POSITIONS:
            players = {pid for src in found for pid in per_source[src].get(pos, ())}
            need = config.vor_baseline[pos]
            mark = "ok " if len(players) >= need else "LOW"
            print(f"  {mark} {pos:<4}{len(players):>4} distinct players "
                  f"(replacement rank {need})")

        _report_scale(rows, config, found)
    return 0


def _report_scale(rows, config, found: list[str]) -> None:
    """Per-source scale, measured on players the sources actually share.

    The naive version of this - median points per source - measures coverage
    depth, not scale. A source carrying only the top ten at each position has a
    median roughly twice that of one carrying a hundred, and a source going two
    hundred deep has a lower one, neither of which says anything about the
    numbers being on the same basis. That version flagged a perfectly sound
    source as off-scale.

    So compare each source against the others *on the same players*: for every
    player at least two sources cover, take that source's points over the mean
    across the sources covering him, and report the median of those ratios.
    Coverage cancels out; a genuine per-game-versus-per-season mismatch does
    not.
    """
    import statistics
    from collections import defaultdict

    from .scoring import score_stats

    by_player: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.pos in ("QB", "RB", "WR", "TE"):
            by_player[row.player_id][row.source] = score_stats(row.stats, config.scoring)

    ratios: dict[str, list[float]] = defaultdict(list)
    shared = 0
    for scores in by_player.values():
        if len(scores) < 2:
            continue
        mean = statistics.fmean(scores.values())
        if mean <= 0:
            continue
        shared += 1
        for src, value in scores.items():
            ratios[src].append(value / mean)

    usable = {
        src: statistics.median(values)
        for src, values in ratios.items()
        if len(values) >= 10
    }
    if len(usable) < 2:
        print("\n  scale check: not enough shared players to compare sources")
        return

    print(f"\n  scale check (points vs the other sources, on {shared} shared players):")
    suspect = []
    for src in sorted(usable, key=lambda s: -usable[s]):
        ratio = usable[src]
        n = len(ratios[src])
        flag = "" if 0.75 <= ratio <= 1.33 else "   <-- OFF SCALE"
        if flag:
            suspect.append(src)
        print(f"    {src:<16}{ratio:>6.2f}x   on {n:>4} shared players{flag}")
    for src in found:
        if src not in usable:
            print(f"    {src:<16}     -    too few shared players to compare")

    if suspect:
        print(f"  {', '.join(suspect)} is on a different scale from its peers.")
        print("  Likely per-game instead of per-season, or a different scoring")
        print("  basis. Equal weighting drags every player it covers toward that")
        print("  scale - drop it from R/pull_projections.R and re-run.")
    else:
        print("  all sources agree on scale within +/-33%")



# Human labels for the stat keys, so the output can be read straight across
# from a league settings page instead of decoded from multipliers.
_STAT_LABELS = {
    "pass_yds": "passing yards", "pass_tds": "passing TD", "pass_int": "interception",
    "pass_comp": "completion", "pass_att": "pass attempt",
    "rush_yds": "rushing yards", "rush_tds": "rushing TD", "rush_att": "rush attempt",
    "rec": "reception", "rec_yds": "receiving yards", "rec_tds": "receiving TD",
    "rec_tgt": "target",
    "fumbles_lost": "fumble lost", "two_pts": "2-pt conversion",
    "return_tds": "kick/punt return TD", "return_yds": "return yards",
    "off_fumble_return_td": "off. fumble return TD",
    "xp": "extra point", "xp_miss": "missed extra point",
    "fg_0019": "FG 0-19", "fg_2029": "FG 20-29", "fg_3039": "FG 30-39",
    "fg_4049": "FG 40-49", "fg_50": "FG 50+", "fg_60": "FG 60+",
    "fg_miss": "missed FG",
    "dst_int": "interception", "dst_fum_rec": "fumble recovery",
    "dst_sacks": "sack", "dst_safety": "safety", "dst_td": "defensive TD",
    "dst_blk": "blocked kick", "dst_forced_fumble": "forced fumble",
    "dst_fourth_down_stop": "4th down stop", "dst_xp_return": "extra point returned",
}

def _compromise(key: str, value: float) -> str | None:
    """Where the config knowingly cannot say what the league actually does.

    Each of these is a place the engine is a little wrong on purpose, and
    knowing which way beats discovering it mid-draft. Conditional on the value,
    not just the key: League 2 penalises every missed field goal at -1, which
    the single field expresses exactly, so it is not a compromise there.
    """
    if key == "fg_miss" and value == 0:
        return (
            "set to 0 because a single value cannot express 'only penalise "
            "misses under 20 yards'. Kickers are slightly over-valued."
        )
    # Rules that are real but that no projection source publishes. Encoded so
    # the config mirrors the settings page, and flagged so it is obvious the
    # engine is blind to them rather than that someone forgot.
    unprojected = {
        "dst_forced_fumble", "dst_fourth_down_stop", "dst_xp_return",
        "off_fumble_return_td",
    }
    if key in unprojected and value:
        return "no projection source provides this stat, so it scores as 0"
    return None


def _fmt_rate(key: str, value: float) -> str:
    """Show yardage rules the way a settings page does: 1 point per N yards."""
    if key.endswith("_yds") and 0 < value < 1:
        return f"1 pt per {round(1 / value)} yds"
    return f"{value:+g} pt" + ("" if abs(value) == 1 else "s")


def cmd_rules(args) -> int:
    """Print a league's settings in a form you can read across from Yahoo/Sleeper.

    The config is the engine's entire model of your league. A wrong `rec` value
    misprices every receiver on the board and nothing downstream would look
    odd - so this exists to be checked against the real settings page once,
    before it matters.
    """
    config = load_by_id(args.league)
    other = load_by_id(args.diff) if args.diff else None

    print(f"{config.name}   [{config.league_id}]")
    if config.source_path:
        print(f"  config: {config.source_path}")
    print()
    print(f"  {config.teams} teams, {config.draft_type} draft, "
          f"{config.rounds} rounds, {config.total_drafted} picks")
    print(f"  starters : {' '.join(config.roster.starters)}")
    print(f"  bench    : {config.roster.bench}    IR: {config.roster.ir}")
    print(f"  playoffs : {config.playoff_teams} of {config.teams}, "
          f"weeks {list(config.playoff_weeks)} "
          f"(regular season through week {config.regular_season_weeks})")
    print(f"  my seat  : {config.my_seat if config.my_seat else 'not set'}")

    groups = [
        ("PASSING", ["pass_yds", "pass_tds", "pass_int", "pass_comp", "pass_att"]),
        ("RUSHING", ["rush_yds", "rush_tds", "rush_att"]),
        ("RECEIVING", ["rec", "rec_yds", "rec_tds", "rec_tgt"]),
        ("MISC", ["fumbles_lost", "two_pts", "return_tds", "return_yds",
                   "off_fumble_return_td"]),
        ("KICKING", ["xp", "xp_miss", "fg_0019", "fg_2029", "fg_3039",
                      "fg_4049", "fg_50", "fg_60", "fg_miss"]),
        ("TEAM DEFENSE", ["dst_int", "dst_fum_rec", "dst_sacks", "dst_safety",
                           "dst_td", "dst_blk", "dst_forced_fumble",
                           "dst_fourth_down_stop", "dst_xp_return"]),
    ]
    values = config.scoring.multipliers
    other_values = other.scoring.multipliers if other else {}

    for title, keys in groups:
        present = [k for k in keys if k in values or k in other_values]
        if not present:
            continue
        print(f"\n  {title}")
        for key in present:
            label = _STAT_LABELS.get(key, key)
            mine = values.get(key)
            shown = _fmt_rate(key, mine) if mine is not None else "not scored"
            line = f"    {label:<22}{shown:>18}"
            if other is not None:
                theirs = other_values.get(key)
                if theirs != mine:
                    theirs_shown = (
                        _fmt_rate(key, theirs) if theirs is not None else "not scored"
                    )
                    line += f"   |  {other.league_id}: {theirs_shown}"
                    line += "   <-- DIFFERS"
            print(line)
            if mine is not None:
                note = _compromise(key, mine)
                if note:
                    print(f"      note: {note}")

    if config.scoring.pts_bracket:
        print("\n  POINTS ALLOWED (team defense)")
        previous = None
        for threshold, points in config.scoring.pts_bracket:
            span = f"{int(previous) + 1}-{int(threshold)}" if previous is not None else f"{int(threshold)}"
            print(f"    {span:<22}{points:>+18g} pts")
            previous = threshold
        if other is not None and other.scoring.pts_bracket != config.scoring.pts_bracket:
            print(f"    (differs from {other.league_id})")

    print("\n  REPLACEMENT LEVEL (the Nth player at each position is free)")
    line = "    "
    for pos in POSITIONS:
        mine = config.vor_baseline[pos]
        cell = f"{pos} {mine}"
        if other is not None and other.vor_baseline[pos] != mine:
            cell += f" (vs {other.vor_baseline[pos]})"
        line += f"{cell:<16}"
    print(line)
    print("    run `ffdraft baselines` for how these are derived")

    if other is not None:
        # Compare only rules a projection source actually provides. A league
        # can score offensive fumble return TDs all it likes; nothing projects
        # them, they are worth 0 to every player, and letting them read as a
        # scoring difference would obscure the claim that matters - that one
        # projection set serves both leagues.
        unprojected = {"off_fumble_return_td"}
        mine = {k: v for k, v in config.scoring.offense.items() if k not in unprojected}
        theirs = {k: v for k, v in other.scoring.offense.items() if k not in unprojected}
        same = mine == theirs
        print(f"\n  vs {other.name}: skill-position scoring is "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
        if same:
            print("    one projection set serves both leagues")
            ignored = sorted(
                (set(config.scoring.offense) | set(other.scoring.offense)) & unprojected
            )
            if ignored:
                print(f"    (ignoring {', '.join(ignored)} - no source projects it)")

    if config.notes:
        print("\n  NOTES")
        for note in config.notes:
            wrapped = " ".join(note.split())
            print(f"    - {wrapped}")
    return 0


# --- wiring ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ffdraft", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, seat=True):
        p.add_argument("--league", required=True)
        p.add_argument("--projections")
        p.add_argument("--market")
        p.add_argument("--season", type=int, default=2026)
        p.add_argument("--pool", type=int, default=DEFAULT_POOL)
        p.add_argument("--sims", type=int)
        if seat:
            p.add_argument("--seat", type=int)

    p = sub.add_parser("rules", help="print a league's settings, readably")
    p.add_argument("--league", required=True)
    p.add_argument("--diff", help="another league_id to compare against")
    p.set_defaults(func=cmd_rules)

    p = sub.add_parser("baselines", help="explain replacement level")
    p.add_argument("--league", required=True)
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("export-r", help="emit the ffanalytics scoring rules")
    p.add_argument("--league", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_export_r)

    p = sub.add_parser("board", help="show the board by VOR")
    common(p)
    p.add_argument("--pos")
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(func=cmd_board)

    p = sub.add_parser("sources", help="per-source coverage of the projection file")
    p.add_argument("--league")
    p.add_argument("--projections")
    p.add_argument("--season", type=int, default=2026)
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("check", help="pre-draft preflight")
    common(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("draft", help="live draft console")
    common(p)
    p.add_argument("--out", help="where to write the pick log")
    p.add_argument("--audit", help="where to write the audit log")
    p.add_argument("--suggest", type=int, default=10,
                   help="how many likely picks to list before each opponent pick")
    p.add_argument("--show", type=int, default=3,
                   help="how many ranked candidates to offer on your own pick")
    p.add_argument("--no-suggest", action="store_true",
                   help="turn the numbered list off")
    p.set_defaults(func=cmd_draft)

    p = sub.add_parser("llm", help="draft console driven by Claude (needs the llm extra)")
    common(p)
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--out", help="where to write the pick log")
    p.set_defaults(func=cmd_llm)

    p = sub.add_parser("mock", help="practice draft against simulated opponents")
    common(p)
    p.add_argument("--seed", type=int, default=None,
                   help="opponent-behaviour seed; omit for a different mock each run")
    p.add_argument("--auto", action="store_true",
                   help="let the engine pick for you too, and just show the roster")
    p.add_argument("--show", type=int, default=3,
                   help="how many ranked candidates to offer on your own pick")
    p.add_argument("--out", help="where to write the mock pick log")
    p.add_argument("--audit", help="where to write the audit log")
    p.set_defaults(func=cmd_mock)

    p = sub.add_parser("lineup", help="set this week's lineup from fresh weekly data")
    common(p)
    p.add_argument("--log", required=True, help="the pick log from the draft")
    p.add_argument("--weekly", required=True,
                   help="this week's projections (multi-source CSV)")
    p.add_argument("--week", type=int, help="the week you are setting")
    p.add_argument("--opponent", type=int,
                   help="the seat you play this week; switches the objective "
                        "from expected points to P(win)")
    p.add_argument("--max-age-hours", type=float, default=24.0,
                   help="a source older than this is dropped")
    p.add_argument("--active-max-age-hours", type=float, default=3.0,
                   help="tighter limit when a player you could start is "
                        "questionable or doubtful")
    p.add_argument("--min-sources", type=int, default=1,
                   help="refuse if fewer than this many sources survive")
    p.add_argument("--sources", action="store_true",
                   help="print per-source coverage and age first")
    p.add_argument("--allow-stale", action="store_true",
                   help="override the freshness refusal, loudly")
    p.add_argument("--usage",
                   help="observed usage from scripts/fetch_nflverse.py; reports "
                        "role changes the consensus may not have priced")
    p.add_argument("--usage-adjust", action="store_true",
                   help="also nudge projections toward observed usage "
                        "(damped, capped, and written to the audit log)")
    p.add_argument("--usage-damping", type=float, default=0.5,
                   help="how much of the usage trend to apply; the consensus "
                        "has already priced some of it")
    p.add_argument("--usage-cap", type=float, default=0.25,
                   help="largest multiplier the adjustment may apply")
    p.add_argument("--usage-max-age-hours", type=float, default=72.0,
                   help="nflverse publishes overnight, so this is looser than "
                        "the projection bar")
    p.add_argument("--audit", help="where to write the audit log")
    p.set_defaults(func=cmd_lineup)

    p = sub.add_parser("trades", help="post-draft trade ideas, ranked by your title odds")
    common(p)
    p.add_argument("--log", required=True, help="the pick log from the draft")
    p.add_argument("--partner", type=int, help="only look at trades with this seat")
    p.add_argument("--top", type=int, default=5, help="how many ideas to print")
    p.add_argument("--max-give", type=int, default=2,
                   help="most players you would send in one trade")
    p.add_argument("--max-get", type=int, default=2,
                   help="most players you would receive in one trade")
    p.add_argument("--min-their-gain", type=float, default=1.0,
                   help="points the other roster must gain, or it is not an offer")
    p.add_argument("--surplus", action="store_true",
                   help="also print who is long and short at each position")
    p.add_argument("--audit", help="where to write the audit log")
    p.set_defaults(func=cmd_trades)

    p = sub.add_parser("replay", help="replay a draft log through the engine")
    common(p)
    p.add_argument("--log", required=True)
    p.add_argument("--check", action="store_true", help="assert determinism instead")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("backtest", help="let the engine draft a seat and score it")
    common(p)
    p.add_argument("--log", required=True)
    p.add_argument("--actuals", required=True)
    p.set_defaults(func=cmd_backtest)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, DraftStateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
