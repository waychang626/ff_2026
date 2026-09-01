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


# --- the live console --------------------------------------------------------
HELP = """
  <number>          record the pick from the numbered list (fastest)
  <name>            record a pick by anyone   (e.g. "bijan gone", "lions d")
  me <name>         record a pick as yours
  go                get a recommendation for the pick on the clock
  undo              take back the last pick
  log [n]           show the last n picks with their numbers
  fix <n> <name>    correct pick n (use when a name resolved to the wrong guy)
  roster [seat]     show a roster
  board [pos]       show the best available
  out <name> : <r>  rule a player out for the season, with a reason
  bump <name> <x> : <r>   multiply a player's projection by x
  save [path]       write the pick log
  help / quit
"""


def cmd_draft(args) -> int:
    config, board = _load(args)
    if config.my_seat is None:
        raise SystemExit("set draft.my_seat in the config or pass --seat")

    log = DraftLog(league_id=config.league_id, my_seat=config.my_seat)
    audit = AuditLog(args.audit) if args.audit else AuditLog()
    log_path = Path(args.out or f"logs/draft_{config.league_id}.jsonl")
    state = DraftState(config=config, drafted=[], my_seat=config.my_seat)

    from .opponents import DraftSimulator

    simulator = DraftSimulator(board, config, config.my_seat)
    suggest_n = 0 if args.no_suggest else args.suggest
    suggestions: list[int] = []
    shown_for: int | None = None

    print(f"{config.name} - seat {config.my_seat} of {config.teams}, "
          f"{config.rounds} rounds, {config.sim.n_sims} sims/pick")
    imputed = int(board.adp_imputed.sum())
    if imputed > len(board) * 0.5:
        print(f"  WARNING: {imputed}/{len(board)} players have no market ADP; "
              f"draft order is VOR rank.\n"
              f"           Survival % and the numbered list are approximate. "
              f"Run `ffdraft check` for the fix.")
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
            shown_for = state.pick_number

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
            state, board, done = _handle(
                line, state, board, config, log, log_path, audit, suggestions
            )
            if done:
                break
        except (DraftStateError, ConfigError, KeyError, ValueError) as exc:
            print(f"  ! {exc}")

    log.save(log_path)
    print(f"\nsaved {len(log.picks)} picks to {log_path}")
    return 0


def _handle(line, state, board, config, log, log_path, audit, suggestions=()):
    lower = line.lower()
    if lower in ("quit", "exit", "q"):
        return state, board, True
    if lower in ("help", "?", "h"):
        print(HELP)
        return state, board, False
    if lower == "save":
        log.save(log_path)
        print(f"  saved {len(log.picks)} picks to {log_path}")
        return state, board, False
    if lower == "undo":
        if not log.picks:
            print("  nothing to undo")
            return state, board, False
        removed = log.picks.pop()
        state = state.undo()
        print(f"  undid pick {removed.pick}: {removed.player_id}")
        return state, board, False
    if lower.startswith("log"):
        parts = line.split()
        _show_log(board, state, config, int(parts[1]) if len(parts) > 1 else 12)
        return state, board, False
    if lower.startswith("fix "):
        state = _fix_pick(line, state, board, log, log_path)
        return state, board, False
    if lower.startswith("roster"):
        parts = line.split()
        seat = int(parts[1]) if len(parts) > 1 else config.my_seat
        _show_roster(board, state, seat, config)
        return state, board, False
    if lower.startswith("board"):
        parts = line.split()
        _show_available(board, state, config, parts[1].upper() if len(parts) > 1 else None)
        return state, board, False
    if lower in ("go", "rec", "."):
        _recommend_now(config, board, state, audit)
        return state, board, False
    if lower.startswith("out ") or lower.startswith("bump "):
        board = _apply_update(line, board, state, config, audit)
        return state, board, False

    # Otherwise: a pick.
    mine = False
    text = line
    if lower.startswith("me "):
        mine, text = True, line[3:]

    # A bare number picks off the suggestion list. This is the whole point of
    # the list: entering someone else's pick is the slowest thing you do in a
    # live draft and the one place a typo silently corrupts the log.
    if text.strip().isdigit() and suggestions:
        choice = int(text.strip())
        if not 1 <= choice <= len(suggestions):
            print(f"  ! pick a number from 1 to {len(suggestions)}, or type a name")
            return state, board, False
        player = board.players[suggestions[choice - 1]]
    else:
        resolution = board.resolver.resolve(text)
        if not resolution.found:
            print(f"  ! {resolution.note}")
            return state, board, False
        if resolution.ambiguous:
            print(f"  ! {resolution.note}")
            print("    say which one - add the position (e.g. 'josh QB') "
                  "or type the full name")
            return state, board, False
        player = resolution.best
    if player.player_id in state.drafted:
        print(f"  ! {player.display} was already taken at pick "
              f"{state.drafted.index(player.player_id) + 1}")
        return state, board, False

    seat = state.on_the_clock
    if mine and seat != config.my_seat:
        print(f"  ! seat {seat} is on the clock, not you (seat {config.my_seat})")
        return state, board, False

    pick_no = state.pick_number
    state = state.record(player.player_id)
    log.append(player.player_id, seat=seat)
    print(f"  {pick_no}. seat {seat}: {player.display}")

    if not state.is_complete and state.on_the_clock == config.my_seat:
        print()
        _recommend_now(config, board, state, audit)
    return state, board, False


def _recommend_now(config, board, state, audit) -> None:
    if state.on_the_clock != config.my_seat:
        print(f"  seat {state.on_the_clock} is on the clock; "
              f"your next pick is {state.my_next_pick()}")
        return
    advice = _recommend(
        config, board, list(state.drafted), state.my_roster, state.pick_number, audit=audit
    )
    print(advice.format_card())
    print()
    print(f"  {'#':<3}{'player':<28}{'VOR':>7}{'surv':>7}{'P(title)':>10}{'delta':>8}")
    for rec in advice.recommendations[:3]:
        surv = f"{rec.survival:.0%}" if rec.survival == rec.survival else "  -"
        print(f"  {rec.rank:<3}{rec.display[:27]:<28}{rec.vor:7.1f}{surv:>7}"
              f"{rec.p_title:10.2%}{rec.delta_p_title * 100:+8.2f}")
    print()


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
        print(f"   {k:>2}  {player.name[:24]:<25}{player.pos:<4}{player.team:<4}"
              f"{prob:>6.0%}")
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



MOCK_HELP = """
  <enter>           take the engine's #1 pick
  <name>            take someone else instead
  board [pos]       show the best available
  roster            show your roster so far
  auto              let the engine finish the draft for you
  quit              stop here
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
    log_path = Path(args.out or f"logs/mock_{config.league_id}.jsonl")
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
        print(advice.format_card())
        print()
        for rec in advice.recommendations[:3]:
            surv = f"{rec.survival:.0%}" if rec.survival == rec.survival else "  -"
            print(f"  {rec.rank}. {rec.display[:30]:<31}VOR {rec.vor:6.1f}  "
                  f"survives {surv:>4}  P(title) {rec.p_title:.1%}")

        choice = None
        while choice is None:
            if auto:
                choice = advice.recommendations[0].player_id
                break
            try:
                line = input("\n  pick> ").strip()
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
            if low == "roster":
                _show_roster(board, state, seat, config)
                continue
            if low.startswith("board"):
                parts = line.split()
                _show_available(board, state, config,
                                parts[1].upper() if len(parts) > 1 else None)
                continue
            if not line:
                choice = advice.recommendations[0].player_id
                break
            resolution = board.resolver.resolve(line)
            if not resolution.found or resolution.ambiguous:
                print(f"  ! {resolution.note or 'no match'}")
                continue
            if resolution.best.player_id in state.drafted:
                print(f"  ! {resolution.best.display} is already gone")
                continue
            choice = resolution.best.player_id

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



# The sources R/pull_projections.R asks for. A source that errors during a
# scrape is skipped with a warning and simply never appears in the CSV, so the
# only way to notice is to compare what arrived against what was requested.
EXPECTED_SOURCES = (
    "CBS", "ESPN", "FantasyPros", "FFToday", "FantasyData",
    "FleaFlicker", "NumberFire", "NFL", "RTSports",
)


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
    """Median scored points per source, to catch one on a different scale.

    This is the failure nothing else here would notice. A source publishing
    per-game rather than per-season numbers, or half-PPR where its peers are
    full, still shows perfect per-position coverage - it just contributes
    numbers roughly 17x or 15% off, and the equal-weighted average silently
    absorbs the error into every player it touches.
    """
    import statistics
    from collections import defaultdict

    from .scoring import score_stats

    # Compare on one position with wide coverage, so the median is meaningful.
    per_source_points: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.pos in ("RB", "WR"):
            per_source_points[row.source].append(score_stats(row.stats, config.scoring))

    usable = {
        src: statistics.median(values)
        for src, values in per_source_points.items()
        if len(values) >= 10
    }
    if len(usable) < 2:
        return

    overall = statistics.median(usable.values())
    print(f"\n  scale check (median RB/WR points, cross-source median {overall:.0f}):")
    suspect = []
    for src in sorted(usable, key=lambda s: -usable[s]):
        value = usable[src]
        ratio = value / overall if overall else 1.0
        flag = "" if 0.7 <= ratio <= 1.4 else "   <-- OFF SCALE"
        if flag:
            suspect.append(src)
        print(f"    {src:<16}{value:>8.0f}  ({ratio:.2f}x){flag}")
    if suspect:
        print(f"  {', '.join(suspect)} projects on a different scale from its peers.")
        print("  Likely per-game instead of per-season, or a different scoring")
        print("  basis. Equal weighting will drag every player it covers toward")
        print("  that scale - drop it from R/pull_projections.R and re-run.")


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
    p.add_argument("--out", help="where to write the mock pick log")
    p.add_argument("--audit", help="where to write the audit log")
    p.set_defaults(func=cmd_mock)

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
