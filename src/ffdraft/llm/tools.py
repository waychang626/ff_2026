"""The tool surface exposed to the model.

The tool contract's whole point is that the model does not choose the
arguments. So `recommend` takes no inputs at all beyond an optional
consistency check: the session already knows the league, the pick log, the
roster and the pick number, and passing them through the model would only
create an opportunity to get them wrong under a clock.

`expected_pick_number` is not a tuning parameter. It is a guard - the model
reports where it thinks the draft is, and the engine refuses if that disagrees
with the log.
"""

from __future__ import annotations

from typing import Any

from .session import DraftSession

# Populated by DraftSession.bind(); module-level because the SDK's tool
# decorators introspect plain functions.
_SESSION: DraftSession | None = None


def bind(session: DraftSession) -> None:
    global _SESSION
    _SESSION = session


def _session() -> DraftSession:
    if _SESSION is None:
        raise RuntimeError("no draft session bound; call ffdraft.llm.tools.bind() first")
    return _SESSION


def resolve_player(query: str) -> dict[str, Any]:
    """Turn a spoken or typed player reference into a player ID.

    Args:
        query: What the user said, verbatim. "bijan gone", "the Lions D",
            "jamarr" are all fine - do not clean it up first.
    """
    return _session().resolve(query)


def record_pick(player_id: str, is_mine: bool = False) -> dict[str, Any]:
    """Record one pick against the draft log.

    Refuses duplicates and picks past the end of the draft. Call resolve_player
    first; never construct a player_id yourself.

    Args:
        player_id: An exact ID returned by resolve_player.
        is_mine: True only if this pick belongs to the user's own seat.
    """
    return _session().record(player_id, is_mine=is_mine)


def undo_last_pick() -> dict[str, Any]:
    """Remove the most recent pick from the log. Use when the user says they
    mis-entered one."""
    return _session().undo()


def recent_picks(count: int = 12) -> dict[str, Any]:
    """List the most recent picks with their pick numbers.

    Call this before correct_pick, to find the number of the pick to fix.

    Args:
        count: How many recent picks to list.
    """
    return _session().recent_picks(count)


def correct_pick(pick_number: int, player_id: str) -> dict[str, Any]:
    """Replace one already-recorded pick with the right player.

    Use this when a name was resolved to the wrong player and it is too far
    back for undo_last_pick to reach. Every other pick is left alone.

    Args:
        pick_number: The overall pick number to correct, from recent_picks.
        player_id: An exact ID returned by resolve_player.
    """
    return _session().correct_pick(pick_number, player_id)


def get_draft_state() -> dict[str, Any]:
    """Where the draft is: pick on the clock, round, whose turn, the user's roster."""
    return _session().state_summary()


def recommend() -> dict[str, Any]:
    """Ask the engine to rank the board for the pick now on the clock.

    Takes no arguments on purpose. The engine derives risk appetite and
    position need from the roster and the live board; there is nothing to tune
    and nothing for you to choose.

    Returns the ranked candidates with VOR, survival probability and title
    odds, plus a preformatted four-line card. Report the card. Do not reorder.
    """
    return _session().recommend()


def update_projection(
    player_id: str,
    reason: str,
    out_for_season: bool = False,
    points_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Feed the engine information it cannot see, as a logged projection change.

    This is the only legitimate way to act on news. Adjusting for an injury by
    mentioning it in your answer instead leaves no record and does not change a
    single number the engine computes.

    Args:
        player_id: An exact ID returned by resolve_player.
        reason: Why, in a few words, and where it came from. Goes in the audit log.
        out_for_season: True if the player is done for the year.
        points_multiplier: Scale the projection, e.g. 0.75 for a lingering injury.
            Ignored when out_for_season is True.
    """
    return _session().update_projection(
        player_id,
        reason=reason,
        out_for_season=out_for_season,
        points_multiplier=points_multiplier,
    )


TOOL_FUNCTIONS = [
    resolve_player,
    record_pick,
    undo_last_pick,
    recent_picks,
    correct_pick,
    get_draft_state,
    recommend,
    update_projection,
]
