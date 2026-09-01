"""Draft order arithmetic and live draft state.

Also the home of the consistency checks behind the LLM's job #2: duplicate
picks, unknown players, and pick-count mismatches are caught here and raised,
never smoothed over. A draft log that quietly accepts a duplicate is worse than
no log, because every downstream number stays plausible while being wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import SLOT_ELIGIBILITY, LeagueConfig
from .ids import POSITIONS


class DraftStateError(ValueError):
    """The pick sequence is not internally consistent."""


def pick_owner(pick_number: int, teams: int, draft_type: str = "snake") -> int:
    """1-indexed seat on the clock for a 1-indexed overall pick number."""
    if pick_number < 1:
        raise ValueError("pick_number is 1-indexed")
    rnd, within = divmod(pick_number - 1, teams)
    if draft_type == "snake" and rnd % 2 == 1:
        return teams - within
    return within + 1


def round_of(pick_number: int, teams: int) -> int:
    return (pick_number - 1) // teams + 1


def seat_picks(seat: int, teams: int, rounds: int, draft_type: str = "snake") -> list[int]:
    """Every overall pick number belonging to a seat."""
    return [
        n
        for n in range(1, teams * rounds + 1)
        if pick_owner(n, teams, draft_type) == seat
    ]


def next_pick_for(
    seat: int, after: int, teams: int, rounds: int, draft_type: str = "snake"
) -> int | None:
    """The seat's next pick strictly after `after`, or None if the draft ends."""
    for n in seat_picks(seat, teams, rounds, draft_type):
        if n > after:
            return n
    return None


@dataclass
class DraftState:
    """Everything that has happened so far, and nothing else.

    `drafted` is every pick in order - the same list the tool contract takes.
    Team assignment is derived from the pick index, not stored, so the state
    cannot disagree with itself.
    """

    config: LeagueConfig
    drafted: list[str] = field(default_factory=list)
    my_seat: int | None = None

    def __post_init__(self) -> None:
        if self.my_seat is None:
            self.my_seat = self.config.my_seat
        self.validate()

    # -- derived views ----------------------------------------------------
    @property
    def pick_number(self) -> int:
        """The pick now on the clock (1-indexed)."""
        return len(self.drafted) + 1

    @property
    def teams(self) -> int:
        return self.config.teams

    @property
    def on_the_clock(self) -> int:
        return pick_owner(self.pick_number, self.teams, self.config.draft_type)

    @property
    def current_round(self) -> int:
        return round_of(self.pick_number, self.teams)

    @property
    def is_complete(self) -> bool:
        return len(self.drafted) >= self.config.total_drafted

    def roster_of(self, seat: int) -> list[str]:
        return [
            pid
            for n, pid in enumerate(self.drafted, start=1)
            if pick_owner(n, self.teams, self.config.draft_type) == seat
        ]

    @property
    def my_roster(self) -> list[str]:
        if self.my_seat is None:
            raise DraftStateError(
                "my_seat is not set. The engine cannot tell which picks are yours, "
                "and team need is meaningless without it. Set draft.my_seat in the "
                "league config or pass --seat."
            )
        return self.roster_of(self.my_seat)

    def my_next_pick(self) -> int | None:
        if self.my_seat is None:
            return None
        if self.on_the_clock == self.my_seat:
            return self.pick_number
        return next_pick_for(
            self.my_seat, len(self.drafted), self.teams,
            self.config.rounds, self.config.draft_type,
        )

    def my_following_pick(self) -> int | None:
        """The pick after the one I am about to make - the survival horizon."""
        current = self.my_next_pick()
        if current is None or self.my_seat is None:
            return None
        return next_pick_for(
            self.my_seat, current, self.teams, self.config.rounds, self.config.draft_type
        )

    def picks_until_my_next(self) -> int:
        """How many other teams pick before I do again."""
        current = self.my_next_pick()
        following = self.my_following_pick()
        if current is None or following is None:
            return 0
        return following - current - 1

    # -- mutation ---------------------------------------------------------
    def record(self, player_id: str) -> "DraftState":
        if player_id in self.drafted:
            raise DraftStateError(
                f"{player_id} was already drafted at pick "
                f"{self.drafted.index(player_id) + 1}. Refusing to record a duplicate."
            )
        if self.is_complete:
            raise DraftStateError(
                f"the draft is already complete ({self.config.total_drafted} picks)"
            )
        return DraftState(
            config=self.config,
            drafted=self.drafted + [player_id],
            my_seat=self.my_seat,
        )

    def replace(self, pick_number: int, player_id: str) -> "DraftState":
        """Correct one already-recorded pick, leaving every other pick alone.

        `undo` only reaches the most recent pick. The common draft-room error
        is noticing at pick 45 that pick 40 went in as the wrong Josh - undoing
        five picks to fix one, under a clock, is how a log gets worse instead
        of better.
        """
        if not (1 <= pick_number <= len(self.drafted)):
            raise DraftStateError(
                f"pick {pick_number} is not in the log; "
                f"{len(self.drafted)} pick(s) recorded so far"
            )
        current = self.drafted[pick_number - 1]
        if player_id == current:
            raise DraftStateError(f"pick {pick_number} is already {player_id}")
        clash = next(
            (
                n for n, pid in enumerate(self.drafted, start=1)
                if pid == player_id and n != pick_number
            ),
            None,
        )
        if clash is not None:
            raise DraftStateError(
                f"{player_id} is already recorded at pick {clash}. Fix that one "
                f"first, or you will have him drafted twice."
            )
        updated = list(self.drafted)
        updated[pick_number - 1] = player_id
        return DraftState(config=self.config, drafted=updated, my_seat=self.my_seat)

    def undo(self) -> "DraftState":
        if not self.drafted:
            raise DraftStateError("nothing to undo")
        return DraftState(
            config=self.config, drafted=self.drafted[:-1], my_seat=self.my_seat
        )

    # -- checks -----------------------------------------------------------
    def validate(self) -> None:
        seen: dict[str, int] = {}
        for n, pid in enumerate(self.drafted, start=1):
            if pid in seen:
                raise DraftStateError(
                    f"{pid} appears twice: picks {seen[pid]} and {n}"
                )
            seen[pid] = n
        if len(self.drafted) > self.config.total_drafted:
            raise DraftStateError(
                f"{len(self.drafted)} picks recorded but the draft is only "
                f"{self.config.total_drafted} picks long"
            )
        if self.my_seat is not None and not (1 <= self.my_seat <= self.teams):
            raise DraftStateError(f"my_seat {self.my_seat} outside 1..{self.teams}")

    def cross_check(self, expected_pick: int) -> None:
        """Confirm the caller and the log agree about where we are.

        Called with the pick number the human believes is on the clock. A
        mismatch means picks were missed or double-entered, and the right move
        is to stop and reconcile rather than recommend against a state that
        does not match the room.
        """
        if expected_pick != self.pick_number:
            raise DraftStateError(
                f"pick count mismatch: {len(self.drafted)} picks are recorded, so "
                f"pick {self.pick_number} is on the clock, but you said pick "
                f"{expected_pick}. Reconcile the log before drafting - "
                f"{'add the missing pick(s)' if expected_pick > self.pick_number else 'undo the extra pick(s)'}."
            )


def position_counts(roster: list[str], pos_lookup) -> dict[str, int]:
    counts = {p: 0 for p in POSITIONS}
    for pid in roster:
        pos = pos_lookup(pid)
        if pos in counts:
            counts[pos] += 1
    return counts


def unfilled_mandatory_slots(
    roster_positions: list[str], config: LeagueConfig
) -> dict[str, int]:
    """Starting slots that only one position can fill and that are still empty.

    Used by the K/DST guard: the policy floor is released exactly when the picks
    you have left equal the mandatory slots you still have to fill.
    """
    remaining = list(roster_positions)
    unfilled: dict[str, int] = {}
    for slot, count in config.roster.slot_counts().items():
        eligible = SLOT_ELIGIBILITY[slot]
        if len(eligible) != 1:
            continue
        pos = eligible[0]
        have = sum(1 for p in remaining if p == pos)
        need = max(0, count - have)
        for _ in range(min(have, count)):
            remaining.remove(pos)
        if need:
            unfilled[pos] = need
    return unfilled
