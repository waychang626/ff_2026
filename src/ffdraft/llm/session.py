"""Draft session state, shared by the CLI and the LLM tool surface.

Holds the board, the config, the pick log and the audit log, and exposes them
as plain dictionaries the model can read. Every method returns a result rather
than raising into the model's face - a tool that throws teaches it to stop
calling the tool, whereas a result that says `ok: false` with a reason teaches
it to ask the user, which is job #2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..audit import AuditLog
from ..board import Board, ProjectionUpdate
from ..config import LeagueConfig
from ..draft import DraftState, DraftStateError
from ..engine import _recommend
from ..replay import DraftLog


class DraftSession:
    def __init__(
        self,
        config: LeagueConfig,
        board: Board,
        audit: AuditLog | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        if config.my_seat is None:
            raise ValueError("config.my_seat must be set before starting a session")
        self.config = config
        self.board = board
        self.audit = audit or AuditLog()
        self.log_path = Path(log_path or f"logs/draft_{config.league_id}.jsonl")
        self.state = DraftState(config=config, drafted=[], my_seat=config.my_seat)
        self.log = DraftLog(league_id=config.league_id, my_seat=config.my_seat)
        self.last_advice = None

    # -- job 1: parse ------------------------------------------------------
    def resolve(self, query: str) -> dict[str, Any]:
        result = self.board.resolver.resolve(query)
        if not result.found:
            return {"ok": False, "reason": result.note, "candidates": []}
        candidates = [
            {
                "player_id": c.player.player_id,
                "name": c.player.name,
                "pos": c.player.pos,
                "team": c.player.team,
                "already_drafted": c.player.player_id in self.state.drafted,
                "match": c.reason,
            }
            for c in result.candidates
        ]
        if result.ambiguous:
            return {
                "ok": False,
                "ambiguous": True,
                "reason": result.note,
                "candidates": candidates,
                "instruction": "Ask the user which one. Do not choose.",
            }
        return {"ok": True, "player_id": result.best.player_id, "candidates": candidates}

    # -- job 2: record and check ------------------------------------------
    def record(self, player_id: str, is_mine: bool = False) -> dict[str, Any]:
        if player_id not in self.board.index:
            return {"ok": False, "reason": f"{player_id} is not on the board"}
        seat = self.state.on_the_clock
        if is_mine and seat != self.config.my_seat:
            return {
                "ok": False,
                "reason": f"seat {seat} is on the clock, not the user's seat "
                          f"{self.config.my_seat}",
            }
        try:
            pick_number = self.state.pick_number
            self.state = self.state.record(player_id)
        except DraftStateError as exc:
            return {"ok": False, "reason": str(exc)}
        self.log.append(player_id, seat=seat)
        self.log.save(self.log_path)
        player = self.board.player(player_id)
        return {
            "ok": True,
            "pick": pick_number,
            "seat": seat,
            "player": player.display,
            "your_turn_now": self.state.on_the_clock == self.config.my_seat,
        }

    def undo(self) -> dict[str, Any]:
        try:
            self.state = self.state.undo()
        except DraftStateError as exc:
            return {"ok": False, "reason": str(exc)}
        removed = self.log.picks.pop() if self.log.picks else None
        self.log.save(self.log_path)
        return {"ok": True, "removed": removed.player_id if removed else None,
                "pick_on_the_clock": self.state.pick_number}

    def state_summary(self) -> dict[str, Any]:
        roster = [self.board.player(p).display for p in self.state.my_roster]
        return {
            "league": self.config.name,
            "pick_number": self.state.pick_number,
            "round": self.state.current_round,
            "on_the_clock_seat": self.state.on_the_clock,
            "your_seat": self.config.my_seat,
            "your_turn": self.state.on_the_clock == self.config.my_seat,
            "your_next_pick": self.state.my_next_pick(),
            "picks_until_your_next": self.state.picks_until_my_next(),
            "your_roster": roster,
            "picks_recorded": len(self.state.drafted),
            "draft_complete": self.state.is_complete,
        }

    # -- the decision ------------------------------------------------------
    def recommend(self, expected_pick_number: int | None = None) -> dict[str, Any]:
        if self.state.on_the_clock != self.config.my_seat:
            return {
                "ok": False,
                "reason": f"seat {self.state.on_the_clock} is on the clock; the "
                          f"user's next pick is {self.state.my_next_pick()}",
            }
        if expected_pick_number is not None and expected_pick_number != self.state.pick_number:
            return {
                "ok": False,
                "reason": f"you said pick {expected_pick_number} but "
                          f"{len(self.state.drafted)} picks are recorded, so pick "
                          f"{self.state.pick_number} is on the clock. Reconcile first.",
            }
        try:
            advice = _recommend(
                self.config, self.board, list(self.state.drafted),
                self.state.my_roster, self.state.pick_number, audit=self.audit,
            )
        except DraftStateError as exc:
            return {"ok": False, "reason": str(exc)}
        self.last_advice = advice
        return {
            "ok": True,
            "card": advice.format_card(),
            "state_hash": advice.state_hash,
            "candidates": [
                {
                    "rank": r.rank,
                    "player": r.display,
                    "player_id": r.player_id,
                    "vor": round(r.vor, 1),
                    "survival_to_your_next_pick": (
                        round(r.survival, 3) if r.survival == r.survival else None
                    ),
                    "p_title": round(r.p_title, 4),
                    "p_playoffs": round(r.p_playoffs, 4),
                    "delta_p_title_vs_top": round(r.delta_p_title, 4),
                    "flags": r.flags,
                }
                for r in advice.recommendations[:3]
            ],
            "notes": advice.notes,
            "instruction": "Report the card verbatim. Do not reorder the candidates.",
        }

    # -- job 3: feed in what the engine cannot see ------------------------
    def update_projection(
        self,
        player_id: str,
        reason: str,
        out_for_season: bool = False,
        points_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        if player_id not in self.board.index:
            return {"ok": False, "reason": f"{player_id} is not on the board"}
        if not reason.strip():
            return {"ok": False, "reason": "a reason is required; it goes in the audit log"}
        update = ProjectionUpdate(
            player_id=player_id,
            out_for_season=out_for_season,
            points_multiplier=points_multiplier,
            reason=reason.strip(),
            source="llm",
        )
        self.board = self.board.apply_update(update)
        self.audit.record(
            state_hash=self.board.fingerprint(),
            league_id=self.config.league_id,
            pick_number=self.state.pick_number,
            kind="projection_update",
            payload={"update": update.describe()},
        )
        return {"ok": True, "applied": update.describe(),
                "board_fingerprint": self.board.fingerprint()}

    def snapshot(self) -> dict[str, Any]:
        return {
            "league_fingerprint": self.config.fingerprint(),
            "board_fingerprint": self.board.fingerprint(),
            "drafted": list(self.state.drafted),
            "updates": [u.describe() for u in self.board.updates],
        }
