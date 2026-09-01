"""The orchestration layer: four jobs, and no fifth one.

These exercise the session and the tool surface directly. They do not call the
Anthropic API - the point of the design is that the model never decides
anything, so there is nothing about the decision to test through it.
"""
from __future__ import annotations

import inspect

import pytest

from ffdraft.audit import AuditLog
from ffdraft.llm import tools
from ffdraft.llm.prompt import SYSTEM_PROMPT
from ffdraft.llm.session import DraftSession


@pytest.fixture
def session(seated_config, sample_board, tmp_path):
    return DraftSession(
        seated_config, sample_board,
        audit=AuditLog(tmp_path / "audit.jsonl"),
        log_path=tmp_path / "draft.jsonl",
    )


# --- the tool surface --------------------------------------------------------
def test_recommend_takes_no_arguments_at_all():
    """The strongest form of the contract: nothing for the model to choose.

    The session already knows the league, the log, the roster and the pick
    number. Routing them through the model would only create a chance to get
    them wrong under a clock.
    """
    assert list(inspect.signature(tools.recommend).parameters) == []


def test_no_tool_exposes_a_tuning_parameter():
    banned = {"risk", "risk_tolerance", "position_need", "aggression", "strategy",
              "upside", "temperature", "weight", "n_sims", "seed"}
    for fn in tools.TOOL_FUNCTIONS:
        params = set(inspect.signature(fn).parameters)
        assert not params & banned, f"{fn.__name__} exposes {params & banned}"


def test_every_tool_is_documented_for_the_model():
    for fn in tools.TOOL_FUNCTIONS:
        assert fn.__doc__ and len(fn.__doc__.strip()) > 40, fn.__name__


def test_the_prompt_states_the_no_reordering_rule():
    lowered = SYSTEM_PROMPT.lower()
    assert "do not reorder" in lowered
    assert "flag" in lowered
    assert "never invent a number" in lowered
    for job in ("parse", "catch errors", "explain the output"):
        assert job in lowered


# --- job 1: parse ------------------------------------------------------------
def test_messy_input_resolves_to_one_id(session, sample_board):
    name = sample_board.players[0].name
    result = session.resolve(f"{name} just went")
    assert result["ok"]
    assert result["player_id"] == sample_board.players[0].player_id


def test_ambiguity_returns_not_ok_with_an_instruction_to_ask(session, sample_board):
    """Job #2. The tool must not pick, and must say so in the payload."""
    first_names = {}
    for player in sample_board.players:
        first_names.setdefault(player.name.split()[0], []).append(player)
    shared = next(v for v in first_names.values() if len(v) > 1)

    result = session.resolve(shared[0].name.split()[0])
    assert not result["ok"]
    assert result["ambiguous"]
    assert "Do not choose" in result["instruction"]
    assert len(result["candidates"]) > 1


def test_unknown_name_is_reported_not_guessed(session):
    result = session.resolve("qqqq zzzz")
    assert not result["ok"]
    assert result["candidates"] == []


# --- job 2: record and check -------------------------------------------------
def test_recording_advances_the_draft_and_writes_the_log(session, sample_board, tmp_path):
    pid = sample_board.players[0].player_id
    result = session.record(pid)
    assert result["ok"] and result["pick"] == 1 and result["seat"] == 1
    assert (tmp_path / "draft.jsonl").exists()


def test_duplicate_pick_is_refused_with_a_reason(session, sample_board):
    pid = sample_board.players[0].player_id
    session.record(pid)
    result = session.record(pid)
    assert not result["ok"]
    assert "already drafted" in result["reason"]


def test_claiming_someone_elses_pick_is_refused(session, sample_board):
    """Seat 3 cannot own pick 1."""
    result = session.record(sample_board.players[0].player_id, is_mine=True)
    assert not result["ok"]
    assert "on the clock" in result["reason"]


def test_unknown_player_id_is_refused(session):
    assert not session.record("nobody|RB")["ok"]


def test_undo_rewinds_both_state_and_log(session, sample_board):
    session.record(sample_board.players[0].player_id)
    result = session.undo()
    assert result["ok"]
    assert result["pick_on_the_clock"] == 1
    assert session.log.picks == []


# --- the decision ------------------------------------------------------------
def test_recommend_refuses_when_it_is_not_the_users_turn(session):
    result = session.recommend()
    assert not result["ok"]
    assert "on the clock" in result["reason"]


def test_recommend_returns_a_card_and_forbids_reordering(session, sample_board):
    for player in sample_board.players[:2]:
        session.record(player.player_id)
    result = session.recommend()
    assert result["ok"]
    assert result["card"].startswith("PICK: ")
    assert "Do not reorder" in result["instruction"]
    assert len(result["candidates"]) <= 3
    assert result["candidates"][0]["rank"] == 1
    assert "vor" in result["candidates"][0]


def test_a_pick_number_the_model_believes_wrong_is_refused(session, sample_board):
    for player in sample_board.players[:2]:
        session.record(player.player_id)
    result = session.recommend(expected_pick_number=17)
    assert not result["ok"]
    assert "Reconcile" in result["reason"]


# --- job 3: news enters through a logged tool --------------------------------
def test_projection_update_changes_the_board_and_is_logged(session, sample_board, tmp_path):
    pid = sample_board.players[10].player_id
    before = session.board.fingerprint()
    result = session.update_projection(pid, reason="ruled out, ACL", out_for_season=True)
    assert result["ok"]
    assert session.board.fingerprint() != before
    assert float(session.board.points[session.board.idx(pid)]) == 0.0

    entries = AuditLog(tmp_path / "audit.jsonl").entries()
    assert any(e.kind == "projection_update" for e in entries)


def test_an_update_without_a_reason_is_refused(session, sample_board):
    """The reason is the audit trail; an unexplained edit is narration."""
    result = session.update_projection(sample_board.players[3].player_id, reason="  ")
    assert not result["ok"]


def test_the_session_snapshot_records_every_update(session, sample_board):
    session.update_projection(
        sample_board.players[5].player_id, reason="limited in practice",
        points_multiplier=0.8,
    )
    snapshot = session.snapshot()
    assert len(snapshot["updates"]) == 1
    assert "limited in practice" in snapshot["updates"][0]


def test_correcting_an_earlier_pick_leaves_the_others_alone(session, sample_board):
    for player in sample_board.players[:4]:
        session.record(player.player_id)
    right = sample_board.players[40].player_id

    result = session.correct_pick(2, right)
    assert result["ok"]
    assert result["was"] == sample_board.players[1].display
    assert session.state.drafted[1] == right
    assert session.state.drafted[0] == sample_board.players[0].player_id
    assert session.state.drafted[3] == sample_board.players[3].player_id
    assert len(session.state.drafted) == 4


def test_a_correction_that_would_duplicate_is_refused(session, sample_board):
    for player in sample_board.players[:3]:
        session.record(player.player_id)
    result = session.correct_pick(3, sample_board.players[0].player_id)
    assert not result["ok"]
    assert "already recorded at pick 1" in result["reason"]


def test_recent_picks_gives_the_model_numbers_to_aim_at(session, sample_board):
    for player in sample_board.players[:5]:
        session.record(player.player_id)
    rows = session.recent_picks(3)["picks"]
    assert [r["pick"] for r in rows] == [3, 4, 5]
    assert all("seat" in r and "player_id" in r for r in rows)
