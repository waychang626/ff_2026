"""The tool contract, its guards, and its determinism."""
from __future__ import annotations

import inspect

import pytest

from ffdraft.audit import AuditLog
from ffdraft.draft import DraftStateError, seat_picks
from ffdraft.engine import Recommendation, _recommend, recommend_pick


@pytest.fixture
def quiet_log(tmp_path):
    return AuditLog(tmp_path / "audit.jsonl")


def advise(config, board, drafted, my_roster, pick, log):
    return _recommend(config, board, drafted, my_roster, pick, audit=log)


# --- the contract ------------------------------------------------------------
def test_signature_has_no_tuning_parameters():
    """Brief section 2. If a parameter encodes an opinion it belongs in a config.

    This is the load-bearing test of the whole design. The failure mode it
    guards against is not a crash - it is the engine slowly turning back into
    the LLM's opinion, one helpful-looking keyword argument at a time.
    """
    params = inspect.signature(recommend_pick).parameters
    positional = [
        name for name, p in params.items()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert positional == ["league_id", "drafted", "my_roster", "pick_number"]

    keyword_only = {n for n, p in params.items() if p.kind == p.KEYWORD_ONLY}
    assert keyword_only <= {"audit"}, (
        f"unexpected keyword arguments {keyword_only - {'audit'}}: a knob here is "
        f"a decision the LLM makes under a 60-second clock"
    )
    banned = {"risk_tolerance", "position_need", "aggression", "strategy",
              "upside", "zero_rb", "temperature"}
    assert not banned & set(params)


def test_recommendation_reports_vor_survival_and_win_probability():
    fields = set(Recommendation.__dataclass_fields__)
    assert {"vor", "survival", "p_title", "delta_p_title"} <= fields


# --- determinism -------------------------------------------------------------
def test_identical_inputs_give_identical_output(seated_config, sample_board, quiet_log):
    drafted = [p.player_id for p in sample_board.players[:2]]
    first = advise(seated_config, sample_board, drafted, [], 3, quiet_log)
    second = advise(seated_config, sample_board, drafted, [], 3, quiet_log)
    assert first.state_hash == second.state_hash
    assert first.seed == second.seed
    assert [r.player_id for r in first.recommendations] == [
        r.player_id for r in second.recommendations
    ]
    assert [r.p_title for r in first.recommendations] == [
        r.p_title for r in second.recommendations
    ]


def test_a_different_board_changes_the_state_hash(seated_config, sample_board, quiet_log):
    from ffdraft.board import ProjectionUpdate

    drafted = [p.player_id for p in sample_board.players[:2]]
    before = advise(seated_config, sample_board, drafted, [], 3, quiet_log)
    updated = sample_board.apply_update(
        ProjectionUpdate(sample_board.players[40].player_id, out_for_season=True,
                         reason="test", source="test")
    )
    after = advise(seated_config, updated, drafted, [], 3, quiet_log)
    assert before.state_hash != after.state_hash


def test_every_call_is_written_to_the_audit_log(seated_config, sample_board, quiet_log):
    drafted = [p.player_id for p in sample_board.players[:2]]
    advice = advise(seated_config, sample_board, drafted, [], 3, quiet_log)
    entries = quiet_log.entries()
    assert len(entries) == 1
    assert entries[0].state_hash == advice.state_hash
    assert entries[0].kind == "recommend_pick"
    assert entries[0].payload["recommendations"][0]["player_id"] == \
        advice.recommendations[0].player_id


# --- the guards (LLM job #2, enforced in the script) -------------------------
def test_unknown_player_id_is_refused(seated_config, sample_board, quiet_log):
    with pytest.raises(DraftStateError, match="not on the board"):
        advise(seated_config, sample_board, ["ghost|RB"], [], 2, quiet_log)


def test_duplicate_pick_is_refused(seated_config, sample_board, quiet_log):
    pid = sample_board.players[0].player_id
    with pytest.raises(DraftStateError, match="appears twice"):
        advise(seated_config, sample_board, [pid, pid], [], 3, quiet_log)


def test_pick_number_disagreeing_with_the_log_is_refused(seated_config, sample_board, quiet_log):
    drafted = [p.player_id for p in sample_board.players[:2]]
    with pytest.raises(DraftStateError, match="mismatch"):
        advise(seated_config, sample_board, drafted, [], 9, quiet_log)


def test_ranking_someone_elses_pick_is_refused(seated_config, sample_board, quiet_log):
    drafted = [p.player_id for p in sample_board.players[:1]]
    with pytest.raises(DraftStateError, match="belongs to seat"):
        advise(seated_config, sample_board, drafted, [], 2, quiet_log)


def test_my_roster_disagreeing_with_the_log_is_refused(seated_config, sample_board, quiet_log):
    drafted = [p.player_id for p in sample_board.players[:2]]
    wrong = [sample_board.players[5].player_id]
    with pytest.raises(DraftStateError, match="disagrees with the pick log"):
        advise(seated_config, sample_board, drafted, wrong, 3, quiet_log)


# --- the K/DST policy guard (brief 3.5) -------------------------------------
def test_kickers_and_defenses_are_withheld_early(seated_config, sample_board, quiet_log):
    drafted = [p.player_id for p in sample_board.players[:2]]
    advice = advise(seated_config, sample_board, drafted, [], 3, quiet_log)
    assert advice.recommendations
    assert not {r.pos for r in advice.recommendations} & {"K", "DST"}


def test_the_guard_releases_when_mandatory_slots_must_be_filled(
    seated_config, sample_board, quiet_log
):
    """The roster still has to be legal at the end, so the floor is not absolute."""
    last_pick = seat_picks(3, seated_config.teams, seated_config.rounds)[-1]
    drafted = [p.player_id for p in sample_board.players[: last_pick - 1]]
    advice = advise(seated_config, sample_board, drafted, None or
                    _derive_roster(seated_config, drafted, 3), last_pick, quiet_log)
    assert advice.recommendations
    assert {r.pos for r in advice.recommendations} <= {"K", "DST"}
    assert any("must still fill" in n for n in advice.notes)


def _derive_roster(config, drafted, seat):
    from ffdraft.draft import pick_owner

    return [
        pid for n, pid in enumerate(drafted, start=1)
        if pick_owner(n, config.teams, config.draft_type) == seat
    ]


# --- the four-line card ------------------------------------------------------
def test_card_is_four_lines_with_the_required_prefixes(seated_config, sample_board, quiet_log):
    drafted = [p.player_id for p in sample_board.players[:2]]
    card = advise(seated_config, sample_board, drafted, [], 3, quiet_log).format_card()
    lines = card.splitlines()
    assert 2 <= len(lines) <= 4, card
    assert lines[0].startswith("PICK: ")
    assert lines[1].startswith("EDGE: ")
    assert any(l.startswith("WHY:  ") for l in lines)
    for line in lines:
        assert len(line) < 160
