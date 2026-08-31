"""Name resolution: LLM jobs #1 (parse) and #2 (refuse when unsure)."""
from __future__ import annotations

import pytest

from ffdraft.ids import Player, Resolver, make_player_id, normalize_name

BOARD = [
    Player("bijan-robinson|RB", "Bijan Robinson", "RB", "ATL"),
    Player("jamarr-chase|WR", "Ja'Marr Chase", "WR", "CIN"),
    Player("justin-jefferson|WR", "Justin Jefferson", "WR", "MIN"),
    Player("josh-allen|QB", "Josh Allen", "QB", "BUF"),
    Player("josh-jacobs|RB", "Josh Jacobs", "RB", "GB"),
    Player("lions-dst|DST", "Lions DST", "DST", "DET"),
    Player("marvin-harrison|WR", "Marvin Harrison Jr.", "WR", "ARI"),
    Player("michael-pittman|WR", "Michael Pittman Jr.", "WR", "IND"),
]


@pytest.fixture
def resolver():
    return Resolver(BOARD)


def test_normalize_strips_accents_punctuation_and_suffixes():
    assert normalize_name("Ja'Marr Chase") == "jamarr chase"
    assert normalize_name("Marvin Harrison Jr.") == "marvin harrison"
    assert normalize_name("Amon-Ra St. Brown") == "amon ra st brown"


def test_player_id_is_stable_and_normalises_defense_aliases():
    assert make_player_id("Ja'Marr Chase", "WR") == "jamarr-chase|WR"
    assert make_player_id("Lions", "DEF") == "lions|DST"
    with pytest.raises(ValueError):
        make_player_id("Nobody", "PUNTER")


@pytest.mark.parametrize(
    "query,expected",
    [
        ("bijan gone", "bijan-robinson|RB"),
        ("Ja'Marr Chase", "jamarr-chase|WR"),
        ("jefferson", "justin-jefferson|WR"),
        ("they took the Lions D", "lions-dst|DST"),
        ("detroit defense", "lions-dst|DST"),
        ("harrison jr", "marvin-harrison|WR"),
    ],
)
def test_messy_draft_room_input_resolves(resolver, query, expected):
    result = resolver.resolve(query)
    assert result.best is not None, f"{query!r}: {result.note}"
    assert result.best.player_id == expected


def test_ambiguous_first_name_refuses_rather_than_guessing(resolver):
    """Two Joshes is exactly the case that silently corrupts a draft log."""
    result = resolver.resolve("josh")
    assert result.ambiguous
    assert result.best is None
    assert {c.player.player_id for c in result.candidates} >= {
        "josh-allen|QB",
        "josh-jacobs|RB",
    }


def test_position_hint_disambiguates(resolver):
    result = resolver.resolve("josh", pos_hint="QB")
    assert result.best is not None
    assert result.best.player_id == "josh-allen|QB"


def test_unknown_name_reports_rather_than_matching_anything(resolver):
    result = resolver.resolve("zzzz qqqq")
    assert not result.found
    assert result.best is None
    assert result.note


def test_resolution_is_deterministic(resolver):
    first = [c.player.player_id for c in resolver.resolve("jr").candidates]
    for _ in range(5):
        assert [c.player.player_id for c in resolver.resolve("jr").candidates] == first
