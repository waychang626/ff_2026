"""League config validation and replacement-level derivation."""
from __future__ import annotations

import pytest
import yaml

from ffdraft.baselines import derive, explain
from ffdraft.config import ConfigError, MissingLeagueInput, parse_league


def test_league_one_shape_matches_the_brief(cuomo_config):
    assert cuomo_config.teams == 8
    assert cuomo_config.rounds == 17
    assert cuomo_config.total_drafted == 136          # brief section 4
    assert cuomo_config.playoff_teams == 4
    assert cuomo_config.roster.starters.count("QWRT") == 1   # superflex
    assert "TE" not in cuomo_config.roster.starters          # no TE slot


def test_league_one_baselines_match_the_brief(cuomo_config):
    assert cuomo_config.vor_baseline == {
        "QB": 17, "RB": 23, "WR": 33, "TE": 10, "K": 9, "DST": 9
    }


def test_starter_demand_derivation_reproduces_every_baseline_except_te(cuomo_config):
    """The brief's table is starter demand + 1 - apart from TE.

    QB, RB, WR, K and DST all fall straight out of the arithmetic. TE does not:
    with no TE slot, starter demand is only ~3, and the brief's 10 is a
    roster-count judgement instead. Worth pinning down, because the two
    definitions disagree by seven ranks at exactly the position the league
    structure makes strange.
    """
    d = derive(cuomo_config)
    for pos in ("QB", "RB", "WR", "K", "DST"):
        assert d.starter_demand[pos] == cuomo_config.vor_baseline[pos], pos
    assert d.starter_demand["TE"] == 3
    assert d.drafted["TE"] == 9
    assert cuomo_config.vor_baseline["TE"] == 10


def test_superflex_doubles_qb_demand(cuomo_config):
    d = derive(cuomo_config)
    assert d.fixed["QB"] == 8      # one per team
    assert d.flex["QB"] == 8       # the Q/W/R/T slot
    assert d.starter_demand["QB"] == 17


def test_explain_renders_without_error(cuomo_config):
    text = explain(cuomo_config)
    assert "Replacement level" in text
    assert "QB" in text and "17" in text


def _minimal(**overrides):
    data = {
        "league_id": "t", "teams": 10,
        "roster": {"starters": ["QB", "RB", "WR", "K", "DEF"], "bench": 5},
        "schedule": {"playoff_teams": 4, "playoff_weeks": [16, 17]},
        "scoring": {"offense": {"rec": 0.5}},
        "baselines": {"explicit": {"QB": 11, "RB": 21, "WR": 31, "TE": 11, "K": 11, "DST": 11}},
        "calibration": {
            "slopes": {p: 1.0 for p in ("QB", "RB", "WR", "TE", "K", "DST")},
            "optimism": {}, "r_squared": {},
        },
        "sim": {"weekly_cv": {p: 0.5 for p in ("QB", "RB", "WR", "TE", "K", "DST")}},
    }
    data.update(overrides)
    return data


def test_missing_roster_slots_is_a_named_blocker():
    """League 2's actual state: everything known except the starting slots."""
    data = _minimal(roster={"bench": 6})
    with pytest.raises(MissingLeagueInput) as exc:
        parse_league(data, "league2.yaml")
    message = str(exc.value)
    assert "roster.starters" in message
    assert "superflex" in message.lower()


def test_missing_baselines_is_a_named_blocker():
    data = _minimal(baselines={})
    with pytest.raises(MissingLeagueInput, match="baselines.explicit"):
        parse_league(data, "x.yaml")


def test_weighted_aggregation_is_refused():
    """Brief 3.3: source accuracy does not persist year to year."""
    data = _minimal()
    data["calibration"]["aggregate"] = "weighted"
    with pytest.raises(ConfigError, match="does not persist"):
        parse_league(data, "x.yaml")


def test_flex_slot_without_a_declared_share_is_refused():
    data = _minimal(roster={"starters": ["QB", "RB", "WRT", "K", "DEF"], "bench": 5})
    with pytest.raises(ConfigError, match="flex_shares"):
        parse_league(data, "x.yaml")


def test_unknown_slot_name_is_refused():
    data = _minimal(roster={"starters": ["QB", "SUPERFLEX"], "bench": 5})
    with pytest.raises(ConfigError, match="unknown roster slot"):
        parse_league(data, "x.yaml")


def test_fingerprint_changes_when_scoring_changes(cuomo_config, tmp_path):
    raw = yaml.safe_load((tmp_path.parent and cuomo_config.source_path and
                          open(cuomo_config.source_path).read()) or "")
    before = cuomo_config.fingerprint()
    raw["scoring"]["offense"]["rec"] = 1.0
    after = parse_league(raw, "modified").fingerprint()
    assert before != after


# --- League 2 ----------------------------------------------------------------
def test_league_two_shape_matches_the_settings_page():
    from ffdraft.config import load_by_id

    config = load_by_id("league2")
    assert config.teams == 12
    assert config.roster.starters == ("QB", "RB", "RB", "WR", "WR", "TE", "WRT", "K", "DEF")
    assert config.roster.bench == 4
    assert config.roster.ir == 1
    assert config.rounds == 13
    assert config.total_drafted == 156
    assert config.playoff_teams == 6
    assert config.playoff_weeks == (15, 16, 17)


def test_league_two_baselines_all_fall_out_of_starter_demand():
    """Unlike League 1, every position here derives cleanly - there is a TE slot."""
    from ffdraft.config import load_by_id

    config = load_by_id("league2")
    d = derive(config)
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        assert d.starter_demand[pos] == config.vor_baseline[pos], pos


def test_replacement_level_differs_by_position_not_uniformly_by_league_size():
    """The brief expected "much lower replacement levels" across the board in
    League 2. That holds for RB, TE, K and DST - and is false for WR and QB.

    League 1 has 8 teams but requires three WRs and two W/R/T flexes, so 32 WRs
    start; League 2 has 12 teams but requires two WRs and one flex, so 30 do.
    Roster shape beats league size. And the superflex makes League 1's QB
    baseline the deeper of the two by a wide margin.

    This matters practically: it means the two boards are NOT related by a
    simple shift, and a WR who is a marginal starter in one league is a
    marginal starter in the other.
    """
    from ffdraft.config import load_by_id

    one, two = load_by_id("cuomo"), load_by_id("league2")

    # Much deeper in League 2, as expected.
    assert two.vor_baseline["RB"] > one.vor_baseline["RB"] + 5
    assert two.vor_baseline["TE"] > one.vor_baseline["TE"]
    assert two.vor_baseline["K"] > one.vor_baseline["K"]
    assert two.vor_baseline["DST"] > one.vor_baseline["DST"]

    # WR is essentially flat, and slightly deeper in the SMALLER league.
    assert abs(two.vor_baseline["WR"] - one.vor_baseline["WR"]) <= 3
    assert one.vor_baseline["WR"] > two.vor_baseline["WR"]

    # Superflex, not team count, drives QB scarcity.
    assert one.vor_baseline["QB"] > two.vor_baseline["QB"] + 3


def test_skill_position_scoring_is_identical_across_leagues(cuomo_config):
    """One QB/RB/WR/TE projection set serves both leagues (brief section 4)."""
    from ffdraft.config import load_by_id

    league2 = load_by_id("league2")
    assert cuomo_config.scoring.offense == league2.scoring.offense
    assert cuomo_config.scoring.pts_bracket == league2.scoring.pts_bracket
    # All the differences live in K and DST.
    assert cuomo_config.scoring.kicking != league2.scoring.kicking
    assert league2.scoring.kicking["fg_60"] == 6
    assert league2.scoring.kicking["fg_miss"] == -1
    assert league2.scoring.dst["dst_forced_fumble"] == 1
    assert "dst_forced_fumble" not in cuomo_config.scoring.dst
