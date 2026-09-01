"""Market data parsing - the ADP column is not a stat column."""
from __future__ import annotations

from ffdraft.projections import load_market, maybe_float


def test_nullish_cells_parse_as_missing_not_zero():
    """An R export writes NA. Coercing that to 0.0 makes every player pick 1.01."""
    for text in ("NA", "N/A", "", "  ", "nan", "NULL", "-", "not a number"):
        assert maybe_float(text) is None, text
    assert maybe_float("12.5") == 12.5
    assert maybe_float(0) == 0.0


def test_an_all_na_adp_column_yields_no_adp_rather_than_all_zeros(tmp_path):
    path = tmp_path / "market.csv"
    path.write_text(
        "player,pos,adp,adp_sd,ecr\n"
        "Josh Allen,QB,NA,NA,NA\n"
        "Bijan Robinson,RB,NA,NA,NA\n"
    )
    market = load_market(path)
    assert market.adp == {}
    assert market.rows_read == 2
    assert market.rows_without_adp == 2
    assert market.looks_empty


def test_a_healthy_file_loads_and_is_not_flagged_empty(tmp_path):
    path = tmp_path / "market.csv"
    rows = "\n".join(f"Player {i},RB,{i + 1},5.0,{i + 1}" for i in range(40))
    path.write_text("player,pos,adp,adp_sd,ecr\n" + rows + "\n")
    market = load_market(path)
    assert len(market.adp) == 40
    assert not market.looks_empty
    assert market.adp["player-0|RB"] == 1.0


def test_zero_and_negative_adp_are_treated_as_missing(tmp_path):
    path = tmp_path / "market.csv"
    path.write_text("player,pos,adp\nA B,RB,0\nC D,WR,-3\nE F,TE,12\n")
    market = load_market(path)
    assert set(market.adp) == {"e-f|TE"}


def test_missing_adp_falls_back_to_vor_rank_not_points_rank(cuomo_config):
    """The fallback has to look like a draft board, not a scoring leaderboard.

    Raw points order drafts every quarterback far too early: a QB outscores a
    running back without being worth more than the next quarterback.
    """
    import csv
    import dataclasses

    from ffdraft.data import build_board_for

    config = dataclasses.replace(cuomo_config, my_seat=1)
    board = build_board_for(
        config,
        "data/samples/projections_synthetic.csv",
        None,                       # no market file at all
    )
    assert board.adp_imputed.all()

    order = sorted(range(len(board)), key=lambda i: board.adp[i])[:10]
    positions = [board.pos_of(i) for i in order]
    # A pure points sort would put quarterbacks first; VOR order must not.
    assert positions[0] in ("RB", "WR"), positions
    assert positions.count("QB") <= 4, positions
    _ = csv
