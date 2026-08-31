"""Apply a league's scoring rules to raw stat lines.

Section 5's selection rule: take raw stat lines, not pre-scored fantasy points.
No vendor scores these exact settings, so we pull yards / TDs / receptions /
INTs once and apply the scoring function once per league. This module is that
function.

Stat keys follow ffanalytics' column names so an R-side pull drops straight in.
"""

from __future__ import annotations

from collections.abc import Mapping

from .config import ScoringRules

# Stat keys the engine understands. Anything else in an input row is ignored,
# but `unknown_stats` will report it so a source-schema change is visible
# rather than silently zeroing a category.
KNOWN_STATS: frozenset[str] = frozenset(
    {
        # passing
        "pass_yds", "pass_tds", "pass_int", "pass_comp", "pass_att", "pass_300_yds",
        # rushing
        "rush_yds", "rush_tds", "rush_att", "rush_100_yds",
        # receiving
        "rec", "rec_yds", "rec_tds", "rec_tgt", "rec_100_yds",
        # misc offense
        "fumbles_lost", "fumbles_total", "two_pts", "return_tds", "return_yds",
        # kicking
        "xp", "xp_att", "xp_miss", "fg", "fg_att", "fg_miss",
        "fg_0019", "fg_2029", "fg_3039", "fg_4049", "fg_50", "fg_60",
        # team defense
        "dst_int", "dst_fum_rec", "dst_forced_fumble", "dst_sacks", "dst_safety",
        "dst_td", "dst_blk", "dst_ret_yds", "dst_pts_allowed", "dst_yds_allowed",
    }
)

# Points allowed is scored through the bracket, not through a multiplier.
_BRACKET_KEY = "dst_pts_allowed"


def bracket_points(pts_allowed: float, bracket: tuple[tuple[float, float], ...]) -> float:
    """Points for a DST given points allowed.

    Bracket semantics match ffanalytics: each entry is (threshold, points) and
    the award is the first entry whose threshold is >= points allowed. So
    `(0, 10)` is the shutout bonus and `(99, -4)` is the catch-all floor.
    """
    if not bracket:
        return 0.0
    for threshold, points in bracket:
        if pts_allowed <= threshold:
            return points
    return bracket[-1][1]


def score_stats(stats: Mapping[str, float], rules: ScoringRules) -> float:
    """Score one stat line. Missing stats count as zero."""
    multipliers = rules.multipliers
    total = 0.0
    for key, value in stats.items():
        if value is None or key == _BRACKET_KEY:
            continue
        weight = multipliers.get(key)
        if weight:
            total += float(value) * weight
    if _BRACKET_KEY in stats and stats[_BRACKET_KEY] is not None:
        total += bracket_points(float(stats[_BRACKET_KEY]), rules.pts_bracket)
    return total


def unknown_stats(stats: Mapping[str, float]) -> list[str]:
    """Stat keys the engine does not recognise. Surfacing beats silence."""
    return sorted(k for k in stats if k not in KNOWN_STATS)


def unscored_stats(stats: Mapping[str, float], rules: ScoringRules) -> list[str]:
    """Recognised keys this league assigns no value to.

    Useful as a pre-draft check: if `rec` shows up here in a half-PPR league,
    the config is wrong and every WR on the board is mispriced.
    """
    multipliers = rules.multipliers
    return sorted(
        k
        for k, v in stats.items()
        if k in KNOWN_STATS and k != _BRACKET_KEY and v and not multipliers.get(k)
    )


# --- R interop ---------------------------------------------------------------
# The YAML config is the single source of truth. ffanalytics needs the same
# rules as an R list; generating it from the YAML is the only way to guarantee
# the two never drift.

_R_SECTIONS = {
    "pass": ("pass_yds", "pass_tds", "pass_int", "pass_comp", "pass_att", "pass_300_yds"),
    "rush": ("rush_yds", "rush_tds", "rush_att", "rush_100_yds"),
    "rec": ("rec", "rec_yds", "rec_tds", "rec_tgt", "rec_100_yds"),
    "misc": ("fumbles_lost", "two_pts"),
    "ret": ("return_tds", "return_yds"),
    "kick": (
        "xp", "xp_miss", "fg_0019", "fg_2029", "fg_3039", "fg_4049",
        "fg_50", "fg_60", "fg_miss",
    ),
    "dst": (
        "dst_int", "dst_fum_rec", "dst_forced_fumble", "dst_sacks",
        "dst_safety", "dst_td", "dst_blk",
    ),
}

# ffanalytics applies these sections to every position unless told otherwise.
_ALL_POS_SECTIONS = {"rush", "rec", "misc", "ret"}


def _fmt(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return repr(round(value, 6))


def to_r_scoring_rules(rules: ScoringRules, var_name: str = "scoring_rules") -> str:
    """Render the scoring config as an ffanalytics `scoring_rules` R list."""
    values = rules.multipliers
    blocks: list[str] = []
    for section, keys in _R_SECTIONS.items():
        entries = [(k, values[k]) for k in keys if k in values]
        if not entries:
            continue
        parts = []
        if section in _ALL_POS_SECTIONS:
            parts.append("all_pos = TRUE")
        parts += [f"{k} = {_fmt(v)}" for k, v in entries]
        blocks.append(f"  {section} = list({', '.join(parts)})")

    if rules.pts_bracket:
        rows = ",\n".join(
            f"    list(threshold = {_fmt(t)}, points = {_fmt(p)})"
            for t, p in rules.pts_bracket
        )
        blocks.append(f"  pts_bracket = list(\n{rows}\n  )")

    body = ",\n".join(blocks)
    return f"{var_name} <- list(\n{body}\n)\n"
