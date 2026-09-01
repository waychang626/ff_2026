"""Ingest raw stat lines, aggregate them, and score them once per league.

Section 5's selection rule governs this module: we take raw stat lines, never
pre-scored fantasy points, because no vendor scores these exact settings.

Aggregation order matters and is a deliberate choice: each source's stat line is
scored first, then the resulting *points* are averaged equal-weight. Averaging
stats first and scoring once is equivalent for every linear rule but not for the
DST points-allowed bracket, which is a step function. Scoring first also yields
the cross-source spread directly on the points scale, which is what the variance
model wants.

Equal weight, always. Source accuracy does not persist year to year: the simple
average beat the historically-weighted average 64% of the time (brief 3.3).
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import ScoringRules
from .ids import Player, make_player_id
from .scoring import score_stats, unknown_stats

# Columns that identify the row rather than describe production.
_META_COLUMNS = {"source", "player", "name", "pos", "position", "team", "bye", "player_id"}


@dataclass
class StatLine:
    source: str
    player_id: str
    name: str
    pos: str
    team: str
    bye: int
    stats: dict[str, float]


@dataclass
class PlayerProjection:
    """One player's aggregated, *uncalibrated* projection."""

    player: Player
    points: float                 # equal-weighted mean across sources
    source_sd: float              # spread of opinion across sources
    n_sources: int
    per_source: dict[str, float] = field(default_factory=dict)

    @property
    def player_id(self) -> str:
        return self.player.player_id


class IngestError(ValueError):
    pass


def _f(value: str | float | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# Values a CSV uses to mean "no number". R writes NA; pandas writes empty or nan.
_NULLISH = {"", "NA", "N/A", "NAN", "NONE", "NULL", "-", "--"}


def maybe_float(value: object) -> float | None:
    """Strict parse: None when the cell does not hold a real number.

    Deliberately not `_f`, which returns 0.0 for anything unparseable. That is
    right for a stat column - a missing rushing line really is zero - and
    catastrophic for ADP, where 0.0 is not "unknown" but "the very first pick".
    An R export whose ADP column is all NA turned every player into the
    consensus 1.01 and the draft-order model into a coin flip.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() in _NULLISH:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def load_stat_lines(path: str | Path) -> list[StatLine]:
    """Read a long-format CSV: one row per (source, player).

    Required columns: source, player, pos. Optional: team, bye.
    Every remaining column is treated as a stat and must be a name the scoring
    engine knows (see `ffdraft.scoring.KNOWN_STATS`); unrecognised columns are
    reported rather than silently ignored, because a source renaming a column
    would otherwise zero out a whole category without a word.
    """
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise IngestError(f"{path}: empty file")
        cols = {c.strip() for c in reader.fieldnames}
        for required in ("source", "pos"):
            if required not in cols:
                raise IngestError(f"{path}: missing required column '{required}'")
        if "player" not in cols and "name" not in cols:
            raise IngestError(f"{path}: needs a 'player' or 'name' column")

        rows: list[StatLine] = []
        unknown: set[str] = set()
        for lineno, raw in enumerate(reader, start=2):
            name = (raw.get("player") or raw.get("name") or "").strip()
            pos = (raw.get("pos") or raw.get("position") or "").strip().upper()
            if not name or not pos:
                continue
            stats = {
                k.strip(): _f(v)
                for k, v in raw.items()
                if k and k.strip() not in _META_COLUMNS
            }
            unknown |= set(unknown_stats(stats))
            try:
                player_id = make_player_id(name, pos)
            except ValueError as exc:
                raise IngestError(f"{path}:{lineno}: {exc}") from exc
            rows.append(
                StatLine(
                    source=(raw.get("source") or "").strip(),
                    player_id=player_id,
                    name=name,
                    pos=pos if pos in ("QB", "RB", "WR", "TE", "K", "DST") else "DST",
                    team=(raw.get("team") or "").strip().upper(),
                    bye=int(_f(raw.get("bye"))),
                    stats={k: v for k, v in stats.items() if k not in unknown},
                )
            )
    if unknown:
        raise IngestError(
            f"{path}: unrecognised stat column(s) {sorted(unknown)}. "
            f"Add them to ffdraft.scoring.KNOWN_STATS or drop them from the file - "
            f"scoring them as zero silently would misprice every affected player."
        )
    if not rows:
        raise IngestError(f"{path}: no usable rows")
    return rows


def aggregate(rows: list[StatLine], rules: ScoringRules) -> list[PlayerProjection]:
    """Score each source, then equal-weight average across sources."""
    grouped: dict[str, list[StatLine]] = defaultdict(list)
    for row in rows:
        grouped[row.player_id].append(row)

    out: list[PlayerProjection] = []
    for player_id, lines in grouped.items():
        per_source: dict[str, float] = {}
        for line in lines:
            # A source appearing twice for one player is a duplicate row, not
            # extra evidence; keep the first and let the count reflect reality.
            per_source.setdefault(line.source or "unnamed", score_stats(line.stats, rules))
        points = list(per_source.values())
        head = lines[0]
        out.append(
            PlayerProjection(
                player=Player(
                    player_id=player_id,
                    name=head.name,
                    pos=head.pos,
                    team=head.team,
                    bye=head.bye,
                ),
                points=statistics.fmean(points),
                source_sd=statistics.stdev(points) if len(points) > 1 else 0.0,
                n_sources=len(points),
                per_source=per_source,
            )
        )
    # Deterministic order: points desc, then PlayerID.
    out.sort(key=lambda p: (-round(p.points, 6), p.player_id))
    return out


@dataclass
class MarketData:
    """ADP and expert-consensus rank, keyed by PlayerID."""

    adp: dict[str, float] = field(default_factory=dict)
    adp_sd: dict[str, float] = field(default_factory=dict)
    ecr: dict[str, float] = field(default_factory=dict)
    rows_read: int = 0
    rows_without_adp: int = 0

    def __len__(self) -> int:
        return len(self.adp)

    @property
    def looks_empty(self) -> bool:
        """True when the file parsed but carried no usable draft order."""
        return self.rows_read > 0 and len(self.adp) < max(20, self.rows_read * 0.2)


def load_market(path: str | Path) -> MarketData:
    """Read ADP/ECR: columns player, pos, adp, and optionally adp_sd, ecr.

    A cell that is not a positive number is treated as missing, not as zero.
    The player then falls through to imputation, which at least preserves a
    sane ordering; a silent 0.0 would make him the consensus first overall
    pick.
    """
    path = Path(path)
    market = MarketData()
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            name = (raw.get("player") or raw.get("name") or "").strip()
            pos = (raw.get("pos") or raw.get("position") or "").strip().upper()
            if not name or not pos:
                continue
            try:
                pid = make_player_id(name, pos)
            except ValueError:
                continue
            market.rows_read += 1

            adp = maybe_float(raw.get("adp"))
            if adp is not None and adp > 0:
                market.adp[pid] = adp
            else:
                market.rows_without_adp += 1

            sd = maybe_float(raw.get("adp_sd"))
            if sd is not None and sd > 0:
                market.adp_sd[pid] = sd
            ecr = maybe_float(raw.get("ecr"))
            if ecr is not None and ecr > 0:
                market.ecr[pid] = ecr
    return market
