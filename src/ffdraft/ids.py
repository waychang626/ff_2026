"""Canonical player identity and messy-name resolution.

The LLM's job #1 is turning "bijan gone" or "they took the Lions D" into a
PlayerID. Its job #2 is refusing when the input is ambiguous rather than
guessing. Both are served here: `Resolver.resolve` returns a *set* of
candidates with scores and an explicit `ambiguous` flag. The resolver never
picks for you when the top two candidates are close.

PlayerID format is `slug|POS` (e.g. `bijan-robinson|RB`). It is derived, stable,
and readable in a log. External IDs (Sleeper, FantasyPros) are carried
alongside as aliases so a board sourced from either joins cleanly.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")

# Suffixes are dropped for matching but kept in display names.
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Ways people type each defense in a draft room. Team code -> spoken forms.
_DST_NICKNAMES = {
    "ARI": ["cardinals", "arizona"], "ATL": ["falcons", "atlanta"],
    "BAL": ["ravens", "baltimore"], "BUF": ["bills", "buffalo"],
    "CAR": ["panthers", "carolina"], "CHI": ["bears", "chicago"],
    "CIN": ["bengals", "cincinnati"], "CLE": ["browns", "cleveland"],
    "DAL": ["cowboys", "dallas"], "DEN": ["broncos", "denver"],
    "DET": ["lions", "detroit"], "GB": ["packers", "green bay"],
    "HOU": ["texans", "houston"], "IND": ["colts", "indianapolis"],
    "JAX": ["jaguars", "jags", "jacksonville"], "KC": ["chiefs", "kansas city"],
    "LAC": ["chargers", "los angeles chargers"], "LAR": ["rams", "los angeles rams"],
    "LV": ["raiders", "las vegas"], "MIA": ["dolphins", "miami"],
    "MIN": ["vikings", "minnesota"], "NE": ["patriots", "pats", "new england"],
    "NO": ["saints", "new orleans"], "NYG": ["giants", "new york giants"],
    "NYJ": ["jets", "new york jets"], "PHI": ["eagles", "philadelphia"],
    "PIT": ["steelers", "pittsburgh"], "SEA": ["seahawks", "seattle"],
    "SF": ["49ers", "niners", "san francisco"], "TB": ["buccaneers", "bucs", "tampa bay"],
    "TEN": ["titans", "tennessee"], "WAS": ["commanders", "washington"],
}

# Tokens that carry no identifying information in draft-room speech.
_NOISE = {
    "gone", "taken", "off", "the", "board", "just", "went", "drafted", "picked",
    "pick", "they", "took", "got", "is", "was", "a", "d", "dst", "def", "defense",
    "st", "team", "his", "my", "i", "we", "there", "goes", "and", "to", "at",
}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_name(name: str) -> str:
    """Lowercase, de-accent, drop punctuation and name suffixes."""
    text = strip_accents(name).lower()
    text = re.sub(r"[.'`’]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    tokens = [t for t in text.split() if t not in _SUFFIXES]
    return " ".join(tokens)


def slugify(name: str) -> str:
    return normalize_name(name).replace(" ", "-")


def make_player_id(name: str, pos: str) -> str:
    pos = pos.upper()
    if pos in ("DEF", "D/ST", "D", "DEFENSE"):
        pos = "DST"
    if pos not in POSITIONS:
        raise ValueError(f"unknown position {pos!r}; expected one of {POSITIONS}")
    return f"{slugify(name)}|{pos}"


def split_player_id(player_id: str) -> tuple[str, str]:
    slug, _, pos = player_id.rpartition("|")
    if not slug or pos not in POSITIONS:
        raise ValueError(f"malformed PlayerID {player_id!r}")
    return slug, pos


@dataclass(frozen=True)
class Player:
    """A board entry. Immutable so a board can be hashed into the audit log."""

    player_id: str
    name: str
    pos: str
    team: str = ""
    bye: int = 0
    aliases: tuple[str, ...] = ()

    @property
    def display(self) -> str:
        team = f", {self.team}" if self.team else ""
        return f"{self.name} ({self.pos}{team})"


@dataclass
class Candidate:
    player: Player
    score: float
    reason: str


@dataclass
class Resolution:
    """Result of parsing one messy input.

    `ambiguous` is the signal the LLM must honour: when True it asks the human
    instead of picking. Ambiguity is decided by *margin*, not by absolute
    score - two Joneses at 0.95 and 0.94 are ambiguous even though both match
    well, and that is exactly the case that silently corrupts a draft log.
    """

    query: str
    candidates: list[Candidate] = field(default_factory=list)
    ambiguous: bool = False
    note: str = ""

    @property
    def best(self) -> Player | None:
        if not self.candidates or self.ambiguous:
            return None
        return self.candidates[0].player

    @property
    def found(self) -> bool:
        return bool(self.candidates)


class Resolver:
    """Name -> PlayerID over a fixed board.

    Deterministic: identical board plus identical query always yields the same
    ordering, ties included (broken by PlayerID).
    """

    # A match must clear this to be offered at all.
    MIN_SCORE = 0.62
    # Top two within this margin -> ambiguous, ask the human.
    AMBIGUITY_MARGIN = 0.08

    def __init__(self, players: list[Player]) -> None:
        self.players = list(players)
        self._by_id = {p.player_id: p for p in self.players}
        self._index: dict[str, list[Player]] = {}
        for player in self.players:
            for key in self._keys_for(player):
                self._index.setdefault(key, []).append(player)

    # -- indexing ---------------------------------------------------------
    def _keys_for(self, player: Player) -> set[str]:
        keys: set[str] = set()
        norm = normalize_name(player.name)
        keys.add(norm)
        tokens = norm.split()
        if tokens:
            keys.add(tokens[-1])  # last name
            keys.add(tokens[0])  # first name
            if len(tokens) > 1:
                keys.add(f"{tokens[0][0]} {tokens[-1]}")  # "j chase"
        for alias in player.aliases:
            keys.add(normalize_name(alias))
        if player.pos == "DST":
            for nick in _DST_NICKNAMES.get(player.team.upper(), []):
                keys.add(normalize_name(nick))
        return {k for k in keys if k}

    def get(self, player_id: str) -> Player | None:
        return self._by_id.get(player_id)

    def __contains__(self, player_id: object) -> bool:
        return player_id in self._by_id

    # -- resolution -------------------------------------------------------
    def resolve(self, query: str, pos_hint: str | None = None) -> Resolution:
        cleaned, hint_from_text = self._clean_query(query)
        pos_hint = (pos_hint or hint_from_text or "").upper() or None
        if not cleaned:
            return Resolution(query=query, note="no identifying tokens in input")

        pool = self.players
        if pos_hint in POSITIONS:
            pool = [p for p in pool if p.pos == pos_hint]
            if not pool:
                return Resolution(query=query, note=f"no {pos_hint} on the board")

        scored: list[Candidate] = []
        for player in pool:
            score, reason = self._score(cleaned, player)
            if score >= self.MIN_SCORE:
                scored.append(Candidate(player=player, score=score, reason=reason))

        # Deterministic order: score desc, then PlayerID asc.
        scored.sort(key=lambda c: (-round(c.score, 6), c.player.player_id))
        if not scored:
            return Resolution(query=query, note=f"no board match for {cleaned!r}")

        ambiguous = (
            len(scored) > 1
            and (scored[0].score - scored[1].score) < self.AMBIGUITY_MARGIN
        )
        note = ""
        if ambiguous:
            names = ", ".join(c.player.display for c in scored[:4])
            note = f"{cleaned!r} matches multiple players: {names}"
        return Resolution(query=query, candidates=scored[:5], ambiguous=ambiguous, note=note)

    def _clean_query(self, query: str) -> tuple[str, str | None]:
        raw = normalize_name(query)
        tokens = raw.split()
        pos_hint = None
        for token in list(tokens):
            upper = token.upper()
            if upper in POSITIONS:
                pos_hint = upper
        if any(t in {"d", "dst", "def", "defense", "st"} for t in tokens):
            pos_hint = "DST"
        kept = [t for t in tokens if t not in _NOISE and t.upper() not in POSITIONS]
        return " ".join(kept), pos_hint

    def _score(self, cleaned: str, player: Player) -> tuple[float, str]:
        keys = self._keys_for(player)
        if cleaned in keys:
            # Exact hit on full name beats an exact hit on a bare first name:
            # "josh" should not tie "josh allen".
            full = normalize_name(player.name)
            if cleaned == full or cleaned in {normalize_name(a) for a in player.aliases}:
                return 1.0, "exact name"
            tokens = full.split()
            if tokens and cleaned == tokens[-1]:
                return 0.93, "last name"
            return 0.88, "partial name"

        best, reason = 0.0, "fuzzy"
        for key in keys:
            ratio = difflib.SequenceMatcher(None, cleaned, key).ratio()
            if ratio > best:
                best, reason = ratio, f"fuzzy~{key}"
        # A clean prefix ("jeff" -> "jefferson") is stronger than raw edit ratio.
        for key in keys:
            if len(cleaned) >= 4 and key.startswith(cleaned):
                best = max(best, 0.85)
                reason = f"prefix~{key}"
        return best * 0.98, reason  # fuzzy never reaches an exact-match score
