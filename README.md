# ffdraft — a draft assistant for season-long fantasy football

Hey Claude, help me win my fantasy leagues, make no mistakes

output example:
```
PICK: Bijan Robinson (RB, ATL)
EDGE: +11.9 VOR over Jahmyr Gibbs (title odds tied)
WHY:  last RB in his tier; 4% to last to your next pick
FLAG: no market ADP; survival is an estimate
```

---

## Quick start

```bash
pip install -e ".[dev]"

# 1. Pull projections (needs R >= 4.1 and ffanalytics — see below)
Rscript R/pull_projections.R --season 2026

# 2. Preflight. Run this the morning of, not five minutes before.
ffdraft check --league league2 --seat 5

# 3. Draft.
ffdraft draft --league league2 --seat 5
```

No projections yet? Every command works against the synthetic board:

```bash
python scripts/make_synthetic_board.py
ffdraft check --league cuomo --seat 3 \
  --projections data/samples/projections_synthetic.csv \
  --market data/samples/market_synthetic.csv
```

---

## What do you want to do?

Every action, by the thing you are trying to do. `<id>` is a league id from
`configs/leagues/` — currently `cuomo` or `league2`. `<n>` is your seat number,
which you can also set once as `draft.my_seat` in the league config and then
leave off every command below.

### Setting up, before draft day

| You want to | Type this |
|---|---|
| Install it | `pip install -e ".[dev]"` |
| Install it with the Claude console too | `pip install -e ".[dev,llm]"` |
| Get projections (needs R ≥ 4.1) | `Rscript R/pull_projections.R --season 2026` |
| Install the R side first | `Rscript R/setup.R` |
| Try everything with no real data | `python scripts/make_synthetic_board.py` |
| Get canonical player IDs and byes | `python scripts/fetch_sleeper.py` |

`R/pull_projections.R` also takes `--out <path>` and `--adp-out <path>`.
`scripts/make_synthetic_board.py` takes `--seed` and `--out-dir`;
`scripts/fetch_sleeper.py` takes `--cache`, `--out` and `--force`.

### Checking things are right

| You want to | Type this |
|---|---|
| Read your league config back, in plain English | `ffdraft rules --league <id>` |
| Compare two leagues' settings | `ffdraft rules --league <id> --diff <other-id>` |
| Confirm you are ready to draft | `ffdraft check --league <id> --seat <n>` |
| See if the projection pull got everything | `ffdraft sources --league <id>` |
| Understand where replacement level came from | `ffdraft baselines --league <id>` |
| Eyeball the board | `ffdraft board --league <id> --top 30` |
| Eyeball one position | `ffdraft board --league <id> --pos RB --top 20` |
| Send scoring rules to the R side | `ffdraft export-r --league <id>` |

`ffdraft check` must print `READY`. Run it the morning of, not five minutes
before.

### Drafting

| You want to | Type this |
|---|---|
| Run the live console | `ffdraft draft --league <id> --seat <n>` |
| Run it faster on a tight clock | add `--sims 800` |
| See more candidates on your pick | add `--show 5` |
| Turn off the likely-pick shortlist | add `--no-suggest` |
| Put the pick log somewhere specific | add `--out logs/mydraft.jsonl` |
| Keep an audit log of projection edits | add `--audit logs/audit.jsonl` |
| Drive it by talking to Claude instead | `ffdraft llm --league <id> --seat <n>` |

### Inside the console — the commands you type at the prompt

These are identical in `ffdraft draft` and `ffdraft mock`.

| You want to | Type this |
|---|---|
| **Record someone else's pick** | `bijan gone` — any messy name works |
| Record it faster | `3` — takes the 3rd player off the numbered list |
| Record a pick as yours | `me josh allen` |
| **See your own roster** | `roster` |
| **See an opponent's roster** | `roster 3` |
| See who has gone recently | `log` or `log 20` |
| See the best players left | `board` or `board RB` |
| Get the recommendation again | `go` (also `rec` or `.`) |
| Take back the last pick | `undo` |
| Fix a pick you got wrong four picks ago | `log` to find the number, then `fix 40 josh allen` |
| **Add a pick you missed entirely** | `insert 40 josh allen` — everything after shifts down |
| Remove a pick that never happened | `drop 40` — everything after shifts up |
| Tell the engine someone is out for the year | `out james cook : ACL, ruled out` |
| Tell it someone is dinged but playing | `bump james cook 0.75 : limited all week` |
| Write the pick log right now | `save` |
| List these commands | `help` (also `?` or `h`) |
| Stop | `quit` (also `exit` or `q`) |

Two things worth knowing about the number shortcut. On **your** pick it selects
from the engine's ranked candidates; on **someone else's** it selects from the
likely-pick shortlist. It is always scoped to the list currently on screen, and
a list never outlives the pick it was built for.

The reason on `out` and `bump` is not optional — it goes in the audit log, and
an unexplained edit to the board is indistinguishable from a typo a week later.

### Practising

| You want to | Type this |
|---|---|
| Practise against simulated opponents | `ffdraft mock --league <id> --seat <n>` |
| Take the engine's top pick | press **Enter** |
| Watch the engine draft a whole roster | `ffdraft mock --league <id> --seat <n> --auto` |
| Hand it the rest mid-draft | `auto` |
| Get the same mock twice | add `--seed 7` |

Every console command in the table above works here too. That is deliberate —
a command you reach for under a clock should be one you have already used.

### After the draft

| You want to | Type this |
|---|---|
| Re-run your draft through the engine | `ffdraft replay --league <id> --seat <n> --log logs/draft_<id>.jsonl` |
| Only replay the first few picks | add `--limit 5` |
| Prove the engine is deterministic | `ffdraft replay --league <id> --seat <n> --log <path> --check` |
| Check determinism harder | add `--runs 5` |
| Score how the engine would have drafted | `ffdraft backtest --league <id> --seat <n> --log <path> --actuals <path>` |

`replay` and `backtest` need `--seat` even though the pick log records it —
they read the seat from the config or the flag, not from the log header.

`--actuals` is a CSV of realised season totals: either a `player_id` column, or
`player` and `pos` columns, plus `points`.

### Trading, once the draft is over

| You want to | Type this |
|---|---|
| Find trades worth proposing | `ffdraft trades --league <id> --log logs/draft_<id>.jsonl` |
| See who is long and short at each position | add `--surplus` |
| Only look at one owner | add `--partner 5` |
| Only 1-for-1 offers | add `--max-give 1 --max-get 1` |
| Insist the other side gains more | add `--min-their-gain 20` |
| See more ideas | add `--top 10` |

This one reads your seat out of the log header, so `--seat` is optional.

### Flags that work on almost every command

| Flag | What it does |
|---|---|
| `--league <id>` | Which league. Required nearly everywhere. |
| `--seat <n>` | Your seat, overriding the config |
| `--sims <n>` | Simulations per pick. Lower is faster and noisier. |
| `--projections <path>` / `--market <path>` | Use specific data files |
| `--season <year>` | Which season's default data files to look for (default 2026) |
| `--pool <n>` | How many players to load onto the board (default 260) |

To run anything against the synthetic board, add:

```bash
--projections data/samples/projections_synthetic.csv \
--market data/samples/market_synthetic.csv
```

---

## The tool contract

```python
recommend_pick(
    league_id: str,
    drafted: list[PlayerID],    # every pick so far, in order
    my_roster: list[PlayerID],
    pick_number: int,
) -> list[Recommendation]       # ranked; VOR, survival, ΔP(title)
```

**No tuning parameters, and adding one is not a small change.** The failure mode
this guards against is not a crash — it is the engine slowly turning back into
the model's opinion, one helpful-looking keyword argument at a time. If a
parameter encodes an opinion it belongs in the league config, written before the
draft, not in a call composed under a 60-second clock.

- **Risk tolerance is derived.** Maximising P(title) buys variance when the
  roster is behind the field and sheds it when ahead — automatically, because
  that is what maximises P(title). `tests/test_variance_appetite.py` measures the
  effect in both directions and pins down that it declines monotonically as the
  roster improves.
- **Position need is derived.** It falls out of the lineup optimiser: a third QB
  in a two-QB league improves no lineup, so it scores zero. No flag had to say so.

`my_roster` is redundant with `drafted` plus the seat, and that is the point — it
is a cross-check. If the two disagree the engine stops rather than recommending
against a state that does not match the room.

The version exposed to the LLM goes one step further and takes **no arguments at
all**: the session already knows the league, the log, the roster and the pick
number.

---

## What the LLM does — exactly four jobs

1. **Parse.** "bijan gone", "they took the Lions D" → player IDs.
2. **Catch errors.** Duplicate picks, ambiguous names, pick-count mismatches.
   Refuse and ask rather than guess — two Joneses at 0.95 and 0.94 is exactly the
   case that silently corrupts a draft log.
3. **Feed in what the engine cannot see.** Late-breaking inactives go through
   `update_projection`, with a reason, into the audit log — never as narration.
4. **Explain the output.** One sentence.

It does not rank players. If it has information contradicting the ranking it
reports the engine's answer and adds a `FLAG`. A model that says "the script says
A but I'd take B" has become the decision-maker again, with worse arithmetic and
no audit trail.

---

## Architecture

```
projections.csv ─┐
                 ├─► scoring ─► calibration ─► board ─┐
market.csv ──────┘   (§3.3)      (§3.3)               │
                                                       ├─► recommend_pick
league.yaml ─► baselines (§3.2) ─► VOR + tiers ────────┤        │
                                                       │        ▼
                        opponents (§3.4) ─► rollout ───┘   audit.jsonl
                                              │
                                              └─► season Monte Carlo (§3.1)
                                                    P(week) P(playoffs) P(title)
```

| Module | What it owns |
|---|---|
| `ids.py` | Player identity, messy-name resolution, ambiguity refusal |
| `config.py` | League configs. Every opinion the engine holds is in one of these |
| `scoring.py` | Stat line → points, per league. Also generates the R scoring block |
| `projections.py` | Ingest raw stat lines; equal-weight aggregation |
| `calibration.py` | §3.3 corrections, and the variance model derived from R² |
| `baselines.py` | Replacement level, both definitions, with the arithmetic shown |
| `vor.py` | VOR, tiers, and team need (`roster_marginal`) |
| `lineup.py` | Optimal starting lineup — greedy, provably, because slots nest |
| `opponents.py` | §3.4 opponent model and the draft rollout |
| `simulate.py` | Season Monte Carlo → P(weekly win), P(playoffs), P(title) |
| `engine.py` | `recommend_pick` and the guards around it |
| `trades.py` | Post-draft trade search: surplus filter, then the season sim |
| `replay.py` | Replay harness and backtester |
| `audit.py` | `(state_hash, output, timestamp)` for every call |
| `cli.py` | The draft-day console |
| `llm/` | The orchestration layer, last and thinnest |

---

## How the pick is actually chosen

The LLM never sees any of this. One function does the whole thing, the same way
every time.

### Before the draft: build the board once

```
raw stat lines (5 sources)
  │  equal weight — source accuracy does not persist year to year
  ▼
mean stat line per player
  │  apply the league's scoring rules
  ▼
projected points
  │  §3.3 corrections, as one transform:
  │    · shrink toward the positional mean   (QB ×0.67 … WR ×0.85)
  │    · subtract optimism                   (−21.6, QB −46.5)
  │    · derive an outcome sd from R² = 0.20
  ▼
each player: a mean AND a spread          ← the spread is the important half
  │  join ADP; impute from VOR rank where missing
  ▼
Board (frozen; changes only via a logged projection update)
```

The board is built once and held fixed. That is what makes a replay meaningful:
if the board could drift between calls, identical picks need not produce
identical recommendations.

### On every pick: six steps

**1. Validate, then seed.**
The pick log, your roster and the pick number are cross-checked against each
other; a disagreement stops the engine rather than producing a plausible wrong
answer. Then

```
state_hash = sha256(league config, board, picks so far, seat, pick number)
seed       = config.sim.seed XOR state_hash
```

Same state, same seed, same answer — always.

**2. Shortlist ~12 candidates.**
Simulating all 250 remaining players is pointless; most cannot be the answer.
Candidates are ranked by an equal blend of

- **VOR** — points above the replacement-level player at that position, *given
  the live board*. As players come off, replacement slides down with them.
- **marginal** — how much the player improves your best legal starting lineup.
  This is where "position need" comes from: a third QB in a two-QB league
  improves nothing, so it scores zero. Nobody had to pass a flag.

Neither alone is enough — VOR alone over-drafts positions you have filled,
marginal alone chases whatever slot happens to be empty.

**3. Apply the one hard policy guard.**
No kicker or defense before its configured round, and never more than the
lineup starts. This is §3.5 — the roster slot that separates the builds that
beat 50% from the ones that do not. It releases automatically when your
remaining picks equal the mandatory slots you still have to fill, so the roster
is always legal at the end. Nothing else is constrained; capping RB or WR would
be choosing your build for you, which §3.6 forbids.

**4. Simulate the rest of the draft, once per candidate.**
Each simulated team gets a *persistent* private ranking of the board — ADP plus
one draw of noise, fixed for the whole draft. Teams have consistent
preferences; re-randomising every pick would model a league of amnesiacs and
would wash out exactly the runs we want to catch. Each team then takes the best
player left that its roster can still use.

Two behaviours are layered on, both from §3.4:

- **Herding.** When the previous pick was a QB (early) or a K/DST (any time),
  that position gets pulled up the board. Implemented as a rank bonus, because
  multiplying a position's odds by *m* under an ADP-plus-noise ranking is the
  same as moving it `log(m) × sd` picks earlier — about 7 picks here, which is
  what a run actually looks like.
- **Handcuffing: absent on purpose.** 793 teams with a handcuff pair won
  51.04% against 50.56% without — a Bayes factor of 4.2 *favouring no
  difference*. A test walks the syntax tree to keep it out.

**5. Simulate the season, once per candidate.**
For every simulated draft, every player draws a season outcome from a lognormal
matched to his mean and spread — right-skewed, because fantasy outcomes are and
because with half the field making the playoffs the ceiling is worth more than
a symmetric distribution would price it. That total is spread across the weeks
he is active (byes removed), with weekly noise on top and a chance of a
season-ending injury at a random week.

Then every team's best legal lineup is filled each week — greedy from the most
restrictive slot outward, which is *provably* optimal here because the slots
nest (QB ⊂ Q/W/R/T, WR ⊂ W/R/T ⊂ Q/W/R/T), and a test checks it against brute
force. Weekly scores go head-to-head on a fixed round-robin schedule, standings
seed a single-elimination bracket, and the bracket produces a champion.

Out of one pass: **P(weekly win)**, **P(playoffs)**, **P(title)**.

This is also where risk appetite comes from, and why there is no setting for it.
Maximising P(title) buys variance when your roster is behind the field and
sheds it when you are ahead — not because anyone coded that, but because that
is what maximises P(title). Measured in both directions in
`tests/test_variance_appetite.py`.

**Common random numbers.** Opponent preferences and player outcomes are drawn
*once* and reused for every candidate, so the candidates are compared in the
same simulated seasons. Without this the differences between them would be
mostly luck.

**6. Rank — and refuse to invent precision.**
P(title) is the objective, but it is estimated from a binary outcome and the
gap between the top few candidates is routinely smaller than its standard
error. Sorting on it anyway would give a ranking that reshuffles on a different
seed: decisive-looking, and arbitrary.

So candidates the simulation *cannot separate* from the leader — within two
paired standard errors — are collected into a tie group and ordered instead by
**cost of waiting**:

```
(1 − P(he survives to your next pick)) × (his VOR − the VOR you expect
                                           to still be there at his position)
```

Among options that win the title equally often, take the one you are most
likely to lose. That is the dynamic program's own logic, applied exactly where
the Monte Carlo runs out of resolution. The card says `title odds tied` when
this happened, which is your signal that your own judgment is cheap to apply.

**Survival is measured honestly.** A candidate's survival has to come from a
world where you did *not* take him — in his own rollout he is gone by
construction. So each candidate is measured in the leader's rollout, and the
leader in the runner-up's.

### What comes back

The top three with VOR, survival, P(title), and the paired standard error of
each delta — plus the four-line card. Every call appends
`(state_hash, output, timestamp)` to `logs/audit.jsonl`.

## Determinism

The simulation seed is derived from a hash of the league config, the board, the
pick log, the seat and the pick number. Identical state produces an identical
ranking, and the replay harness asserts it:

```bash
ffdraft replay --league cuomo --log logs/draft_cuomo.jsonl --check
```

Every call appends `(state_hash, output, timestamp)` to `logs/audit.jsonl`. The
board is built once and held fixed; news enters only through
`Board.apply_update`, which records what changed and changes the fingerprint.

---

## Draft-day runbook

1. **Check the config against the settings page.** `ffdraft rules --league <id>`
   prints the whole config the way Yahoo or Sleeper reads it. This is the
   highest-value check here — a wrong reception value misprices every receiver
   and nothing downstream looks odd. It has already caught one real error (see
   the changelog).
2. **Morning of.** `ffdraft check --league <id> --seat <n>`. It must print
   `READY`. Then `ffdraft sources --league <id>` to confirm the pull got what
   it asked for.
3. **Sanity-read the board.** `ffdraft board --league <id> --top 30`. If a name
   near the top is wrong, the projections are wrong, and no amount of simulation
   fixes that.
4. **Draft.** `ffdraft draft --league <id> --seat <n>`. Type every pick as it
   happens — including everyone else's. The recommendation appears automatically
   when you are on the clock. `?` lists the commands.
5. **Afterwards.** The pick log in `logs/` is the first backtest dataset and the
   seed for the opponent model. Free to collect live, expensive to reconstruct.

A recommendation takes about 13 s in League 1 and 15 s in League 2 at 2,500
sims. Lower it with `--sims` if your clock is tighter than that.

---

## Leagues

|  | League 1 "Cuomo" | League 2 |
|---|---|---|
| Teams | 8 | 12 |
| Starters | QB WR WR WR RB RB TE W/R/T **Q/W/R/T** K DEF | QB RB RB WR WR TE W/R/T K DEF |
| Bench / IR | 6 / 2 | 4 / 1 |
| Draft | 17 rounds, 136 picks | 13 rounds, 156 picks |
| Playoffs | 4 of 8, weeks 16–17 | 6 of 12, weeks 15–17 |
| Baselines | QB 17 · RB 20 · WR 29 · TE 10 · K 9 · DST 9 | QB 13 · RB 30 · WR 31 · TE 14 · K 13 · DST 13 |

Skill-position scoring is identical, so **one projection set serves both**. All
differences are in K and DST — the two positions being punted anyway.

Two things about this table are worth staring at:

- **League 1's superflex is the whole story at QB.** With 4-pt passing TDs a
  second quarterback beats a flex-level WR comfortably, so assume all 8 teams
  start two. It makes QB *deeper in the smaller league* — 17 against League 2's
  13 — which is the one place team count points the wrong way.
- **The two boards are not related by a simple shift.** League 2 is deeper
  everywhere else, but by wildly different amounts: RB moves 10 ranks (20→30)
  while WR moves 2 (29→31), because League 1 requires three receivers and
  League 2 requires two. A running back who is a comfortable starter in League 1
  is replacement level in League 2; a receiver is worth about the same in both.

> **The brief had League 1's roster wrong** — it described no TE slot and two
> W/R/T flexes, where the settings page has a TE slot and one W/R/T. Its RB 23 /
> WR 33 baselines counted eight flex spots that do not exist. Corrected here
> after checking the live settings; `ffdraft rules --league cuomo` prints the
> config back in settings-page form so this is checkable in a minute.

---

## Data

Raw stat lines only, never pre-scored fantasy points — no vendor scores either
league's exact settings, so we pull yards/TDs/receptions/INTs once and apply the
scoring function twice.

- **ffanalytics (R)** is the primary aggregator. `R/pull_projections.R` writes
  the ingest CSV. Its scoring block is generated from the YAML —
  `ffdraft export-r --league cuomo` — so the Python engine and the R pull cannot
  drift apart.
- Never pass `avg_type = "weighted"`. Source accuracy does not persist year to
  year; the simple average beat the historically-weighted one 64% of the time.
  The config parser rejects `aggregate: weighted` outright.
- ffanalytics' default `vor_baseline` assumes a 12-team league. It is wrong for
  both of these. The engine always computes its own.

---

## Verified vs. not

Built and tested in a container with no R and no outbound access to the
projection sources, so:

**Verified** — the whole Python engine, end to end, against a synthetic board:
263 tests, determinism, greedy-vs-brute-force lineups, the calibration
arithmetic, the opponent model's herding, the endogenous variance appetite, both
league configs, the replay harness and the backtester.

**Not executed here** — `R/pull_projections.R` (no R) and
`scripts/fetch_sleeper.py` (no network). Both follow the documented APIs; run
each once locally before draft day. The R script's verification snippet is at
the bottom of the file.

**Placeholders, flagged in the configs** — `sim.weekly_cv` and
`sim.injury_game_loss` are estimates pending nflverse historical variance. They
affect week-to-week noise, not the season-level spread that dominates the
result.

---

## Testing

```bash
python -m pytest              # 263 tests, ~95s
python -m pytest tests/test_variance_appetite.py -v   # the §3.1 behaviour
```

The tests worth reading first, because they are the design:

- `test_engine.py::test_signature_has_no_tuning_parameters`
- `test_variance_appetite.py` — risk appetite emerging, not configured
- `test_lineup.py::test_greedy_matches_brute_force`
- `test_replay.py::test_replaying_the_same_log_twice_gives_identical_recommendations`
- `test_opponents.py::test_handcuffing_is_not_implemented_anywhere`

See `docs/research-map.md` for where each research finding lives in the code.

---

## Changelog

Newest first. Every push updates this section.

### Fix: the shortlist was over-drafting quarterbacks
Reported from a superflex mock that kept recommending back-to-back QBs. The
shortlist blended VOR with "how much this player improves your lineup" — but on
an empty roster every player improves the lineup by exactly his own projection,
so that second term silently *was* raw projected points, which is precisely the
cross-position bias VOR exists to remove. Four quarterbacks sat in the top six
while running backs with higher VOR ranked below them.

Lineup improvement is now measured against what a freely available player at
the same position would have contributed. On an empty roster it reduces exactly
to VOR; it diverges only where it should, when the slots a position can fill
are already taken. A superflex second QB still counts; a third scores zero.

### Numbers select on your own pick; fix a stale-shortlist bug
On your pick the console now lists three ranked candidates and takes `1`, `2`
or `3` for them (`--show 5` for more), the same one-keystroke path that already
existed for opponent picks.

Building it surfaced a latent bug worth more than the feature: the shortlist
variable persisted across turns, so on your own pick it still held the previous
opponent's list. A bare number would have silently recorded a player from a
list no longer on screen. Shortlists are now scoped to the pick they were built
for and are ignored the moment that pick passes.

### Fix: the first seat got no recommendation
Reported from a live mock. The console only recommended *after* recording a
pick, so whoever holds 1.01 opened it, saw the prompt, and got nothing —
`go` worked, but nothing said so. Recommending is now driven by the loop and
keyed on `(pick number, board fingerprint)`, which also fixes two cases that
were failing silently: `undo` and `fix` did not refresh, and ruling a player
out left the previous recommendation on screen as though it still held.
Covered by `tests/test_cli_draft_console.py`, which drives the console through
stdin — the bug was in *when* it acted, not in what the engine computed.

### Correct League 1's scoring against the settings page
Checked line by line against Yahoo. Ten of thirteen offensive rules already
matched. Added **missed PAT at −2**, which is real, expressible, and was simply
missing. Encoded three more the league scores but no projection source
publishes — offensive fumble return TD, 4th down stops, extra point returned —
so the config mirrors the settings page and `ffdraft rules` flags them as known
blind spots rather than leaving them to look like oversights. Together they are
worth ~5–8 points a season at positions the engine punts anyway.

Also expanded the README with a full walkthrough of how `recommend_pick`
decides.

### Correct League 1's roster; add `ffdraft rules`
The brief described League 1 as having **no TE slot and two W/R/T flexes**. The
live settings page has a **TE slot and one W/R/T**. Every baseline downstream
was wrong: RB 23→20, WR 33→29. TE stays at 10 but now *derives* rather than
being a judgement call, which also dissolves the "TE is inconsistent" finding
reported earlier — it was an artifact of the missing slot.

Added `ffdraft rules --league <id> [--diff <id>]`, which prints a config back in
settings-page form (`1 pt per 25 yds`, not `0.04`) so this class of error is
catchable in a minute. A config error is invisible downstream: every number the
engine produced from the wrong roster was internally consistent, well-tested,
and wrong.

### Fix a false positive in the scale check; drop four dead sources
The scale check compared median points per source, which measures coverage
depth rather than scale — it flagged a sound source at 1.93× purely for
covering only the top ten at each position. Now compares sources on the players
they share.

Dropped FantasyData, FleaFlicker, NumberFire and NFL from the pull after a real
2026 run showed each failing identically every time (paywall, no season-long
data, and a parser broken by an nfl.com page change). Five sources remain.

### `ffdraft sources`: prove the pull got what it asked for
A source that errors mid-scrape is skipped silently and simply never appears in
the CSV. Reports per-source, per-position coverage, names what is missing,
flags sources present but thin, and checks each position is deep enough to
reach its replacement rank.

### Fix ADP parsing as zero, which made draft order random
`load_market` ran ADP through a parser returning `0.0` for anything
unparseable. Correct for a stat column; catastrophic for ADP, where 0.0 means
"first overall pick". An export with an `NA` column made every player the
consensus 1.01 and the pick-order model pure noise — with no error. Now parses
strictly, and imputation falls back to **VOR rank** rather than points rank,
which produces a plausible draft order from projections alone.

### Numbered shortlist for opponent picks
Typing other people's picks is the slowest thing in a live draft and the one
place a typo silently corrupts the log. Before each opponent pick the console
lists the ten likeliest players with probabilities; type `3` to record the
third. Comes from the opponent model, so it accounts for who is gone, what that
seat still needs, and whether a run is on.

### Targeted pick correction: `log` and `fix <n> <name>`
`undo` only reaches the last pick, but the real mistake is noticing at pick 45
that pick 40 went in as the wrong Josh. `fix` swaps one recorded pick and
refuses anything that would corrupt the log.

### Fix: the shortlist collapsed in the rounds it was needed most
Reported from a live practice draft: by round 10 the likely-pick list was
showing one or two names instead of ten. `likely_next_picks` built its list from
`np.unique(argmin)`, which only returns players who *won* a simulation. When one
faller's ADP sits far enough ahead of everyone left he wins all 4,000 draws, so
`unique` returns a single row — and the list that exists to save you typing
disappeared exactly where the board is widest and you least remember who is
left. Now ranked by simulated probability and backfilled in ADP order, so it is
always as long as asked for while that many players remain legal. A name the
console is actively offering also no longer prints as `0%`, which read as a
broken list; below the simulation's resolution it says `<1%`.

### `insert` and `drop`: repair a log that lost its place
`fix` handles the wrong player at the right pick. A pick you never entered is a
different failure: every later pick is recorded one slot early, and because the
seat that owns a pick is derived from its number, the whole tail is attributed
to the wrong teams — wrong rosters, wrong replacement level, wrong
recommendations. `undo` can only reach that by discarding the correct picks
after it. `insert 40 josh allen` slots the missing pick in and shifts the tail
back into alignment; `drop 40` is the mirror, for a pick entered twice. Both
renumber the saved log and refuse anything that would duplicate a player or push
a pick past the end of the draft.

### Fix: every mock draft overwrote the last one's log
`mock` wrote to `logs/mock_<league>.jsonl` on every run, so a rehearsal you
wanted to compare against was already gone. Each run now gets its own
timestamped file — with a counter, because two mocks started inside the same
second collided and the test caught it. The live console keeps its stable
filename, since that is where the runbook says to look, but no longer truncates
an existing log: a pick log is the only record of the room and cannot be rebuilt
afterwards, so a previous one is moved aside and the console says where it went.

### `ffdraft trades`: what to offer, to whom, and why they would accept
A trade creates value out of positional imbalance, not out of one side being
fooled: your RB4 scores you nothing on Sunday, and the WR-heavy owner has the
mirror-image problem. The search is therefore a surplus search, run in the two
stages `engine.py` already uses — a cheap filter, then the expensive truth.

Stage 1 recomputes both teams' optimal starting lineups on point estimates for
every candidate trade (~164k on an eight-team board, under a second, because it
is the `lineup.py` primitive vectorised over candidates) and keeps only those
that raise *both*. Stage 2 runs the season Monte Carlo on the survivors and
reports the change in your P(title), on common random numbers so the differences
between trades are differences in the trades.

Stage 2 is where bye weeks and depth get priced: `draw_season` zeroes a player's
bye outright, so a trade that stacks three receivers into week 11 shows up as
lower title odds without anything special being done about it. Because a number
moving by half a point does not *say* what went wrong, every finalist is also
profiled week by week and a trade that deepens your worst week is flagged in
words.

Two things it refuses to pretend. Uneven trades force a cut, so a 1-for-2 is not
a free extra player. And the other owner does not run a lineup optimiser — they
value players by name and draft position — so every idea carries the ADP gap
next to the projection gap, and asking for the earlier-drafted name is labelled
a hard sell however good the arithmetic is.

### One console surface, so practice matches draft day
`mock` accepted a strict subset of the live console: no `log`, no `undo`, no
`fix`, `roster` with no seat argument, and a bare `2` resolved as a *player
name* rather than taking the engine's second choice. A rehearsal that teaches
commands the real thing does not have is worse than no rehearsal. Both consoles
now dispatch through one `_shared_command`, so the vocabularies cannot drift; a
parity test drives every verb through both. Also fixes `startswith("log")`
matching a pick for Logan, and `roster 99` reporting an empty seat instead of
saying the seat does not exist.

### Mock draft mode; fix a K/DST over-draft it exposed
`ffdraft mock` fills every other seat from the opponent model. Running it
immediately showed the engine filling its whole bench with kickers and
defenses: the round floors kept them out early but nothing capped the count,
and late on a backup kicker's VOR beats the 60th receiver's. Now capped at what
the lineup starts — and only K and DST, since capping everything would decide
the build for you.

### `docs/RUNBOOK.md`, `R/setup.R`
A standalone draft-day guide, and a one-command ffanalytics install that ends
with the brief's open-item verification.

### Initial build
The engine end to end in the brief's build order: replay harness first, then
projections, calibration, VOR, opponent model, Monte Carlo, `recommend_pick`,
and the LLM layer last. 263 tests.
