# Draft-day runbook

Everything you need to run this without help. Start at the top.

---

## The 60-second version

```bash
Rscript R/pull_projections.R --season 2026        # once, before draft day
ffdraft check  --league league2 --seat 5          # must print READY
ffdraft board  --league league2 --top 30          # eyeball it
ffdraft draft  --league league2 --seat 5          # draft day
```

Replace `5` with your actual draft slot. `--league cuomo` for League 1.

**`zsh: command not found: ffdraft`?** Install it once, in the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Then give yourself the short name, running this **from the repo root** so
`$PWD` resolves to the real path:

```bash
grep -q 'alias ffdraft=' ~/.zshrc || echo "alias ffdraft=\"$PWD/.venv/bin/ffdraft\"" >> ~/.zshrc
source ~/.zshrc
```

Or, if you want a real command that also works in scripts:

```bash
sudo ln -sf "$PWD/.venv/bin/ffdraft" /usr/local/bin/ffdraft
```

Either works from any directory - the installed script's shebang points at the
venv's Python, so it finds numpy wherever you call it from.

Don't rely on `source .venv/bin/activate`. On draft day you will open a fresh
terminal, forget to activate, and hit `command not found` mid-draft.

Skipping the install entirely? Every command works as
`PYTHONPATH=src python3 -m ffdraft.cli ...` from the repo root. (Plain
`python3 -m ffdraft.cli` does *not* work - this repo uses a `src/` layout, so
the package is not importable until it is installed or `src` is on the path.)

---

## 1. Before draft day

### 1.1 Pull projections — 5 to 15 minutes

```bash
Rscript R/pull_projections.R --season 2026
```

Writes `data/projections_2026.csv` and `data/market_2026.csv`. The last two
lines tell you whether it worked:

```
wrote data/projections_2026.csv: 2847 rows, 312 players, 9 sources
wrote data/market_2026.csv: 298 rows
```

| What you see | Verdict |
|---|---|
| 250+ players, 8–9 sources | Good. Move on. |
| 150–250 players | Usable but thin. Check `ffdraft check` passes. |
| Under 150 players | Not enough. See Troubleshooting. |
| No market file written | Fine — ADP gets imputed from projection rank, and the board flags those players with `*`. Survival numbers get less reliable. |

Re-run any time to refresh. It overwrites.

### 1.2 Confirm all nine sources arrived — 5 seconds

```bash
ffdraft sources --league league2
```

`scrape_data` skips a source that errors and carries on, so a failed source
does not announce itself — it just never appears in the file. This prints a
per-source, per-position table and names anything requested but absent.

What to look for:

| Output | Meaning |
|---|---|
| `all 5 sources present with comparable coverage` | Good. |
| `MISSING (2): FantasyData, RTSports` | Those failed. Re-run the pull; if one keeps failing, drop it from the `sources` vector in `R/pull_projections.R`. Seven good sources beat nine where two are broken. |
| `THIN (marked !): CBS/WR` | Present but covering a fraction of its peers. The engine equal-weights whatever it finds, so this shifts the average at that position only. |
| `LOW  RB  12 distinct players (replacement rank 30)` | Not enough players to reach replacement level. VOR at that position is meaningless until fixed. |

Four or five healthy sources is fine. The equal-weighted average is robust and
the gain is in averaging *at all*, with sharp diminishing returns after a few;
it is a source that is *half* there that quietly distorts things.

The pull asks for five, not the brief's nine. Four were dropped after a real
2026 pull showed each failing identically on every position and every re-run:
FantasyData is behind a paywall, FleaFlicker and NumberFire/FanDuel return no
season-long data, and fantasy.nfl.com changed its page structure in a way
ffanalytics' parser does not handle. Re-add them if a later release fixes it.

The **scale check** at the bottom is the one that catches a source on a
different basis — per-game instead of per-season, say. It compares sources only
on players they both cover, so a source carrying just the top ten is not
penalised for having a high median.

### 1.3 Preflight — 5 seconds

```bash
ffdraft check --league league2 --seat 5
```

Must end with `READY`. It prints your league shape, the board depth per
position, and the replacement baselines. **Read the `starters` line** and
confirm it matches your league settings page. If it doesn't, everything
downstream is wrong.

`NOT READY` prints a `FAIL` line saying exactly what's missing.

### 1.4 Sanity-check the board — 2 minutes

```bash
ffdraft board --league league2 --top 30
ffdraft board --league league2 --pos RB --top 20
```

You are looking for one thing: **does the top of the board look sane?** If a
player you know is injured or retired is sitting at #3, the projections are
stale and no amount of simulation fixes that. Fix it with `out` during the
draft (section 3.3), or re-run the pull.

`*` next to an ADP means it was imputed, not from the market.

---

## 2. Draft day

```bash
ffdraft draft --league league2 --seat 5
```

### The one rule

**Type every pick, including everyone else's.** The engine's survival model,
opponent model, and replacement level all depend on knowing the full board. If
you only enter your own picks, the numbers become fiction — silently, with no
error.

When your turn comes up, the recommendation appears **automatically**. You
don't have to ask for it.

### Console commands

| Type this | It does |
|---|---|
| `3` | Takes numbered player #3 — **the fast path, use this** |
| | On *your* pick that is the engine's 3rd choice; on someone else's it is the 3rd likely pick |
| `bijan gone` | Records a pick by whoever is on the clock |
| `me josh allen` | Records the pick as yours |
| `go` | Re-run the recommendation for your current pick |
| `undo` | Take back the last pick |
| `log` / `log 20` | Recent picks **with their numbers** |
| `fix 40 josh allen` | Correct pick 40, leaving every other pick alone |
| `insert 40 josh allen` | Add a pick you missed at 40; everything after shifts down |
| `drop 40` | Remove a pick that never happened; everything after shifts up |
| `roster` / `roster 3` | Show your roster / seat 3's roster |
| `board` / `board RB` | Best available overall / at a position |
| `out <name> : <reason>` | Rule a player out for the season |
| `bump <name> 0.8 : <reason>` | Scale a projection (0.8 = 20% haircut) |
| `save` | Write the pick log now (it also auto-saves every pick) |
| `help` | Reprint this list |
| `quit` | Exit (saves first) |

Names are fuzzy. `bijan`, `lions d`, `jamarr`, `harrison jr` all work.

The same table applies to `ffdraft mock` — both consoles dispatch through one
function, so practice and draft day cannot drift apart.

### Fixing a pick you got wrong

`undo` only reaches the **last** pick. The usual mistake is noticing at pick 45
that pick 40 went in as the wrong Josh — undoing five picks to fix one, under a
clock, makes the log worse.

```
log            # find the number
fix 40 josh allen
```

It refuses to create a duplicate, refuses a pick number that isn't in the log,
and records what it changed in the pick log. Everything downstream —
replacement level, survival, your roster — recomputes from the corrected state.

### A pick you missed entirely

`fix` swaps a player at a pick that exists. If a pick never got entered at all,
the count itself is wrong: every later pick sits one slot early, and since the
seat that owns a pick comes from its number, the whole tail is on the wrong
team. That is what `insert` is for.

```
log                      # find where the gap is
insert 40 josh allen     # everything from 40 on shifts down one
```

`drop 40` is the mirror, for a pick that got entered twice. Both rewrite the
pick numbers and seats in the saved log, and both refuse a change that would
duplicate a player or push a pick past the end of the draft.

The symptom that sends you here is usually `! pick count mismatch`, or a
`roster <seat>` that has someone else's player in it.

### Numbers work on your own pick too

When it is your turn the console prints the three ranked candidates and a
reminder:

```
  #  player                          VOR   surv  P(title)   delta
  1  Bijan Robinson (RB, ATL)      127.4     1%    13.00%   +0.00
  2  Jahmyr Gibbs (RB, DET)        116.8     4%    12.10%   -0.90
  3  Puka Nacua (WR, LAR)          104.9    22%    11.60%   -1.40

  type 1-3 to take one, or type a name
```

Type `2` and it drafts Gibbs to your roster. `--show 5` offers five instead.

Numbers are always scoped to the list currently on screen — the engine's
ranking on your pick, the likely-pick list on someone else's. A list never
survives past the pick it was built for.

### The numbered list — use it

Before every opponent pick the console prints the ten players most likely to go
next, with probabilities:

```
  likely for seat 2 (type the number; 93% of the time it is one of these)
    1  Quentin Carter           RB  WAS    22%
    2  Roman Harrison           RB  HOU    18%
    3  Troy Thomas              RB  CAR    13%
    ...
```

Type `3` and it records Troy Thomas. One keystroke instead of a name.

The real pick is on that list about 90% of the time in the early rounds. When
it isn't, type the name as usual — both always work.

The list comes from the same opponent model the recommender uses, so it
accounts for who is already gone, what that specific seat still needs, and
whether a positional run is underway. It is not just ADP order.

Turn it off with `--no-suggest`, or change the length with `--suggest 15`.

### Timing

A recommendation takes **~13 seconds** (League 1) or **~15 seconds** (League 2)
at the default 2,500 simulations.

If your clock is tight, start the console with fewer sims:

```bash
ffdraft draft --league league2 --seat 5 --sims 1000    # ~6 seconds
```

Fewer sims means more candidates land in the "statistical tie" bucket, which is
handled explicitly — see section 3.2. It does not make the engine wrong, just
less able to separate close calls.

**Enter opponent picks as they happen, not in batches.** The recommendation
starts computing the moment the pick before yours is entered, so it's ready
when the clock reaches you.

---

## 2b. Rehearse before Thursday

Two different things, and you probably want the first.

### Rehearsal against a real Sleeper mock — recommended

Open a mock draft on Sleeper (or wherever), and run the console alongside it,
typing in **every** pick as it happens. This is the real thing with fake
stakes: same commands, same clock pressure, same typing.

```bash
ffdraft draft --league league2 --seat 5 --out logs/rehearsal_1.jsonl
```

`--out` is the only difference from draft day. It keeps the rehearsal out of
`logs/draft_league2.jsonl`, which is the file you want clean for the real
thing.

Set `--seat` to whatever slot the mock gives you. If you don't know it until
the draft starts, start the console once you do.

What you are rehearsing:

1. Typing opponent picks fast enough to keep up — this is the whole skill
2. What an ambiguous name looks like, and how you resolve it (`josh QB`)
3. Recovering from a missed pick without panicking (`undo`, `roster 7`)
4. Reading the four-line card in the two seconds you will actually have

Do it once. Twenty minutes. It is worth more than any amount of reading.

### Solo practice with no draft room — `ffdraft mock`

If you just want to see the engine build a roster, or practice the commands
with nobody else involved, the opponent model can fill the other seats:

```bash
ffdraft mock --league league2 --seat 5              # you pick, it drafts the rest
ffdraft mock --league league2 --seat 5 --auto       # it picks for you too, ~20s
```

At each of your picks: press **Enter** to take the engine's #1, type `2` to
take its second choice, or type a name to take someone else. **Every console
command above works here** — `roster 3`, `log 20`, `fix 12 josh allen`, `undo`,
`out <name> : <reason>`, `board RB`, `go`. That is the point of the mode: a
command you reach for under a clock should be one you have already used. `auto`
hands the rest of the draft to the engine.

What it does *not* rehearse is the part that actually goes wrong on draft day,
which is keeping up with entering other people's picks — here they enter
themselves. Use it to learn the commands and sanity-check the engine; use a
real mock draft room to practise typing.

---

## 2.5 After the draft — trades

```bash
ffdraft trades --league league2 --log logs/draft_league2.jsonl --surplus
```

`--surplus` prints the position-by-position map first: who is long, who is
short. That is where the trades come from, and it is worth reading before the
ideas so you know why each one exists.

Each idea shows what you send, what you get, what your starting lineup gains,
**what theirs gains** — that is the reason they say yes — the effect on your
title odds, and your worst bye week after the deal.

Read the `SELL` line before you send anything. It compares the draft position of
what you are giving up against what you are asking for. A trade the numbers love
and the other owner reads as a robbery does not get accepted, and the tool says
so rather than letting you find out.

`--partner 5` narrows it to one owner. `--max-give 1 --max-get 1` restricts it
to straight one-for-ones, which are far easier to get agreed.

---

## 2.6 During the season — setting a lineup

```bash
Rscript R/pull_projections.R --season 2026 --week 3      # pull, every week
ffdraft lineup --league league2 --log logs/draft_league2.jsonl \
    --weekly data/weekly_2026_w03.csv --week 3 --opponent 5 --sources
```

**Pull it the morning you set the lineup, not the night before.** The tool
refuses data older than 24 hours, and refuses anything older than 3 hours when
one of your players is QUESTIONABLE or DOUBTFUL — those tags resolve about 90
minutes before kickoff and are the most valuable thing you will learn all week.
The refusal is deliberate: a warning printed above a lineup gets read after the
lineup.

`--sources` shows which sources survived and how old each is. A source past the
limit is dropped rather than averaged in, so a five-source consensus can quietly
become a two-source one — this is how you see that. `--min-sources 3` makes it
an error instead.

`--opponent <seat>` changes what "best" means: without it the lineup maximises
projected points, with it it maximises your chance of beating that specific
team. Those differ. Against a much stronger opponent it will start the volatile
player over the steady one, because losing by less is still losing.

If it refuses and you disagree, `--allow-stale` proceeds and prints exactly what
it overrode.

---

## 3. Reading the output

```
PICK: Bijan Robinson (RB, ATL)
EDGE: +11.9 VOR over Jahmyr Gibbs (title odds tied)
WHY:  last RB in his tier; 4% to last to your next pick
FLAG: no market ADP; survival is an estimate
```

### 3.1 What the numbers mean

- **VOR** — projected points above the replacement-level player at that
  position, given the live board. Bigger is better, but it is *not* what the
  ranking is based on.
- **survival** — probability the player is still there at your next pick. Low
  survival is the argument for taking him now.
- **P(title)** — the actual objective. Probability you win the league if you
  take this player, from simulating the rest of the draft and the whole season.

The ranking is by P(title). VOR is shown because it's interpretable; it can be
negative in the EDGE line and that is not a bug.

### 3.2 "title odds tied" — the most important thing to understand

P(title) is estimated from simulations, and the gap between the top few
candidates is often smaller than the margin of error. When that happens the
engine says so and breaks the tie by **cost of waiting** — how much value you
expect to lose by passing and hoping he comes back.

**When you see "title odds tied", the #2 option is genuinely almost as good.**
That is exactly where your own judgment is cheapest to apply. If you know
something about a player that the projections don't, this is the moment to use
it.

When you *don't* see it, the engine has separated them statistically, and
overriding is a real bet against the math.

### 3.3 When you disagree

The engine does not know about this morning's news. Feed it in rather than
overriding in your head:

```
out mike evans : ruled out week 1, hamstring
bump james cook 0.75 : limited in practice all week
```

Both take effect immediately, change every subsequent number, and go in the
audit log with your reason. Then run `go` for a fresh recommendation.

If you decide to override anyway, just draft the player and type him in. The
engine will keep working from the real state.

---

## 4. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `! 'josh' matches multiple players` | Ambiguous name | Add the position: `josh QB`, or type the full name |
| `! no board match for 'x'` | Name not on the board | `board <POS>` to find the exact spelling |
| `! pick count mismatch` | A pick wasn't entered | `roster <seat>` to find the gap, then enter the missing pick or `undo` |
| `! X was already taken at pick N` | Duplicate | Nothing to do — the engine already refused it |
| `NOT READY: only 12 RB on the board but replacement rank is 30` | Thin projections | Re-run the pull; try `--pool 400`; check the pull's row counts |
| `no projections at data/projections_2026.csv` | Pull never ran | Run section 1.1, or pass `--projections <path>` |
| Recommendation too slow | Default 2,500 sims | Restart with `--sims 1000` |
| Everything looks wrong | Wrong league or seat | `ffdraft check` and read the `starters` and `seat` lines |

### If the pull returns too few players

Try one source at a time to find the broken one:

```bash
Rscript -e 'library(ffanalytics); x <- scrape_data(src="CBS", pos="RB", season=2026, week=0); nrow(x$RB)'
```

Swap `CBS` for `ESPN`, `FFToday`, `NumberFire`, `NFL`. Any source returning
60+ RBs is healthy. Then edit the `sources` vector near the top of
`R/pull_projections.R` to keep only the healthy ones and re-run. Six good
sources beat nine where three are broken.

### Emergency fallback

If the projections are unusable an hour before the draft, the engine still runs
on the synthetic board — but **that is fake data and will give you fake
advice**. Use `ffdraft board` off a real source and draft manually instead.

---

## 5. After the draft

The pick log is at `logs/draft_<league>.jsonl`. **Keep it.** It is:

1. The backtest dataset for next season
2. The data to fit the opponent model to your actual leagues
3. The record of what the engine said, alongside `logs/audit.jsonl`

### Check the engine was deterministic

```bash
ffdraft replay --league league2 --log logs/draft_league2.jsonl --check
```

### Score the engine against what you actually did

Needs a CSV of realized season points (`player,pos,points`), so this waits
until the season ends:

```bash
ffdraft backtest --league league2 \
  --log logs/draft_league2.jsonl --actuals data/actuals_2026.csv
```

It re-drafts your seat with the engine, leaves the other eleven teams as they
really were, and compares the two rosters.

### Improve it for next year — in priority order

1. **Fit the flex shares** from your own logs. Count what positions actually
   filled W/R/T slots. Currently an assumption in `configs/leagues/*.yaml`.
2. **Replace `sim.weekly_cv`** with measured values from nflverse. Currently a
   placeholder.
3. **Fit the opponent model** (`adp_noise_picks`, `herding_multiplier`) to your
   leagues instead of the published priors.

---

## 6. What to trust, and what not to

**Trust:** the relative ordering of players, the survival probabilities, the
replacement-level arithmetic, and the fact that the same inputs always produce
the same answer.

**Treat with care:**

- **Absolute P(title) numbers.** ~10% in an 8-team league is roughly what
  chance gives you anyway. The *differences* between candidates are the signal;
  the levels are not calibrated against reality.
- **League 1's TE baseline.** The brief's value of 10 flatters every tight end
  by about a tier compared to what starter demand implies (~3). See
  `docs/baselines.md`.
- **Bye weeks.** If the projection pull returned no `bye` column, byes are 0
  and the simulator never sits anyone. Minor, but it slightly overrates players
  on the same bye.
- **`weekly_cv` and `injury_game_loss`.** Placeholders. They affect week-to-week
  noise, not the season-level spread that dominates the result.

**Doesn't exist:** trade modeling, in-season waiver advice, lineup-setting.
This is a draft tool.

---

## Command reference

```bash
ffdraft sources   --league <id>                     # per-source coverage
ffdraft check     --league <id> --seat <n>          # preflight
ffdraft board     --league <id> [--pos RB] [--top 30]
ffdraft baselines --league <id>                     # replacement-level arithmetic
ffdraft draft     --league <id> --seat <n> [--sims 1000] [--out <path>]
                  [--suggest 10 | --no-suggest]
ffdraft mock      --league <id> --seat <n> [--auto] [--seed N]
ffdraft replay    --league <id> --log <path> [--check]
ffdraft backtest  --league <id> --log <path> --actuals <path>
ffdraft export-r  --league <id>                     # regenerate the R scoring block
```

Every command takes `--projections <path>` and `--market <path>` to point at
files other than the defaults.

Leagues: `cuomo` (8-team superflex), `league2` (12-team).
