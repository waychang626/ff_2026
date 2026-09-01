# Replacement level

ffanalytics' default `vor_baseline` — `QB 13, RB 35, WR 36, TE 13, K 8, DST 3` —
assumes a 12-team league and is wrong for both of ours. Always pass baselines
explicitly.

```bash
ffdraft baselines --league cuomo
ffdraft baselines --league league2
```

## Two definitions, and they are not the same question

**Starter demand** — `fixed + flex + 1`. The worst starter in the league. Answers
*what do I give up by not starting this player?*

**Drafted** — `fixed + flex + bench + 1`. The best free agent after the draft.
Answers *what can I replace this player with for nothing?*

For a punted position they coincide: nobody benches a second kicker, so the 9th
kicker in an 8-team league is both the worst starter and the best free agent. For
a position with no starting slot they diverge sharply — and League 1 has exactly
that position.

## League 1

```
  pos    fixed   flex  bench  starter  drafted  config     gap
  QB         8    8.0    5.8       17       23      17      +0
  RB        16    3.0   16.8       20       37      20      +0
  WR        24    4.0   16.8       29       46      29      +0
  TE         8    1.0    5.8       10       16      10      +0
  K          8    0.0    1.4        9       10       9      +0
  DST        8    0.0    1.4        9       10       9      +0
```

Every position derives cleanly. Nothing here is a judgement call.

**This table was wrong until it was checked against the live settings page.**
The project brief described League 1 as having no TE slot and two W/R/T
flexes; it has a TE slot and one. Three consequences, all of which had
propagated into the engine:

1. RB and WR baselines counted eight flex spots that do not exist — 23 and 33
   where the truth is 20 and 29.
2. Tight end looked like a position you skip. It is a normal starting position
   here, and TE10 is its honest replacement level.
3. TE appeared to be the one entry in the brief's table that did not fall out
   of the arithmetic — starter demand said ~3 where the table said 10. That gap
   was entirely an artifact of the missing slot. With the slot in place, TE
   derives to exactly 10 like everything else.

The lesson is cheap to state and was expensive to find: **check the config
against the settings page before trusting anything downstream of it.**
`ffdraft rules --league cuomo` prints it back in settings-page form.

## League 2

```
  pos    fixed   flex  bench  starter  drafted  config     gap
  QB        12    0.0    5.8       13       19      13      +0
  RB        24    4.8   16.8       30       47      30      +0
  WR        24    6.0   16.8       31       48      31      +0
  TE        12    1.2    5.8       14       20      14      +0
  K         12    0.0    1.4       13       14      13      +0
  DST       12    0.0    1.4       13       14      13      +0
```

Everything derives cleanly here, because there is a TE slot.

## The assumption worth revisiting

Flex allocation. Both configs declare `flex_shares` explicitly — the split of a
W/R/T slot across the positions eligible to fill it — because it is an
assumption, not an observation, and assumptions belong in a file you can read
before the draft rather than in code.

- League 1: `WRT: {WR 0.500, RB 0.375, TE 0.125}` over 8 slots → 4 WR, 3 RB, 1 TE
- League 2: `WRT: {WR 0.50, RB 0.40, TE 0.10}` over 12 slots → 6 WR, 4.8 RB, 1.2 TE

Shifting these moves the RB and WR baselines a rank or two each. The right way
to settle it is data: once you have pick logs, count what positions actually
filled flex slots in your leagues and refit. That is one of the reasons the
brief asks you to log every pick.

## Cross-league comparison

|  | League 1 | League 2 | gap | |
|---|---|---|---|---|
| QB | 17 | 13 | **−4** | League 1 deeper — **superflex**, not team count |
| RB | 20 | 30 | +10 | the biggest move on the board |
| WR | 29 | 31 | +2 | barely moves |
| TE | 10 | 14 | +4 | |
| K/DST | 9 | 13 | +4 | |

Two rows are worth understanding.

**QB runs backwards.** League 1 has four fewer teams and a deeper quarterback
board, because the superflex means 16 QBs start instead of 12. Team count is
not what makes a position scarce; starting slots are.

**WR barely moves while RB moves ten ranks.** League 1 requires three receivers
and League 2 requires two, so the 50% larger league is only two ranks deeper at
WR. At running back, where both require two, the team count comes through in
full.

The practical consequence: **the two boards are not related by a simple shift.**
A receiver is worth about the same in both. A running back who is a comfortable
starter in League 1 is replacement level in League 2. Do not carry intuitions
from the first draft into the second.
