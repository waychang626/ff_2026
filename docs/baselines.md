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
  RB        16    6.0   16.8       23       40      23      +0
  WR        24    8.0   16.8       33       50      33      +0
  TE         0    2.0    5.8        3        9      10      +7
  K          8    0.0    1.4        9       10       9      +0
  DST        8    0.0    1.4        9       10       9      +0
```

Every position is starter demand — except TE, which matches the *drafted*
definition instead. This is the brief's table and the engine uses it as given;
the point of printing both columns is that the discrepancy is visible rather
than buried.

Which one is right for TE depends on what you do with the number. If you are
asking whether an elite TE beats your fourth receiver for a flex spot, starter
demand (≈3) is the honest baseline and the brief's 10 flatters every tight end
by roughly a full tier. If you are asking what you can stream off waivers in
week 3, 10 is right. The engine's `roster_marginal` handles the first question
directly — a TE who cannot beat a flex-eligible WR adds nothing to the lineup —
so the inflated VOR is partly compensated for downstream.

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

- League 1: `WRT: {WR 0.500, RB 0.375, TE 0.125}` over 16 slots → 8 WR, 6 RB, 2 TE
- League 2: `WRT: {WR 0.50, RB 0.40, TE 0.10}` over 12 slots → 6 WR, 4.8 RB, 1.2 TE

Shifting these moves the RB and WR baselines a rank or two each. The right way
to settle it is data: once you have pick logs, count what positions actually
filled flex slots in your leagues and refit. That is one of the reasons the
brief asks you to log every pick.

## Cross-league comparison

|  | League 1 | League 2 | |
|---|---|---|---|
| QB | 17 | 13 | League 1 deeper — **superflex**, not team count |
| RB | 23 | 30 | as expected |
| WR | 33 | 31 | **League 1 deeper**, despite 4 fewer teams |
| TE | 10 | 14 | as expected |
| K/DST | 9 | 13 | as expected |

The WR row is the surprising one and it is not a rounding artifact: League 1
starts 32 WRs (24 fixed + 8 flex across 8 teams), League 2 starts 30 (24 + 6
across 12). Three required receivers and two flexes in a small league outweighs
two and one in a large league.

The practical consequence: **the two boards are not related by a simple shift.**
A receiver who is a marginal starter in one is a marginal starter in the other,
while a running back who is a comfortable starter in League 1 is replacement
level in League 2. Do not carry intuitions from the first draft into the second.
