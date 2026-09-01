# Research findings → code

Every rule from the project brief's section 3, and where it is implemented,
enforced, or deliberately absent. If a finding is not traceable to a line here,
it did not make it into the engine.

## 3.1 The objective is win probability, not points

*Becker & Sun (JQAS 2016); Lee & Liu (JDM 2022), 1,350 Sleeper leagues.*

| Claim | Where |
|---|---|
| Optimise P(winning matchups), not total points | `simulate.py` → `SeasonResult`; `engine.py` ranks on `p_title` |
| Every player needs a distribution, not a point estimate | `calibration.py` → `CalibratedProjection.sd` |
| P(week), P(playoffs), P(title) are three readouts of one sim | `SeasonSimulator.run` returns all three from one pass |
| Variance appetite is endogenous, never a knob | Nothing implements it — it emerges. `tests/test_variance_appetite.py` |
| Half the field makes the playoffs → tilts toward ceiling | Lognormal outcome distributions; both configs' `playoff_teams` |

The variance result is the one worth checking yourself. Holding a roster's
strength relative to the field as the only variable, preference for the
higher-variance of two equal-mean players declines monotonically and crosses
zero at roughly the playoff threshold:

| Roster strength | 0.70× | 0.85× | 0.95× | 1.00× | 1.05× | 1.15× | 1.25× | 1.35× |
|---|---|---|---|---|---|---|---|---|
| ΔP(playoffs), volatile − steady | +1.14pp | +1.85pp | +1.22pp | +0.88pp | +0.45pp | −0.32pp | −0.17pp | −0.14pp |

The buying side is individually significant; the shedding side is small in
magnitude — one flex spot is a modest share of a team's variance — so the test
asserts the monotone trend and the sign change rather than significance at each
point.

## 3.2 VOR with league-specific baselines

*Fry, Lundberg & Ohlmann (JQAS 2007).*

| Claim | Where |
|---|---|
| Value of player, of others available, and team need | `vor.py` (`vor_array`, `roster_marginal`), `opponents.py` (rollout) |
| Reduce the intractable DP to a tractable one | Rollout + Monte Carlo instead of a full stochastic DP |
| Baselines are league-specific | `baselines.py`; `configs/leagues/*.yaml` |

## 3.3 Correct third-party projections before use

*Fantasy Football Analytics, 12 seasons, 11 sources.*

| Correction | Where | Enforcement |
|---|---|---|
| Equal-weight aggregation | `projections.aggregate` | `config.py` **rejects** `aggregate: weighted` |
| Shrink the spread (QB .67, TE .72, RB .79, WR .85) | `calibration.shrink` | `test_calibration.py` |
| Subtract optimism (+21.6 overall, QB +46.5) | `calibration.calibrate` | commutativity test |
| Accept the ceiling (R² 14–26%) | `calibration.residual_sd` | drives the whole variance model |

The fourth row is the one that shapes the engine. R² ≈ 0.20 is not a caveat to
note and move past — it is a number to use. Under `outcome = a + slope·proj + ε`
it fixes the residual scale at `slope · sd(proj) · √((1−ρ)/ρ)`, a factor of 2.0
at ρ = 0.20. That is why cross-source disagreement alone is a poor variance
estimate: sources agreeing with each other says nothing about them agreeing with
reality. Disagreement enters only as a unit-mean per-player modulation, measured
*relative* to each player's projection.

## 3.4 Model opponents

| Claim | Where |
|---|---|
| Herding is real and exploitable (QB early; K/DST throughout) | `opponents.DraftSimulator._herd_mask`; `test_opponents.py::test_herding_creates_position_runs` |
| **Handcuffing does not work** (BF 4.2 favouring no difference) | Deliberately absent. `test_handcuffing_is_not_implemented_anywhere` walks the AST to keep it that way |

Herding is expressed as a rank bonus rather than a probability multiplier: under
an ADP-plus-noise ranking, multiplying a position's odds by *m* is equivalent to
moving it `log(m)·sd` picks up the board — about 7 picks at the configured
values, which is what a run actually looks like.

## 3.5 Common roster builds are mediocre

| Claim | Where |
|---|---|
| Winning builds carry more RB/WR at the expense of K and DST | `DraftPolicy.min_round_k` / `min_round_dst`, applied in `engine._apply_policy_guard` |
| The roster must still be legal at the end | The guard releases automatically when picks remaining equal mandatory slots unfilled |

## 3.6 Do NOT hard-code strategy labels

No module mentions Zero RB, Robust RB, or Hyperfragile, and none has a strategy
input. The engine optimises against the live board; strategy labels are
emergent outputs you observe after the fact.

## 3.7 Caveats carried, not encoded

- Lee & Liu is one platform, one season (2017). The opponent-model parameters
  live in config precisely so they can be refit from your own pick logs.
- Their draft-slot finding (early slots outperformed, except pick 1) was
  explicitly not interpreted pending replication, and is **not encoded
  anywhere**. Draft position enters only through the snake arithmetic.

---

## Where the brief and the arithmetic disagreed

Both entries here were **errors in the brief's description of League 1**, found
by checking the config against the live Yahoo settings page. The brief had the
roster wrong — it described no TE slot and two W/R/T flexes, where there is a TE
slot and one W/R/T — and both apparent findings dissolve once that is fixed.

**League 1's TE baseline.** Against the brief's roster, TE looked like the one
entry in its table that did not derive: starter demand said ~3 where the table
said 10, and the two definitions of replacement (worst starter vs. best free
agent) appeared to diverge by seven ranks. With the TE slot in place, starter
demand *is* 10. There was never a judgement call.

**RB and WR baselines.** The brief's RB 23 / WR 33 counted eight W/R/T slots
that do not exist. The real numbers are RB 20 and WR 29 — a full three and four
ranks shallower, which moves every VOR on the board.

**"Expect much lower replacement levels" in League 2.** True at every position
except QB, where League 1 is deeper because of the superflex. But the size of
the move varies enormously: RB goes 20→30 while WR goes 29→31, because League 1
requires three receivers and League 2 requires two. League size alone does not
tell you which board is deeper at a given position.

The general lesson, and the reason `ffdraft rules` exists: a config error is
invisible downstream. Every number the engine produced from the wrong roster
was internally consistent, well-tested, and wrong.
