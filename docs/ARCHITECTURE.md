# NBA Prediction Engine — Architecture, Data Audit, Execution Plan

## 1. Data-source audit

| Need | Source used | Coverage | Cost / licence | Limitations |
|---|---|---|---|---|
| Game results, schedule, team box scores | NBA.com stats via `nba_api` (`LeagueGameLog`) | 2007-08 → 2026-27 to date, regular season | Free; NBA.com terms of use (personal / non-commercial research) | Unofficial API, rate-limited. Records are *final, as of today* — no revision history. One cancelled game (BOS–IND 2013-04-16) is listed with zeros and dropped. |
| Closing spread / total / moneyline, 2007-08 → 2025-26 | Kaggle "NBA betting data, Oct 2007 → Jun 2026" (`cviaxmiwnptr`, derived from SportsbookReviewsOnline) | Spread + total: every regular-season game. Moneyline: through Jan 2023 only. | Free download; Kaggle dataset licence (check before redistribution) | One line per game, **not timestamped**, book not specified. Validated against ESPN/DraftKings closers (see `outputs/data_validation.json`) to establish it is a closing line. No juice on spreads/totals → standard −110 assumed. |
| Open + close lines with prices, 2022-23 → 2026-27 | ESPN core API, DraftKings book (`sports.core.api.espn.com/.../odds`) | Open+close spread/total/ML with prices from 2023-24; single line (assumed close) for 2022-23 | Free, unofficial, undocumented; no licence granted — research use, responses cached in `data/raw/espn` | Opener timestamp unknown (posted "before tip" by an unspecified margin). Only DraftKings used (one book ≠ best line). |
| Rest days, back-to-backs, travel | Derived from the schedule + hard-coded arena coordinates | All seasons | — | Current arena location used for all years (Clippers/Warriors moved within metro; Seattle used for OKC 2007-08). Neutral-site games (Mexico City, Paris, Cup semis/final in Las Vegas) treated as home-arena. |
| Player stats | **Not used** | — | — | Team-level model only. Player box scores exist (nba_api) but without availability data they mostly add roster noise. |
| Injuries / player availability | **Gap — not available** | — | Official NBA injury-report PDFs exist from 2021-22 but are not timestamped per decision point in an accessible dataset; ESPN `injuries` field is a current snapshot, not historical | Biggest known gap. The model sees availability only indirectly through recent team performance. The market prices injuries; the model cannot. Pretending otherwise (e.g. using who actually played) would be leakage. |
| Rosters / transactions | **Not used** | — | — | Same reason; trades show up only through subsequent results. |

**Do not assume today's database equals what was known then.** Each record carries an availability time:

| Table | event time | knowable at |
|---|---|---|
| `schedule` | `game_date`, `tip_utc` | `known_at` = Aug 1 before the season (schedule release; ponytail assumption, rescheduled dates taken as final) |
| `team_box`, `results` | `game_date` | `available_at` = game_date + 1 day 06:00 ET |
| `odds` | — | `line_type` ∈ {open, close, close_assumed}; `observed_at` = tip time for closers, NULL where unknown |

## 2. Architecture

```
nba/
  config.py        paths, seasons, splits, team-code map, arena coords
  ingest.py        sources -> data/raw (cached) -> data/nba.sqlite
  validate.py      hard integrity asserts + coverage / odds sanity report
  features.py      streaming point-in-time feature engine (Elo, EWMA four factors, form, schedule)
  models.py        elo | logreg(+ridge heads) | random forest | LightGBM
  backtest.py      walk-forward engine, ensemble, bet recommendation, prediction LOCK (sha256)
  evaluate.py      verifies lock, joins outcomes, all metrics, betting, CLV
  leakage_test.py  future-scramble / truncation / column audits
  viz.py           calibration, bankroll, monthly, per-season, importance charts
run.sh             full pipeline
```

**Prediction cutoff.** For games on date D the information cutoff is D 09:00 ET. The feature engine
walks dates in order and only folds in a box score once its `available_at` ≤ cutoff, so a game's own
result can never influence its features. This is enforced structurally *and* tested by scrambling all
future box scores and verifying features are bit-identical.

**Walk-forward.** Models are refit at the first game of each calendar month (expanding window) using
only results whose `available_at` ≤ refit cutoff; asserted on every refit. Elo and EWMA states update
every day. Odds are **never** model inputs — they are joined only after prediction to choose bets
at the line that exists at bet time (open = morning-of decision, close = at tip).

**Separation.** `backtest final` writes `outputs/predictions.csv` (no outcome columns) + `.sha256`, and
records the hash in `model_registry.json`. `evaluate` refuses to run if the file changed, and is the
only stage that reads final scores into an evaluation dataset (`outputs/evaluation.csv`).

## 3. Splits (season = year it ends)

| Split | Seasons | Use |
|---|---|---|
| warm-up | 2007-08 | ratings burn-in, never scored |
| dev | 2008-09 → 2018-19 | training history; reported for reference |
| valid | 2019-20 → 2021-22 | **all** tuning: EWMA half-life, ensemble members, σ for cover/over probabilities, bet thresholds, ablation |
| test | 2022-23 → 2025-26 | frozen config, run once |
| holdout | 2026-27 (through 2026-11-16) | untouched until the final run |

## 4. Outputs per game
date, matchup, cutoff, model version, predicted score, winner, win probabilities, model spread and
total, cover / over / under probability, recommendation and EV at open and close, market lines —
then, in the separate evaluation file, actual score and errors.
