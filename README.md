# NBA Prediction Engine

A reproducible NBA game prediction and backtesting system that asks one question: **can a model find
information the betting market hasn't already priced in?**

It predicts winners, point margins and totals for every regular-season game since 2008-09 using only
information available before tip-off, then compares itself against sportsbook closing lines and tracks
live picks in a tamper-evident forward-test ledger.

> **Current answer: not yet.** The model beats Elo and naive baselines and is well calibrated, but the
> closing market is more accurate in every season, and simulated betting loses 4–6% after the
> sportsbook margin. See [Results](#results).

## Results

Test seasons 2022-23 → 2025-26 (4,920 games), walk-forward, out-of-sample. Ranges are 95% bootstrap intervals.

| Model | Winners correct | Brier | Log loss | Margin MAE | Total MAE |
|---|---|---|---|---|---|
| Always pick home | 55.5% | 0.248 | 0.690 | – | – |
| Elo | 65.1% | 0.218 | 0.627 | 11.00 | 15.08 |
| Random forest | 65.9% | 0.213 | 0.615 | 10.88 | 14.95 |
| LightGBM | 66.3% | 0.213 | 0.615 | 10.89 | 14.98 |
| **v1 (logistic regression)** | **66.1%** [64.8, 67.5] | **0.212** | **0.613** | **10.85** | **15.00** |
| **Market closing line** | **68.4%** [67.1, 69.7] | **0.202** | **0.589** | **10.43** | **14.27** |

**Betting at closing prices (flat stakes, after the margin):** moneyline −6.0%, spread −5.6%, total −4.0% ROI.
The only positive signal: bets at the opening line saw the line move toward the model before tip
(+0.3 pts on spreads, +0.7 pts on totals). That's real but far smaller than the margin.

**Injury reports (v2.1)**, on test games with an official report: margin MAE improves from 10.94 to
10.79 (−0.15 [−0.24, −0.07]). That's a real gain, but the market (10.56) is still ahead.

Full write-ups: [REPORT.md](REPORT.md) (v1 results), [docs/AUDIT_V3.md](docs/AUDIT_V3.md) (audit,
bugs found, v2.1), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (design and data audit).

## How leakage is prevented

- **Point-in-time features.** The feature engine walks forward one date at a time. A box score only
  counts once it's available (06:00 ET the day after the game); the cutoff for a game is 09:00 ET on
  game day.
- **Injury reports by publish time.** Each game uses the latest official report published at least
  60 minutes before its real tip time.
- **Walk-forward training.** Models are refit monthly on results available at that moment.
- **Odds are never model inputs.** They're joined only afterwards, to evaluate and choose bets.
- **Locked predictions.** Predictions are written with a SHA-256 hash before any outcome is joined.
  The evaluator refuses to run if the file changed.
- **Automated leakage tests.** These scramble or delete every future box score and player game and
  confirm features don't change, and check the timing of every injury report used.

## Repository layout

```
nba/
  config.py        paths, season splits, team codes
  ingest.py        NBA.com game logs + historical odds -> data/nba.sqlite
  validate.py      data integrity checks
  injuries.py      official injury-report PDFs -> point-in-time player-availability features
  features.py      streaming point-in-time team features and Elo
  models.py        Elo, logistic regression, random forest, LightGBM
  backtest.py      walk-forward engine, bet selection, prediction locking, model registry
  evaluate.py      metrics, calibration, betting results, CLV, version comparisons
  leakage_test.py  automated temporal-leakage checks
  audit.py         independent reproduction of reported metrics
  today.py         live predictions for today's games
  track.py         hash-chained forward-test ledger and scorer
  viz.py           charts and HTML dashboard
outputs/           v1 locked predictions, evaluation CSVs, charts, dashboard
outputs_v2/        v2 (injury parser bug, kept for the record)
outputs_v2.1/      v2.1 (injury features, parser fixed)
forward/           live forward-test log
docs/              architecture, data audit, V3 audit
tests/             parser and ledger self-checks
```

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/QuangTran-ye/nba-engine.git && cd nba-engine
```

```bash
./run.sh
```

`run.sh` creates a virtual environment, downloads and caches the data, rebuilds the database, then
runs validation, the backtest, leakage tests, evaluation and charts. The first run downloads a few
thousand files and takes a while; later runs use the cache in `data/raw/`.

Run a specific model version by pointing `NBA_OUT` at its output folder:

```bash
NBA_OUT=outputs_v2.1 .venv/bin/python -m nba.backtest final
```

### Daily forward testing

Predict today's games and log the picks before tip-off:

```bash
NBA_OUT=outputs_v2.1 .venv/bin/python -m nba.today
```

Score finished games and print the scorecard (ROI, closing-line value, accuracy vs market):

```bash
.venv/bin/python -m nba.track
```

Picks can only be logged before tip, the first pick per game is final, and every row is hash-chained,
so later edits are detectable.

### Tests

```bash
.venv/bin/python tests/test_injury_parser.py
```

```bash
PYTHONPATH=. .venv/bin/python tests/test_track.py
```

```bash
.venv/bin/python -m nba.leakage_test
```

## Data sources

| Data | Source | Coverage |
|---|---|---|
| Schedules, team and player box scores | NBA.com stats via [`nba_api`](https://github.com/swar/nba_api) | 2007-08 → present |
| Historical spreads, totals, moneylines | [Kaggle NBA betting data](https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024) (SportsbookReviewsOnline-derived) | 2007-08 → 2025-26 |
| Opening/closing lines with prices | ESPN odds API (DraftKings and others) | 2022-23 → present |
| Injury reports | [Official NBA injury reports](https://official.nba.com/) (PDF) | 2021-22 → present |

Raw downloads and the SQLite database are not committed; `run.sh` rebuilds them. NBA.com and ESPN
endpoints are unofficial and undocumented, so check each source's terms before reusing the data.

## Season splits

| Split | Seasons | Purpose |
|---|---|---|
| Warm-up | 2007-08 | rating burn-in, never scored |
| Development | 2008-09 → 2018-19 | training history |
| Validation | 2019-20 → 2021-22 | all tuning and model selection |
| Test | 2022-23 → 2025-26 | frozen config, scored once |
| Holdout | 2026-27 | untouched until the final run |

## Known limitations

- No projected minutes, lineups or player-impact ratings yet; injuries are valued with a simple
  Game Score average.
- Odds history has opening and closing lines only; no intraday prices.
- Neutral-site games use the designated home arena for travel distance.
- The 2026-27 holdout is small (205 games), so its intervals are wide.

## Roadmap

- **Market-residual model:** start from the market's probability and learn only where it's
  systematically wrong, evaluated by log-loss and Brier skill against the market.
- **Player impact:** point-in-time plus-minus ratings with minutes projections, replacing the Game
  Score injury values.
- **Multi-book consensus and best-price lines**, timestamped price snapshots, and line-movement models.
- **Better uncertainty:** conditional-variance and empirical score distributions for spread and
  total probabilities.

## Disclaimer

This is a research project. Nothing here is betting or financial advice, and the results above show
the model does **not** beat the market. If you bet, only risk money you can afford to lose.
