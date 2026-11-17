# V3 Technical & Statistical Audit (Phase 1)

Date: 2026-11-17. Scope: everything in this repository as of commit `c372e74` plus the fixes listed below.
Reproduce: `python -m nba.audit` (independent recomputation, writes `outputs/audit/`).

## 1. System map

| Module | Role | Reads | Writes |
|---|---|---|---|
| `config.py` | paths, splits, team codes, arenas; `NBA_OUT` selects a model version's output dir | – | – |
| `ingest.py` | NBA.com game logs, Kaggle/SBR odds, ESPN DraftKings odds → SQLite | network, `data/raw` cache | `data/nba.sqlite` (schedule, team_box, results, odds) |
| `validate.py` | integrity asserts + odds sanity | DB | `outputs/data_validation.json` |
| `injuries.py` | official injury PDFs, player logs, point-in-time player values | network, DB | DB (`injury_entries`, `injury_report_choice`) |
| `features.py` | streaming point-in-time team features + Elo | DB frames | in-memory / `features.csv` |
| `models.py` | Elo pass-through, logreg+ridge, RF, LightGBM | features | – |
| `backtest.py` | walk-forward, ensemble, bet recommendation, SHA-256 lock, registry | DB, features | `predictions.csv`, `.sha256`, registry |
| `evaluate.py` | verifies lock, joins outcomes, metrics, betting, CLV, `compare` | locked file, DB | evaluation CSVs |
| `leakage_test.py` | future scramble/truncation, column audits, injury timing | DB, locked file | log |
| `today.py` / `track.py` | live predictions; hash-chained forward ledger and scorer | network, DB | `forward/` |
| `audit.py` | this audit's independent reproduction | locked file, DB | `outputs/audit/` |
| `viz.py` | charts + dashboard | outputs | `charts/`, `dashboard.html` |

Dependencies: pandas, numpy, scikit-learn, LightGBM, SciPy, matplotlib, nba_api, requests, pdfplumber (pinned in `requirements.txt`).

## 2. Data sources and coverage

| Source | Coverage | Timestamp quality |
|---|---|---|
| NBA.com team & player logs | 2007-08 → today (team); 2020-21 → today (player) | game date only; availability modeled as next day 06:00 ET |
| Kaggle/SBR odds | 2007-08 → 2025-26 spreads/totals; ML to Jan 2023 | **none**; validated as closing (0.27-pt mean gap to DK close vs 1.73 to open) |
| ESPN core odds | 2022-23 → today; 14 providers cached raw, **only DraftKings ingested** | close = at tip; open = time unknown; 2022-23 single line |
| Official injury reports | 2021-22 → today, hourly (15-min in 2026-27) | exact publish time printed on each PDF |
| Polymarket US | live order book only; NBA = winner, full-game spreads & totals, halves/quarters, team totals, player props (combos enabled); taker fee coeff 0.095 | live; **no history** — must be collected forward |
| Polymarket international | public price history | proxy only, not the execution venue |

Market benchmark mix (`outputs/audit/market_source_mix.csv`): seasons 2008-09 → 2021-22 use the
Kaggle consensus line; 2022-23 → today use DraftKings. ML available for every game except 6 in 2025-26.

## 3. Reproduction of reported baseline

All 20 headline numbers (v1 and market; test and holdout; accuracy, Brier, log loss, margin MAE,
total MAE) reproduce **exactly** with independent code (`outputs/audit/reproduction.csv`).
Sample counts are consistent: 4,920 test, 205 holdout, 4,911 test games with a moneyline.

## 4. Bugs found (all fixed unless noted)

| # | Bug | Effect | Status |
|---|---|---|---|
| B1 | Injury parser rejected names with spaces (2021-22, 2022-23 format) and accented names | v2 had **no** injury rows for 2021-22/2022-23; 2,050 games silently zero-filled. v2's "worse than v1" result was this bug | Fixed → v2.1; regression test `tests/test_injury_parser.py` |
| B2 | Old-format report files are stamped HH:30; code discarded instead of walking back | 2,673 games lost a valid report | Fixed (expected publish time per URL) |
| B3 | Name suffixes (Jr., III) blocked player matches | stars (e.g. Jimmy Butler III) valued at 0 | Fixed |
| B4 | Random-baseline accuracy CI computed from the wrong vector | cosmetic, wrong CI in report | Fixed |
| B5 | Cancelled game (BOS–IND 2013-04-16) with zero stats; one misrecorded team-minutes row; 11 neutral-site games had no home team | NaNs / wrong pace | Fixed at ingestion |
| B6 | `inj_known` flag is 0 for every pre-2021-22 game and 1 after → acts as an **era indicator**, not injury information | likely cause of v2.1's +0.98-pt total MAE regression in 2021-22 | **Open** — fix in v3 (drop flag / train injury effect only within injury era) |

## 5. Statistical weaknesses

1. **Bet thresholds tuned under different pricing than tested.** Validation used Kaggle consensus lines with an assumed −110 price; test used DraftKings with real prices. The tuned thresholds are not transferable.
2. **Threshold rule could select a losing strategy.** "Best validation ROI with ≥150 bets" picked a spread threshold of 0 even though every spread threshold lost money. The rule must allow "no bet".
3. **Model EV is inflated.** The model's claimed edge does not rank bets (ROI by edge bucket is flat-to-negative). EV computed from an independent model ignores that the market is the better forecaster — the core reason for the Phase 3 market-residual design.
4. **No market-relative metrics.** Selection used raw log loss. Must add: log-loss and Brier skill vs market, encompassing regression, paired date-blocked tests.
5. **Fixed-normal cover/over probabilities.** σ (13.8 margin, 19.0 total) is constant; heteroscedasticity untested (Phase 6).
6. **Near-tie model selection.** Validation log loss spread across all candidates was 0.002; "winner" choice is noise. Prefer simplest model unless a paired test separates them.
7. **Heuristic injury weights.** Doubtful ×0.75, Questionable ×0.4 are guesses, not estimated.
8. **CLV headline was misleading** (§7).

Bootstrap: date-blocked and iid intervals are nearly identical for headline quantities (e.g. v1 −
market log loss +0.0233: iid [0.0176, 0.0291], blocked [0.0179, 0.0290]). Reported CIs stand; future
comparisons use the blocked version by default.

## 6. Leakage review

| Risk | Finding |
|---|---|
| Features use the game's own or later results | Ruled out structurally and by scramble/truncation tests (pass) |
| Injury report after tip | Ruled out: every report used published ≥60 min before tip; **all 5,216 tip times are real ESPN times** (no defaults; 284 afternoon tips handled) |
| Player values use later games | Ruled out by truncation test (pass) |
| Closing odds as model input | Ruled out (no odds columns among features); close used only for bet-at-close simulation |
| Rescheduled games (e.g. COVID postponements) | **Minor residual risk**: schedule is taken as final, so rest/travel for make-up games uses a date that was only announced days in advance. Not exploitable in a pregame sense but not strictly as-of. |
| Box-score revisions | Today's corrected records; corrections are rare and small |
| Kaggle line timestamp | Unknown; treated as close; used only for at-close decisions |

## 7. Report discrepancies investigated

| Item | Finding |
|---|---|
| "Only 43% of open-line spread bets beat the close" despite +0.27-pt mean CLV | 21% of lines never moved (ties). Among moved lines the model's side won 55.5% (spread) and 56.9% (total). The 43% statistic understated it; report beat/tie/worse separately |
| Sample counts (test ML 4,911 of 4,920; open ML bets 1,972 vs CLV 1,970) | consistent: missing closing ML on 9 games; 2 open bets lack a close |
| Holdout size | 205 locked = 205 in DB |
| ROI CIs | correct per bet; date-blocked equivalent within ±0.003 |
| "Ensemble" label | is logistic regression alone (selection near-tie); renamed in reports going forward |

## 8. Model results so far (matched samples)

| Comparison (games with an injury report) | v1 | v2 (parser bug) | v2.1 (fixed) | Market |
|---|---|---|---|---|
| Test log loss (n=4,006) | 0.615 | – | 0.611 (−0.004 [−0.009, +0.002]) | 0.593 |
| Test margin MAE | 10.94 | – | **10.79 (−0.15 [−0.24, −0.07])** | 10.56 |
| Test total MAE | 15.00 | – | 14.94 (−0.06 [−0.12, +0.00]) | – |
| 2021-22 total MAE (n=1,044) | 14.74 | – | **15.72 (+0.98 [+0.61, +1.33])** ← B6 | – |

v2.1 test numbers are a second look at those seasons (they shaped v1), so they are evidence, not proof.

## 9. Redundant features
Home/away/difference triplets for every rating; `elo_p`, `elo_diff`, `elo_margin` are monotone
transforms of one number; `d_margin` vs `d_net` ≈ 0.95 correlated. Harmless for regularized logistic
regression; inflates tree importance splits. Ablation showed the efficiency/form group adds nothing
beyond Elo + margin.

## 10. Missing information
Projected minutes and rotations; lineups and starters; player impact beyond Game Score; referee
assignments; intraday odds history; multi-book consensus (cached but not ingested); Polymarket price
history for the user's venue.

## 11. Performance bottlenecks
- Injury report discovery probes ~10k URLs (most return 403); cache makes reruns fast.
- Feature engine ~27 s per build; leakage test rebuilds it 8×.
- Walk-forward ~60 s per season with four models; RF dominates.

## 12. Improvements without rewrites (queued)
1. Ingest all 14 cached ESPN providers → consensus + best price (open/close).
2. Market-relative metrics + date-blocked paired comparisons in `evaluate.py`.
3. Drop `inj_known` era flag; restrict injury coefficients to the injury era (B6).
4. Threshold selection with a "no bet" option and consistent pricing.
5. Polymarket US snapshot collector feeding the forward ledger.
