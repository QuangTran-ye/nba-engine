# NBA Prediction Engine — Results (model v1, run 2026-11-17)

**Bottom line: the model is a competent, well-calibrated forecaster, but it does not beat the
sportsbook market on any measure, and it loses money after vig in every market at closing lines.
It should not be used for betting.** Everything below is out-of-sample, walk-forward, with locked
predictions (sha256 `b17f86da…cba26`) and passing leakage tests.

Design and data audit: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Reproduce: `./run.sh`.

## 1. Protocol actually followed
1. Tuned **only** on 2019-20 → 2021-22 (`outputs/tuning/`). Selection rule declared in code before
   running: lowest validation log loss. Winner: logistic regression (+ ridge heads), EWMA half-life
   24 games. Every candidate was within noise (log loss 0.6329–0.6349); logreg+RF was 0.00001 behind.
   So the "ensemble" in the output files is logistic regression alone; RF, LightGBM and Elo are
   still scored side by side.
2. Froze config as `v1` in `outputs/model_registry.json`.
3. Ran the walk-forward once over 2008-09 → 2026-27 (monthly refits, expanding window), locked the
   predictions, ran the leakage tests, then evaluated. The test and holdout seasons were never used for
   any choice. Rerunning the evaluator to fix a bug in the random baseline's CI didn't change
   any prediction.

## 2. Game prediction (test = 2022-23 → 2025-26, n = 4,920; holdout = 2026-27 to Nov 16, n = 205)

| | Winner acc. | Brier | Log loss | Margin MAE | Total MAE |
|---|---|---|---|---|---|
| Random | 50.4% | 0.250 | 0.693 | – | – |
| Always home | 55.5% | 0.248 | 0.690 | – | – |
| Elo (538-style) | 65.1% | 0.218 | 0.627 | 11.00 | 15.08 |
| Random forest | 65.9% | 0.213 | 0.615 | 10.88 | 14.95 |
| LightGBM | 66.3% | 0.213 | 0.615 | 10.89 | 14.98 |
| **Model v1 (logistic)** | **66.1%** [64.8, 67.5] | **0.212** | **0.613** [0.603, 0.624] | **10.85** [10.62, 11.08] | **15.00** [14.69, 15.32] |
| **Market (no-vig close ML / closing spread / closing total)** | **68.4%** [67.1, 69.7] | **0.202** | **0.589** [0.578, 0.600] | **10.43** [10.20, 10.66] | **14.27** [13.95, 14.59] |
| *Holdout v1* | 69.3% [62.4, 75.6] | 0.203 | 0.589 | 11.83 | 14.86 |
| *Holdout market* | 72.7% [66.3, 79.0] | 0.187 | 0.553 | 11.22 | 14.10 |

95% bootstrap CIs in brackets. The model beats every naive baseline and Elo clearly. It trails the
market by about 2.3 pts of accuracy, 0.024 of log loss, 0.42 pts of margin MAE and 0.73 pts of total
MAE. **The market wins in every one of the 19 seasons** for margin and total MAE
(`outputs/per_season.csv`, `charts/per_season_mae.png`).

**Calibration** (`charts/calibration.png`, `outputs/calibration_table.csv`): model v1 and the market
both sit on the diagonal. By confidence bucket, the 0.75–0.80 bucket hit 77.9% and the 0.80–0.90
bucket hit 82.8%. Elo overrates home teams in recent seasons because its fixed home-court bonus
predates the post-2020 drop in home advantage; the learned models absorb this.

## 3. Betting (1-unit flat stakes, after vig)

| Split / line | Market | Bets | Win % | Units | ROI [95% CI] | Max DD |
|---|---|---|---|---|---|---|
| Test, close | Moneyline | 2,931 | 35.3% | −175.2 | −6.0% [−11.4, −0.1] | 216 |
| Test, close | Spread | 3,970 | 49.3% | −223.2 | −5.6% [−8.5, −2.9] | 242 |
| Test, close | Total | 3,501 | 50.2% | −140.1 | −4.0% [−7.1, −0.9] | 161 |
| Test, open (2023-24+) | Spread | 2,811 | 49.6% | −140.1 | −5.0% [−8.4, −1.0] | 155 |
| Test, open (2023-24+) | Total | 2,584 | 52.2% | +3.5 | +0.1% [−3.7, +3.9] | 70 |
| Holdout, close | all three | 413 | – | −67.5 | −11% to −31% | – |

Picking a side against the closing spread in every game: 49.6% in test, 45.4% in holdout. Break-even
at −110 is 52.4%. Over/under on every game: 50.4% in test.

**Closing-line value** (bets at the DraftKings opener, `outputs/clv.csv`): spread picks gained
+0.27 pts on average [0.18, 0.35] and totals picks +0.72 pts [0.52, 0.96]. So the market tends to
move *toward* the model after the open, which is real but small information. It's far below the
~4.5% vig: only the totals-at-open slice breaks even, and its CI spans −3.7% to +3.9%. That slice
was not pre-selected, so treat it as a hypothesis for the next model version, not a finding.

## 4. What drives the model
LightGBM gain (`charts/feature_importance.png`): EWMA point-differential, opponent-adjusted net
rating, net rating, then Elo dominate (~60% of gain). Validation ablation
(`outputs/tuning/validation_ablation.csv`): removing schedule features (rest, back-to-back,
travel) costs the most log loss (+0.0016) and margin MAE (+0.03). Removing four-factor features
costs +0.0013. Removing pace/expected-total inflates total MAE from 15.09 to 15.83. Removing
efficiency/form features alone changes nothing, because they're redundant with Elo and margin. None
of these effects is large: team-level box-score features are close to saturated.

## 5. Why the model fails against the market
1. **No player availability.** No free, timestamped historical injury feed exists (see the audit).
   The market prices a star's absence the moment it's announced; the model only sees it afterwards,
   through results. This is the largest single gap. (Both the model and the market have higher
   errors in the early-season holdout, 11.8 vs 11.2 margin MAE, so the gap itself is similar.)
2. **The closing line already contains everything the model knows.** Team efficiency, rest and
   home court are public and fully priced.
3. **One book, no line shopping.** Real bettors would take the best of several books. That would
   add perhaps 1–2% ROI, still not enough to turn −5% positive.

## 6. Limitations
- Kaggle/SBR lines carry no timestamp or book. Validation shows they match DraftKings closing lines
  (mean |Δ| 0.27 pts spread, 0.38 total) rather than openers (1.73 / 3.07), so they're treated as
  closers. Spread and total prices are assumed −110 before 2022-23.
- Moneylines are missing from Jan 2023 until ESPN coverage starts. 32 source moneylines that
  contradict the spread were discarded.
- Opener timestamps are unknown. "Bet at open" assumes the opener is available at the 09:00 ET
  cutoff, which is usually true (lines post the night before) but not guaranteed.
- Box scores are today's corrected versions, not what was published that night. Corrections are rare
  and small. The schedule is assumed known from Aug 1, and rescheduled dates are taken as final.
- The data contains no player stats, rosters or transactions.
- Simplifications: arenas use current coordinates; neutral sites use the designated home arena for
  travel; models are refit monthly rather than daily, while Elo and EWMA states update daily.
- The holdout has only 205 games, so its CIs are wide (accuracy ±6.6 pts).
- Data licences: NBA.com and ESPN are unofficial and undocumented APIs used for research. Check the
  Kaggle dataset's licence before redistributing it.

## 7. Files
| File | Contents |
|---|---|
| `outputs/predictions.csv` (+ `.sha256`) | every locked prediction: date, matchup, cutoff, version, predicted score, winner, win probs, model spread and total, cover/over probs, open and close lines, recommendations, EV. **No outcomes.** |
| `outputs/evaluation.csv` | actual scores and errors per game (separate dataset) |
| `outputs/predictions_all_models.csv` | Elo / logreg / RF / LightGBM per game |
| `outputs/model_comparison.csv`, `per_season.csv`, `accuracy_by_bucket.csv`, `calibration_table.csv` | metrics with CIs |
| `outputs/betting_performance.csv`, `bets_ledger.csv`, `monthly_betting.csv`, `ats_all_games.csv`, `clv.csv` | betting |
| `outputs/feature_importance.csv`, `outputs/tuning/*` | importance, validation grid, ablation, thresholds |
| `outputs/charts/*.png` | calibration, bankroll, monthly units, per-season MAE, accuracy by bucket, importance |
| `outputs/model_registry.json` | every model version, its config and hash |

## 8. Next steps (each needs a new version, tuned on validation only)
- Player availability from the official NBA injury-report PDFs (2021-22+), timestamped per report.
  This is the one addition plausibly able to close the gap.
- A model of the *difference* from the opening line (market as a prior), aimed at CLV rather than raw accuracy.
- A time-varying home-court term.
