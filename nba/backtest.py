"""Walk-forward backtest + prediction locking.

  python -m nba.backtest tune    # dev+valid seasons only; picks config by validation metrics
  python -m nba.backtest final   # frozen config, all seasons -> locked predictions (no results)

The predictor gets outcomes ONLY through `available_results(cutoff)`, which filters on
results.available_at. Market odds are never model inputs; they are joined afterwards only to
turn a locked prediction into a bet recommendation at the line available at bet time.
"""
import hashlib
import json
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import DB_PATH, OUT, TEST_SEASONS, HOLDOUT_SEASONS, VALID_SEASONS, split_of
from .features import build_features, feature_columns
from .models import Model

REGISTRY = OUT / "model_registry.json"
MODEL_NAMES = ["elo", "logreg", "rf", "lgbm"]


def load_db():
    con = sqlite3.connect(DB_PATH)
    t = {n: pd.read_sql(f"select * from {n}", con) for n in ("schedule", "team_box", "results", "odds")}
    con.close()
    return t


def available_results(results, cutoff):
    return results[results.available_at <= cutoff]


def refit_starts(season_dates):
    """First game date of the season and of every later calendar month."""
    d = pd.Series(sorted(season_dates))
    return d.groupby(d.str[:7]).first().tolist()


FEATURE_GROUPS = {
    "elo": ["elo"],
    "four_factors": ["efg", "tov", "orb", "ftr", "fg3ar"],
    "efficiency_form": ["off_rtg", "def_rtg", "_net", "margin"],
    "schedule": ["rest", "b2b", "g7", "travel"],
    "pace": ["pace", "exp_total"],
}


def group_cols(cols, group):
    return [c for c in cols if any(k in c for k in FEATURE_GROUPS[group])]


def entry_features(db, entry, schedule=None, injuries=None):
    """Box-score features, plus injury-report features when the model version uses them."""
    f = build_features(db["schedule"] if schedule is None else schedule, db["team_box"], entry["feature_params"])
    if entry.get("injuries"):
        from .injuries import injury_features
        inj = injury_features() if injuries is None else injuries
        f = f.merge(inj, on="game_id", how="left")
        num = [c for c in inj.columns if c not in ("game_id", "inj_published", "tip_et")]
        f[num] = f[num].fillna(0.0)      # no report (pre 2021-22 or missing) -> zeros, inj_known = 0
    return f


def walk_forward(feats, results, seasons, hp, models=MODEL_NAMES, log=print, drop=()):
    cols = [c for c in feature_columns(feats) if c not in set(drop)]
    out = []
    feats = feats.sort_values(["game_date", "game_id"])
    for season in seasons:
        sd = feats[feats.season == season]
        starts = refit_starts(sd.game_date.unique()) + ["9999-99-99"]
        for lo, hi in zip(starts[:-1], starts[1:]):
            cutoff = f"{lo} 09:00"
            lab = available_results(results, cutoff)
            train = feats.merge(lab, on="game_id")
            train = train[~((train.season == feats.season.min()) & (train[["h_gp", "a_gp"]].min(axis=1) < 5))]
            # --- leakage guards -------------------------------------------------------
            assert train.game_date.max() < lo, "training game on/after prediction date"
            assert (train.available_at <= cutoff).all()
            # --------------------------------------------------------------------------
            test = sd[(sd.game_date >= lo) & (sd.game_date < hi)]
            y_w = (train.home_pts > train.away_pts).astype(int)
            y_m = train.home_pts - train.away_pts
            y_t = train.home_pts + train.away_pts
            for name in models:
                m = Model(name, hp).fit(train[cols], y_w, y_m, y_t)
                p, mg, tt = m.predict(test[cols])
                out.append(pd.DataFrame({"game_id": test.game_id.values, "model": name, "p_home": p,
                                         "pred_margin": mg, "pred_total": tt, "train_n": len(train),
                                         "train_through": train.game_date.max()}))
            log(f"  {season} {lo}: train={len(train)} predict={len(test)}")
    return pd.concat(out, ignore_index=True)


def ensemble(long, members):
    e = long[long.model.isin(members)].groupby("game_id")[["p_home", "pred_margin", "pred_total"]].mean()
    meta = long.drop_duplicates("game_id").set_index("game_id")[["train_n", "train_through"]]
    return e.join(meta).reset_index().assign(model="ensemble")


# ---------------------------------------------------------------- market + recommendations
def american_to_decimal(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a > 0, 1 + a / 100, 1 + 100 / a.abs())


def market_lines(odds):
    """Closing line per game (ESPN DraftKings preferred, else Kaggle/SBR) and ESPN opener."""
    pri = {"espn": 0, "kaggle_sbr": 1}
    c = odds[odds.line_type != "open"].assign(pri=lambda d: d.source.map(pri)).sort_values("pri")
    close = c.groupby("game_id").first()
    # fill moneylines missing from the preferred source
    for col in ("ml_home", "ml_away"):
        close[col] = close[col].fillna(c.groupby("game_id")[col].first())
    # moneylines whose favourite contradicts a >=2 pt spread are source errors: drop them
    bad = (close.home_spread.abs() >= 2) & ((close.home_spread < 0) != (close.ml_home < close.ml_away)) & close.ml_home.notna()
    close.loc[bad, ["ml_home", "ml_away"]] = np.nan
    close = close[["source", "line_type", "home_spread", "total", "ml_home", "ml_away",
                   "spread_price_home", "spread_price_away", "over_price", "under_price"]]
    opn = odds[odds.line_type == "open"].set_index("game_id")[
        ["home_spread", "total", "ml_home", "ml_away", "spread_price_home", "spread_price_away",
         "over_price", "under_price"]]
    return close.add_prefix("close_"), opn.add_prefix("open_")


def recommend(pred, mkt, prefix, cfg):
    """Bet selection at a given line. Uses predictions + line only (never the result)."""
    r = pd.DataFrame(index=pred.index)
    line, tot = mkt[f"{prefix}home_spread"], mkt[f"{prefix}total"]
    sm, stt = cfg["sigma_margin"], cfg["sigma_total"]
    r["cover_p_home"] = 1 - norm.cdf((-line - pred.pred_margin) / sm)
    r["over_p"] = 1 - norm.cdf((tot - pred.pred_total) / stt)
    d = lambda c, default=-110: american_to_decimal(mkt[c].fillna(default)) if c in mkt else american_to_decimal(pd.Series(default, index=mkt.index))
    dh, da = american_to_decimal(mkt[f"{prefix}ml_home"]), american_to_decimal(mkt[f"{prefix}ml_away"])
    r["mkt_p_home"] = (1 / dh) / (1 / dh + 1 / da)               # vig-free implied probability
    ev_h, ev_a = pred.p_home * dh - 1, (1 - pred.p_home) * da - 1
    sh, sa = d(f"{prefix}spread_price_home"), d(f"{prefix}spread_price_away")
    ev_sh, ev_sa = r.cover_p_home * sh - 1, (1 - r.cover_p_home) * sa - 1
    do, du = d(f"{prefix}over_price"), d(f"{prefix}under_price")
    ev_o, ev_u = r.over_p * do - 1, (1 - r.over_p) * du - 1

    def pick(a, b, la, lb, thr):
        return np.where((a > thr) & (a >= b), la, np.where((b > thr) & (b > a), lb, "none"))
    r["ml_bet"] = np.where(np.isnan(dh) | np.isnan(da), "no_line", pick(ev_h, ev_a, "home", "away", cfg["thr_ml"]))
    r["spread_bet"] = np.where(line.isna(), "no_line", pick(ev_sh, ev_sa, "home", "away", cfg["thr_spread"]))
    r["total_bet"] = np.where(tot.isna(), "no_line", pick(ev_o, ev_u, "over", "under", cfg["thr_total"]))
    r["ml_ev"] = np.maximum(ev_h, ev_a)
    r["spread_ev"] = np.maximum(ev_sh, ev_sa)
    r["total_ev"] = np.maximum(ev_o, ev_u)
    r["ml_dec_home"], r["ml_dec_away"] = dh, da
    r["spread_dec_home"], r["spread_dec_away"] = sh, sa
    r["over_dec"], r["under_dec"] = do, du
    return r.add_prefix(prefix)


def lock(primary, schedule, feats, odds, cfg, version, path):
    """Assemble the prediction dataset (no outcomes) and write it with a sha256 manifest."""
    teams = pd.read_csv(OUT / "teams.csv").set_index("team_id").abbr if (OUT / "teams.csv").exists() else None
    p = primary.merge(schedule[["game_id", "season", "game_date", "home_id", "away_id", "tip_utc"]], on="game_id")
    p = p.merge(feats[["game_id", "cutoff"]], on="game_id")
    p["split"] = p.season.map(split_of)
    if teams is not None:
        p["matchup"] = p.away_id.map(teams) + " @ " + p.home_id.map(teams)
    p["model_version"] = version
    p["pred_home_pts"] = (p.pred_total + p.pred_margin) / 2
    p["pred_away_pts"] = (p.pred_total - p.pred_margin) / 2
    p["pred_winner"] = np.where(p.p_home >= 0.5, "home", "away")
    p["p_away"] = 1 - p.p_home
    p["pred_spread_home"] = -p.pred_margin          # model's own line, sportsbook sign convention
    close, opn = market_lines(odds)
    p = p.set_index("game_id")
    mc, mo = close.reindex(p.index), opn.reindex(p.index)
    p = pd.concat([p, mc, mo, recommend(p, mc, "close_", cfg), recommend(p, mo, "open_", cfg)], axis=1)
    p = p.reset_index().sort_values(["game_date", "game_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    p.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "\n")
    return p, digest


def registry():
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []


def save_version(entry):
    reg = registry()
    reg.append(entry)
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2, default=str))


def write_teams(tb):
    from nba_api.stats.static import teams
    pd.DataFrame([(t["id"], t["abbreviation"]) for t in teams.get_teams()], columns=["team_id", "abbr"]) \
        .to_csv(OUT / "teams.csv", index=False)


# ---------------------------------------------------------------------- commands
def cmd_tune():
    """Model/feature selection using validation seasons ONLY. Test/holdout are never predicted here."""
    from .evaluate import prob_metrics, point_metrics, betting_table
    db = load_db()
    res = db["results"]
    seasons = list(range(min(VALID_SEASONS), max(VALID_SEASONS) + 1))
    assert not set(seasons) & (set(TEST_SEASONS) | set(HOLDOUT_SEASONS))
    rows, best = [], None
    cfg_base = {"sigma_margin": 12.5, "sigma_total": 18.0, "thr_ml": 0.03, "thr_spread": 0.03, "thr_total": 0.03}
    for hl in (6.0, 12.0, 24.0):
        feats = build_features(db["schedule"], db["team_box"], {"halflife": hl})
        long = walk_forward(feats, res, seasons, {}, log=lambda *_: None)
        for members in (["logreg"], ["lgbm"], ["rf"], ["elo"], ["logreg", "lgbm"], ["logreg", "lgbm", "rf"],
                        ["logreg", "lgbm", "elo"], ["logreg", "rf"]):
            e = ensemble(long, members).merge(res, on="game_id")
            pm, qm = prob_metrics(e), point_metrics(e)
            rows.append({"halflife": hl, "members": "+".join(members), **pm, **qm})
            print(rows[-1], flush=True)
            if best is None or pm["log_loss"] < best[0]:
                best = (pm["log_loss"], hl, members, long, feats)
    tune = pd.DataFrame(rows)
    # feature-group ablation (logistic/ridge, validation seasons, best halflife so far)
    hl_best = tune.sort_values("log_loss").halflife.iloc[0]
    feats_b = build_features(db["schedule"], db["team_box"], {"halflife": hl_best})
    abl = []
    for grp in [None, *FEATURE_GROUPS]:
        drop = group_cols(feature_columns(feats_b), grp) if grp else []
        lw = walk_forward(feats_b, res, seasons, {}, models=["logreg"], log=lambda *_: None, drop=drop)
        e = lw.merge(res, on="game_id")
        abl.append({"dropped_group": grp or "(none)", "n_dropped": len(drop), **prob_metrics(e), **point_metrics(e)})
        print(abl[-1], flush=True)
    (OUT / "tuning").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(abl).to_csv(OUT / "tuning" / "validation_ablation.csv", index=False)
    (OUT / "tuning").mkdir(parents=True, exist_ok=True)
    tune.to_csv(OUT / "tuning" / "validation_grid.csv", index=False)
    _, hl, members, long, feats = best
    e = ensemble(long, members).merge(res, on="game_id")
    resid_m = (e.home_pts - e.away_pts) - e.pred_margin
    resid_t = (e.home_pts + e.away_pts) - e.pred_total
    cfg = {**cfg_base, "sigma_margin": float(resid_m.std()), "sigma_total": float(resid_t.std())}
    # betting thresholds: pick from a coarse grid on validation, require >= 150 bets
    write_teams(None)
    pv, _ = lock(e[["game_id", "p_home", "pred_margin", "pred_total", "train_n", "train_through", "model"]],
                 db["schedule"], feats, db["odds"], {**cfg, "thr_ml": 0, "thr_spread": 0, "thr_total": 0},
                 "tune", OUT / "tuning" / "valid_predictions.csv")
    thr_rows = []
    for thr in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        c = {**cfg, "thr_ml": thr, "thr_spread": thr, "thr_total": thr}
        pp, _ = lock(e[["game_id", "p_home", "pred_margin", "pred_total", "train_n", "train_through", "model"]],
                     db["schedule"], feats, db["odds"], c, "tune", OUT / "tuning" / "tmp.csv")
        bt = betting_table(pp.merge(res, on="game_id"), "close_")
        for _, r in bt.iterrows():
            thr_rows.append({"thr": thr, **r.to_dict()})
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(OUT / "tuning" / "validation_thresholds.csv", index=False)
    for mk in ("ml", "spread", "total"):
        cand = thr_df[(thr_df.market == mk) & (thr_df.bets >= 150)]
        cfg[f"thr_{mk}"] = float(cand.sort_values("roi", ascending=False).thr.iloc[0]) if len(cand) else 0.05
    entry = {"version": f"v{len(registry()) + 1}", "created": datetime.now().isoformat(timespec="seconds"),
             "stage": "tuned_on_validation", "feature_params": {"halflife": hl}, "ensemble_members": members,
             "hyperparams": {}, "bet_config": cfg, "valid_seasons": seasons,
             "selection_rule": "lowest validation log loss; thresholds = best validation ROI with >=150 bets"}
    save_version(entry)
    print(json.dumps(entry, indent=2))


def cmd_final():
    reg = [r for r in registry() if r["stage"] in ("tuned_on_validation", "final")]
    entry = reg[-1]
    version = entry["version"]
    db = load_db()
    write_teams(None)
    feats = entry_features(db, entry)
    feats.to_csv(OUT / "features.csv", index=False)
    seasons = sorted(s for s in feats.season.unique()
                     if split_of(s) != "warmup" and s >= entry.get("first_predict_season", 0))
    long = walk_forward(feats, db["results"], seasons, entry["hyperparams"])
    long.to_csv(OUT / "predictions_all_models.csv", index=False)
    # feature importance: fit on everything known before the holdout season starts
    cols = feature_columns(feats)
    first_hold = feats[feats.season == max(HOLDOUT_SEASONS)].game_date.min()
    tr = feats.merge(available_results(db["results"], f"{first_hold} 09:00"), on="game_id")
    imp = {}
    for name in ("lgbm", "rf", "logreg"):
        m = Model(name, entry["hyperparams"]).fit(tr[cols], (tr.home_pts > tr.away_pts).astype(int),
                                                  tr.home_pts - tr.away_pts, tr.home_pts + tr.away_pts)
        v = pd.Series(m.importance(cols))
        imp[f"{name}_gain_share" if name == "lgbm" else f"{name}_importance"] = v / v.sum()
    pd.DataFrame(imp).rename_axis("feature").reset_index().sort_values("lgbm_gain_share", ascending=False) \
        .to_csv(OUT / "feature_importance.csv", index=False)
    prim = ensemble(long, entry["ensemble_members"])
    _, digest = lock(prim, db["schedule"], feats, db["odds"], entry["bet_config"], version,
                     OUT / "predictions.csv")
    save_version({**entry, "stage": "final", "created": datetime.now().isoformat(timespec="seconds"),
                  "predictions_sha256": digest, "note": "frozen config run once over test + holdout"})
    print("locked predictions sha256", digest)


if __name__ == "__main__":
    {"tune": cmd_tune, "final": cmd_final}[sys.argv[1]]()
