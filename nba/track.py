"""Forward (paper-trading) tracker: the only clean test of the model from here on.

  python -m nba.today            # predicts AND logs each game's pick before tip
  python -m nba.track            # scores finished games, prints the scorecard

Log rules: a game is logged once per model version (first pick is final), only before tip, with
the exact line and price available at that moment. Rows are hash-chained so later edits are
detectable. Outcomes and closing lines live in a separate results file.
"""
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from .config import ROOT
from .evaluate import boot_ci, clv, settle
from .ingest import ESPN_ODDS, ESPN_SB, ET, _num

DIR = ROOT / "forward"
LOG, RESULTS = DIR / "log.csv", DIR / "results.csv"
LOG_COLS = ["logged_at", "model_version", "game_id", "game_date", "matchup", "tip_et", "p_home", "pred_margin",
            "pred_total", "now_home_spread", "now_total", "now_ml_home", "now_ml_away", "now_spread_price_home",
            "now_spread_price_away", "now_over_price", "now_under_price", "now_mkt_p_home", "now_cover_p_home",
            "now_over_p", "now_ml_bet", "now_ml_ev", "now_ml_dec_home", "now_ml_dec_away", "now_spread_bet",
            "now_spread_ev", "now_spread_dec_home", "now_spread_dec_away", "now_total_bet", "now_total_ev",
            "now_over_dec", "now_under_dec", "inj_published"]


def _canon(v):
    """Representation that survives a CSV round trip (int vs float, None vs NaN)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return f"{float(v):.6f}"
    return str(v)


def _hash(prev, row):
    body = {k: _canon(row.get(k)) for k in LOG_COLS}
    return hashlib.sha256((prev + json.dumps(body, sort_keys=True)).encode()).hexdigest()


def read_log():
    return pd.read_csv(LOG, dtype={"game_id": str}) if LOG.exists() else pd.DataFrame(columns=LOG_COLS + ["prev_hash", "row_hash"])


def verify_chain(log):
    prev = ""
    for r in log.to_dict("records"):
        body = r
        assert r["prev_hash"] == (prev or "genesis") and r["row_hash"] == _hash(prev, body), \
            f"forward log edited or reordered at {r['game_id']} ({r['logged_at']})"
        prev = r["row_hash"]
    return prev


def log_picks(out, version):
    """Append today's picks. `out` is the frame `nba.today` builds (one row per game)."""
    DIR.mkdir(exist_ok=True)
    log = read_log()
    prev = verify_chain(log)
    now = datetime.now(ET).replace(tzinfo=None)
    done = set(zip(log.model_version, log.game_id))
    rows, skipped = [], []
    for r in out.to_dict("records"):
        tip = pd.Timestamp(r["tip_utc"]).tz_convert(ET).tz_localize(None)
        if (version, r["game_id"]) in done:
            skipped.append(f"{r['matchup']} (already logged)"); continue
        if now >= tip:
            skipped.append(f"{r['matchup']} (tipped off)"); continue
        body = {k: r.get(k) for k in LOG_COLS}
        body.update(logged_at=now.strftime("%Y-%m-%d %H:%M:%S"), model_version=version, tip_et=str(tip))
        body = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in body.items()}
        h = _hash(prev, body)
        rows.append({**body, "prev_hash": prev or "genesis", "row_hash": h})
        prev = h
    if rows:
        pd.concat([log, pd.DataFrame(rows)], ignore_index=True).to_csv(LOG, index=False)
    return len(rows), skipped


# ---------------------------------------------------------------------------- scoring
def fetch_outcome(game_id, date):
    eid = game_id.split("_", 1)[1]
    sb = requests.get(ESPN_SB.format(date.replace("-", "")), timeout=60).json()
    ev = next((e for e in sb.get("events", []) if e["id"] == eid), None)
    if ev is None or not ev["competitions"][0]["status"]["type"].get("completed"):
        return None
    sc = {c["homeAway"]: int(c["score"]) for c in ev["competitions"][0]["competitors"]}
    o = requests.get(ESPN_ODDS.format(eid), timeout=60).json()
    it = next((i for i in o.get("items", []) if i["provider"]["name"].replace(" ", "").lower() == "draftkings"), None)
    h, a = (it or {}).get("homeTeamOdds", {}), (it or {}).get("awayTeamOdds", {})
    hc, ac, tc = h.get("close") or {}, a.get("close") or {}, (it or {}).get("close") or {}
    return {"game_id": game_id, "home_pts": sc["home"], "away_pts": sc["away"],
            "close_home_spread": _num(hc.get("pointSpread")), "close_total": _num(tc.get("total")),
            "close_ml_home": _num(hc.get("moneyLine")), "close_ml_away": _num(ac.get("moneyLine")),
            "scored_at": datetime.now(ET).strftime("%Y-%m-%d %H:%M")}


def score():
    log = read_log()
    if log.empty:
        print("forward log is empty: run `python -m nba.today` before games tip off")
        return
    verify_chain(log)
    res = pd.read_csv(RESULTS, dtype={"game_id": str}) if RESULTS.exists() else pd.DataFrame(columns=["game_id"])
    todo = log[~log.game_id.isin(res.game_id)].drop_duplicates("game_id")
    new = [x for x in (fetch_outcome(r.game_id, r.game_date) for r in todo.itertuples()) if x]
    if new:
        res = pd.concat([res, pd.DataFrame(new)], ignore_index=True)
        res.to_csv(RESULTS, index=False)
    print(f"log verified: {len(log)} picks; {len(res)} games scored ({len(new)} new); "
          f"{log.game_id.nunique() - len(res)} awaiting results\n")
    report(log, res)


def report(log, res):
    d = log.merge(res, on="game_id")
    if d.empty:
        return
    dec = lambda x: np.where(x > 0, 1 + x / 100, 1 + 100 / np.abs(x))
    dh, da = dec(d.close_ml_home), dec(d.close_ml_away)
    d["close_mkt_p_home"] = (1 / dh) / (1 / dh + 1 / da)
    rows = []
    for v, g in d.groupby("model_version"):
        y = (g.home_pts > g.away_pts).to_numpy()
        p = np.clip(g.p_home.to_numpy(float), 1e-6, 1 - 1e-6)
        q = np.clip(g.now_mkt_p_home.to_numpy(float), 1e-6, 1 - 1e-6)
        ll = lambda x: float(np.mean(-(y * np.log(x) + (1 - y) * np.log(1 - x))))
        print(f"== {v}: {len(g)} games  winner accuracy model {np.mean((p >= .5) == y):.1%} vs market "
              f"{np.mean((q >= .5) == y):.1%}  |  log loss model {ll(p):.3f} vs market {ll(q):.3f} (lower is better)")
        s = settle(g, "now_")
        c = clv(g.rename(columns=lambda c: c.replace("now_", "open_") if c.startswith("now_") else c)).set_index("market")
        for mk in ("ml", "spread", "total"):
            b = s[s.market == mk]
            n = len(b)
            roi_lo, roi_hi = boot_ci(b.profit) if n > 1 else (np.nan, np.nan)
            cv = c.loc[mk] if mk in c.index else None
            rows.append({"model": v, "market": mk, "bets": n, "win_rate": b.win[~b.push].mean() if n else np.nan,
                         "units": b.profit.sum(), "roi": b.profit.mean() if n else np.nan, "roi_lo": roi_lo, "roi_hi": roi_hi,
                         "clv": cv["mean_clv"] if cv is not None else np.nan,
                         "clv_lo": cv["ci"][0] if cv is not None else np.nan,
                         "clv_hi": cv["ci"][1] if cv is not None else np.nan,
                         "clv_unit": cv["unit"] if cv is not None else ""})
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    print("\nEvidence bars (from the backtest discussion):")
    for r in t.itertuples():
        clv_ok = r.bets >= 300 and r.clv_lo > 0
        roi_ok = r.bets >= 1600 and r.roi_lo > 0
        print(f"  {r.model} {r.market:6s}: {r.bets:4d} bets | CLV bar (>=300 bets, CI above 0): "
              f"{'PASS' if clv_ok else 'not yet'} | ROI bar (>=1,600 bets, CI above 0): {'PASS' if roi_ok else 'not yet'}")
    t.to_csv(DIR / "scorecard.csv", index=False)


if __name__ == "__main__":
    score()
