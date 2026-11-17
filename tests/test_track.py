# forward-log self-check (run: PYTHONPATH=. .venv/bin/python tests/test_track.py): chain verifies after CSV round trip, and detects an edit
import pandas as pd, numpy as np, tempfile, pathlib
import nba.track as T
T.DIR = pathlib.Path(tempfile.mkdtemp()); T.LOG = T.DIR / "log.csv"
fut = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=3)).isoformat()
rows = pd.DataFrame([{"game_id": f"live_{i}", "game_date": "2099-01-01", "matchup": "A @ B", "tip_utc": fut,
                      "p_home": 0.6123456789, "now_ml_home": np.int64(-150), "now_ml_away": 130, "now_home_spread": -3.5,
                      "now_ml_bet": "home", "inj_published": None} for i in range(3)])
print(T.log_picks(rows, "vT"))
print(T.log_picks(rows, "vT"))                     # duplicates refused
T.verify_chain(T.read_log()); print("chain ok after round trip")
log = T.read_log(); log.loc[1, "now_home_spread"] = -5.5; log.to_csv(T.LOG, index=False)
try:
    T.verify_chain(T.read_log()); print("EDIT NOT DETECTED")
except AssertionError as e:
    print("edit detected:", e)
past = rows.assign(game_id="live_9", tip_utc=(pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)).isoformat()).head(1)
T.LOG.unlink(); print(T.log_picks(past, "vT"))     # after tip refused
