"""Shared constants: paths, seasons, split boundaries, team codes."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DB_PATH = DATA / "nba.sqlite"
OUT = ROOT / os.environ.get("NBA_OUT", "outputs")   # v2: NBA_OUT=outputs_v2

# Season label = year the season ends (2027 == 2026-27), matching the odds dataset.
FIRST_SEASON = 2008          # 2007-08: warm-up only, never evaluated
LAST_SEASON = 2027           # 2026-27: in progress, untouched holdout
DEV_SEASONS = range(2009, 2020)       # 2008-09 .. 2018-19  model development
VALID_SEASONS = range(2020, 2023)     # 2019-20 .. 2021-22  tuning / selection
TEST_SEASONS = range(2023, 2027)      # 2022-23 .. 2025-26  frozen-config test, run once
HOLDOUT_SEASONS = range(2027, 2028)   # 2026-27            final holdout


def split_of(season: int) -> str:
    if season in DEV_SEASONS: return "dev"
    if season in VALID_SEASONS: return "valid"
    if season in TEST_SEASONS: return "test"
    if season in HOLDOUT_SEASONS: return "holdout"
    return "warmup"


def season_str(season: int) -> str:
    return f"{season-1}-{str(season)[-2:]}"


# NBA franchise TEAM_ID -> codes used by other sources (Kaggle/SBR and ESPN, case-insensitive).
TEAM_CODES = {
    1610612737: ["atl"], 1610612738: ["bos"], 1610612751: ["bkn", "nj", "njn"],
    1610612766: ["cha"], 1610612741: ["chi"], 1610612739: ["cle"], 1610612742: ["dal"],
    1610612743: ["den"], 1610612765: ["det"], 1610612744: ["gs", "gsw"], 1610612745: ["hou"],
    1610612754: ["ind"], 1610612746: ["lac"], 1610612747: ["lal"], 1610612763: ["mem"],
    1610612748: ["mia"], 1610612749: ["mil"], 1610612750: ["min"],
    1610612740: ["no", "nop", "noh", "nok"], 1610612752: ["ny", "nyk"],
    1610612760: ["okc", "sea"], 1610612753: ["orl"], 1610612755: ["phi"],
    1610612756: ["phx", "pho"], 1610612757: ["por"], 1610612758: ["sac"],
    1610612759: ["sa", "sas"], 1610612761: ["tor"], 1610612762: ["utah", "uta"],
    1610612764: ["wsh", "was"],
}
CODE_TO_ID = {c: tid for tid, cs in TEAM_CODES.items() for c in cs}

# Home arena coordinates (lat, lon) for travel distance. Seattle used for OKC franchise in 2007-08.
ARENAS = {
    1610612737: (33.757, -84.396), 1610612738: (42.366, -71.062), 1610612751: (40.683, -73.975),
    1610612766: (35.225, -80.839), 1610612741: (41.881, -87.674), 1610612739: (41.497, -81.688),
    1610612742: (32.790, -96.810), 1610612743: (39.749, -105.008), 1610612765: (42.341, -83.055),
    1610612744: (37.768, -122.388), 1610612745: (29.751, -95.362), 1610612754: (39.764, -86.156),
    1610612746: (33.945, -118.343), 1610612747: (34.043, -118.267), 1610612763: (35.138, -90.051),
    1610612748: (25.781, -80.188), 1610612749: (43.045, -87.917), 1610612750: (44.979, -93.276),
    1610612740: (29.949, -90.082), 1610612752: (40.751, -73.994), 1610612760: (35.463, -97.515),
    1610612753: (28.539, -81.384), 1610612755: (39.901, -75.172), 1610612756: (33.446, -112.071),
    1610612757: (45.532, -122.667), 1610612758: (38.580, -121.500), 1610612759: (29.427, -98.438),
    1610612761: (43.643, -79.379), 1610612762: (40.768, -111.901), 1610612764: (38.898, -77.021),
}
SEATTLE = (47.622, -122.354)
