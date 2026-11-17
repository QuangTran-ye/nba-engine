#!/usr/bin/env bash
# Full reproducible pipeline. Raw downloads are cached in data/raw; delete them to refetch.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
PY=.venv/bin/python
$PY -m nba.ingest          # sources -> data/nba.sqlite
$PY -m nba.validate        # integrity asserts + outputs/data_validation.json
$PY -m nba.injuries        # official injury reports 2021-22+ (rebuilt after every ingest)
if [ "${1:-}" = "--retune" ] || [ ! -f outputs/model_registry.json ]; then
  $PY -m nba.backtest tune # validation seasons only
fi
$PY -m nba.backtest final  # frozen config, all seasons, locks outputs/predictions.csv
$PY -m nba.leakage_test    # temporal leakage checks
$PY -m nba.evaluate        # verifies lock, then joins outcomes
$PY -m nba.viz             # charts
# v2 (injury features): NBA_OUT=outputs_v2 for backtest final / leakage_test / evaluate
# Daily: NBA_OUT=outputs_v2 $PY -m nba.today   (predict + log picks before tip)
#        $PY -m nba.track                       (score finished games, scorecard)
