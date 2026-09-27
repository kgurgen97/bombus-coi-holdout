#!/usr/bin/env bash
# Rebuild every table and figure of the paper. Needs the prepared genera in work/
# (see README: prepare + run_scenarios.py for each genus, and the -conflicts reruns).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
run() { echo "== $*"; "$PY" -W ignore "$@"; }

# Calibration (hours on a laptop). Skipped when its outputs exist; CALIBRATION=1 forces a rerun.
R=${HOLDOUT_WORK:-../work}/bombus/results
need() { [ "${CALIBRATION:-0}" = 1 ] || [ ! -e "$R/$1" ]; }
need v9/toy_model_grid.tsv && run toy_model.py
need v10/query_features.tsv.gz && run calibration.py
need v10_1/query_features_plus.tsv.gz && run calibration_nested.py
need v11/calibration/oob_predictions.tsv.gz && run calibration_unseen_blocks.py
need v11/calibration_continent/oob_predictions.tsv.gz && run calibration_unseen_continents.py

run report_outcomes.py
run compare_cleaning.py
run threshold_rules.py
run influential_records.py
run forward_in_time.py
run rule_bombus.py
run rule_features.py Bombus
run rule_features.py Bombus --conflicts
for g in Andrena Lasioglossum Megachile Eupithecia Xestia Aedes Hylaeus Pardosa; do
  run rule_features.py $g
done
run rule_by_genus.py
run rule_tradeoff.py
run same_weights.py
run forward_availability.py
run figures.py
run figure_mechanism.py
