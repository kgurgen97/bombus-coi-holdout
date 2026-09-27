#!/usr/bin/env bash
# Download, clean and align every genus of the paper, then run all scenarios.
# Each genus also gets a rerun that keeps records with conflicting labels (work/<genus>-conflicts).
# Needs MAFFT; the "blast" scenario also needs BLAST+. Takes most of a day.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
for g in Bombus Andrena Lasioglossum Megachile Eupithecia Xestia Aedes Hylaeus Pardosa; do
  w=work/$(echo $g | tr A-Z a-z)
  "$PY" -c "import holdout; holdout.prepare('$g', '$w')"
  HOLDOUT_GENUS=$g HOLDOUT_WORKDIR=$w "$PY" -W ignore paper/run_scenarios.py all
  mkdir -p $w-conflicts && ln -sfn ../$(basename $w)/raw $w-conflicts/raw
  "$PY" -c "import holdout; holdout.prepare('$g', '$w-conflicts', keep_conflicts=True)"
  HOLDOUT_GENUS=$g HOLDOUT_WORKDIR=$w-conflicts "$PY" -W ignore paper/run_scenarios.py all
done
