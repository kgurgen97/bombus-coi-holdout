"""Geographic hold-out audit of COI reference libraries."""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path

from holdout.pipeline import prepare

_MODULES = ("config", "qc", "distances", "geography", "refsets", "stats", "scenarios", "blast", "audit", "plots")


def use(genus, workdir=None, keep_conflicts=False):
    """Select the genus and working folder for this session (reloads the modules that read them)."""
    workdir = Path(workdir or f"work/{genus.lower()}").resolve()
    os.environ.update(HOLDOUT_GENUS=genus, HOLDOUT_WORKDIR=str(workdir),
                      HOLDOUT_KEEP_CONFLICTS="1" if keep_conflicts else "0")
    for m in _MODULES:
        name = f"holdout.{m}"
        if name in sys.modules:
            importlib.reload(sys.modules[name])
    return workdir


def run(**kw):
    """Run the audit (see holdout.audit.run)."""
    from holdout.audit import run as _run
    return _run(**kw)
