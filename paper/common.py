"""Folders shared by the paper scripts. Each genus lives in work/<genus>/ and its
conflict-kept rerun in work/<genus>-conflicts/. Bombus is the default project."""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("HOLDOUT_WORK", ROOT / "work"))
FIGURES = ROOT / "figures"
os.environ.setdefault("HOLDOUT_GENUS", "Bombus")
os.environ.setdefault("HOLDOUT_WORKDIR", str(WORK / "bombus"))
sys.path.insert(0, str(ROOT))

GENERA = ["Bombus", "Andrena", "Lasioglossum", "Megachile", "Eupithecia", "Xestia", "Aedes", "Hylaeus", "Pardosa"]


def workdir(genus, conflicts=False):
    return WORK / (genus.lower() + ("-conflicts" if conflicts else ""))


def code_files(*names):
    """Files whose content keys the calibration caches."""
    return [ROOT / "holdout" / "params.json", ROOT / "holdout" / "scenarios.py"] + [ROOT / "paper" / n for n in names]
