"""Download, clean and align COI records for one genus, then compute distances and geography."""
from __future__ import annotations
import os, shutil, subprocess, sys, time
from pathlib import Path

STEPS = ["download_bold", "download_genbank", "build_dataset", "cap_species", "align",
         "encode_alignment", "flag_outliers", "specimen_keys", "eligibility"]


def env(genus, workdir, keep_conflicts=False):
    e = dict(os.environ, HOLDOUT_GENUS=genus, HOLDOUT_WORKDIR=str(Path(workdir).resolve()),
             HOLDOUT_KEEP_CONFLICTS="1" if keep_conflicts else "0")
    e["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + e.get("PYTHONPATH", "")
    return e


def _done(step, wd, genus):
    raw = {"download_bold": wd / "raw" / f"bold_{genus.lower()}_v5.tsv",
           "download_genbank": wd / "raw" / f"genbank_{genus.lower()}_coi.gb"}
    return raw.get(step, wd / "logs" / f"{step}.done").exists()


def prepare(genus, workdir, keep_conflicts=False, threads=8, redo=False, quiet=True):
    wd = Path(workdir).resolve(); (wd / "logs").mkdir(parents=True, exist_ok=True)
    e = env(genus, wd, keep_conflicts)
    if shutil.which("mafft") is None:
        raise RuntimeError("MAFFT is not on PATH (e.g. `brew install mafft` or `conda install -c bioconda mafft`)")
    for step in STEPS:
        if not redo and _done(step, wd, genus):
            print(f"{step:18s} already done"); continue
        t0 = time.time(); log = open(wd / "logs" / f"{step}.log", "w")
        if step == "align":
            with open(wd / "alignment" / "main_coi.aln.fasta", "w") as out:
                subprocess.run(["mafft", "--auto", "--thread", str(threads), str(wd / "clean" / "main_coi.fasta")],
                               stdout=out, stderr=log, check=True)
        else:
            subprocess.run([sys.executable, "-W", "ignore", "-m", f"holdout.steps.{step}"], env=e, cwd=wd,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        (wd / "logs" / f"{step}.done").touch()
        print(f"{step:18s} {time.time() - t0:6.0f} s")
    return wd
