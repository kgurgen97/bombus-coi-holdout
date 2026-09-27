"""Shared loader for precomputed distances + aligned metadata (Phases 4-6)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from holdout import config as C

MIN_OVERLAP = 300      # minimum comparable core columns for a valid p-distance


def load():
    ids = (C.ALIGNMENT / "encoded_ids.txt").read_text().splitlines()
    matches = np.load(C.ALIGNMENT / "matches.npy")
    valid = np.load(C.ALIGNMENT / "valid.npy")
    core_cover = np.load(C.ALIGNMENT / "core_cover.npy")
    meta = pd.read_csv(C.CLEAN / "clean_metadata.tsv", sep="\t", low_memory=False)
    meta = meta[meta["dataset"] == "main"].copy()
    meta = meta.set_index("record_id")
    # align metadata to the encoded id order
    meta = meta.loc[ids]
    return {
        "ids": ids, "matches": matches, "valid": valid,
        "core_cover": core_cover, "meta": meta.reset_index(),
    }


def pdist_matrix(matches, valid, min_overlap=MIN_OVERLAP):
    """Full N x N float32 p-distance matrix; invalid pairs -> np.nan."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1.0 - (matches.astype(np.float32) / valid.astype(np.float32))
    d[valid < min_overlap] = np.nan
    np.fill_diagonal(d, np.nan)
    return d
