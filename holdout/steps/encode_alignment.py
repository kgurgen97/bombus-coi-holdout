"""Phase 4 prep: trim the MAFFT alignment to the core COI-5P barcode region and
precompute pairwise distance components for fast reuse.

We keep alignment columns with >= CORE_OCC occupancy (the Folmer window), one-hot
encode bases, and precompute two N x N integer matrices via BLAS matmul:
  matches[i,j] = # columns where i and j share the same base (both non-missing)
  valid[i,j]   = # columns where both i and j are non-missing (comparable sites)
p-distance(i,j) = 1 - matches/valid, computed on demand (pairwise deletion).
Saved so the exploratory and benchmark steps reuse identical distances.
"""
from __future__ import annotations
import json
import numpy as np
from Bio import SeqIO
from holdout import config as C

CORE_OCC = 0.5
MIN_CORE_COVER = 300   # a sequence needs >= this many core, non-missing columns


def main():
    ids, rows = [], []
    for r in SeqIO.parse(str(C.ALIGNMENT / "main_coi.aln.fasta"), "fasta"):
        ids.append(r.id)
        rows.append(np.frombuffer(str(r.seq).upper().encode("ascii"), dtype=np.uint8))
    arr = np.array(rows)                       # N x L uint8 (ascii)
    N, L = arr.shape
    DASH = ord("-")
    occ = (arr != DASH).mean(axis=0)
    core_cols = np.where(occ >= CORE_OCC)[0]
    core = arr[:, core_cols]                   # N x Lc
    Lc = core.shape[1]
    print(f"alignment {N}x{L} -> core {N}x{Lc} (cols occ>={CORE_OCC})")

    # base codes: A,C,G,T -> 0..3 ; everything else (gap, N, IUPAC ambig) -> missing
    base_map = {ord("A"): 0, ord("C"): 1, ord("G"): 2, ord("T"): 3}
    code = np.full(core.shape, -1, dtype=np.int8)
    for b, k in base_map.items():
        code[core == b] = k
    present = (code >= 0)                       # N x Lc bool
    core_cover = present.sum(axis=1)

    # one-hot (N x Lc x 4) -> (N x 4Lc) float32; missing rows are all-zero
    onehot = np.zeros((N, Lc, 4), dtype=np.float32)
    idx = np.where(present)
    onehot[idx[0], idx[1], code[idx]] = 1.0
    X = onehot.reshape(N, Lc * 4)
    P = present.astype(np.float32)

    print("computing matches = X @ X.T ...")
    matches = (X @ X.T).astype(np.int16)
    print("computing valid = P @ P.T ...")
    valid = (P @ P.T).astype(np.int16)

    np.save(C.ALIGNMENT / "matches.npy", matches)
    np.save(C.ALIGNMENT / "valid.npy", valid)
    (C.ALIGNMENT / "encoded_ids.txt").write_text("\n".join(ids))
    np.save(C.ALIGNMENT / "core_cover.npy", core_cover)

    # write the trimmed core alignment (for the record / tree)
    with open(C.ALIGNMENT / "main_coi.core.aln.fasta", "w") as fh:
        core_chars = core.view("S1").reshape(N, Lc)
        for i, sid in enumerate(ids):
            fh.write(f">{sid}\n{core_chars[i].tobytes().decode()}\n")

    meta = {
        "n_seqs": int(N), "alignment_cols": int(L), "core_cols": int(Lc),
        "core_occupancy_threshold": CORE_OCC,
        "min_core_cover_for_analysis": MIN_CORE_COVER,
        "seqs_below_min_core_cover": int((core_cover < MIN_CORE_COVER).sum()),
        "median_core_cover": int(np.median(core_cover)),
    }
    (C.ALIGNMENT / "core_encoding_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
