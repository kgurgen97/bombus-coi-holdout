"""v8.1: BLAST top-hit identification (spec addendum v8.1 in docs_v8/FROZEN_SPEC_v8.md).

Builds a BLAST database of all analysis records (unaligned main sequences),
runs blastn for the query records in chunks, and converts hits into a
dissimilarity matrix d = -bitscore (query rows only; NaN = no hit; self-hits
removed), so that the v8 NN machinery (min = best, ties = equal bitscore)
applies unchanged. Raw tabular output is kept gzipped in blast/.
"""
from __future__ import annotations
import gzip, hashlib, subprocess, time
import numpy as np
import pandas as pd
from Bio import SeqIO
from holdout import config as C

from pathlib import Path
CHUNK = 500


def run(sh, cwd):
    subprocess.run(sh, cwd=cwd, check=True, capture_output=True)


def blast_matrix(D, query_idx, threads=8):
    BDIR = C.PROJECT_ROOT / "blast"
    WORK = Path.home() / "coi_blast_work" / C.PROJECT_ROOT.name   # makeblastdb fails on paths with spaces
    BDIR.mkdir(exist_ok=True)
    ids = list(D["ids"]); pos = {r: i for i, r in enumerate(ids)}
    key = hashlib.sha256(("|".join(ids) + "#" + ",".join(map(str, sorted(query_idx)))).encode()).hexdigest()[:16]
    cache = BDIR / f"bitscores_{key}.npz"
    N = len(ids)
    if cache.exists():
        z = np.load(cache)
        qi, bits = z["query_idx"], z["bits"]
    else:
        WORK.mkdir(parents=True, exist_ok=True)
        seqs = {r.id: r for r in SeqIO.parse(str(C.CLEAN / "main_coi.fasta"), "fasta")}
        SeqIO.write([seqs[i] for i in ids], str(WORK / "db.fasta"), "fasta")
        run(["makeblastdb", "-in", "db.fasta", "-dbtype", "nucl", "-out", "db"], WORK)
        qi = np.array(sorted(query_idx)); qpos = {q: k for k, q in enumerate(qi)}
        bits = np.full((len(qi), N), np.nan, np.float32)
        t0 = time.time()
        for c0 in range(0, len(qi), CHUNK):
            chunk = qi[c0:c0 + CHUNK]
            SeqIO.write([seqs[ids[q]] for q in chunk], str(WORK / "q.fasta"), "fasta")
            out = f"hits_{c0:06d}.tsv"
            run(["blastn", "-task", "blastn", "-query", "q.fasta", "-db", "db", "-evalue", "1e-5", "-dust", "no",
                 "-max_hsps", "1", "-max_target_seqs", str(N), "-outfmt", "6 qseqid sseqid bitscore pident length",
                 "-num_threads", str(threads), "-out", out], WORK)
            h = pd.read_csv(WORK / out, sep="\t", header=None, names=["q", "s", "bits", "pident", "len"],
                            dtype={"q": str, "s": str})
            r = h.q.map(lambda x: qpos[pos[x]]).to_numpy(); c = h.s.map(pos).to_numpy()
            # keep the best bitscore per (query, subject) (max_hsps 1 already, defensive)
            np.fmax.at(bits, (r, c), h.bits.to_numpy(np.float32))
            with open(WORK / out, "rb") as fi, gzip.open(BDIR / (out + ".gz"), "wb") as fo:
                fo.write(fi.read())
            (WORK / out).unlink()
            print(f"  BLAST chunk {c0}-{c0+len(chunk)} of {len(qi)} [{time.time()-t0:.0f}s]", flush=True)
        np.savez_compressed(cache, query_idx=qi, bits=bits)
        (WORK / "q.fasta").unlink(missing_ok=True)
    d = np.full((N, N), np.nan, np.float32)
    d[qi] = -bits
    d[qi, qi] = np.nan   # self-hits
    return d
