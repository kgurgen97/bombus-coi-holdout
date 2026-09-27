"""v9 computational cap (docs_v9/PROTOCOL_v9.md, pre-registered).
If the main dataset has > 15,000 records, keep a seeded random 200 records of
every species with > 200 main records; the rest become excluded
('computational_cap'). Uncapped copies are kept. No-op otherwise."""
import shutil
import numpy as np
import pandas as pd
from Bio import SeqIO
from holdout import config as C

MAX_MAIN, CAP = 15000, 200


def main():
    meta_p, fa_p = C.CLEAN / "clean_metadata.tsv", C.CLEAN / "main_coi.fasta"
    m = pd.read_csv(meta_p, sep="\t", low_memory=False, dtype=str)
    n = int((m.dataset == "main").sum())
    if n <= MAX_MAIN:
        print(f"main={n} <= {MAX_MAIN}: no cap applied"); return
    for src, dst in ((meta_p, "clean_metadata_uncapped.tsv"), (fa_p, "main_coi_uncapped.fasta")):
        if not (C.CLEAN / dst).exists():
            shutil.copy2(src, C.CLEAN / dst)
    m = pd.read_csv(C.CLEAN / "clean_metadata_uncapped.tsv", sep="\t", low_memory=False, dtype=str)
    rng = np.random.default_rng(C.RANDOM_SEED)
    drop = []
    main = m[m.dataset == "main"]
    for sp, g in sorted(main.groupby("species"), key=lambda t: t[0]):
        if len(g) > CAP:
            ids = np.sort(g.record_id.to_numpy())
            keep = set(rng.choice(ids, CAP, replace=False))
            drop += [i for i in ids if i not in keep]
    sel = m.record_id.isin(drop)
    m.loc[sel, "dataset"] = "excluded"; m.loc[sel, "exclusion_reason"] = "computational_cap"
    m.to_csv(meta_p, sep="\t", index=False)
    keep_ids = set(m.record_id[m.dataset == "main"])
    recs = [r for r in SeqIO.parse(str(C.CLEAN / "main_coi_uncapped.fasta"), "fasta") if r.id in keep_ids]
    SeqIO.write(recs, str(fa_p), "fasta")
    print(f"main {n} -> {len(keep_ids)} (fasta {len(recs)}); capped species: "
          f"{int((main.species.value_counts() > CAP).sum())}")


if __name__ == "__main__":
    main()
