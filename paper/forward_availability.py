"""v15: forward-in-time test with corrected availability dates.
Availability = earliest of BOLD sequence_upload_date and GenBank 'Submitted' date (process-ID creation date not used).
Reuses forward_in_time.py unchanged otherwise. Outputs: results/v15/prospective/."""
import re
import numpy as np
import pandas as pd
import common
from holdout import config as C
import forward_in_time as m74
OUT = C.RESULTS / "v15" / "prospective"; OUT.mkdir(parents=True, exist_ok=True)


def availability_dates(meta):
    b = pd.read_csv(C.RAW / "bold_bombus_v5.tsv", sep="\t", dtype=str, usecols=["processid", "sequence_upload_date"])
    bold = b.assign(d=pd.to_datetime(b.sequence_upload_date, errors="coerce")).groupby("processid").d.min()
    sub, acc = {}, None
    with open(C.RAW / "genbank_bombus_coi.gb") as fh:
        for line in fh:
            if line.startswith("ACCESSION"):
                acc = line.split()[1]
            m = re.search(r"Submitted \((\d{2}-[A-Z]{3}-\d{4})\)", line)
            if m and acc:
                d = pd.to_datetime(m.group(1), format="%d-%b-%Y"); sub[acc] = min(sub.get(acc, d), d)
    out = []
    for r, a in zip(meta.record_id, meta.accession.fillna("")):
        ds = [bold.get(r, pd.NaT)] + [sub[x] for x in {str(a).split(".")[0], str(r).split(".")[0]} if x in sub]
        ds = [d for d in ds if pd.notna(d)]
        out.append(min(ds) if ds else pd.NaT)
    return pd.Series(out, index=meta.index)


if __name__ == "__main__":
    m74.deposit_dates = availability_dates
    m74.OUT = OUT
    m74.main()
    # trade-off counts (as rule_tradeoff.py) for the corrected split
    rows = []
    for cut in ("2021", "2019"):
        d = pd.read_csv(OUT / f"queries_cutoff_{cut}.tsv.gz", sep="\t")
        m0, m1, U = d.one_P0.to_numpy(bool), d.one_P1.to_numpy(bool), d.U.to_numpy()
        mv = m0 & ~m1
        rows.append({"cutoff": cut, "n_queries": len(d), "P0_wrong_among_single": (m0 & (U == 0)).sum() / m0.sum(),
                     "P1_wrong_among_single": (m1 & (U == 0)).sum() / m1.sum(), "wrong_moved": int((mv & (U == 0)).sum()),
                     "correct_moved": int((mv & (U == 1)).sum()), "ratio": (mv & (U == 1)).sum() / max((mv & (U == 0)).sum(), 1)})
    T = pd.DataFrame(rows); T.to_csv(OUT / "tradeoff_forward.tsv", sep="\t", index=False); print(T.round(4).to_string(index=False))
