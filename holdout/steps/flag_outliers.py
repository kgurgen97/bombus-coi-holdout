"""Post-hoc sequence-outlier audit (contamination / mislabel candidates).

For every analysis-set record: distance to nearest conspecific and nearest
heterospecific record (core p-distance, overlap >= MIN_OVERLAP).
Flags (records are FLAGGED, never deleted):
  isolated            nearest record of ANY species > 8%  (non-Bombus-like / contaminant / NUMT)
  conspecific_outlier nearest conspecific > 5% while a heterospecific is < 2%
                      (label likely wrong or cryptic lineage)
  bin_multispecies    BOLD BIN carries >1 species name in the analysis set
Outputs: results/outlier_flags.tsv, results/outlier_summary.tsv
"""
import numpy as np, pandas as pd
from holdout import config as C
from holdout import distances as A

D = A.load(); m = D["meta"].reset_index(drop=True)
ok = D["core_cover"] >= A.MIN_OVERLAP
d = A.pdist_matrix(D["matches"], D["valid"])
sp = m.species.to_numpy(); idx = np.where(ok)[0]
sub = d[np.ix_(idx, idx)]; ssp = sp[idx]
same = ssp[:, None] == ssp[None, :]
with np.errstate(all="ignore"):
    nn_any = np.nanmin(sub, axis=1)
    nn_con = np.nanmin(np.where(same, sub, np.nan), axis=1)
    nn_het = np.nanmin(np.where(~same, sub, np.nan), axis=1)
    het_j = np.nanargmin(np.where(~same, np.nan_to_num(sub, nan=9), 9), axis=1)
df = m.iloc[idx][["record_id", "source", "accession", "species", "country", "region",
                  "bold_bin", "voucher"]].copy()
df["nn_any"] = nn_any.round(4); df["nn_conspecific"] = nn_con.round(4)
df["nn_heterospecific"] = nn_het.round(4); df["nearest_heterospecific_species"] = ssp[het_j]
bins = df[df.bold_bin.notna() & (df.bold_bin != "")].groupby("bold_bin").species.nunique()
multi = set(bins[bins > 1].index)
df["flag_isolated"] = df.nn_any > 0.08
df["flag_conspecific_outlier"] = (df.nn_conspecific.fillna(1) > 0.05) & (df.nn_heterospecific < 0.02)
df["flag_bin_multispecies"] = df.bold_bin.isin(multi)
df["any_flag"] = df.flag_isolated | df.flag_conspecific_outlier
df.to_csv(C.RESULTS / "outlier_flags.tsv", sep="\t", index=False)

rows = [{"metric": k, "value": int(df[k].sum())} for k in
        ("flag_isolated", "flag_conspecific_outlier", "flag_bin_multispecies", "any_flag")]
rows.append({"metric": "n_bins_multispecies", "value": len(multi)})
top = df[df.any_flag].groupby(["species", "region"]).size().sort_values(ascending=False).head(10)
for (s, r), n in top.items():
    rows.append({"metric": f"flagged::{s}::{r}", "value": int(n)})
out = pd.DataFrame(rows); out.to_csv(C.RESULTS / "outlier_summary.tsv", sep="\t", index=False)
print(out.to_string(index=False))
