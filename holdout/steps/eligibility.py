"""v3 step 1: sub-continental geography, library coverage and species eligibility.

Eligibility (primary benchmark), per species, using v3 analysis-set records in
resolved blocks:
  * >= 20 usable sequences;
  * occurs (>= 2 records) in >= 3 blocks;
  * >= 2 testable blocks (>= 2 queries in the block AND >= 1 conspecific
    record in another resolved block);
  * reliable labels: resolvable name and < 10% of records flagged as
    conspecific outliers.
Outputs:
  metadata/v3_record_geography.tsv
  results/v3/region_blocks.tsv
  results/library_coverage.tsv          (species x block counts + species summary)
  results/v3/eligibility_v3.tsv, results/v3/eligibility_v3.json
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from holdout import config as C
from holdout import geography as V

OUT = C.RESULTS / "v3"; OUT.mkdir(exist_ok=True)


def gc_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2 +
         np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    D, meta, dist = V.load_v3()
    meta[["record_id", "species", "country", "province_state", "lat", "lon", "region", "block",
          "continent", "specimen_key", "study_key", "in_v3", "flag_isolated",
          "flag_conspecific_outlier"]].to_csv(C.METADATA / "v3_record_geography.tsv", sep="\t", index=False)
    use = meta[meta.in_v3 & V.resolved(meta.block.to_numpy())]
    blocks = (meta[meta.in_v3].groupby(["continent", "block"])
              .agg(n_records=("record_id", "size"), n_species=("species", "nunique"),
                   n_countries=("country", "nunique"),
                   frac_coords=("coord_valid", "mean")).reset_index())
    blocks.to_csv(OUT / "region_blocks.tsv", sep="\t", index=False)

    # ---- species x block coverage ----
    cov = (use.groupby(["species", "block"])
           .agg(n_records=("record_id", "size"), n_haplotypes=("seq_hash", "nunique"),
                n_studies=("study_key", "nunique"),
                frac_coords=("coord_valid", "mean")).reset_index())
    cov["continent"] = cov.block.map(V.continent_of)
    sp_rows = []
    for s, g in use.groupby("species"):
        bc = g.block.value_counts()
        occ = bc[bc >= 2]
        testable = [b for b in occ.index if (bc.drop(b).sum() >= 1)]
        ll = g[g.coord_valid.fillna(False).astype(bool)][["lat", "lon"]].astype(float).to_numpy()
        if len(ll) >= 2:
            sel = ll if len(ll) <= 400 else ll[np.random.default_rng(C.RANDOM_SEED).choice(len(ll), 400, replace=False)]
            dd = gc_km(sel[:, None, 0], sel[:, None, 1], sel[None, :, 0], sel[None, :, 1])
            range_km = float(np.percentile(dd[np.triu_indices(len(sel), 1)], 95))
        else:
            range_km = np.nan
        n_all = int(((meta.species == s) & meta.in_v3).sum())
        sp_rows.append({
            "species": s, "n_usable_resolved": len(g), "n_all_v3": n_all,
            "n_blocks_ge1": int((bc >= 1).sum()), "n_blocks_ge2": int(len(occ)),
            "n_continents": int(g.continent.nunique()),
            "blocks": ";".join(f"{b}:{n}" for b, n in bc.items()),
            "n_testable_blocks": len(testable), "testable_blocks": ";".join(testable),
            "dominant_block_share": round(float(bc.iloc[0] / bc.sum()), 3),
            "n_undersampled_blocks_lt5": int(((bc >= 1) & (bc < 5)).sum()),
            "range_km_p95": round(range_km, 0) if range_km == range_km else np.nan,
            "frac_coords": round(float(g.coord_valid.fillna(False).astype(bool).mean()), 3),
            "frac_voucher": round(float(g.voucher.fillna("").astype(str).str.len().gt(0).mean()), 3),
            "frac_conspecific_outlier": round(float(g.flag_conspecific_outlier.mean()), 3),
            "name_unresolved": bool(g.name_unresolved.any()),
        })
    sdf = pd.DataFrame(sp_rows)
    sdf["eligible_v3"] = ((sdf.n_usable_resolved >= C.MIN_SEQS_PER_SPECIES) &
                          (sdf.n_blocks_ge2 >= 3) & (sdf.n_testable_blocks >= 2) &
                          (sdf.frac_conspecific_outlier < 0.10) & ~sdf.name_unresolved)
    sdf["eligible_loco"] = sdf.eligible_v3 & (sdf.n_continents >= 2)
    sdf = sdf.sort_values(["eligible_v3", "n_usable_resolved"], ascending=False)
    sdf.to_csv(OUT / "eligibility_v3.tsv", sep="\t", index=False)
    # library_coverage.tsv = species x block table joined with species summary
    lc = cov.merge(sdf[["species", "n_usable_resolved", "n_blocks_ge2", "n_continents",
                        "dominant_block_share", "eligible_v3"]], on="species")
    lc.to_csv(C.RESULTS / "library_coverage.tsv", sep="\t", index=False)
    el = sdf[sdf.eligible_v3]
    summary = {
        "n_records_v3_analysis": int(meta.in_v3.sum()),
        "n_records_resolved_block": int(len(use)),
        "n_records_unresolved_or_unknown": int(meta.in_v3.sum() - len(use)),
        "n_excluded_isolated": int((meta.flag_isolated).sum()),
        "n_blocks": int(use.block.nunique()), "n_species_resolved": int(use.species.nunique()),
        "n_eligible_v3": int(len(el)), "n_eligible_loco": int(sdf.eligible_loco.sum()),
        "eligible_by_continent_presence": {c: int(sum(c in str(b) for b in el.blocks))
                                           for c in ("EU_", "AS_", "NA_", "SA_", "OC_", "AF_")},
        "eligible_species": el.species.tolist(),
    }
    (OUT / "eligibility_v3.json").write_text(json.dumps(summary, indent=2))
    print(blocks.to_string(index=False)); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
