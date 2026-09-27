"""Shared helpers for the v3 (expanded) analysis.

Geography: two nested levels
  continent  : Europe | Asia | North_America | South_America | Africa | Oceania
  block      : sub-continental biogeographic block (the unit of hold-out)
Assignment priority: valid coordinates (large countries) -> BOLD province/state
-> GenBank geo_loc_name ("Country: State") -> country. Records of large
countries without sub-national information get block "<country>_unresolved";
they are never used as test queries, and are excluded from the reference
whenever their country overlaps the held-out block.

Leakage rule v3 (reference for a split excludes any record that):
  * is in the test split;
  * has a raw sequence identical to any test query (seq_hash);
  * is at core p-distance 0 (>= MIN_OVERLAP columns) from any test query;
  * shares a normalised specimen key with any test query;
  * optionally (study-blocked sensitivity) shares a study key with any test query.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from holdout import config as C
from holdout import distances as A

UN = "__UNASSIGNED__"

EU_NORTH = {"Norway", "Sweden", "Finland", "Denmark", "Iceland", "Estonia", "Latvia", "Lithuania"}
EU_WEST = {"United Kingdom", "Ireland", "France", "Belgium", "Netherlands", "Luxembourg"}
EU_CENTRAL = {"Germany", "Austria", "Switzerland", "Liechtenstein", "Poland", "Czech Republic",
              "Czechia", "Slovakia", "Hungary", "Slovenia"}
EU_SOUTH = {"Spain", "Portugal", "Andorra", "Italy", "Greece", "Croatia", "Serbia", "Bulgaria",
            "Albania", "North Macedonia", "Bosnia and Herzegovina", "Montenegro", "Malta", "San Marino"}
EU_EAST = {"Belarus", "Ukraine", "Romania", "Moldova"}
AF_NORTH = {"Morocco", "Algeria", "Tunisia"}
AS_WEST = {"Turkey", "Turkiye", "Iran", "Georgia", "Armenia", "Azerbaijan", "Israel", "Lebanon",
           "Cyprus", "Syria", "Iraq", "Jordan"}
AS_CENTRAL = {"Kazakhstan", "Kyrgyzstan", "Tajikistan", "Uzbekistan", "Turkmenistan"}
AS_SIB = {"Mongolia"}
AS_FAREAST = {"South Korea", "Korea, South", "North Korea"}
AS_JAPAN = {"Japan"}
AS_CHINA = {"Taiwan"}
AS_SOUTH = {"India", "Nepal", "Pakistan", "Bhutan", "Afghanistan"}
AS_SE = {"Thailand", "Myanmar", "Vietnam", "Laos", "Malaysia", "Indonesia", "Philippines"}
MESO = {"Mexico", "Guatemala", "Honduras", "Costa Rica", "Panama", "El Salvador", "Nicaragua", "Belize"}
SA_ANDES = {"Ecuador", "Colombia", "Peru", "Bolivia", "Chile", "Argentina", "Venezuela"}
SA_EAST = {"Brazil", "Paraguay", "Uruguay"}
NA_GREEN = {"Greenland"}
OCEANIA = {"New Zealand", "Australia", "Tasmania"}

STATIC = {}
for s, b in [(EU_NORTH, "EU_North"), (EU_WEST, "EU_West"), (EU_CENTRAL, "EU_Central"),
             (EU_SOUTH, "EU_South"), (EU_EAST, "EU_East"), (AF_NORTH, "AF_North"),
             (AS_WEST, "AS_West"), (AS_CENTRAL, "AS_Central"), (AS_SIB, "AS_Siberia_Mongolia"),
             (AS_FAREAST, "AS_FarEast"), (AS_JAPAN, "AS_Japan"), (AS_CHINA, "AS_China"),
             (AS_SOUTH, "AS_South_Himalaya"), (AS_SE, "AS_Southeast"), (MESO, "NA_Mesoamerica"),
             (SA_ANDES, "SA_Andes"), (SA_EAST, "SA_East"), (NA_GREEN, "NA_North"),
             (OCEANIA, "OC_Oceania")]:
    for c in s:
        STATIC[c] = b

CONTINENT = {"EU": "Europe", "AF": "Africa", "AS": "Asia", "NA": "North_America",
             "SA": "South_America", "OC": "Oceania"}

US_WEST = {"washington", "oregon", "california", "nevada", "idaho", "montana", "wyoming", "utah",
           "colorado", "arizona", "new mexico", "hawaii"}
CA_WEST = {"british columbia", "alberta", "saskatchewan"}
CA_NORTH = {"yukon", "northwest territories", "nunavut"}
RU_FAREAST = ("primor", "sakhalin", "kamchat", "khabarov", "amur", "magadan", "chukot", "sakha",
              "yakut", "jewish", "kuril")
RU_SIB = ("novosib", "omsk", "krasnoyar", "altai", "tomsk", "kemerov", "irkut", "buryat", "tuva",
          "tyva", "zabaykal", "chita", "yamal", "khanty", "tyumen", "taymyr", "khakas", "kurgan")


def _sub(prov: str, gbloc: str) -> str:
    p = prov.strip().lower() if isinstance(prov, str) else ""
    if not p and isinstance(gbloc, str) and ":" in gbloc:
        p = gbloc.split(":", 1)[1].split(",")[0].strip().lower()
    return p


def assign_block(country, lat, lon, coord_ok, prov, gbloc):
    c = country if isinstance(country, str) else ""
    if not c or c in ("Unspecified country", "Atlantic Ocean"):
        return "Unknown"
    has = bool(coord_ok) and lat == lat and lon == lon
    sub = _sub(prov, gbloc)
    if c in ("United States", "USA"):
        if sub == "alaska" or (has and lon < -140):
            return "NA_North"
        if has:
            return "NA_West" if lon < -102 else "NA_East"
        if sub:
            return "NA_West" if sub in US_WEST else "NA_East"
        return "United States_unresolved"
    if c == "Canada":
        if has:
            if lat >= 60:
                return "NA_North"
            return "NA_West" if lon < -102 else "NA_East"
        if sub:
            if sub in CA_NORTH: return "NA_North"
            return "NA_West" if sub in CA_WEST else "NA_East"
        return "Canada_unresolved"
    if c == "Russia":
        if has:
            return "EU_East" if lon < 60 else ("AS_Siberia_Mongolia" if lon < 110 else "AS_FarEast")
        if sub:
            if any(k in sub for k in RU_FAREAST): return "AS_FarEast"
            if any(k in sub for k in RU_SIB): return "AS_Siberia_Mongolia"
            return "EU_East"
        return "Russia_unresolved"
    if c == "China":
        if has:
            if lon < 80: return "AS_Central"
            if lat > 45 and lon > 115: return "AS_FarEast"
            return "AS_China"
        if sub:
            if "xinjiang" in sub: return "AS_Central"
            if "inner mongolia" in sub or "nei mongol" in sub: return "AS_Siberia_Mongolia"
            if any(k in sub for k in ("heilong", "jilin", "liaoning")): return "AS_FarEast"
            return "AS_China"
        return "China_unresolved"
    return STATIC.get(c, "Unknown")


def continent_of(block: str) -> str:
    if block == "Unknown" or block.endswith("_unresolved"):
        c = block.replace("_unresolved", "")
        return {"United States": "North_America", "Canada": "North_America",
                "Russia": "Russia_unresolved", "China": "Asia"}.get(c, "Unknown")
    return CONTINENT[block.split("_")[0]]


def block_country_sets(meta):
    """countries contributing to each block (for excluding unresolved records)."""
    return meta.groupby("block")["country"].apply(lambda s: set(s.dropna())).to_dict()


# ---------------------------------------------------------------------------
def load_v3():
    """Main analysis set + v3 geography + aux keys + QC flags, aligned to the
    precomputed distance matrix order."""
    D = A.load()
    meta = D["meta"].reset_index(drop=True)
    aux = pd.read_csv(C.METADATA / "record_aux.tsv", sep="\t", low_memory=False,
                      dtype=str).set_index("record_id").loc[meta.record_id].reset_index()
    meta["province_state"] = aux.province_state
    meta["specimen_key"] = aux.specimen_key.fillna("")
    meta["study_key"] = aux.study_key.fillna("")
    meta["bold_project"] = aux.bold_project.fillna("")
    meta["block"] = [assign_block(c, la, lo, ok, p, g) for c, la, lo, ok, p, g in
                     zip(meta.country, meta.lat, meta.lon, meta.coord_valid.fillna(False),
                         aux.province_state, aux.gb_geo_loc_name)]
    meta["continent"] = meta.block.map(continent_of)
    fl = pd.read_csv(C.RESULTS / "outlier_flags.tsv", sep="\t").set_index("record_id")
    meta["flag_isolated"] = meta.record_id.map(fl.flag_isolated).fillna(False).astype(bool)
    meta["flag_conspecific_outlier"] = meta.record_id.map(fl.flag_conspecific_outlier).fillna(False).astype(bool)
    meta["name_unresolved"] = meta.species.str.contains(r"[?]", regex=True)
    ok = D["core_cover"] >= A.MIN_OVERLAP
    # v3 analysis set: comparable core + not isolated (>8% from every record:
    # label-free QC) + resolvable species name
    meta["in_v3"] = ok & ~meta.flag_isolated.to_numpy() & ~meta.name_unresolved.to_numpy()
    dist = A.pdist_matrix(D["matches"], D["valid"])
    return D, meta, dist


def resolved(block):
    return (block != "Unknown") & ~pd.Series(block).str.endswith("_unresolved").to_numpy()


def blocked_reference(test_idx, base_mask, meta, dist, study=False, haplotype=True):
    """Apply the v3 leakage rule. base_mask: candidate reference records.
    haplotype=False gives the 'haplotype-shared' variant: identical sequences
    from OTHER specimens stay in the reference (specimen duplicates and the
    test records themselves are still removed)."""
    m = base_mask.copy()
    m[test_idx] = False
    hh = meta.seq_hash.to_numpy()
    if haplotype:
        m &= ~np.isin(hh, list(set(hh[test_idx])))
    sk = meta.specimen_key.to_numpy()
    tk = set(sk[test_idx]) - {""}
    if tk:
        m &= ~np.isin(sk, list(tk))
    if study:
        st = meta.study_key.to_numpy()
        ts = set(st[test_idx]) - {""}
        if ts:
            m &= ~np.isin(st, list(ts))
    if haplotype:
        cand = np.where(m)[0]
        zero = np.zeros(len(cand), bool)
        for k in range(0, len(test_idx), 512):
            zero |= np.any(dist[np.ix_(test_idx[k:k + 512], cand)] == 0, axis=0)
        m[cand[zero]] = False
    return np.where(m)[0]


def nn_scores(dist, ref_idx, test_idx, sp):
    """NN1 prediction plus confidence features for each query:
    d1 (nearest distance), pred (species of nearest), d2 (nearest distance to a
    species other than pred), d_con (nearest conspecific-to-TRUE-label ref),
    n_con (number of true-label refs), ref_j (index of nearest ref)."""
    sub = dist[np.ix_(test_idx, ref_idx)]
    sub = np.where(np.isnan(sub), np.inf, sub)
    rsp = sp[ref_idx]
    j = np.argmin(sub, axis=1)
    d1 = sub[np.arange(len(test_idx)), j]
    pred = np.where(np.isinf(d1), UN, rsp[j])
    same_pred = rsp[None, :] == pred[:, None]
    d2 = np.where(same_pred, np.inf, sub).min(axis=1)
    tsp = sp[test_idx]
    con = rsp[None, :] == tsp[:, None]
    d_con = np.where(con, sub, np.inf).min(axis=1)
    d_het = np.where(con, np.inf, sub).min(axis=1)
    n_con = con.sum(axis=1)
    return pd.DataFrame({"pred": pred, "d1": d1, "d2": d2, "d_con": d_con, "d_het": d_het,
                         "n_con": n_con, "ref_j": np.asarray(ref_idx)[j]})


def grouped_folds(idx, labels, groups, k, rng):
    """Group-aware, label-balanced k folds (whole groups kept together)."""
    gdf = pd.DataFrame({"i": idx, "lab": labels, "g": groups})
    gl = gdf.groupby("g").agg(lab=("lab", lambda s: s.mode().iloc[0]), n=("i", "size"))
    gl = gl.sample(frac=1, random_state=int(rng.integers(1e9)))
    folds = [[] for _ in range(k)]; load = {}
    for lab, sub in gl.groupby("lab", sort=True):
        cnt = np.zeros(k)
        for g, row in sub.iterrows():
            f = int(np.argmin(cnt + 1e-3 * np.array([len(x) for x in folds])))
            folds[f].append(g); cnt[f] += row.n
    g2f = {g: f for f, gs in enumerate(folds) for g in gs}
    fid = gdf.g.map(g2f).to_numpy()
    return [np.asarray(idx)[fid == f] for f in range(k)]


def union_groups(meta, idx):
    """Group id = connected components of (same seq_hash OR same specimen key),
    so identical sequences and duplicate specimen records stay together."""
    import scipy.sparse as sps
    from scipy.sparse.csgraph import connected_components
    idx = np.asarray(idx); n = len(idx)
    keys = {}
    rows, cols = [], []
    for pos, i in enumerate(idx):
        for key in (("h", meta.seq_hash.iat[i]), ("s", meta.specimen_key.iat[i])):
            if key[1]:
                if key in keys:
                    rows.append(pos); cols.append(keys[key])
                else:
                    keys[key] = pos
    g = sps.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab
