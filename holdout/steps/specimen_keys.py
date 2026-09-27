"""Auxiliary per-record metadata recovered from the unchanged raw files.

03_build_datasets.py parsed but did not keep several fields needed for the
leakage audit and the suspicious-record review (BOLD project codes, specimen
identifiers, province/state, identification method, sequencing site; GenBank
voucher/isolate, full geo_loc_name and publication references). This script
re-reads raw/ (read-only) and writes one row per clean record_id, merging the
BOLD and GenBank sides of 'both' records via clean_metadata.merged_ids.

Output: metadata/record_aux.tsv
"""
from __future__ import annotations
import re
import pandas as pd
from Bio import SeqIO
from holdout import config as C


def norm_voucher(v: str) -> str:
    v = (v or "").upper().strip()
    return re.sub(r"[^A-Z0-9]", "", v)


def bold_table():
    cols = ["processid", "sampleid", "fieldid", "museumid", "specimenid", "bin_uri",
            "bold_recordset_code_arr", "province/state", "region", "site",
            "identification_method", "identified_by", "sequence_run_site", "insdc_acs",
            "inst", "marker_code", "coord", "collection_date_start", "nuc_basecount",
            "species", "identification", "taxonomy_notes", "notes"]
    df = pd.read_csv(C.RAW / f"bold_{C.GENUS.lower()}_v5.tsv", sep="\t", dtype=str,
                     usecols=cols, low_memory=False).fillna("")
    df = df[df.marker_code == "COI-5P"]
    out = pd.DataFrame({
        "bold_processid": df.processid,
        "bold_project": df.processid.str.extract(r"^([A-Z]+)", expand=False).fillna(""),
        "bold_recordsets": df.bold_recordset_code_arr.str.replace(r"[\[\]'\" ]", "", regex=True),
        "bold_sampleid": df.sampleid, "bold_fieldid": df.fieldid,
        "bold_museumid": df.museumid, "bold_specimenid": df.specimenid,
        "bold_inst": df.inst, "province_state": df["province/state"],
        "bold_region_field": df.region, "bold_site": df.site,
        "identification_method": df.identification_method,
        "bold_identified_by": df.identified_by,
        "sequence_run_site": df.sequence_run_site, "bold_insdc_acs": df.insdc_acs,
        "bold_species_raw": df.species, "bold_identification_raw": df.identification,
        "bold_taxonomy_notes": df.taxonomy_notes, "bold_notes": df.notes,
    })
    return out.drop_duplicates("bold_processid")


def genbank_table():
    rows = []
    for rec in SeqIO.parse(str(C.RAW / f"genbank_{C.GENUS.lower()}_coi.gb"), "genbank"):
        src = next((f for f in rec.features if f.type == "source"), None)
        q = src.qualifiers if src else {}
        refs = rec.annotations.get("references", []) or []
        r0 = next((r for r in refs if r.title and r.title != "Direct Submission"), refs[0] if refs else None)
        rows.append({
            "gb_accession_version": rec.id,
            "gb_accession": rec.id.split(".")[0],
            "gb_specimen_voucher": q.get("specimen_voucher", [""])[0],
            "gb_isolate": q.get("isolate", [""])[0],
            "gb_geo_loc_name": q.get("geo_loc_name", q.get("country", [""]))[0],
            "gb_note": q.get("note", [""])[0][:200],
            "gb_ref_title": (r0.title if r0 else "")[:250],
            "gb_ref_journal": (r0.journal if r0 else "")[:200],
            "gb_ref_pubmed": (r0.pubmed_id if r0 else ""),
            "gb_ref_authors": (r0.authors if r0 else "")[:120],
            "gb_keywords": ";".join(rec.annotations.get("keywords", []))[:120],
            "gb_definition": rec.description[:200],
        })
    return pd.DataFrame(rows).drop_duplicates("gb_accession_version")


def main():
    meta = pd.read_csv(C.CLEAN / "clean_metadata.tsv", sep="\t", low_memory=False,
                       dtype={"merged_ids": str, "accession": str})
    b = bold_table().set_index("bold_processid")
    g = genbank_table()
    g_by_ver = g.set_index("gb_accession_version")
    g_by_acc = g.drop_duplicates("gb_accession").set_index("gb_accession")
    rows = []
    for _, r in meta.iterrows():
        ids = [r.record_id] + (str(r.merged_ids).split(";") if isinstance(r.merged_ids, str) else [])
        row = {"record_id": r.record_id}
        for i in ids:
            if i in b.index and "bold_project" not in row:
                row.update(b.loc[i].to_dict()); row["bold_processid"] = i
            if i in g_by_ver.index and "gb_accession_version" not in row:
                row.update(g_by_ver.loc[i].to_dict()); row["gb_accession_version"] = i
        acc = str(r.accession).split("-")[0] if isinstance(r.accession, str) else ""
        if "gb_accession_version" not in row and acc in g_by_acc.index:
            row.update(g_by_acc.loc[acc].to_dict()); row["gb_accession"] = acc
        rows.append(row)
    aux = pd.DataFrame(rows)
    # specimen key: normalised voucher from any source (BOLD museumid/sampleid, GenBank voucher/isolate)
    cand = aux[["bold_museumid", "bold_sampleid", "gb_specimen_voucher"]].fillna("")
    aux["specimen_key_raw"] = [next((norm_voucher(x) for x in row if norm_voucher(x)), "")
                               for row in cand.itertuples(index=False)]
    # a usable specimen key must look like a real identifier: >=5 chars, contains a
    # digit, not a placeholder (XXX, NA, NONE...), and shared by <=3 records (keys
    # shared by many records are placeholders or institution-level codes)
    k = aux["specimen_key_raw"]
    cnt = k.map(k.value_counts())
    bad = (k.str.len() < 5) | ~k.str.contains(r"\d") | k.str.fullmatch(r"(X+|NA|NONE|UNKNOWN|NOVOUCHER)") | (cnt > 3)
    aux["specimen_key"] = k.where(~bad, "")
    # publication/project key: BOLD project code, else GenBank PubMed id, else ref title
    aux["study_key"] = [
        (f"BOLD:{p}" if isinstance(p, str) and p else
         (f"PMID:{pm}" if isinstance(pm, str) and pm else
          (f"REF:{t[:60]}" if isinstance(t, str) and t else "")))
        for p, pm, t in zip(aux.get("bold_project"), aux.get("gb_ref_pubmed"), aux.get("gb_ref_title"))]
    aux.to_csv(C.METADATA / "record_aux.tsv", sep="\t", index=False)
    print(aux.shape, "specimen_key filled:", (aux.specimen_key != "").mean().round(3),
          "study_key filled:", (aux.study_key != "").mean().round(3),
          "province filled:", aux.province_state.fillna("").ne("").mean().round(3))


if __name__ == "__main__":
    main()
