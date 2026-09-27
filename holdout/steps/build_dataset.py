"""Phase 2: metadata + sequence QC, dedup, and dataset construction.

Reads raw BOLD v5 TSV and raw GenBank flat file, normalizes fields, applies
record-level QC with an explicit reason for every exclusion, deduplicates within
and across databases, and writes the main/partial datasets plus audit tables.

Outputs:
  clean/clean_metadata.tsv         all parsed records + normalized fields + flags
  clean/main_coi.fasta             high-quality near-full-length species-level COI
  clean/partial_coi.fasta          shorter usable COI (sensitivity only)
  results/qc_summary.tsv           record counts at each QC step
  results/conflicting_records.tsv  identical sequences with conflicting species
  results/duplicate_records.tsv    cross-DB / specimen duplicates collapsed
  clean/build_manifest.json        provenance + checksums
"""
from __future__ import annotations
import json, datetime, hashlib
from collections import defaultdict, Counter
import pandas as pd
from Bio import SeqIO
from holdout import config as C
from holdout import qc as Q


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
def parse_bold():
    df = pd.read_csv(C.RAW / f"bold_{C.GENUS.lower()}_v5.tsv", sep="\t", dtype=str,
                     low_memory=False).fillna("")
    df = df[df["marker_code"] == "COI-5P"].copy()
    recs = []
    for _, r in df.iterrows():
        lat, lon = Q.parse_coord_bold(r.get("coord", ""))
        recs.append({
            "source": "BOLD",
            "record_id": r.get("processid", ""),
            "accession": (r.get("insdc_acs", "") or "").strip(),
            "raw_taxon": r.get("species", "") or r.get("identification", ""),
            "raw_country": r.get("country/ocean", ""),
            "lat_raw": lat, "lon_raw": lon,
            "raw_year": r.get("collection_date_start", ""),
            "collector": r.get("collectors", ""),
            "voucher": r.get("museumid", "") or r.get("sampleid", ""),
            "bold_bin": r.get("bin_uri", ""),
            "project": r.get("bold_recordset_code_arr", ""),
            "identified_by": r.get("identified_by", ""),
            "id_rank": r.get("identification_rank", ""),
            "seq": Q.clean_seq(r.get("nuc", "")),
        })
    return recs


def _feature_country(feat):
    for key in ("geo_loc_name", "country"):
        if key in feat.qualifiers:
            return feat.qualifiers[key][0]
    return ""


def extract_coi_region(rec):
    """For long records (mitogenome/multigene) extract the COI CDS/gene span."""
    full = str(rec.seq)
    if len(full) <= 1500:
        return full
    best = None
    for feat in rec.features:
        if feat.type not in ("CDS", "gene"):
            continue
        q = " ".join(feat.qualifiers.get("gene", []) +
                     feat.qualifiers.get("product", [])).lower()
        if "cox1" in q or "coi" in q or "cytochrome c oxidase subunit i" in q \
                or "cytochrome oxidase subunit 1" in q or "co1" in q:
            try:
                sub = str(feat.extract(rec.seq))
                if best is None or len(sub) > len(best):
                    best = sub
            except Exception:
                continue
    return best if best else full


def parse_genbank():
    recs = []
    path = C.RAW / f"genbank_{C.GENUS.lower()}_coi.gb"
    for rec in SeqIO.parse(str(path), "genbank"):
        src = next((f for f in rec.features if f.type == "source"), None)
        quals = src.qualifiers if src else {}
        organism = quals.get("organism", [rec.annotations.get("organism", "")])[0]
        country = _feature_country(src) if src else ""
        lat, lon = Q.parse_latlon_genbank(quals.get("lat_lon", [""])[0])
        coi = extract_coi_region(rec)
        recs.append({
            "source": "GenBank",
            "record_id": rec.id,
            "accession": rec.id.split(".")[0],
            "raw_taxon": organism,
            "raw_country": country,
            "lat_raw": lat, "lon_raw": lon,
            "raw_year": quals.get("collection_date", [""])[0],
            "collector": quals.get("collected_by", [""])[0],
            "voucher": quals.get("specimen_voucher", [""])[0],
            "bold_bin": "",
            "project": (rec.annotations.get("references", [{}]) and ""),
            "identified_by": quals.get("identified_by", [""])[0],
            "id_rank": "",
            "seq": Q.clean_seq(coi),
        })
    return recs


# ---------------------------------------------------------------------------
def normalize(recs):
    for r in recs:
        g, binom, subsp, is_sp, reason = Q.normalize_species(r["raw_taxon"])
        r["genus"] = g
        r["species"] = binom
        r["subspecies"] = subsp
        r["is_species_level"] = is_sp
        r["taxon_reason"] = reason
        r["country"] = Q.normalize_country(r["raw_country"])
        r["region"] = C.region_for_country(r["country"])
        lat, lon = r["lat_raw"], r["lon_raw"]
        r["coord_valid"] = Q.valid_coord(lat, lon)
        r["lat"] = lat if r["coord_valid"] else ""
        r["lon"] = lon if r["coord_valid"] else ""
        r["year"] = Q.parse_year(r["raw_year"])
        r["seq_len"] = len(r["seq"])
        r["ambig_frac"] = round(Q.ambiguity_fraction(r["seq"]), 4) if r["seq"] else 1.0
        frame, stops = Q.min_internal_stops(r["seq"]) if r["seq"] else (0, -1)
        r["best_frame"] = frame
        r["internal_stops"] = stops
        r["seq_hash"] = Q.seq_md5(r["seq"]) if r["seq"] else ""
    return recs


# ---------------------------------------------------------------------------
def dedup_cross_db(recs):
    """Collapse BOLD/GenBank records that share an INSDC accession into one
    'both'-source record (prefer BOLD metadata, which carries richer geography).
    Returns (kept_records, duplicate_rows)."""
    by_acc = defaultdict(list)
    no_acc = []
    for r in recs:
        acc = (r["accession"] or "").strip()
        if acc:
            by_acc[acc].append(r)
        else:
            no_acc.append(r)
    kept, dup_rows = [], []
    for r in no_acc:
        kept.append(r)
    for acc, group in by_acc.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        bold = [x for x in group if x["source"] == "BOLD"]
        gb = [x for x in group if x["source"] == "GenBank"]
        primary = (bold[0] if bold else group[0])
        primary = dict(primary)
        primary["source"] = "both" if (bold and gb) else primary["source"]
        # fill missing geo from the other record
        for other in group:
            if other is primary:
                continue
            if not primary["country"] and other["country"]:
                primary["country"] = other["country"]
                primary["region"] = C.region_for_country(other["country"])
            if primary["lat"] == "" and other.get("lat", "") != "":
                primary["lat"], primary["lon"] = other["lat"], other["lon"]
                primary["coord_valid"] = True
        primary["merged_ids"] = ";".join(sorted(x["record_id"] for x in group))
        kept.append(primary)
        for other in group:
            dup_rows.append({
                "accession": acc,
                "kept_record_id": primary["record_id"],
                "duplicate_record_id": other["record_id"],
                "duplicate_source": other["source"],
                "reason": "shared_INSDC_accession",
            })
    return kept, dup_rows


def flag_conflicts(recs):
    """Identical sequences with conflicting species labels."""
    by_hash = defaultdict(list)
    for r in recs:
        if r["seq_hash"] and r["is_species_level"]:
            by_hash[r["seq_hash"]].append(r)
    conflict_rows = []
    conflicted_hashes = set()
    for h, group in by_hash.items():
        species = {r["species"] for r in group}
        if len(species) > 1:
            conflicted_hashes.add(h)
            for r in group:
                conflict_rows.append({
                    "seq_hash": h, "record_id": r["record_id"],
                    "source": r["source"], "species": r["species"],
                    "country": r["country"], "n_species_in_group": len(species),
                    "species_in_group": "|".join(sorted(species)),
                })
    return conflict_rows, conflicted_hashes


# ---------------------------------------------------------------------------
def assign_datasets(recs, conflicted_hashes):
    """Assign each record to main / partial / excluded with a reason."""
    for r in recs:
        reasons = []
        if not r["is_species_level"]:
            reasons.append(f"no_species_id:{r['taxon_reason']}")
        if r["genus"] and r["genus"] != C.GENUS:
            reasons.append(f"contradictory_taxonomy:genus={r['genus']}")
        if r["seq_len"] == 0:
            reasons.append("no_sequence")
        r["conflict_label"] = r["seq_hash"] in conflicted_hashes
        if r["conflict_label"] and not C.KEEP_CONFLICTS:
            reasons.append("conflicting_species_same_sequence")
        if r["internal_stops"] is not None and r["internal_stops"] > 0:
            reasons.append(f"internal_stops:{r['internal_stops']}")
        # length / ambiguity tiers
        too_long = r["seq_len"] > C.MAX_BP
        if too_long:
            reasons.append(f"too_long:{r['seq_len']}")

        main_ok = (not reasons
                   and r["seq_len"] >= C.MAIN_MIN_BP
                   and r["ambig_frac"] <= C.MAX_AMBIG_FRAC)
        partial_ok = (not reasons
                      and r["seq_len"] >= C.PARTIAL_MIN_BP
                      and r["ambig_frac"] <= C.MAX_AMBIG_FRAC_PARTIAL)

        if main_ok:
            r["dataset"] = "main"
            r["exclusion_reason"] = ""
        elif partial_ok:
            r["dataset"] = "partial"
            r["exclusion_reason"] = "below_main_length_or_ambiguity" if not reasons else ";".join(reasons)
        else:
            r["dataset"] = "excluded"
            if not reasons:
                if r["seq_len"] < C.PARTIAL_MIN_BP:
                    reasons.append(f"too_short:{r['seq_len']}")
                if r["ambig_frac"] > C.MAX_AMBIG_FRAC_PARTIAL:
                    reasons.append(f"too_ambiguous:{r['ambig_frac']}")
            r["exclusion_reason"] = ";".join(reasons) if reasons else "unmet_criteria"
    return recs


# ---------------------------------------------------------------------------
def write_fasta(recs, dataset, path):
    n = 0
    with open(path, "w") as fh:
        for r in recs:
            if r["dataset"] == dataset:
                fh.write(f">{r['record_id']}\n{r['seq']}\n")
                n += 1
    return n


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    print("Parsing BOLD ...")
    bold = parse_bold()
    print(f"  BOLD COI-5P rows: {len(bold)}")
    print("Parsing GenBank ...")
    gb = parse_genbank()
    print(f"  GenBank records: {len(gb)}")

    all_recs = normalize(bold + gb)
    n_before_dedup = len(all_recs)

    kept, dup_rows = dedup_cross_db(all_recs)
    print(f"  after cross-DB dedup: {len(kept)} (removed {n_before_dedup-len(kept)})")

    conflict_rows, conflicted_hashes = flag_conflicts(kept)
    print(f"  conflicting-label sequence groups: "
          f"{len({c['seq_hash'] for c in conflict_rows})}")

    kept = assign_datasets(kept, conflicted_hashes)

    # ---- outputs ----
    cols = ["record_id", "source", "accession", "genus", "species", "subspecies",
            "is_species_level", "taxon_reason", "country", "region",
            "lat", "lon", "coord_valid", "year", "collector", "voucher",
            "bold_bin", "identified_by", "id_rank", "seq_len", "ambig_frac",
            "best_frame", "internal_stops", "seq_hash", "dataset",
            "exclusion_reason", "merged_ids", "conflict_label"]
    for r in kept:
        r.setdefault("merged_ids", "")
    dfk = pd.DataFrame(kept)[cols]
    dfk.to_csv(C.CLEAN / "clean_metadata.tsv", sep="\t", index=False)

    n_main = write_fasta(kept, "main", C.CLEAN / "main_coi.fasta")
    n_partial = write_fasta(kept, "partial", C.CLEAN / "partial_coi.fasta")

    pd.DataFrame(dup_rows).to_csv(C.RESULTS / "duplicate_records.tsv",
                                  sep="\t", index=False)
    pd.DataFrame(conflict_rows).to_csv(C.RESULTS / "conflicting_records.tsv",
                                       sep="\t", index=False)

    # ---- QC summary (step-by-step counts) ----
    src_counts = Counter(r["source"] for r in kept)
    ds_counts = Counter(r["dataset"] for r in kept)
    excl_reasons = Counter()
    for r in kept:
        if r["dataset"] == "excluded":
            excl_reasons[r["exclusion_reason"].split(";")[0].split(":")[0]] += 1
    qc_rows = [
        ("bold_coi5p_raw", len(bold)),
        ("genbank_raw", len(gb)),
        ("combined_before_dedup", n_before_dedup),
        ("cross_db_duplicates_collapsed", n_before_dedup - len(kept)),
        ("records_after_dedup", len(kept)),
        ("source_BOLD_only", src_counts.get("BOLD", 0)),
        ("source_GenBank_only", src_counts.get("GenBank", 0)),
        ("source_both", src_counts.get("both", 0)),
        ("dataset_main", ds_counts.get("main", 0)),
        ("dataset_partial", ds_counts.get("partial", 0)),
        ("dataset_excluded", ds_counts.get("excluded", 0)),
        ("conflicting_seq_groups", len({c["seq_hash"] for c in conflict_rows})),
    ]
    for reason, n in excl_reasons.most_common():
        qc_rows.append((f"excluded__{reason}", n))
    pd.DataFrame(qc_rows, columns=["metric", "count"]).to_csv(
        C.RESULTS / "qc_summary.tsv", sep="\t", index=False)

    manifest = {
        "built_utc": now(),
        "random_seed": C.RANDOM_SEED,
        "inputs": {
            "bold": str((C.RAW / f"bold_{C.GENUS.lower()}_v5.tsv")),
            "genbank": str((C.RAW / f"genbank_{C.GENUS.lower()}_coi.gb")),
        },
        "thresholds": {
            "MAIN_MIN_BP": C.MAIN_MIN_BP, "PARTIAL_MIN_BP": C.PARTIAL_MIN_BP,
            "MAX_BP": C.MAX_BP, "MAX_AMBIG_FRAC": C.MAX_AMBIG_FRAC,
            "MAX_AMBIG_FRAC_PARTIAL": C.MAX_AMBIG_FRAC_PARTIAL,
            "MITO_TABLE": C.MITO_TABLE,
        },
        "counts": dict(qc_rows),
        "outputs": {
            "clean_metadata": "clean/clean_metadata.tsv",
            "main_fasta": "clean/main_coi.fasta",
            "partial_fasta": "clean/partial_coi.fasta",
        },
        "checksums": {
            "main_coi.fasta": sha256(C.CLEAN / "main_coi.fasta"),
            "partial_coi.fasta": sha256(C.CLEAN / "partial_coi.fasta"),
            "clean_metadata.tsv": sha256(C.CLEAN / "clean_metadata.tsv"),
        },
    }
    (C.CLEAN / "build_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== QC SUMMARY ===")
    for k, v in qc_rows:
        print(f"  {k}: {v}")
    print(f"\nmain={n_main} partial={n_partial}")


if __name__ == "__main__":
    main()
