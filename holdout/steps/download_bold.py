"""Phase 1: download Bombus records from BOLD v5 public portal API, unchanged.

The legacy BOLD v4 Public API (API_Public/combined) is offline ("BOLD Public
Offline"). BOLD has migrated to the v5 portal at portal.boldsystems.org. The
public workflow is:
  1. GET /api/query?query=tax:genus:Bombus&extent=full  -> query_id
  2. GET /api/documents/{query_id}/download?format=tsv    -> full TSV export

The TSV includes nucleotides (nuc), marker_code, INSDC accession (insdc_acs),
coordinates, country, collectors, identification metadata, and a per-record
md5 hash. We save the raw TSV verbatim and write a manifest with a checksum.
Marker filtering (COI-5P) happens later in QC, not here, so raw stays complete.
"""
from __future__ import annotations
import hashlib, json, datetime, urllib.request, urllib.parse, sys, time
from holdout import config as C

PORTAL = "https://portal.boldsystems.org"
TRIPLET = f"tax:genus:{C.GENUS}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(url, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": C.NCBI_TOOL})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    t0 = time.time()
    # 1) resolve query -> query_id
    q_url = f"{PORTAL}/api/query?" + urllib.parse.urlencode(
        {"query": TRIPLET, "extent": "full"})
    print("Query:", q_url)
    qid = json.loads(get(q_url, timeout=300))["query_id"]
    print("query_id:", qid)

    # 2) counts (for the manifest / sanity)
    counts = json.loads(get(f"{PORTAL}/api/counts?" +
                            urllib.parse.urlencode({"query": TRIPLET}), 120))
    print("counts:", counts)

    # 3) download full TSV
    dl_url = f"{PORTAL}/api/documents/{urllib.parse.quote(qid)}/download?" + \
             urllib.parse.urlencode({"format": "tsv"})
    print("Downloading:", dl_url)
    data = get(dl_url, timeout=900)
    raw_path = C.RAW / f"bold_{C.GENUS.lower()}_v5.tsv"
    raw_path.write_bytes(data)

    text = data.decode("utf-8", errors="replace")
    n_records = max(text.count("\n") - 1, 0)
    header = text.split("\n", 1)[0].split("\t") if text else []
    dt = time.time() - t0

    manifest = {
        "source": "BOLD v5 Public Portal API",
        "portal": PORTAL,
        "query_triplet": TRIPLET,
        "query_endpoint": q_url,
        "download_endpoint": dl_url,
        "query_id": qid,
        "counts_reported": counts,
        "download_utc": now(),
        "raw_file": str(raw_path.relative_to(C.PROJECT_ROOT)),
        "bytes": len(data),
        "n_records_downloaded": n_records,
        "n_columns": len(header),
        "columns": header,
        "sha256": sha256(raw_path),
        "elapsed_s": round(dt, 1),
        "notes": ("Raw export retained unchanged and complete (all markers). "
                  "COI-5P marker filtering happens in QC, not here."),
    }
    (C.RAW / "bold_download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "columns"}, indent=2))
    if n_records < 1000:
        print("WARNING: fewer BOLD records than expected; inspect raw file.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
