"""Phase 1: download Bombus COI records from GenBank (NCBI E-utilities), unchanged.

Workflow:
  1. esearch (usehistory=y) for Bombus COI -> WebEnv/query_key/count
  2. efetch rettype=gb retmode=text in batches -> raw/genbank_bombus_coi.gb
GenBank flat files carry source-feature qualifiers (country, lat_lon,
collection_date, specimen_voucher, organism) parsed later in QC. We save the raw
concatenated flat file verbatim and write a manifest with a checksum.

No personal email is sent to NCBI; we identify the tool and throttle politely.
"""
from __future__ import annotations
import hashlib, json, datetime, urllib.request, urllib.parse, sys, time, re
from holdout import config as C


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_bytes_stream(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eutils_get(endpoint, params, timeout=180, retries=4):
    params = dict(params)
    params.setdefault("tool", C.NCBI_TOOL)
    url = f"{C.EUTILS}/{endpoint}?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": C.NCBI_TOOL})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # transient NCBI hiccups -> backoff
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"E-utilities failed after {retries} tries: {url}\n{last}")


def main():
    t0 = time.time()
    # 1) esearch with history. NCBI history-server efetch fails at retstart >= 10,000, so larger
    # result sets are split into disjoint publication-date (PDAT) ranges, each < 10,000 records.
    def esearch(term):
        es = eutils_get("esearch.fcgi", {"db": "nucleotide", "term": term, "usehistory": "y", "retmax": 0})
        t = es.decode()
        return (es, int(re.search(r"<Count>(\d+)</Count>", t).group(1)),
                re.search(r"<WebEnv>(\S+)</WebEnv>", t).group(1), re.search(r"<QueryKey>(\d+)</QueryKey>", t).group(1))

    es, count, webenv, qkey = esearch(C.GENBANK_QUERY)
    (C.RAW / "genbank_esearch.xml").write_bytes(es)
    print(f"esearch: count={count} query_key={qkey}")
    LIMIT = 9999
    parts = []
    if count <= LIMIT:
        parts = [(C.GENBANK_QUERY, count, webenv, qkey, "all")]
    else:
        todo = [(1900, 2030)]
        while todo:
            a, b = todo.pop(0)
            term = f"({C.GENBANK_QUERY}) AND {a}:{b}[PDAT]"
            _, n, w, k = esearch(term); time.sleep(C.NCBI_DELAY_S)
            if n > LIMIT and b > a:
                m = (a + b) // 2; todo = [(a, m), (m + 1, b)] + todo
            elif n > LIMIT:
                raise RuntimeError(f"cannot split below one year: {a} has {n} records")
            elif n:
                parts.append((term, n, w, k, f"{a}-{b}"))
        print("date split:", [(p[4], p[1]) for p in parts], "sum", sum(p[1] for p in parts))
        if sum(p[1] for p in parts) != count:
            raise RuntimeError(f"date-split total {sum(p[1] for p in parts)} != esearch count {count}")

    # 2) batched efetch -> concatenated gb flat file
    raw_path = C.RAW / f"genbank_{C.GENUS.lower()}_coi.gb"
    batch = 300
    fetched = 0
    with open(raw_path, "wb") as out:
        for term, n, w, k, lab in parts:
            for start in range(0, n, batch):
                data = eutils_get("efetch.fcgi", {
                    "db": "nucleotide", "query_key": k, "WebEnv": w,
                    "retstart": start, "retmax": batch,
                    "rettype": "gb", "retmode": "text"}, timeout=300)
                out.write(data)
                fetched += data.count(b"\nLOCUS ") + (1 if data.startswith(b"LOCUS ") else 0)
                print(f"  [{lab}] fetched batch {start}-{start+batch} "
                      f"(cumulative LOCUS ~{fetched})", flush=True)
                time.sleep(C.NCBI_DELAY_S)

    # count LOCUS lines exactly
    n_locus = 0
    with open(raw_path, "rb") as fh:
        for line in fh:
            if line.startswith(b"LOCUS "):
                n_locus += 1
    dt = time.time() - t0

    manifest = {
        "source": "NCBI GenBank (E-utilities esearch+efetch)",
        "eutils_base": C.EUTILS,
        "db": "nucleotide",
        "query": C.GENBANK_QUERY,
        "esearch_count": count,
        "webenv": webenv,
        "query_key": qkey,
        "rettype": "gb", "retmode": "text", "batch_size": batch,
        "pdat_split": [{"range": p[4], "count": p[1]} for p in parts],
        "download_utc": now(),
        "raw_file": str(raw_path.relative_to(C.PROJECT_ROOT)),
        "bytes": raw_path.stat().st_size,
        "n_records_locus": n_locus,
        "sha256": sha256_bytes_stream(raw_path),
        "elapsed_s": round(dt, 1),
        "email_sent_to_ncbi": False,
    }
    (C.RAW / "genbank_download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    if n_locus < count * 0.95:
        print(f"WARNING: got {n_locus} LOCUS vs esearch {count}.", file=sys.stderr)


if __name__ == "__main__":
    main()
