"""Shared parsing / normalization / QC helpers for Phase 2."""
from __future__ import annotations
import re, hashlib
from Bio.Seq import Seq
from holdout import config as C

# ---- sequence helpers -------------------------------------------------------
_VALID = set("ACGT")
_IUPAC = set("ACGTRYSWKMBDHVN")


def clean_seq(s: str) -> str:
    if not s:
        return ""
    s = s.upper().replace("-", "").replace(".", "").replace("~", "")
    s = re.sub(r"\s+", "", s)
    return s


def ambiguity_fraction(s: str) -> float:
    if not s:
        return 1.0
    n_amb = sum(1 for b in s if b not in _VALID)
    return n_amb / len(s)


def seq_md5(s: str) -> str:
    return hashlib.md5(clean_seq(s).encode()).hexdigest()


def min_internal_stops(nuc: str, table: int = C.MITO_TABLE):
    """Try 3 forward frames; return (best_frame, internal_stop_count).

    Internal stop = a '*' anywhere except the final translated residue. Uses the
    invertebrate mito code. Ambiguous codons translate to 'X' (not a stop).
    """
    s = clean_seq(nuc)
    best = (0, 10**9)
    for frame in (0, 1, 2):
        sub = s[frame:]
        sub = sub[: len(sub) - (len(sub) % 3)]
        if len(sub) < 30:
            continue
        try:
            aa = str(Seq(sub).translate(table=table))
        except Exception:
            continue
        internal = aa[:-1].count("*")  # allow a terminal stop
        if internal < best[1]:
            best = (frame, internal)
    return best if best[1] < 10**9 else (0, -1)


# ---- taxonomy normalization -------------------------------------------------
_BAD_EPITHET = {"sp", "sp.", "spp", "spp.", "cf", "cf.", "aff", "aff.", "nr",
                "nr.", "near", "complex", "group", "gr", "gr.", "indet",
                "indet.", "unknown", "environmental", "sample"}


def normalize_species(raw: str):
    """Return (genus, species_binomial, subspecies, is_species_level, reason).

    species_binomial is 'Bombus epithet' when resolvable, else ''.
    """
    if not raw or str(raw).strip().lower() in ("nan", "none", ""):
        return ("", "", "", False, "missing_taxon")
    txt = re.sub(r"\s+", " ", str(raw).strip())
    # strip author years / parentheses
    txt = re.sub(r"\(.*?\)", "", txt).strip()
    parts = txt.split(" ")
    genus = parts[0].capitalize()
    if len(parts) < 2:
        return (genus, "", "", False, "genus_only")
    epithet = parts[1].strip().lower().strip(".,")
    if epithet in _BAD_EPITHET or any(ch.isdigit() for ch in epithet) or len(epithet) < 3:
        return (genus, "", "", False, "not_species_level")
    binomial = f"{genus} {epithet}"
    subspecies = ""
    if len(parts) >= 3:
        third = parts[2].strip().lower().strip(".,")
        if third not in _BAD_EPITHET and third.isalpha() and len(third) >= 3:
            subspecies = f"{binomial} {third}"
    return (genus, binomial, subspecies, True, "ok")


# ---- country normalization --------------------------------------------------
_COUNTRY_FIX = {
    "USA": "United States", "U.S.A.": "United States", "US": "United States",
    "United States of America": "United States",
    "Korea, South": "South Korea", "Republic of Korea": "South Korea",
    "Korea": "South Korea", "Korea, Republic of": "South Korea",
    "Russian Federation": "Russia", "Viet Nam": "Vietnam",
    "UK": "United Kingdom", "Great Britain": "United Kingdom",
    "Czech Republic": "Czech Republic", "Czechia": "Czech Republic",
    "The Netherlands": "Netherlands", "Holland": "Netherlands",
}
_NON_COUNTRY = {"", "nan", "none", "unrecoverable", "unknown", "none:none",
                "not collected", "n/a", "na", "exception"}


def normalize_country(raw: str) -> str:
    if raw is None:
        return ""
    c = str(raw).strip()
    # GenBank geo_loc_name is "Country: locality" -> take part before ':'
    if ":" in c:
        c = c.split(":", 1)[0].strip()
    if c.lower() in _NON_COUNTRY:
        return ""
    return _COUNTRY_FIX.get(c, c)


# ---- coordinates ------------------------------------------------------------
def parse_latlon_genbank(s: str):
    """GenBank /lat_lon like '30.9751 N 77.1874 E' -> (lat, lon) floats."""
    if not s:
        return (None, None)
    m = re.match(r"\s*([\d.]+)\s*([NS])\s+([\d.]+)\s*([EW])", str(s))
    if not m:
        return (None, None)
    lat = float(m.group(1)) * (1 if m.group(2) == "N" else -1)
    lon = float(m.group(3)) * (1 if m.group(4) == "E" else -1)
    return (lat, lon)


def parse_coord_bold(s: str):
    """BOLD 'coord' like '45.1,-73.2' or '45.1 -73.2' -> (lat, lon)."""
    if not s or str(s).strip().lower() in ("nan", "none", ""):
        return (None, None)
    nums = re.findall(r"-?\d+\.?\d*", str(s))
    if len(nums) >= 2:
        try:
            return (float(nums[0]), float(nums[1]))
        except ValueError:
            return (None, None)
    return (None, None)


def valid_coord(lat, lon):
    if lat is None or lon is None:
        return False
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return False
    if abs(lat) < 1e-6 and abs(lon) < 1e-6:  # null island
        return False
    return True


# ---- dates ------------------------------------------------------------------
def parse_year(s):
    if not s:
        return None
    m = re.search(r"(19|20)\d{2}", str(s))
    if not m:
        return None
    y = int(m.group(0))
    return y if 1900 <= y <= 2026 else None
