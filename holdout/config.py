"""Paths and constants. The genus and the working folder come from environment
variables so that the pipeline steps (run as separate processes) and the notebook
see the same settings; use holdout.use() in the notebook."""
from __future__ import annotations
import os
from pathlib import Path

GENUS = os.environ.get("HOLDOUT_GENUS", "Bombus")
PROJECT_ROOT = Path(os.environ.get("HOLDOUT_WORKDIR", f"work/{GENUS.lower()}")).resolve()
KEEP_CONFLICTS = os.environ.get("HOLDOUT_KEEP_CONFLICTS", "0") == "1"

RAW = PROJECT_ROOT / "raw"
METADATA = PROJECT_ROOT / "metadata"
CLEAN = PROJECT_ROOT / "clean"
ALIGNMENT = PROJECT_ROOT / "alignment"
RESULTS = PROJECT_ROOT / "results"
for _d in (RAW, METADATA, CLEAN, ALIGNMENT, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20260924
MITO_TABLE = 5                 # invertebrate mitochondrial code
BARCODE_TARGET_BP = 658
MAIN_MIN_BP = 500
PARTIAL_MIN_BP = 300
MAX_BP = 2000
MAX_AMBIG_FRAC = 0.01
MAX_AMBIG_FRAC_PARTIAL = 0.02
MIN_SEQS_PER_SPECIES = 20

# NCBI asks for a tool name; no e-mail address is sent.
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_TOOL = "coi_holdout_audit"
NCBI_DELAY_S = 0.4
GENBANK_QUERY = (f'{GENUS}[Organism] AND (COI[Gene] OR COX1[Gene] OR '
                 '"cytochrome oxidase subunit 1"[All Fields] OR '
                 '"cytochrome c oxidase subunit I"[All Fields] OR '
                 '"cytochrome oxidase subunit I"[All Fields])')

COUNTRY_TO_REGION = {
    # --- Europe (West/Central) ---
    "Austria": "Europe", "Belgium": "Europe", "Switzerland": "Europe",
    "Germany": "Europe", "France": "Europe", "United Kingdom": "Europe",
    "Ireland": "Europe", "Netherlands": "Europe", "Luxembourg": "Europe",
    "Denmark": "Europe", "Norway": "Europe", "Sweden": "Europe",
    "Finland": "Europe", "Iceland": "Europe", "Poland": "Europe",
    "Czech Republic": "Europe", "Czechia": "Europe", "Slovakia": "Europe",
    "Hungary": "Europe", "Italy": "Europe", "Spain": "Europe",
    "Portugal": "Europe", "Greece": "Europe", "Romania": "Europe",
    "Bulgaria": "Europe", "Croatia": "Europe", "Slovenia": "Europe",
    "Serbia": "Europe", "Estonia": "Europe", "Latvia": "Europe",
    "Lithuania": "Europe", "Belarus": "Europe", "Ukraine": "Europe",
    "Andorra": "Europe", "Liechtenstein": "Europe",
    # --- East Palearctic / Asia ---
    "Russia": "East_Palearctic", "China": "East_Palearctic",
    "Mongolia": "East_Palearctic", "Japan": "East_Palearctic",
    "South Korea": "East_Palearctic", "Korea, South": "East_Palearctic",
    "North Korea": "East_Palearctic", "Kazakhstan": "East_Palearctic",
    "Kyrgyzstan": "East_Palearctic", "Tajikistan": "East_Palearctic",
    "Uzbekistan": "East_Palearctic", "Turkmenistan": "East_Palearctic",
    "India": "South_Asia", "Nepal": "South_Asia", "Pakistan": "South_Asia",
    "Bhutan": "South_Asia", "Afghanistan": "South_Asia",
    "Turkey": "West_Asia", "Iran": "West_Asia", "Georgia": "West_Asia",
    "Armenia": "West_Asia", "Azerbaijan": "West_Asia", "Israel": "West_Asia",
    # --- North America ---
    "United States": "North_America", "USA": "North_America",
    "Canada": "North_America", "Mexico": "North_America",
    "Greenland": "North_America",
    # --- South America ---
    "Argentina": "South_America", "Chile": "South_America",
    "Peru": "South_America", "Bolivia": "South_America",
    "Colombia": "South_America", "Ecuador": "South_America",
    "Brazil": "South_America",
    # --- Africa / Oceania (rare for Bombus) ---
    "South Africa": "Africa", "New Zealand": "Oceania",
    "Australia": "Oceania", "Tasmania": "Oceania",
}


def region_for_country(country: str) -> str:
    if not country:
        return "Unknown"
    c = country.strip()
    return COUNTRY_TO_REGION.get(c, "Other")
