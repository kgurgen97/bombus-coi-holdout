# bombus-coi-holdout

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22996972.svg)](https://doi.org/10.5281/zenodo.22996972)

Code and data for the paper *Geographic hold-out reveals turnover hidden by aggregate COI identification accuracy in bumblebees*.

When the region a specimen comes from is missing from the reference library, COI identification can look as accurate as before and still give different answers for many specimens. Losing close relatives of the query's own species turns some correct answers into wrong ones. Losing competing species turns some ambiguous answers into correct ones. The two effects almost cancel in the overall numbers. In *Bombus*, the net change is close to zero while about a fifth of the queries (species-weighted) change outcome.

## Try it on your genus

```bash
pip install -r requirements.txt
jupyter notebook holdout_audit.ipynb
```

Set `GENUS` in the first cell and run everything. The notebook downloads public COI records from BOLD and GenBank, cleans and aligns them, then identifies the same queries with the query's region removed from the library and with its whole continent removed. You get:

- how many identifications changed, and in which direction;
- how the net change splits into conspecific loss and competitor removal;
- which reference records cause the most failures (candidates to check, not proven errors);
- what the 1% rule costs in correct answers per error it removes.

MAFFT must be on the PATH (`brew install mafft` or `conda install -c bioconda mafft`). Everything goes into `work/<genus>/`.

The same thing from Python:

```python
import holdout as hd
hd.use("Andrena")
hd.prepare("Andrena", "work/andrena")
res = hd.run()
res.summary
```

## Reproduce the paper

```bash
paper/prepare_all.sh    # download and run all scenarios for the nine genera (most of a day)
paper/make_paper.sh     # all tables and figures (about 2 hours)
```

`make_paper.sh` skips the calibration models (several hours) when their outputs already exist. Set `CALIBRATION=1` to rerun them. The BLAST scenario needs BLAST+.

BOLD and GenBank change over time, so a fresh download will not match ours record for record. `data/records/` lists every record we used for each genus, with the reason for each exclusion. We downloaded the data in September 2026.

## What is where

```
holdout/                the package: download, cleaning, alignment, scenarios, audit, plots
holdout_audit.ipynb     the notebook
paper/                  scripts behind every table and figure in the paper
data/records/           record IDs used per genus, with QC outcome
figures/                figures of the paper
```

## Licence

Code: MIT. Tables and figures: CC BY 4.0.
