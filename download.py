"""Download NADAC yearly CSVs to data/raw/nadac_<year>.csv. Skips files that already exist.

Usage: python download.py 2022 2023 2024 2025 2026   (no arguments = every year below)

URLs come from each year's dataset record in the data.medicaid.gov catalog
(https://data.medicaid.gov/api/1/metastore/schemas/dataset/items/<dataset id>).
The filenames follow no single pattern, so they are listed here rather than guessed.
Note: medicaid.gov blocks some non-US/Canada IP addresses (403 Forbidden).
"""
import sys
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://download.medicaid.gov/data/"
URLS = {  # year: (dataset id, file name)
    2022: ("dfa2ab14-06c2-457a-9e36-5cb6d80f8d93", "nadac-national-average-drug-acquisition-cost-2022.csv"),
    2023: ("4a00010a-132b-4e4d-a611-543c9521280f", "nadac-national-average-drug-acquisition-cost-12-27-2023.csv"),
    2024: ("99315a95-37ac-4eee-946a-3c523b4c481e", "nadac-national-average-drug-acquisition-cost-12-25-2024.csv"),
    2025: ("f38d0706-1239-442c-a3cc-40ef1b686ac0", "nadac-national-average-drug-acquisition-cost-12-31-2025.csv"),
    # 2026 is the current, partial year; CMS republishes it weekly under a new name (this one: as of 11/04/2026)
    2026: ("fbb83258-11c7-47f5-8b18-5f8e79f7e704", "nadac-national-average-drug-acquisition-cost-11-04-2026.csv"),
}


def download(year):
    dest = Path(f"data/raw/nadac_{year}.csv")
    if dest.exists():
        print(f"{year}: already have {dest}, skipping download.")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = BASE + URLS[year][1]
        tmp = dest.with_suffix(".part")  # download to a temp name first...
        print(f"{year}: downloading {url} ...")
        urlretrieve(url, tmp)
        tmp.rename(dest)  # ...and rename only once it finished, so a half-download never looks complete
    print(f"{year}: {dest} {dest.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or sorted(URLS)
    for year in years:
        download(year)
