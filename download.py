"""Download one year of NADAC data to data/raw/. Skips if the file already exists."""
from pathlib import Path
from urllib.request import urlretrieve

# Official 2025 file, from the dataset's catalog record on data.medicaid.gov:
# https://data.medicaid.gov/dataset/f38d0706-1239-442c-a3cc-40ef1b686ac0
# Note: medicaid.gov blocks non-US IP addresses (403 Forbidden).
URL = "https://download.medicaid.gov/data/nadac-national-average-drug-acquisition-cost-12-31-2025.csv"
DEST = Path("data/raw/nadac_2025.csv")

if DEST.exists():
    print(f"Already have {DEST}, skipping download.")
else:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")  # download to a temp name first...
    print(f"Downloading {URL} ...")
    urlretrieve(URL, tmp)
    tmp.rename(DEST)  # ...and rename only once it finished, so a half-download never looks complete

print(f"{DEST}: {DEST.stat().st_size / 1_000_000:.1f} MB")
