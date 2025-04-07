import pandas as pd
from pathlib import Path

# Load your combined GDSC file
project_root = Path(__file__).resolve().parent.parent  # or hardcode if outside scr/
gdsc_path = project_root / "datasets" / "combined_gdsc.csv"

df = pd.read_csv(gdsc_path)

# Find drug IDs that match a drug name
search_name = "Dabrafenib"  # ← change this to drug name

# Case-insensitive match
matches = df[df["DRUG_NAME"].str.lower() == search_name.lower()]["DRUG_ID"].unique()
print(f"Drug IDs for '{search_name}':", matches)