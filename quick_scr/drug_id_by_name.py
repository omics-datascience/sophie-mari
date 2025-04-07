import pandas as pd
from pathlib import Path

# Define project root as the parent of the quick_scr folder
project_root = Path(__file__).resolve().parent.parent  # ← goes up to sophie-mari
gdsc_path = project_root / "datasets" / "combined_gdsc.csv"

# Load the dataset
df = pd.read_csv(gdsc_path)

# Find drug IDs that match a drug name
search_name = "Etoposide"  # ← change this to drug name

# Case-insensitive match
matches = df[df["DRUG_NAME"].str.lower() == search_name.lower()]["DRUG_ID"].unique()
print(f"Drug IDs for '{search_name}':", matches)