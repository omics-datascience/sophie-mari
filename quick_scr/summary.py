import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

def summarize_correlations(
    results_dir: Path, 
    gdsc_path: Path,
    save_csv: bool = True, 
    threshold: float = None
) -> pd.DataFrame:
    """Summarize Pearson and Spearman correlations for each drug in results_dir."""
    
    # Load drug name mappings
    try:
        gdsc_df = pd.read_csv(gdsc_path)
        drug_id_to_name = pd.Series(gdsc_df["DRUG_NAME"].values, index=gdsc_df["DRUG_ID"]).to_dict()
    except Exception as e:
        print(f"❌ Could not load drug names from {gdsc_path}: {e}")
        drug_id_to_name = {}

    drug_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    print(f"🔍 Found {len(drug_dirs)} drug result folders.")

    summaries = []
    
    for drug_dir in drug_dirs:
        drug_id = drug_dir.name
        final_path = drug_dir / "final_predictions.csv"

        if not final_path.exists():
            print(f"⚠️  Skipping {drug_id}: no final_predictions.csv")
            continue

        try:
            df = pd.read_csv(final_path).dropna()
            pearson = pearsonr(df["actual"], df["predicted"])[0]
            spearman = spearmanr(df["actual"], df["predicted"])[0]

            summaries.append({
                "drug_id": drug_id,
                "drug_name": drug_id_to_name.get(int(drug_id), "Unknown"),
                "pearson": round(pearson, 4),
                "spearman": round(spearman, 4)
            })

        except Exception as e:
            print(f"❌ Error processing {drug_id}: {e}")

    summary_df = pd.DataFrame(summaries)

    if threshold is not None:
        summary_df = summary_df[
            (summary_df["pearson"] >= threshold) & (summary_df["spearman"] >= threshold)
        ]
        print(f"📊 Filtered to {len(summary_df)} drugs with both correlations ≥ {threshold}")

    summary_df = summary_df.sort_values("drug_id")

    if save_csv:
        out_path = results_dir / "summary_correlations.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"✅ Saved summary to {out_path}")

    return summary_df

from pathlib import Path
from summary import summarize_correlations

# Adjust paths to your setup
project_root = Path(__file__).resolve().parent.parent
results_path = project_root / "results"
gdsc_path = project_root / "datasets" / "combined_gdsc.csv"

# Run without threshold (just view everything)
summary_df = summarize_correlations(results_path, gdsc_path, threshold=None)

# View first few entries
print(summary_df.head())