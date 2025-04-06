import os
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# Settings
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
THRESHOLD = 0.4  # you can adjust this to 0.5 or 0.6
predictable_drugs = []

# Get sorted list of drug folders
drug_dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
print(f"🔍 Found {len(drug_dirs)} drug result folders")

for i, drug_dir in enumerate(drug_dirs, 1):
    try:
        pred_path = drug_dir / "predictions.csv"
        actual_path = drug_dir / "actuals.csv"
        
        if not pred_path.exists() or not actual_path.exists():
            print(f"⚠️ Missing files for drug {drug_dir.name}, skipping")
            continue

        preds = pd.read_csv(pred_path)
        actuals = pd.read_csv(actual_path)

        # Flatten all predictions and actuals
        y_pred = preds.values.flatten()
        y_true = actuals.values.flatten()

        # Drop NaNs (just in case)
        mask = ~pd.isna(y_pred) & ~pd.isna(y_true)
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        # Compute correlations
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)

        print(f"[{i}/{len(drug_dirs)}] Drug {drug_dir.name}: "
              f"Pearson={pearson_corr:.3f}, Spearman={spearman_corr:.3f}")

        if pearson_corr >= THRESHOLD and spearman_corr >= THRESHOLD:
            predictable_drugs.append({
                "drug_id": drug_dir.name,
                "pearson": pearson_corr,
                "spearman": spearman_corr
            })

    except Exception as e:
        print(f"❌ Error processing {drug_dir.name}: {e}")

# Save results
df_predictable = pd.DataFrame(predictable_drugs)
df_predictable.to_csv("predictable_drugs.csv", index=False)
print(f"\n✅ Saved {len(df_predictable)} predictable drugs to predictable_drugs.csv")
