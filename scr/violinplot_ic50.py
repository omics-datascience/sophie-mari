import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Config ---
drug_id = 5
drug_name = "Sunitinib"  # You can make this dynamic later if needed
results_path = Path("results") / str(drug_id)

# --- Load predictions and actuals ---
preds_df = pd.read_csv(results_path / "predictions.csv")
actuals_df = pd.read_csv(results_path / "actuals.csv")

# --- Assign sample IDs (as strings) for x-axis ---
num_samples = preds_df.shape[0]
sample_labels = [f"Sample {i}" for i in range(num_samples)]
preds_df["Sample"] = sample_labels
actuals_df["Sample"] = sample_labels

# --- Melt predictions into long format for violinplot ---
melted_preds = preds_df.melt(id_vars="Sample", var_name="Run", value_name="Predicted IC50")

# --- Calculate mean actuals per sample ---
mean_actuals = actuals_df.drop(columns="Sample").mean(axis=1).values

# --- Plot ---
plt.figure(figsize=(14, 6))
sns.violinplot(data=melted_preds, x="Sample", y="Predicted IC50", inner="quartile")

# Overlay actual IC50s as red dots
sns.stripplot(x=sample_labels,
              y=mean_actuals,
              color='red', marker='o', size=6, jitter=False, label='Actual IC50')

# --- Final formatting ---
plt.title(f"Predicted IC50 Distributions for Drug {drug_id} ({drug_name})")
plt.xlabel("Cell Line (Sample)")
plt.ylabel("Predicted IC50")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()