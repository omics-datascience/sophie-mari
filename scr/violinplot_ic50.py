import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Config ---
drug_id = 5
drug_name = "Sunitinib"  # Optional: manually set or load dynamically
results_path = Path("results") / str(drug_id)

# --- Load predictions ---
preds_df = pd.read_csv(results_path / "predictions.csv")
actuals_df = pd.read_csv(results_path / "actuals.csv")

# --- Reshape for violin plot ---
melted = preds_df.melt(var_name="Run", value_name="Predicted IC50")
melted["Sample"] = melted.index

# --- Plot ---
plt.figure(figsize=(12, 6))
sns.violinplot(data=melted, x="Sample", y="Predicted IC50", inner="quartile")

# Overlay actual values
mean_actuals = actuals_df.mean(axis=1).values
sns.stripplot(x=list(range(len(mean_actuals))), y=mean_actuals,
              color='red', marker='o', size=6, jitter=False, label='Actual IC50')

# Format
plt.title(f"Predicted IC50 Distributions for Drug {drug_id} ({drug_name})")
plt.xlabel("Sample Index")
plt.ylabel("Predicted IC50")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

mean_actuals = actuals_df.mean(axis=1).values  # Average across runs

# After sns.violinplot(...)
sns.stripplot(x=list(range(len(mean_actuals))),
              y=mean_actuals,
              color='red', marker='o', size=6, jitter=False, label='Actual IC50')

plt.legend()