import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

drug_id = "1372"  # change to whichever drug you want to check
# Get absolute path to results directory relative to this script
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results" / drug_id

# Load prediction and actual CSVs
preds = pd.read_csv(results_dir / "predictions.csv")
actuals = pd.read_csv(results_dir / "actuals.csv")

# Flatten predictions and actuals
y_pred = preds.values.flatten()
y_true = actuals.values.flatten()

# Drop NaNs if any
mask = ~pd.isna(y_pred) & ~pd.isna(y_true)
y_pred = y_pred[mask]
y_true = y_true[mask]

# Plot
plt.figure(figsize=(6, 5))
plt.scatter(y_true, y_pred, alpha=0.7)
plt.xlabel("Actual IC50")
plt.ylabel("Predicted IC50")
plt.title(f"Drug {drug_id} — Actual vs Predicted IC50")
plt.grid(True)
plt.tight_layout()

# Save plot
plot_path = results_dir / f"drug_{drug_id}_scatter.png"
plt.savefig(plot_path)
print(f"✅ Saved scatter plot to {plot_path}")