import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Reuse your PROJECT_ROOT definition
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
GDSC_CSV_PATH = PROJECT_ROOT / "datasets" / "combined_gdsc.csv"

def load_drug_mapping():
    """Load drug ID to name mapping (identical to your main script)"""
    try:
        gdsc_df = pd.read_csv(GDSC_CSV_PATH)
        return (
            pd.Series(gdsc_df["DRUG_NAME"].values, index=gdsc_df["DRUG_ID"])
            .dropna()
            .to_dict()
        )
    except Exception as e:
        print(f"⚠️ Could not load drug mapping: {e}")
        return {}

def create_violin_plots(drug_id: int, drug_name: str, max_samples=10):
    """Create violin plots with simplified legend"""
    drug_dir = RESULTS_DIR / str(drug_id)
    
    try:
        preds = pd.read_csv(drug_dir / "predictions.csv")
        actuals = pd.read_csv(drug_dir / "actuals.csv")
    except FileNotFoundError:
        print(f"❌ No results found for drug {drug_id}")
        return

    # Calculate prediction variance and select top samples
    top_samples = preds.var().nlargest(max_samples).index.tolist()
    
    # Prepare plot data
    plot_data = []
    for sample in top_samples:
        # Add predictions
        plot_data.extend([{
            'Sample': sample, 
            'Value': pred,
            'Type': 'Predicted'
        } for pred in preds[sample].dropna()])
        
        # Add single actual value marker
        plot_data.append({
            'Sample': sample,
            'Value': actuals[sample].dropna().mean(),
            'Type': 'Actual'
        })
    
    df = pd.DataFrame(plot_data)

    # Create plot
    plt.figure(figsize=(max_samples*0.8, 5))  # Compact width
    sns.set_style("whitegrid")
    ax = plt.gca()
    
    # Violin plot (predictions)
    sns.violinplot(
        x='Sample', y='Value', 
        data=df[df['Type'] == 'Predicted'],
        color='lightblue',
        cut=0,
        inner=None,
        ax=ax
    )
    
    # Actual values (single legend entry)
    actual_points = sns.stripplot(
        x='Sample', y='Value',
        data=df[df['Type'] == 'Actual'],
        color='red', size=8,
        edgecolor='black', linewidth=1,
        ax=ax,
        label='Actual IC50'
    )
    
    # Custom legend (only show one red dot)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles[-1:], labels=['Actual IC50'], 
              loc='upper right', frameon=True)
    
    # Clean formatting
    plt.title(f'{drug_name}\nPrediction Distribution vs Actual', pad=15)
    plt.xlabel('')
    plt.ylabel('IC50 Value', labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    
    # Save
    plot_path = drug_dir / f"{drug_id}_clean_violin.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved clean plot to {plot_path}")

if __name__ == "__main__":
    drug_id_to_name = load_drug_mapping()
    
    # Get all processed drugs (folders in results directory)
    processed_drugs = [
        int(folder.name) 
        for folder in RESULTS_DIR.glob("*") 
        if folder.is_dir() and folder.name.isdigit()
    ]
    
    for drug_id in processed_drugs:
        drug_name = drug_id_to_name.get(drug_id, f"Drug {drug_id}")
        print(f"\nProcessing {drug_name}...")
        create_violin_plots(drug_id, drug_name)