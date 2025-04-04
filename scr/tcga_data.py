import pandas as pd
from pathlib import Path

'project_root = Path(__file__).resolve().parent.parent
tcga_path = project_root / "datasets" / "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"

'''# Load just first few lines
df_tcga = pd.read_csv(tcga_path, sep='\t', nrows=5)

print("✅ TCGA data (head):")
print(df_tcga.head())
print("Columns:", df_tcga.columns[:10])'''

cols_only = pd.read_csv(tcga_path, sep='\t', nrows=0)
print("Column names:", cols_only.columns[:10])
print("Total columns:", len(cols_only.columns))