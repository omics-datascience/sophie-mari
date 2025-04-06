import pandas as pd
import gzip
from pathlib import Path
from collections import Counter
import numpy as np

# Set up file paths
project_root = Path(__file__).resolve().parent.parent
datasets_dir = project_root / "datasets"
tcga_path = datasets_dir / "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
gtex_path = datasets_dir / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
tcga_tumor_path = datasets_dir / "tcga_tumor.txt"
tcga_normal_path = datasets_dir / "tcga_normal.txt"

# Load barcodes
barcodes_df = pd.read_csv(tcga_path, sep='\t', nrows=5)
barcodes = barcodes_df.columns[1:]  # skip first column only
print("Sample barcodes:", barcodes[:5].tolist())

# Classify sample types
def classify_tcga_sample(barcode: str) -> str:
    sample_type = barcode.split('-')[3][:2]
    if sample_type == '11':
        return 'normal'
    elif sample_type == '01':
        return 'tumor'
    else:
        return 'other'

sample_classes = {barcode: classify_tcga_sample(barcode) for barcode in barcodes}
print("Sample type counts:", Counter(sample_classes.values()))

# Load full TCGA expression matrix
print("Loading full TCGA data...")
df_tcga_full = pd.read_csv(tcga_path, sep='\t', index_col=0)
print(f"✅ Loaded TCGA with shape: {df_tcga_full.shape}")

# Extract gene symbols from "?|symbol" format in the first column
df_tcga_full.index = df_tcga_full.index.to_series().astype(str).str.split('|').str[-1]

# Subset tumor/normal samples
normal_barcodes = [s for s, t in sample_classes.items() if t == 'normal']
tumor_barcodes = [s for s, t in sample_classes.items() if t == 'tumor']
df_tcga_normal = df_tcga_full[normal_barcodes].copy()
df_tcga_tumor = df_tcga_full[tumor_barcodes].copy()
print(f"TCGA tumor shape: {df_tcga_tumor.shape}")
print(f"TCGA normal shape: {df_tcga_normal.shape}")

# Drop potential duplicates by averaging
df_tcga_tumor = df_tcga_tumor.groupby(df_tcga_tumor.index).mean(numeric_only=True)
df_tcga_normal = df_tcga_normal.groupby(df_tcga_normal.index).mean(numeric_only=True)

# Log2 transform
df_tcga_tumor = np.log2(df_tcga_tumor + 1)
df_tcga_normal = np.log2(df_tcga_normal + 1)

# Save if not already saved
if not tcga_tumor_path.exists():
    df_tcga_tumor.to_csv(tcga_tumor_path, sep='\t')
    print(f"✅ Saved tumor data to: {tcga_tumor_path}")
else:
    print(f"ℹ️ {tcga_tumor_path.name} already exists — skipping save.")

if not tcga_normal_path.exists():
    df_tcga_normal.to_csv(tcga_normal_path, sep='\t')
    print(f"✅ Saved normal data to: {tcga_normal_path}")
else:
    print(f"ℹ️ {tcga_normal_path.name} already exists — skipping save.")

# Load CCLE and TCGA again (post-saved)
ccle_df = pd.read_csv(datasets_dir / "ccle.txt", sep='\t', index_col=0)
tcga_tumor_df = pd.read_csv(tcga_tumor_path, sep='\t', index_col=0)
tcga_normal_df = pd.read_csv(tcga_normal_path, sep='\t', index_col=0)

print("✅ Loaded CCLE data:", ccle_df.shape)
print("✅ Loaded TCGA tumor:", tcga_tumor_df.shape)
print("✅ Loaded TCGA normal:", tcga_normal_df.shape)

# Intersect genes
common_genes = ccle_df.index.intersection(tcga_tumor_df.index).intersection(tcga_normal_df.index)
print(f"🧬 Common genes between CCLE and TCGA: {len(common_genes)}")

# Filter to common genes
ccle_df = ccle_df.loc[common_genes]
tcga_tumor_df = tcga_tumor_df.loc[common_genes]
tcga_normal_df = tcga_normal_df.loc[common_genes]

print("📀 Filtered shapes:")
print("CCLE:", ccle_df.shape)
print("TCGA Tumor:", tcga_tumor_df.shape)
print("TCGA Normal:", tcga_normal_df.shape)

'''# === Load TCGA ===
df_tcga = pd.read_csv(tcga_path, sep='\t', nrows=5)
print("✅ TCGA data (head):")
print(df_tcga.head())
print("\nTCGA Columns:", df_tcga.columns[:10].tolist())

# === Load GTEx ===
df_gtex = pd.read_csv(gtex_path, sep='\t', skiprows=2)
df_gtex_head = df_gtex.head()

print("\n✅ GTEx data (head):")
print(df_gtex_head)
print("\nGTEx Columns:", df_gtex.columns[:10].tolist())'''
