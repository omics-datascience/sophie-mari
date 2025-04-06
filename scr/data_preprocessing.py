# read CCLE

import pandas as pd
import os
import gzip
import numpy as np
from pathlib import Path

# CCLE has the gene expression

# File path
folder = "./datasets/"
ccle_path = os.path.join(folder, "CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz")
# Define the save path for processed CCLE data
ccle_save_path = Path("datasets") / "ccle.txt"
ccle_save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure 'datasets/' exists


if not os.path.exists(ccle_path):
    print("File not found. Please check the file path.")
else:
    print("File exists.")

# Read the gzipped file
ccle_data = pd.read_csv(ccle_path, compression='gzip', sep='\t', comment='#')

# Print the head of the dataset
print("CCLE data head:", ccle_data.head())

# transform ENSENBL IDs into official gene symbols

# read gene annotation

# File path
gencode_path = os.path.join(folder, "gencode.v19.annotation.gtf.gz")

if not os.path.exists(gencode_path):
    print("Gencode file not found. Please check the file path.")
else:
    print("Gencode file exists.")

# Read the gzipped file
gencode_data = pd.read_csv(gencode_path, compression='gzip', sep='\t', comment='#')

# Print the head of the dataset
print("Gencode data head:", gencode_data.head(10))
print("Gencode data shape:", gencode_data.shape)



# Read the GTF file (only necessary columns for gene_id, gene_name, and gene_type)
with gzip.open(gencode_path, 'rt') as f:
    # Parse only rows containing "gene" and extract the relevant column
    gtf_data = pd.read_csv(f, sep='\t', comment='#', header=None, usecols=[8], names=['attributes'])

# Extract gene_id, gene_name, and gene_type from the attributes column
def parse_attributes(attributes):
    """Extract gene_id, gene_name, and gene_type from the attributes string."""
    fields = attributes.split(';')
    gene_id = next((field.split('"')[1] for field in fields if field.strip().startswith("gene_id")), None)
    gene_name = next((field.split('"')[1] for field in fields if field.strip().startswith("gene_name")), None)
    gene_type = next((field.split('"')[1] for field in fields if field.strip().startswith("gene_type")), None)
    return gene_id, gene_name, gene_type

# Apply the parsing function
mapping = gtf_data['attributes'].apply(parse_attributes)
gene_mapping_df = pd.DataFrame(mapping.tolist(), columns=['gene_id', 'gene_name', 'gene_type'])

# Filter only protein-coding genes
protein_coding_genes = gene_mapping_df[gene_mapping_df['gene_type'] == 'protein_coding']

# Create a dictionary mapping Ensembl IDs to gene names
gene_mapping = dict(zip(protein_coding_genes['gene_id'], protein_coding_genes['gene_name']))

print("Length of mapping dictionary:", len(gene_mapping))

# Print the first 10 items of the dictionary
for i, (key, value) in enumerate(gene_mapping.items()):
    print(f"{key}: {value}")
    if i == 9:  # Stop after printing 10 items
        break

# Replace gene IDs with gene names using the mapping dictionary
ccle_data['gene_id'] = ccle_data['gene_id'].map(gene_mapping)

# Display the first few rows of the transformed DataFrame
print("Transformed CCLE data", ccle_data.head())
print("Transformed data CCLE before average:", ccle_data.shape)

# Average duplicates 

# Group by the 'gene_id' column (now containing gene names)
# and calculate the mean for all other columns
averaged_data = ccle_data.groupby('gene_id', as_index=False).mean(numeric_only=True)

# Display the first few rows of the resulting DataFrame
print("Averaged data head:", averaged_data.head())
print("Averaged data shape:", averaged_data.shape)

# Final save: ensure gene_id is index and write cleanly
ccle_output_path = Path("datasets") / "ccle.txt"
if not ccle_output_path.exists():
    print("🔄 Saving clean CCLE file...")
    averaged_data.set_index("gene_id", inplace=True)
    averaged_data.to_csv(ccle_output_path, sep='\t')
    print(f"✅ Saved processed CCLE data to: {ccle_output_path}")
else:
    print(f"ℹ️ {ccle_output_path.name} already exists — skipping save.")

# read GDSC 1

# File path
gdsc1_path = os.path.join(folder, "GDSC1_fitted_dose_response_27Oct23.xlsx")

if not os.path.exists(gdsc1_path):
    print("File not found. Please check the file path.")
else:
    print("File exists.")

gdsc1_data = pd.read_excel(gdsc1_path)

# Print the head of the dataset
print(gdsc1_data.head(10))

# read GDSC 2

# File path
gdsc2_path = os.path.join(folder, "GDSC2_fitted_dose_response_27Oct23.xlsx")

if not os.path.exists(gdsc2_path):
    print("File not found. Please check the file path.")
else:
    print("File exists.")

gdsc2_data = pd.read_excel(gdsc2_path)

# Print the head of the dataset
print(gdsc2_data.head(10))

# Ensure both files have the same column names (if needed)
assert list(gdsc1_data.columns) == list(gdsc2_data.columns), "Column names must match!"

# Drop duplicates from the first file if they exist in the second file
# Keeping rows from df2 when duplicates are found
gdsc1_filtered = gdsc1_data[~gdsc1_data[['CELL_LINE_NAME', 'DRUG_ID']].isin(gdsc2_data[['CELL_LINE_NAME', 'DRUG_ID']].to_dict(orient='list')).all(axis=1)]

# Combine the filtered rows from df1 with all rows from df2
combined_gdsc_df = pd.concat([gdsc1_filtered, gdsc2_data], ignore_index=True)

# Display the result
print(combined_gdsc_df.head())

# Check the shape of the combined DataFrame
num_rows, num_columns = combined_gdsc_df.shape

print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# ✅ Save combined GDSC as CSV (if it doesn't already exist)
project_root = Path(__file__).resolve().parent.parent
csv_path = project_root / "datasets" / "combined_gdsc.csv"

if not csv_path.exists():
    combined_gdsc_df.to_csv(csv_path, index=False)
    print(f"✅ Saved combined GDSC dataset to: {csv_path}")
else:
    print(f"ℹ️ combined_gdsc.csv already exists — skipping save.")


# create data matrix and IC50 vector for each drug

def create_matrix_and_ic50_for_drug(drug_id, combined_gdsc_df, ccle_data):
    """
    Create gene expression data matrix and IC50 vector for a specific DRUG_ID.

    Parameters:
        drug_id (int): The DRUG_ID for which to create the matrix and vector.
        combined_gdsc_df (pd.DataFrame): GDSC dataset with DRUG_ID, CELL_LINE_NAME, and LN_IC50 columns.
        ccle_data (pd.DataFrame): CCLE dataset with gene expression profiles as rows and cell line columns.

    Returns:
        tuple: (gene_expression_matrix, ic50_vector)
            - gene_expression_matrix (pd.DataFrame): Gene expression matrix (G x N, CCLE).
            - ic50_vector (list): Corresponding IC50 values (length N, CCLE).
    """
    # Create mapping from CELL_LINE_ID to CCLE columns
    ccle_columns = {col.split('_')[0]: col for col in ccle_data.columns if '_' in col}

    # Filter rows for the given DRUG_ID
    drug_data = combined_gdsc_df[combined_gdsc_df['DRUG_ID'] == drug_id]

    # Initialize lists to store matrix columns and IC50 values
    gene_expression_matrix = []
    ic50_vector = []
    matching_cell_lines = []

    # Match CELL_LINE_NAMEs to CCLE columns
    for cell_line in drug_data['CELL_LINE_NAME']:
        if cell_line in ccle_columns:
            ccle_column_name = ccle_columns[cell_line]

            # Append the gene expression profile to the matrix
            gene_expression_matrix.append(ccle_data[ccle_column_name].values)

            # Append IC50 value for this cell line
            ic50_value = float(drug_data[drug_data['CELL_LINE_NAME'] == cell_line]['LN_IC50'].values[0])
            ic50_vector.append(ic50_value)
            matching_cell_lines.append(cell_line)

    ic50_vector = np.array(ic50_vector)

    # If no matching cell lines, return None
    if not gene_expression_matrix:
        print(f"No matching cell lines found for DRUG_ID {drug_id}.")
        return None, None

    # Convert the gene expression matrix to a DataFrame with rows as genes and columns as cell lines
    gene_expression_matrix = pd.DataFrame(gene_expression_matrix).T
    gene_expression_matrix.columns = matching_cell_lines

    return gene_expression_matrix, ic50_vector


# Get all unique drug IDs
unique_drug_ids = combined_gdsc_df['DRUG_ID'].unique()
print(unique_drug_ids[:10]) #first 10 unique drug IDs

# Count the number of unique drug IDs
num_unique_drugs = combined_gdsc_df['DRUG_ID'].nunique()
print(f"Number of unique DRUG_IDs: {num_unique_drugs}")

# Count the occurrences of each DRUG_ID
drug_id_counts = combined_gdsc_df['DRUG_ID'].value_counts()

# Find the drug IDs with the maximum and minimum counts
max_drug_id, max_count = drug_id_counts.idxmax(), drug_id_counts.max()
min_drug_id, min_count = drug_id_counts.idxmin(), drug_id_counts.min()

print(f"Drug ID with the maximum count: {max_drug_id}, Count: {max_count}")
print(f"Drug ID with the minimum count: {min_drug_id}, Count: {min_count}")


# Specify a DRUG_ID
drug_id = 1

# Call the function with the specified drug ID
gene_expression_matrix, ic50_vector = create_matrix_and_ic50_for_drug(drug_id, combined_gdsc_df, averaged_data)

# Check the output
if gene_expression_matrix is not None:
    print(f"Gene Expression Matrix for DRUG_ID {drug_id}:")
    print(gene_expression_matrix.head())
    print("\nIC50 Vector:")
    print(ic50_vector[:10])  # Print the first 10 values
else:
    print(f"No data available for DRUG_ID {drug_id}.")

print(f"Number of cell lines in gene_expression_matrix: {gene_expression_matrix.shape[1]}")
print(f"Length of ic50_vector: {len(ic50_vector)}")

#Exporting Drug 1 to .csv

# Export matrix to CSV 
gene_expression_matrix.to_csv('gene_expression_drug1.csv', index=True, header=True)

# Convert to Pandas Series
ic50_series = pd.Series(ic50_vector)

# Save as csv
ic50_series.to_csv('ic50_drug1.csv', index=True, header=True)

