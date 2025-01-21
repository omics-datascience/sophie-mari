# read CCLE

import pandas as pd
import os
import gzip


# CCLE has the gene expression

# File path
folder = "./datasets/"
ccle_path = os.path.join(folder, "CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz")


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

# transform ENSENBL IDs into official gene symbols


# Read the GTF file (only necessary columns for gene_id and gene_name)
with gzip.open(gencode_path, 'rt') as f:
    # Parse only rows containing "gene" and extract the relevant column
    gtf_data = pd.read_csv(f, sep='\t', comment='#', header=None, usecols=[8], names=['attributes'])

# Extract gene_id and gene_name from the attributes column
def parse_attributes(attributes):
    """Extract gene_id and gene_name from the attributes string."""
    fields = attributes.split(';')
    gene_id = next((field.split('"')[1] for field in fields if field.strip().startswith("gene_id")), None)
    gene_name = next((field.split('"')[1] for field in fields if field.strip().startswith("gene_name")), None)
    return gene_id, gene_name

# Apply the parsing function to extract gene_id and gene_name
mapping = gtf_data['attributes'].apply(parse_attributes)
gene_mapping_df = pd.DataFrame(mapping.tolist(), columns=['gene_id', 'gene_name'])

# Drop duplicates to ensure unique mapping
gene_mapping_df = gene_mapping_df.drop_duplicates()

# Create a dictionary for mapping
gene_mapping = dict(zip(gene_mapping_df['gene_id'], gene_mapping_df['gene_name']))

# Print the first 10 items of the dictionary
for i, (key, value) in enumerate(gene_mapping.items()):
    print(f"{key}: {value}")
    if i == 9:  # Stop after printing 10 items
        break
# Replace gene IDs with gene names using the mapping dictionary
ccle_data['gene_id'] = ccle_data['gene_id'].map(gene_mapping)


# Display the first few rows of the transformed DataFrame
print("Transformed CCLE data", ccle_data.head())

# Average duplicates 

# Group by the 'gene_id' column (now containing gene names)
# and calculate the mean for all other columns
averaged_data = ccle_data.groupby('gene_id', as_index=False).mean()

# Display the first few rows of the resulting DataFrame
print(averaged_data.head())

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

    # Remove duplicate CELL_LINE_NAME entries for this drug
    drug_data = drug_data.groupby('CELL_LINE_NAME', as_index=False).mean()

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
            ic50_value = drug_data[drug_data['CELL_LINE_NAME'] == cell_line]['LN_IC50'].values[0]
            ic50_vector.append(ic50_value)
            matching_cell_lines.append(cell_line)

    # If no matching cell lines, return None
    if not gene_expression_matrix:
        print(f"No matching cell lines found for DRUG_ID {drug_id}.")
        return None, None

    # Convert the gene expression matrix to a DataFrame with rows as genes and columns as cell lines
    gene_expression_matrix = pd.DataFrame(gene_expression_matrix).T
    gene_expression_matrix.columns = matching_cell_lines

    return gene_expression_matrix, ic50_vector



# Specify a DRUG_ID
drug_id = 1

# Call the function with the specified drug ID
gene_expression_matrix, ic50_vector = create_matrix_and_ic50_for_drug(drug_id, combined_gdsc_df, averaged_data)

# Check the output
if gene_expression_matrix is not None:
    print(f"Gene Expression Matrix for DRUG_ID {drug_id}:")
    print(gene_expression_matrix.head())
    print("\nIC50 Vector:")
    print(ic50_vector[:10])  # Print the first 5 values
else:
    print(f"No data available for DRUG_ID {drug_id}.")