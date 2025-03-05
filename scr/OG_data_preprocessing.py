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
print(ccle_data.shape)

# Average duplicates 

# Group by the 'gene_id' column (now containing gene names)
# and calculate the mean for all other columns
averaged_data = ccle_data.groupby('gene_id', as_index=False).mean(numeric_only=True)

# Display the first few rows of the resulting DataFrame
print(averaged_data.head())
print(averaged_data.shape)

# read GDSC 1

# File path
folder2 = "./datasets_OG/"
gdsc1_path = os.path.join(folder2, "GDSC1_fitted_dose_response_17Jul19.csv")

if not os.path.exists(gdsc1_path):
    print("File not found. Please check the file path.")
else:
    print("File exists.")

gdsc1_data = pd.read_csv(gdsc1_path)

# Print the head of the dataset
print(gdsc1_data.head(10))

# read GDSC 2

# File path
gdsc2_path = os.path.join(folder2, "GDSC2_fitted_dose_response_15Oct19.csv")

if not os.path.exists(gdsc2_path):
    print("File not found. Please check the file path.")
else:
    print("File exists.")

gdsc2_data = pd.read_csv(gdsc2_path)

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

def create_matrix_and_ic50_for_drug(drug_id, combined_gdsc_df, averaged_data):
    """
    Create gene expression data matrix and IC50 vector for a specific DRUG_ID.

    Parameters:
        drug_id (int): The DRUG_ID for which to create the matrix and vector.
        combined_gdsc_df (pd.DataFrame): GDSC dataset with DRUG_ID, CELL_LINE_NAME, and LN_IC50 columns.
        averaged_data (pd.DataFrame): CCLE dataset with gene expression profiles as rows and cell line columns.

    Returns:
        tuple: (gene_expression_matrix, ic50_vector)
            - gene_expression_matrix (pd.DataFrame): Gene expression matrix (G x N, CCLE).
            - ic50_vector (list): Corresponding IC50 values (length N, CCLE).
    """
    # Create mapping from CELL_LINE_ID to CCLE columns
    ccle_columns = {col.split('_')[0]: col for col in averaged_data.columns if '_' in col}

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
            gene_expression_matrix.append(averaged_data[ccle_column_name].values)

            # Append IC50 value for this cell line
            ic50_value = float(drug_data[drug_data['CELL_LINE_NAME'] == cell_line]['LN_IC50'].values[0])
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
print(f"Number of genes in gene_expression_matrix: {gene_expression_matrix.shape[0]}")
print(f"Length of ic50_vector: {len(ic50_vector)}")

# Export all gene expression matrices and IC50 vectors
# Define the list of drug IDs
'''
drug_ids = [
    1, 3, 5, 6, 9, 11, 17, 29, 30, 32, 34, 35, 37, 38, 41, 45, 51, 52, 53, 54, 55, 
    56, 59, 60, 62, 63, 64, 71, 83, 86, 87, 88, 89, 91, 94, 104, 106, 110, 111, 119, 
    127, 133, 134, 135, 136, 140, 147, 150, 151, 152, 153, 154, 155, 156, 157, 158, 
    159, 163, 164, 165, 166, 167, 170, 171, 172, 173, 175, 176, 177, 178, 179, 180, 
    182, 184, 185, 186, 190, 192, 193, 194, 196, 197, 199, 200, 201, 202, 203, 204, 
    205, 206, 207, 208, 211, 219, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 
    235, 236, 238, 245, 249, 252, 253, 254, 255, 256, 257, 258, 260, 261, 262, 263, 
    264, 265, 266, 268, 269, 271, 272, 273, 274, 275, 276, 277, 279, 281, 282, 283, 
    284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 298, 299, 300, 301, 
    302, 303, 304, 305, 306, 308, 309, 310, 312, 317, 326, 328, 329, 330, 331, 332, 
    333, 341, 342, 344, 345, 346, 356, 362, 363, 366, 371, 372, 374, 375, 376, 380, 
    381, 382, 406, 407, 408, 409, 410, 412, 415, 416, 427, 428, 431, 432, 435, 436, 
    437, 438, 439, 442, 446, 447, 449, 461, 474, 476, 477, 478, 546, 552, 562, 563, 
    573, 574, 576, 1001, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 
    1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 
    1028, 1029, 1030, 1031, 1032, 1033, 1036, 1037, 1038, 1039, 1042, 1043, 1046, 
    1047, 1048, 1049, 1050, 1052, 1053, 1054, 1057, 1058, 1059, 1060, 1061, 1062, 
    1066, 1067, 1069, 1072, 1091, 1114, 1129, 1133, 1142, 1143, 1149, 1158, 1161, 
    1164, 1166, 1170, 1175, 1192, 1194, 1199, 1203, 1218, 1219, 1230, 1236, 1239, 
    1241, 1242, 1243, 1248, 1259, 1261, 1262, 1263, 1264, 1266, 1268, 1371, 1372, 
    1373, 1375, 1377, 1378, 1494, 1495, 1498, 1502, 1526, 1527, 1529, 1530, 1003, 
    1051, 1068, 1073, 1079, 1080, 1083, 1084, 1085, 1086, 1088, 1089, 1093, 1096, 
    1131, 1168, 1177, 1179, 1180, 1190, 1191, 1200, 1237, 1249, 1250, 1507, 1510, 
    1511, 1512, 1549, 1553, 1557, 1558, 1559, 1560, 1561, 1563, 1564, 1576, 1578, 
    1593, 1594, 1598, 1613, 1614, 1615, 1617, 1618, 1620, 1621, 1622, 1624, 1625, 
    1626, 1627, 1629, 1630, 1631, 1632, 1634, 1635, 1786, 1799, 1802, 1804, 1806, 
    1807, 1808, 1809, 1810, 1811, 1813, 1814, 1816, 1818, 1819, 1825, 1827, 1830, 
    1835, 1838, 1849, 1852, 1853, 1854, 1855, 1866, 1873, 1908, 1909, 1910, 1911, 
    1912, 1913, 1915, 1916, 1917, 1918, 1919, 1922, 1924, 1925, 1926, 1927, 1928, 
    1930, 1931, 1932, 1933, 1936, 1939, 1940, 1941, 1996, 1997, 1998, 2040, 2043, 
    2044, 2045, 2046, 2047, 2048, 2096, 2106, 2107, 2109, 2110, 2111, 2169, 2170, 
    2171, 2172
]

# Create directories if they don’t exist
os.makedirs("gene_expression_matrices", exist_ok=True)
os.makedirs("drug_id_vectors", exist_ok=True)

for drug_id in drug_ids:
    gene_expression_matrix, ic50_vector = create_matrix_and_ic50_for_drug(drug_id, combined_gdsc_df, averaged_data)

    if gene_expression_matrix is not None:
        gene_expression_matrix.to_csv(f"gene_expression_matrices/{drug_id}_matrix.csv")
        pd.Series(ic50_vector).to_csv(f"drug_id_vectors/{drug_id}_vector.csv", index=False)

        print(f"Saved files for DRUG_ID {drug_id}")
    else:
        print(f"No data for DRUG_ID {drug_id}")
        '''

#Exporting Drug 1 to .csv

# Create a DataFrame for the IC50 row with the same columns as the gene expression data
#ic50_row = pd.Series(ic50_vector, index=gene_expression_matrix.columns)

# Insert IC50 row at the top of the DataFrame
#gene_expression_matrix.loc['IC50'] = ic50_row  # Adding IC50 row

# Export to CSV (with IC50 as the first row)
#gene_expression_matrix.to_csv('gene_expression_and_ic50.csv', index=True, header=True)

# Export to JSON (with IC50 as the first row)
#gene_expression_matrix.to_json('gene_expression_and_ic50.json', orient='records', lines=True)

