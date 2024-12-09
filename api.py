import requests
import json
import time

# Define the API endpoint and headers
API_URL = "https://bioapi.multiomix.org/expression-of-genes"
HEADERS = {"Content-Type": "application/json"}

# List of tissues and genes to query
# Tissues used in the article
# OBS: double check if the tissues have the same name in the API
tissues = ['Bladder', 'Blood', 'Breast', 'Colon', 'Intestine', 'Kidney', 'Liver', 'Lung', 'Ovary', 'Pancreas', 'Prostate', 'Skin', 'Spleen', 'Stomach', 'Uterus']  
genes = ["BRCA1", "BRCA2", "TP53", "EGFR"]  # Replace with your gene list

# Function to get gene expression data for a specific tissue and list of genes
def get_gene_expression(tissue, gene_list, response_type="json"):
    """
    Fetch gene expression data for a specific tissue and list of genes.
    Args:
        tissue (str): Name of the tissue to query.
        gene_list (list): List of gene symbols to query.
        response_type (str): Response format type ('json' or 'gzip').

    Returns:
        dict or None: Response JSON if successful, None otherwise.
    """
    # Prepare request body
    body = {
        "tissue": tissue,
        "gene_ids": gene_list,
        "type": response_type
    }
    
    # Send the request
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(body))
        
        # Check if response is successful
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Loop over tissues and fetch data
for tissue in tissues:
    print(f"Fetching data for tissue: {tissue}")
    
    # Get gene expression data
    expression_data = get_gene_expression(tissue, genes)
    
    # Check and process response
    if expression_data:
        print(f"Data received for {tissue} - Showing sample records:")
        # Display the first few records for verification
        for sample in expression_data[:3]:  # Show only first 3 samples for brevity
            print(sample)
    
    # Pause between requests to respect API limits
    time.sleep(1)