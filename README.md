# Tumor Drug Sensitivity Prediction Project

## Overview

This project replicates and extends the study presented in **"[Predicting Tumor Response to Drugs Based on Gene-Expression Biomarkers of Sensitivity Learned from Cancer Cell Lines](https://doi.org/10.1186/s12864-021-07581-7)"**. The study aims to predict tumor response to drugs using gene-expression biomarkers learned from cancer cell lines, primarily by leveraging a Genetic Algorithm (GA) for feature selection and k-Nearest Neighbors (KNN) for prediction. This project allows users to predict drug sensitivity based on gene-expression data.

## Requirements

- Python 3.8 or above
- Recommended hardware: CPU or GPU if large datasets are used
- Recommended OS: Ubuntu, MacOS, or Windows

## Set up environment

   - Clone this repository and navigate to the project directory.
   - Install dependencies using:
     ```bash
     pip install -r requirements.txt
     ```
## Download Datasets

- TCGA RNA-seq: https://gdc.cancer.gov/about-data/publications/pancanatlas (filename: EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv)
- GTEx RNA-seq: https://www.gtexportal.org/home/datasets (filename: GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz)
- CCLE RNA-seq: https://depmap.org/portal/data_page/?tab=allData (filename: CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz)
- Drug Sensitivity (IC50 values): https://www.cancerrxgene.org/downloads/bulk_download (filenames: GDSC1_fitted_dose_response_17Jul19.txt & GDSC2_fitted_dose_response_15Oct19.txt)
