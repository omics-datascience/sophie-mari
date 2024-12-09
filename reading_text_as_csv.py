#This is my Python script for reading in .txt files
import pandas as pd

#Read the tab-delimited text file
data = pd.read_csv('data_clinical_patient.txt', sep='\t')
print(data)
