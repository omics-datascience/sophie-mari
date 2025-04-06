import pandas as pd
from scipy.stats import pearsonr, spearmanr

df = pd.read_csv("results/1372/final_predictions.csv").dropna()
print("Pearson:", pearsonr(df["actual"], df["predicted"])[0])
print("Spearman:", spearmanr(df["actual"], df["predicted"])[0])
