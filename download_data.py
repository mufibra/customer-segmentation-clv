"""Download the UCI Online Retail dataset."""
import os
import pandas as pd

os.makedirs("data/raw", exist_ok=True)

output_path = "data/raw/online_retail.csv"

if os.path.exists(output_path):
    print(f"Already exists: {output_path}")
else:
    print("Downloading UCI Online Retail dataset (~20MB Excel file)...")
    print("This may take a minute...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = pd.read_excel(url)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
