import pandas as pd
# Ganti dengan path atau link ke dataset Anda
df = pd.read_csv("advertising.csv")
print("--- 5 Baris Pertama ---")
print(df.head())
print("\n--- Nama Kolom ---")
print(df.columns)