# eda_preprocess.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Baca data ===
df = pd.read_csv("advertising.csv")  # sesuaikan nama file

# === 2. Info dasar dataset ===
print(df.head())
print("\nInfo:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive stats:\n", df.describe())

# === 3. Visualisasi distribusi ===
df.hist(figsize=(10,6), bins=20)
plt.suptitle("Distribusi Variabel")
plt.savefig("distribusi.jpg", dpi=300, bbox_inches='tight')
plt.show()

# === 4. Boxplot untuk cek outlier ===
plt.figure(figsize=(8,4))
sns.boxplot(data=df)
plt.title("Boxplot Semua Variabel")
plt.savefig("boxplot.jpg", dpi=300, bbox_inches='tight')
plt.show()

# === 5. Scatterplot X vs y untuk validasi linearitas ===
for col in ['TV', 'Radio', 'Newspaper']:
    plt.figure(figsize=(6,4))
    plt.scatter(df[col], df['Sales'], alpha=0.6)
    plt.xlabel(col)
    plt.ylabel("Sales")
    plt.title(f"{col} vs Sales")

    # Tambahkan garis tren linear
    m, b = np.polyfit(df[col], df['Sales'], 1)
    plt.plot(df[col], m*df[col] + b, color='red')

    plt.savefig(f"scatterplot_{col}.jpg", dpi=300, bbox_inches='tight')
    plt.show()

# === 6. Korelasi antar variabel ===
plt.figure(figsize=(5,4))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Korelasi")
plt.savefig("korelasi.jpg", dpi=300, bbox_inches='tight')
plt.show()

# === 7. Pembersihan Data: Outlier (IQR Method) ===
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
df_clean = df[mask].copy()
print(f"\nJumlah data awal: {len(df)}, setelah pembersihan: {len(df_clean)}")

# === 8. Simpan dataset bersih ===
df_clean.to_csv("Advertising_clean.csv", index=False)
print(" Saved Advertising_clean.csv")
