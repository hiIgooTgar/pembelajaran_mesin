import pandas as pd

# 1. Membaca file CSV
# Jika file berada di folder yang sama dengan script ini:
try:
    df = pd.read_csv('housing.csv')
    print("Data berhasil di-import!\n")
except FileNotFoundError:
    print("Error: File 'housing.csv' tidak ditemukan. Pastikan folder kerja sudah benar.")

# 2. Menampilkan 5 data teratas
print("--- 5 Data Teratas ---")
print(df.head())

# 3. Menampilkan informasi struktur data (Tipe data, kolom, dll)
print("\n--- Informasi Dataset ---")
print(df.info())

# 4. Menampilkan ringkasan statistik (Mean, Min, Max, dll)
print("\n--- Ringkasan Statistik ---")
print(df.describe())

# 5. Menampilkan jumlah data berdasarkan kedekatan dengan laut
print("\n--- Distribusi Ocean Proximity ---")
print(df['ocean_proximity'].value_counts())