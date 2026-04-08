import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

# --- 1. IMPORT DATASET ---
dataset = pd.read_csv('dataset/california_housing_dataset.csv')
df = pd.DataFrame(dataset)

# --- 2. PENANGANAN MISSING VALUES (SOLUSI: MEAN BERDASARKAN BLOK) ---
df['block'] = pd.cut(df['households'], bins=20, labels=False)

def fill_nan_with_block_mean(group):
    return group.fillna(group.mean())

df['total_bedrooms'] = df.groupby('block')['total_bedrooms'].transform(fill_nan_with_block_mean)
df.drop(columns=['block'], inplace=True)

# --- 3. LABEL ENCODING UNTUK DATA KATEGORIKAL ---
label_encoder = LabelEncoder()
df['ocean_proximity'] = label_encoder.fit_transform(df['ocean_proximity'])

# --- 4. FEATURE SCALING (MINMAXSCALER) ---
scaler = MinMaxScaler()
# Scaling dilakukan pada fitur numerik agar PCA dan Chi-Square bekerja optimal
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# --- 5. EKSPERIMEN SELEKSI FITUR (KODE LAMA) ---
print("--- HASIL EKSPERIMEN SELEKSI FITUR (Top 7) ---")

# Model 1: F-Regression (Filter Method)
selector_f = SelectKBest(score_func=f_regression, k=7)
selector_f.fit(X_scaled, y)
features_f = X_scaled.columns[selector_f.get_support()].tolist()
print(f"1. F-Regression: {features_f}")

# Model 2: Recursive Feature Elimination / RFE (Wrapper Method)
rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=7)
rfe_selector.fit(X_scaled, y)
features_rfe = X_scaled.columns[rfe_selector.support_].tolist()
print(f"2. RFE (Linear Reg): {features_rfe}")

# Model 3: Random Forest Importance (Embedded Method)
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_scaled, y)
importances = pd.Series(rf_model.feature_importances_, index=X_scaled.columns)
features_rf = importances.nlargest(7).index.tolist()
print(f"3. Random Forest: {features_rf}")

# Tambahan: Chi-Square (Untuk input PCA sesuai instruksi)
selector_chi2 = SelectKBest(score_func=chi2, k=7)
X_selected = selector_chi2.fit_transform(X_scaled, y)
features_chi2 = X_scaled.columns[selector_chi2.get_support()].tolist()
print(f"4. Chi-Square   : {features_chi2}")

# --- 6. REDUKSI DIMENSI MENGGUNAKAN PCA (KODE BARU) ---
print("\n--- PROSES REDUKSI DIMENSI PCA ---")
# Inisialisasi PCA untuk mengambil 5 komponen utama dari hasil Chi-Square
pca = PCA(n_components=5)

# Transformasi data X_selected (hasil Chi-Square) menjadi X_pca
X_pca = pca.fit_transform(X_selected)

# Melihat rasio varians yang dijelaskan oleh setiap komponen
print("Rasio varians tiap komponen:", pca.explained_variance_ratio_)

# Melihat total informasi yang dipertahankan
total_info = sum(pca.explained_variance_ratio_) * 100
print(f"Total informasi yang dipertahankan: {total_info:.2f}%")
print(f"Total informasi yang tereliminasi: {100 - total_info:.2f}%")

print(f"\nUkuran data setelah Chi-Square: {X_selected.shape}")
print(f"Ukuran data setelah PCA: {X_pca.shape}")

# --- 7. DATA SPLITTING ---

# A. DATA SPLITTING TANPA PCA (Menggunakan 7 Fitur Terbaik dari Random Forest)
# Kita ambil fitur asli hasil seleksi Random Forest (X_final)
X_final_original = X_scaled[features_rf]
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_final_original, y, test_size=0.15, random_state=0
)

# B. DATA SPLITTING DENGAN PCA (Menggunakan 5 Komponen Utama)
# Menggunakan X_pca yang sudah dibuat di langkah sebelumnya
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.15, random_state=0
)

# --- 8. OUTPUT STATUS DATA ---

print("\n--- STATUS DATA SPLITTING (TANPA PCA) ---")
print(f"Fitur yang digunakan      : {features_rf}")
print(f"Jumlah data training (85%): {len(X_train_orig)}")
print(f"Jumlah data testing (15%) : {len(X_test_orig)}")

print("\n--- STATUS DATA SPLITTING (DENGAN PCA) ---")
print(f"Jumlah komponen PCA       : 5 Komponen")
print(f"Jumlah data training (85%): {len(X_train_pca)}")
print(f"Jumlah data testing (15%) : {len(X_test_pca)}")

# --- 9. VISUALISASI ---
plt.figure(figsize=(10, 6))
importances.sort_values().plot(kind='barh', color='skyblue')
plt.title('Feature Importance - Random Forest (Fitur Asli Sebelum PCA)')
plt.show()