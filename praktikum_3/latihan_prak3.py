# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

# 1. Import dataset
# Pastikan file csv berada di folder yang sesuai
dataset = pd.read_csv('dataset/california_housing_dataset.csv')
df = pd.DataFrame(dataset)

# 2. Identifikasi missing values
print("Info dataset awal:")
print(df.info())
mv = df.isna().sum()
print('\nJumlah missing value tiap kolom:\n', mv)

'''
Solusi missing value:
Isi missing value dengan nilai median berdasarkan blok 'households'
'''
# Membagi dataframe menjadi 20 blok berdasarkan 'households'
df['block'] = pd.cut(df['households'], bins=20, labels=False)

# Fungsi untuk mengisi NaN dengan median dari setiap blok
# Diperbaiki: Menggunakan lambda untuk memastikan fillna berjalan di tiap group
df['total_bedrooms'] = df.groupby('block')['total_bedrooms'].transform(lambda x: x.fillna(x.median()))

# Validasi cadangan: jika ada blok yang seluruhnya NaN, isi dengan median global
if df['total_bedrooms'].isna().sum() > 0:
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# Menghapus kolom 'block' setelah proses selesai
df.drop(columns=['block'], inplace=True)

print('\nMissing value setelah perbaikan:', df['total_bedrooms'].isna().sum())

'''
Solusi Categorical Data:
Pengodean kolom ocean_proximity dengan LabelEncoder
'''
label_encoder_x = LabelEncoder()
df['ocean_proximity'] = label_encoder_x.fit_transform(df['ocean_proximity'])

# Visualisasi (Opsional - Scatter plot)
sns.scatterplot(x='median_income', y='median_house_value', data=df)
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Income vs House Value')
plt.show()

'''
Solusi feature scaling:
PENTING: chi2 membutuhkan nilai non-negatif. 
Kita skalakan semua fitur X agar aman untuk SelectKBest.
'''
# Ekstraksi variabel independen (X) dan dependen (y)
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

# Melakukan scaling pada seluruh fitur X
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

'''
Seleksi fitur menggunakan chi-square
'''
# Memilih 7 fitur terbaik
selector = SelectKBest(score_func=chi2, k=7)
X_selected = selector.fit_transform(X_scaled, y)

# Mendapatkan nama fitur yang tereliminasi
eliminated_features_index = ~selector.get_support()
eliminated_features = X.columns[eliminated_features_index]

print("\nFitur-fitur yang tereliminasi:")
print(eliminated_features)

# Mendapatkan nama fitur yang terpilih
selected_features = X.columns[selector.get_support()]
print("\nFitur-fitur yang terpilih:")
print(selected_features)

'''
Data splitting
'''
# Pembagian dataset menjadi training dan test set
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=0
)

print("\nData splitting berhasil.")
print(f"Jumlah data training: {X_train.shape[0]}")
print(f"Jumlah data testing: {X_test.shape[0]}")