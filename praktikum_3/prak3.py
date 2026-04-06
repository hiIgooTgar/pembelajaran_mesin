#import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#import dataset
dataset=pd.read_csv('california_housing_dataset.csv')
df=pd.DataFrame(dataset)

#identifikasi missing values
print(df.info())
mv=df.isna().sum()
print('\nJumlah missing value tiap kolom:\n', mv)

'''
Solusi missing value:
Isi missing value dengan nilai median
'''
#membagi dataframe menjadi 20 blok berdasarkan 'households'
df['block'] = pd.cut(df['households'], bins=20, labels=False)

#fungsi untuk mengisi NaN dengan median dari setiap blok
def fill_nan_with_block_median(group):
return group.fillna(group.median())

#mengisi NaN pada kolom 'total_bedrooms' dengan median dari
#blok 'total_bedrooms' dari blok record tersebut berasal

df['total_bedrooms'] =
df.groupby('block')['total_bedrooms'].transform(fill_nan_with_block_median)

#kode bantu untuk cek blok dan median secara manual
block_0_records = df.loc[df['block'] == 0]
median_households_block_0 = block_0_records['total_bedrooms'].median()
print("Nilai median kolom 'total_bedrooms' pada block 0:",
median_households_block_0)

#menghapus kolom 'block' setelah proses pengisian NaN selesai
df.drop(columns=['block'], inplace=True)

'''
Solusi Categorical Data:
Pengodean kolom ocean_proximity dengan LabelEncoder
'''
label_encoder_x= LabelEncoder()
df['ocean_proximity']= label_encoder_x.fit_transform(df['ocean_proximity'])

#identifikasi korelasi antar variabel
correlation = df.corr(method='pearson')

#scatter plot:'median_income' vs 'median_house_value'
sns.scatterplot(x='median_income', y='median_house_value', data=df)
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Income vs House Value')
plt.show()

#scatter plot: 'latitude' & 'longitude'
plt.figure(figsize=(15,8))
sns.scatterplot(x='latitude', y='longitude', data=df,
hue='median_house_value')

'''
Solusi feature scaling dengan MinMaxScaler
'''
scaler = MinMaxScaler()
df[['latitude', 'longitude']] = scaler.fit_transform(df[['latitude',
'longitude']])

#ekstraksi variabel independen
X = df.drop(columns=['median_house_value'])

#ekstraksi variabel dependen
y = df['median_house_value']

'''
seleksi fitur menggunakan chi-square
'''
# Memilih 7 fitur terbaik
selector = SelectKBest(score_func=chi2, k=7)
X_selected = selector.fit_transform(X, y)

#mendapatkan indeks fitur yang tereliminasi
eliminated_features_index = ~selector.get_support()

#mendapatkan nama fitur yang tereliminasi
eliminated_features = X.columns[eliminated_features_index]

#menampilkan fitur-fitur yang tereliminasi
print("Fitur-fitur yang tereliminasi:")
print(eliminated_features)

'''
Data spliting
'''
#pembagian dataset menjadi training dan test set
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size
= 0.2, random_state = 0)