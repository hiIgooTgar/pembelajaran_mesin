#import libraries
import numpy as np
import pandas as pd
import sklearn 

dataset = pd.read_csv('data.csv')
df = pd.DataFrame(dataset)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(df.info())
mv = df.isna().sum()
print('\nJumlah missing value tiap kolom:\n', mv)

df1 = df.copy()
print("Sebelum: ", df1.shape)
df1.dropna(inplace=True)
print("Setelah: ", df1.shape)

'''
Solusi 2: Isi missing value dengan nilai MEDIAN (Latihan 1)
'''
df2 = df.copy()
df2['Age'] = df2['Age'].fillna(df2['Age'].median())
df2['Salary'] = df2['Salary'].fillna(df2['Salary'].median())

'''
Isi missing value pada variabel X dengan strategi MEDIAN (Latihan 2)
'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder
label_encoder_x = LabelEncoder()
X[:, 0] = label_encoder_x.fit_transform(X[:, 0])
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
X_train[:, 1:] = st_x.fit_transform(X_train[:, 1:])
X_test[:, 1:] = st_x.transform(X_test[:, 1:])

'''
Standarisasi variabel X_train dan X_test menggunakan MinMaxScaler (Latihan 3)
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[:, 1:] = scaler.fit_transform(X_train[:, 1:])
X_test[:, 1:] = scaler.transform(X_test[:, 1:])

# Tambahan: Menampilkan statistik deskriptif (Mean, Median, Mode/Modus) untuk analisis
print("\n--- Analisis Statistik Deskriptif (Data Asli) ---")
print("Mean Age:", df['Age'].mean())
print("Median Age:", df['Age'].median()) 
print("Mode Age:", df['Age'].mode()[0]) 
print("")
print("Mean Salary:", df['Salary'].mean()) 
print("Median Salary:", df['Salary'].median()) 
print("Mode Salary:", df['Salary'].mode()[0]) 

print("\n--- Latihan 3: Output X_train (MinMaxScaler) ---")
print(X_train)