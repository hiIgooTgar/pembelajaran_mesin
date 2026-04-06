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

df2 = df.copy()

# perbaikan disini
df2['Age'] = df2['Age'].fillna(df2['Age'].mean())
df2['Salary'] = df2['Salary'].fillna(df2['Salary'].mean())

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
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