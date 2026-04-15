#import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

#import dataset
dataset=pd.read_csv('dataset/california_housing_dataset.csv')
df=pd.DataFrame(dataset)

#identifikasi missing values
print(df.info())
mv=df.isna().sum()
print('\nJumlah missing value tiap kolom:\n', mv)

'''
Solusi Categorical Data:
Pengodean kolom ocean_proximity dengan LabelEncoder
'''
label_encoder_x= LabelEncoder()
df['ocean_proximity']= label_encoder_x.fit_transform(df['ocean_proximity'])

#identifikasi korelasi antar variabel
correlation = df.corr(method='pearson')

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
df['total_bedrooms'] = df.groupby('block')['total_bedrooms'].transform(fill_nan_with_block_median)

#menghapus kolom 'block' setelah proses pengisian NaN selesai
df.drop(columns=['block'], inplace=True)

#scatter plot:'median_income' vs 'median_house_value'
sns.scatterplot(x='median_income', y='median_house_value', data=df)
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Income vs House Value')
plt.show()

'''
Solusi feature scaling dengan MinMaxScaler
'''
scaler = MinMaxScaler()
df[['latitude', 'longitude']] = scaler.fit_transform(df[['latitude', 'longitude']])

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

'''
Data spliting
'''
#pembagian dataset menjadi training dan test set
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size = 0.2, random_state = 0)

'''
Evaluasi model Regresi
'''
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    return y_pred

'''
Model Simple Linear Regression
'''
#membuat model
simple_lr = LinearRegression()

#melatih model dengan mengambil value dari kolom median_income (X_train)
simple_lr.fit(X_train[:, 5].reshape(-1, 1), y_train)

#evaluasi model dengan mengambil value dari kolom median_income (X_test)
print("Simple Linear Regression:")
y_pred = evaluate_model(simple_lr, X_test[:, 5].reshape(-1, 1), y_test)

#plot hasil
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Simple Linear Regression: ActualPredicted Values for Median HouseValue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

'''
Multiple Linear Regression
'''
#membuat model
multiple_lr = LinearRegression()

#melatih model
multiple_lr.fit(X_train, y_train)

#evaluasi model
print("Multiple Linear Regression:")
y_pred = evaluate_model(multiple_lr, X_test, y_test)

#plot hasil
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Multiple Linear Regression: ActualPredicted Values for Median HouseValue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()