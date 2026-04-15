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
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures

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
print("\n \nSimple Linear Regression:")
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
print("\nMultiple Linear Regression:")
y_pred = evaluate_model(multiple_lr, X_test, y_test)

#plot hasil
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Multiple Linear Regression: ActualPredicted Values for Median HouseValue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

'''
Tugas Nomor 2: 
Multiple Linear Regression untuk memprediksi 'households'
'''
# 1. Identifikasi korelasi untuk menentukan 2 variabel terbaik
# Berdasarkan dataset California, variabel dengan korelasi tertinggi 
# terhadap 'households' adalah 'total_bedrooms' dan 'population'
X_task = df[['total_bedrooms', 'population']]
y_task = df['households']

# 2. Data Splitting
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_task, y_task, test_size=0.2, random_state=0
)

# 3. Membuat dan melatih model Multiple Linear Regression
model_households = LinearRegression()
model_households.fit(X_train_t, y_train_t)

# 4. Evaluasi Model
print("\nMultiple Linear Regression (Target: Households):")
y_pred_t = evaluate_model(model_households, X_test_t, y_test_t)

# 5. Plot Hasil Prediksi
plt.figure(figsize=(10, 6))
plt.scatter(y_test_t, y_pred_t, alpha=0.5, color='green')
plt.xlabel('Actual Households')
plt.ylabel('Predicted Households')
plt.title('Actual vs Predicted Households (2 Variables)')
plt.plot([min(y_test_t), max(y_test_t)], [min(y_test_t), max(y_test_t)], color='red')
plt.show()


'''
Tugas Nomor 3: Support Vector Regression (SVR)
'''
# Daftar kernel yang akan diuji
kernels = ['linear', 'poly', 'rbf']

for k in kernels:
    print(f"\nSVR dengan Kernel: {k}")
    # Inisialisasi model SVR dengan kernel terkait
    # Catatan: Untuk kernel 'poly', derajat default adalah 3
    svr_model = SVR(kernel=k)
    # Melatih model menggunakan seluruh variabel independen (X_train)
    svr_model.fit(X_train, y_train)
    # Evaluasi model
    evaluate_model(svr_model, X_test, y_test)


'''
Tugas Nomor 4: Polynomial Regression (Degree = 2)
'''
print("\nPolynomial Regression (Degree 2):")

# 1. Transformasi fitur menjadi polinomial derajat 2
poly_reg = PolynomialFeatures(degree=2)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.transform(X_test)

# 2. Membuat dan melatih model Linear Regression pada fitur polinomial
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# 3. Evaluasi Model
evaluate_model(poly_model, X_poly_test, y_test)