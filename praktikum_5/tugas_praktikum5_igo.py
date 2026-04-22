# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

#import dataset
dataset=pd.read_csv('dataset/IceCreamData.csv')
X = dataset['Temperature'].values
y = dataset['Revenue'].values

# Splitting the dataset into the Training set and Test set
# print("Data Uji : 20% \n")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
'''
Tugas Nomor 1: Data Uji Menjadi 30%
'''
print("Data Uji : 30% \n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

'''
Decision Tree Regressor Model
'''
# Training the Decision Tree Regression model on the training set
DTregressor = DecisionTreeRegressor()

# reshape variables to a single column vector
DTregressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

# Predicting the Results
y_pred = DTregressor.predict(X_test.reshape(-1,1))

# Comparing the Real Values with Predicted Values
df = pd.DataFrame({'Real Values':y_test.reshape(-1), 'Predicted Values':y_pred.reshape(-1)})

# Visualising the Decision Tree Regression Results
# Real values: Red & Predicted values: Green
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'red')
plt.scatter(X_test, y_pred, color = 'green')
plt.title('Decision Tree Regression')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()

plt.plot(X_grid, DTregressor.predict(X_grid), color = 'black')
plt.title('Decision Tree Regression')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()

'''
Evaluasi model Regresi
'''
print("--- Evaluasi Model Decision Tree ---")
print("Mean Squared Error (MSE)       : ", mean_squared_error(y_test, y_pred))
print("R-Squared (R2 Score)           : ", r2_score(y_test, y_pred))
print("Mean Absolute Percentage Error : ", mean_absolute_percentage_error(y_test, y_pred))
print("-" * 64)
print("\n")

'''
Random Forest Regression Model
'''
RFregressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
RFregressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

y_pred = RFregressor.predict(X_test.reshape(-1,1))
df2 = pd.DataFrame({'Real Values':y_test.reshape(-1), 'Predicted Values':y_pred.reshape(-1)})

# Visualising the Random Forest Regression Results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'red')
plt.scatter(X_test, y_pred, color = 'green')
plt.title('Random Forest Regression')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()
plt.plot(X_grid, RFregressor.predict(X_grid), color = 'black')
plt.title('Random Forest Regression')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()

'''
Evaluasi model Regresi
'''
print("\n--- Evaluasi Model Random Forest ---")
print("Mean Squared Error (MSE)       : ", mean_squared_error(y_test, y_pred))
print("R-Squared (R2 Score)           : ", r2_score(y_test, y_pred))
print("Mean Absolute Percentage Error : ", mean_absolute_percentage_error(y_test, y_pred))
print("-" * 64)

'''
Tugas Nomor 2: Simple Linear Regression Model
'''
# 3. Training the Simple Linear Regression model
SLRegressor = LinearRegression()

# Reshape karena hanya menggunakan satu fitur (Simple Regression)
SLRegressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

# 4. Predicting the Results
y_pred_slr = SLRegressor.predict(X_test.reshape(-1,1))

# 5. Visualising the Simple Linear Regression Results
plt.scatter(X_test, y_test, color = 'red', label='Actual Data')
plt.plot(X_train, SLRegressor.predict(X_train.reshape(-1,1)), color = 'blue', label='Regression Line')
plt.title('Simple Linear Regression (Test Set 30%)')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.legend()
plt.show()

'''
Evaluasi Model Simple Linear Regression
'''
print("\n--- Evaluasi Model Simple Linear Regression ---")
print("Mean Squared Error (MSE)         : ", mean_squared_error(y_test, y_pred_slr))
print("R-Squared (R2 Score)             : ", r2_score(y_test, y_pred_slr))
print("Mean Absolute Percentage Error   : ", mean_absolute_percentage_error(y_test, y_pred_slr))
print("-" * 48)


'''
Tugas Nomor 3: Support Vector Regression (SVR)
'''
# 1. Feature Scaling (Penting untuk SVR)
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train.reshape(-1, 1))
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1))
X_test_scaled = sc_X.transform(X_test.reshape(-1, 1))

# 2. Inisialisasi list untuk menyimpan hasil
svr_kernels = ['linear', 'poly', 'rbf']
svr_results = []

print("\n--- Evaluasi Model SVR Berdasarkan Kernel ---")
for kernel in svr_kernels:
    # Training model
    regressor = SVR(kernel = kernel)
    regressor.fit(X_train_scaled, y_train_scaled.ravel())
    
    # Prediksi dan kembalikan ke skala awal
    y_pred_scaled = regressor.predict(X_test_scaled)
    y_pred_svr = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    # Metrik Evaluasi
    mse = mean_squared_error(y_test, y_pred_svr)
    r2 = r2_score(y_test, y_pred_svr)
    mape = mean_absolute_percentage_error(y_test, y_pred_svr)
    
    print(f"Kernel: {kernel}")
    print(f"  Mean Squared Error (MSE)       : {mse:.4f}")
    print(f"  R-Squared (R2 Score)           : {r2:.4f}")
    print(f"  Mean Absolute Percentage Error : {mape:.4f}")
    print("-" * 30)