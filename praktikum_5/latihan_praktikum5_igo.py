# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#import dataset
dataset=pd.read_csv('dataset/IceCreamData.csv')
X = dataset['Temperature'].values
y = dataset['Revenue'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

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
