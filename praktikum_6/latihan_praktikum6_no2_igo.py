# importing the libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load the Iris flower dataset
iris=pd.read_csv("dataset/Iris.csv")

# checking for null values
iris.isna().sum()
# check for any duplicated rows
iris.duplicated().sum()

# jumlah member tiap class: Iris-Setosa, Iris-Versicolour, Iris-Virginica
count_setosa = len(iris[iris['Species'] == 'Iris-setosa'])
count_versicolour = len(iris[iris['Species'] == 'Iris-versicolor'])
count_virginica = len(iris[iris['Species'] == 'Iris-virginica'])
print(f'Jumlah Setosa di dataset: {count_setosa}')
print(f'Jumlah Versicolour di dataset: {count_versicolour}')
print(f'Jumlah Virginica di dataset: {count_virginica}')

# data visualization 1
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris)
plt.show()

# data visualization 2
data = iris.drop(['Species','Id'], axis=1)
sns.heatmap(data.corr(method='pearson'), annot=True)
plt.show()

# dependent variables
y = iris['Species']

# split the train and test dataset
data_train, data_test, y_train, y_test = train_test_split(data, y, test_size = 0.3)

# using Logistic Regression
model = LogisticRegression(max_iter=5000)
model.fit(data_train, y_train)
prediction = model.predict(data_test)

# accuracy metric
print('Accuracy:', accuracy_score(y_test, prediction))

# classification report
print(classification_report(y_test, prediction))

# confusion_matrix
cm = confusion_matrix(y_test, prediction)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
xticklabels=['Predicted 0', 'Predicted 1', 'Predicted 2'],
yticklabels=['Actual 0', 'Actual 1', 'Actual 2'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()