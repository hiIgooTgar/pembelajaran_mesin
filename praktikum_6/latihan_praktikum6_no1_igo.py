# importing the libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# load the breast cancer dataset
cancer_data=pd.read_csv('dataset/breast_cancer_wisconsin.csv')
# dataset information
print(cancer_data.info())

# check missing values
cancer_data.isnull().sum()
# check for any duplicated rows
cancer_data.duplicated().sum()

# create 'x' as an independent variables
columns_to_drop = ["id", "Unnamed: 32"]
for column in columns_to_drop:
    cancer_data.drop(column, axis=1, inplace=True)

X=cancer_data.drop("diagnosis", axis=1)
# create 'y' as target/dependent variable
y=cancer_data['diagnosis']

# create graphic for class distribution
# Mengambil data kelas target
class_counts = cancer_data['diagnosis'].value_counts()
# create plot
plot = class_counts.plot(kind='bar', title="Class distributions : Benign | Malignant")

# Add text label to plot
for i, count in enumerate(class_counts): plt.text(i, count, str(count), ha='center', va='bottom')

plt.show()

# split the train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Logistic Regression
clf = LogisticRegression(max_iter = 10000)
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# accuracy evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# classification report
print(classification_report(y_test,y_pred))

# confusion_matrix
confusion_matrix(y_test, y_pred)