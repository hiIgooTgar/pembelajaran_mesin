# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

# import dataset
univ_data = pd.read_csv("dataset/university_data.csv", delimiter=';')
# checking for null values
univ_data.isna().sum()

# check for any duplicated rows
univ_data.duplicated().sum()

# distribusi data variabel 'Target'
univ_data['Target'].value_counts()

# visualisasi distribusi data variabel 'Target'
# Dropout vs Graduate vs Enrolled
categories_counts = univ_data['Target'].value_counts()
plt.pie(categories_counts.values, labels=categories_counts.index, autopct='%1.2f%%')

plt.title('Educational Outcomes: Distribution of Graduates (2), Enrolled Students(1), and Dropouts (0).')
plt.show()

# transformasi value variabel 'Target ke dalam format numerik
univ_data['Target'] = univ_data['Target'].map({'Dropout' : 0, 'Enrolled': 1, 'Graduate': 2})
print(univ_data['Target'].unique())

# korelasi antar variabel dengan Pearson untuk linear relationships
pearson_corr = univ_data.corr(method='pearson')['Target'].sort_values(ascending=False)
print("Pearson's Rank Correlation with 'Target':")
print(pearson_corr)

# korelasi antar variabel dengan Spearman untuk non-linear relationships
spearman_corr = univ_data.corr(method='spearman')['Target'].sort_values(ascending=False)
print("Spearman's Rank Correlation with 'Target':")
print(spearman_corr)

# fase optimasi dataset dengan mengeliminasi variabel yang tidak relevan
# berdasarkan analisis pearson & spearman
# eliminasi nilai korelasi -0.05 s0.05 dengan variabel 'Target'
univ_data = univ_data.drop(columns=['Curricular units 1st sem (grade)',
'Tuition fees up to date', 'Scholarship holder',
'Curricular units 2nd sem (enrolled)', 'Curricular units 1st sem (enrolled)',
'Admission grade', 'Displaced', 'Previous qualification (grade)',
'Curricular units 2nd sem (evaluations)', 'Application order',
'Age at enrollment', 'Debtor', 'Gender', 'Nacionality', 'Course',
'Curricular units 2nd sem (without evaluations)', 'GDP',
'Application mode', 'Curricular units 1st sem (without evaluations)',
'Curricular units 2nd sem (credited)', 'International',
'Curricular units 1st sem (evaluations)', 'Inflation rate',
'Educational special needs', 'Marital status', 'Previous qualification',
'Mother\'s qualification', 'Mother\'s occupation', 'Father\'s occupation',
'Father\'s qualification','Curricular units 1st sem (credited)',
'Unemployment rate','Daytime/evening attendance\t'])

# create independent variables
X=univ_data.drop("Target", axis=1)

# create target/dependent variable
y=univ_data['Target']

# split the train and test data
kf = KFold(n_splits = 10, shuffle = True, random_state=43)
for tr_idx, te_idx in kf.split(X):
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

# model ML dengan original data (imbalanced class)
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# prediction
y_pred = classifier.predict(X_test)

'''
Evaluasi Machine Learning (imbalanced class)
'''
# accuracy evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# classification report
print(classification_report(y_test,y_pred))

# confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
xticklabels=['Predicted Dropout', 'Predicted Enrolled', 'Predicted Graduate'],
yticklabels=['Actual Dropout', 'Actual Enrolled', 'Actual Graduate'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

'''
implementasi SMOTE (Synthetic Minority Over-sampling Technique)
untuk menangani imbalanced class
'''
# transformasi dataset dengan SMOTE
oversample = SMOTE()
X_2, y_2 = oversample.fit_resample(X, y)

# summarize distribution
counter = Counter(y_2)

# plot the distribution
plt.figure(figsize=(5, 3))
plt.bar(counter.keys(), counter.values())

# menambahkan label angka di atas grafik
for x, y in counter.items():
    plt.text(x, y + 0.5, str(y), ha='center')

# menetapkan label pada sumbu x
plt.xticks([0, 1, 2])
plt.show()

# split the train and test data on balanced data
kf2 = KFold(n_splits = 10, shuffle = True, random_state=43)
for tr_idx, te_idx in kf2.split(X_2):
    X_train_2, X_test_2 = X_2.iloc[tr_idx], X_2.iloc[te_idx]
    y_train_2, y_test_2 = y_2.iloc[tr_idx], y_2.iloc[te_idx]

# model ML dengan modifikasi data (balanced class)
classifier2 = GaussianNB()
classifier2.fit(X_train_2, y_train_2)

# prediction
y_pred_2 = classifier2.predict(X_test_2)

'''
Evaluasi Machine Learning (balanced class)
'''
# accuracy evaluation
accuracy = accuracy_score(y_test_2, y_pred_2)
print(f'Accuracy: {accuracy:.4f}')

# classification report
print(classification_report(y_test_2,y_pred_2))

# confusion_matrix
cm2 = confusion_matrix(y_test_2, y_pred_2)
plt.figure(figsize=(8, 6))
sns.heatmap(cm2, annot=True, cmap='Blues', fmt='g',
xticklabels=['Predicted Dropout', 'Predicted Enrolled', 'Predicted Graduate'],
yticklabels=['Actual Dropout', 'Actual Enrolled', 'Actual Graduate'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()