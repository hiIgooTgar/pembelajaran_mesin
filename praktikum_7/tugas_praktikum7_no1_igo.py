import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
df = pd.read_csv('dataset/university_data.csv', delimiter=';')

# 2. Preprocessing
# Mengubah target menjadi numerik: Dropout(0), Enrolled(1), Graduate(2)
df['Target'] = df['Target'].map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})
df.columns = df.columns.str.strip()

# 3. Feature Selection (Menghapus variabel tidak relevan sesuai modul)
cols_to_drop = [
    'Curricular units 1st sem (grade)', 'Tuition fees up to date', 'Scholarship holder',
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
    'Father\'s qualification', 'Curricular units 1st sem (credited)',
    'Unemployment rate'
]

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

# Memisahkan fitur dan target
X = df.drop('Target', axis=1)
y = df['Target']

# 4. Split Data (Data Uji 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

# 5. Inisialisasi Model
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "k-NN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=43),
    "Naïve Bayes": GaussianNB()
}

# 6. Evaluasi dan Plotting Confusion Matrix
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    # Training
    model.fit(X_train, y_train)
    # Prediction
    y_pred = model.predict(X_test)
    
    # Print Evaluation Report
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Dropout', 'Enrolled', 'Graduate']))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=axes[i],
                xticklabels=['Dropout', 'Enrolled', 'Graduate'],
                yticklabels=['Dropout', 'Enrolled', 'Graduate'])
    axes[i].set_title(f'Confusion Matrix: {name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()