import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Import Dataset
data_sms = pd.read_csv("dataset/spam.csv", delimiter=',', encoding='latin1')

# 2. Data Cleaning & Preprocessing
data_sms.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
data_sms = data_sms.drop_duplicates(keep='first')
data_sms.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)

encoder = LabelEncoder()
data_sms['target'] = encoder.fit_transform(data_sms['target'])

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join([ps.stem(i) for i in text])

data_sms['transformed_text'] = data_sms['text'].apply(transform_text)

# 3. Vectorization & Split Data (30% Test Size)
tfid = TfidfVectorizer(max_features=3000)
X = tfid.fit_transform(data_sms['transformed_text']).toarray()
y = data_sms['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# 4. Inisialisasi Model
models = {
    "Logistic Regression": LogisticRegression(),
    "k-NN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Naïve Bayes": GaussianNB()
}

# 5. Eksekusi & Pengukuran
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Output Console (Seperti di gambar)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    # Visualisasi Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f"CM: {name}")
    axes[i].set_xlabel("Predicted labels")
    axes[i].set_ylabel("True labels")
    axes[i].set_xticklabels(['ham', 'spam'])
    axes[i].set_yticklabels(['ham', 'spam'])

plt.tight_layout()
plt.show()