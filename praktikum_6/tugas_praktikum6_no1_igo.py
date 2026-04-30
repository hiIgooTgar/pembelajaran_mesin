import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pandas import DataFrame, ExcelWriter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 1. IMPORT & MERGE DATASET ---
dataset1 = pd.read_csv('dataset/500_berita_indonesia.csv', delimiter=';')
dataset2 = pd.read_csv('dataset/600_news_with_valid_hoax_label.csv', delimiter=';', encoding='ISO-8859-1')

df = pd.merge(dataset1, dataset2, on=['kategori', 'berita'], how='outer')
df['kategori'] = df['kategori'].str.lower()

# --- 2. PRE-PROCESSING SETUP ---
factory_sw = StopWordRemoverFactory()
stopword = factory_sw.create_stop_word_remover()
factory_st = StemmerFactory()
stemmer = factory_st.create_stemmer()

corpus = []
df_toList = df.values.tolist()


for (kategori, berita) in df_toList:
    berita = str(berita)
    # Punctuation removal & Lowercase
    kalimat = re.sub('[^a-zA-Z]', ' ', berita).lower()
    # Stopwords removal (Disederhanakan tanpa loop while agar tidak macet)
    kalimat = re.sub(' +', ' ', kalimat).strip()
    kalimat = stopword.remove(kalimat)
    # Stemming
    kalimat = stemmer.stem(kalimat)
    corpus.append((kategori, kalimat))

# Konversi hasil ke DataFrame
list_to_df = pd.DataFrame(corpus, columns=['kategori', 'berita'])
list_to_df.dropna(subset=['berita'], inplace=True)

'''

Tugas No. 1 - Skenario Penggunaan Data Uji 
(0.2 untuk 20% / 0.3 untuk 30% / 0.4 untuk 40%)

'''
# ukuran_tes = 0.2  -> Data Uji 20%
# ukuran_tes = 0.3 -> Data Uji 30%
ukuran_tes = 0.4

# --- 3. PREPARASI DATA ---
X = list_to_df['berita']
y = list_to_df['kategori']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=ukuran_tes, stratify=y, random_state=42
)

# Feature Extraction: Tf-IDF
vectorization = TfidfVectorizer()
x_train_tfidf = vectorization.fit_transform(X_train)
x_test_tf_idf = vectorization.transform(X_test)

# --- 4. PEMODELAN ---
# A. Model Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(x_train_tfidf, y_train)
y_pred_lr = model_lr.predict(x_test_tf_idf)

# B. Model k-Nearest Neighbors
model_knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
model_knn.fit(x_train_tfidf, y_train)
y_pred_knn = model_knn.predict(x_test_tf_idf)

# --- 5. OUTPUT EVALUASI ---
print(f"\n{'='*15} HASIL DATA UJI {int(ukuran_tes*100)}% {'='*15}")
print("\n[1] PERFORMA LOGISTIC REGRESSION")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr))

print("\n[2] PERFORMA k-NEAREST NEIGHBORS")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(classification_report(y_test, y_pred_knn))

# Visualisasi Confusion Matrix (Logistic Regression)
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=model_lr.classes_, yticklabels=model_lr.classes_)
plt.title(f'Confusion Matrix LR (Data Uji {int(ukuran_tes*100)}%)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()