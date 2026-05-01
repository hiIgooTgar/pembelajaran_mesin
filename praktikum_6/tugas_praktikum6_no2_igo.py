import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 1. LOAD & MERGE DATA ---
dataset1 = pd.read_csv('dataset/500_berita_indonesia.csv', delimiter=';')
dataset2 = pd.read_csv('dataset/600_news_with_valid_hoax_label.csv', delimiter=';', encoding='ISO-8859-1')
df = pd.merge(dataset1, dataset2, on=['kategori', 'berita'], how='outer')
df['kategori'] = df['kategori'].str.lower()

# --- 2. PRE-PROCESSING SETUP ---
stopword = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

def preprocess_text(text, use_stemming=True):
    # Cleaning & Lowercase
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    text = re.sub(' +', ' ', text).strip()
    # Stopword Removal
    text = stopword.remove(text)
    # Stemming (Kondisional)
    if use_stemming:
        text = stemmer.stem(text)
    return text

# --- 3. EKSPERIMEN ---
skenario = [True, False] # True = Dengan Stemming, False = Tanpa Stemming
hasil_evaluasi = []

for status_stem in skenario:
    label_skenario = "DENGAN Stemming" if status_stem else "TANPA Stemming"
    print(f"\n{'='*20} MEMPROSES: {label_skenario} {'='*20}")
    
    # Apply Preprocessing
    df_temp = df.copy()
    df_temp['berita_bersih'] = df_temp['berita'].apply(lambda x: preprocess_text(x, use_stemming=status_stem))
    
    # Split Data (20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        df_temp['berita_bersih'], df_temp['kategori'], test_size=0.2, stratify=df_temp['kategori'], random_state=42
    )
    
    # TF-IDF
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Model 1: Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train_tfidf, y_train)
    y_pred_lr = lr.predict(X_test_tfidf)
    
    # Model 2: K-NN (K=3)
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(X_train_tfidf, y_train)
    y_pred_knn = knn.predict(X_test_tfidf)
    
    # Simpan Hasil untuk Analisis
    hasil_evaluasi.append({
        'Skenario': label_skenario,
        'LR': (y_test, y_pred_lr, accuracy_score(y_test, y_pred_lr)),
        'KNN': (y_test, y_pred_knn, accuracy_score(y_test, y_pred_knn))
    })

# --- 4. TAMPILKAN PERBANDINGAN ---
for h in hasil_evaluasi:
    print(f"\n--- HASIL {h['Skenario']} ---")
    print(f"LR Accuracy : {h['LR'][2]:.4f}")
    print(f"KNN Accuracy: {h['KNN'][2]:.4f}")
    # Cetak detail salah satu model sebagai contoh laporan
    print(f"\nDetail LR ({h['Skenario']}):\n", classification_report(h['LR'][0], h['LR'][1]))