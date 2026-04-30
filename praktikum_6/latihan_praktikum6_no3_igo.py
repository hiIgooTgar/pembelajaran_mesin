# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from pandas import ExcelWriter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import dataset
dataset1 = pd.read_csv('dataset/500_berita_indonesia.csv', delimiter=';')
dataset2 = pd.read_csv('dataset/600_news_with_valid_hoax_label.csv', delimiter=';', encoding='ISO-8859-1')

# merge dataset
df = pd.merge(dataset1, dataset2, on=['kategori', 'berita'], how='outer')
# ubah value kolom kategori menjadi huruf kecil
df['kategori'] = df['kategori'].str.lower()

'''
Create graphic for class distribution
'''
# Mengambil data kelas target
class_counts = df['kategori'].value_counts()
# create plot
plot = class_counts.plot(kind='bar', title="Class distributions : Valid | Hoax")
# Add text label to plot
for i, count in enumerate(class_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.show()

'''
Tahap pra-pemrosesan teks:
1. Punctuation removal
2. Lowercase operation
3. Stopwords removal
4. Stemming
'''
# import Regular expression
import re
# import StopWordRemoverFactory class
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
# stemmer Creation
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# create an empty list of corpus
corpus = []
# converting Dataframe into List
df_toList = df.values.tolist()

for (kategori, berita) in df_toList:
    # 1. Tahap Punctuation removal
    kalimat = re.sub('[^a-zA-Z]', ' ', berita)

    # 2. Tahap Lowercase operation
    kalimat = kalimat.lower()

    # 3. Tahap Stopwords removal
    kalimat = stopword.remove(kalimat)
    jumlah_kata_awal = len(kalimat.split())
    kondisi = True

    while(kondisi):
        kalimat = re.sub(' +', ' ', kalimat)
        kalimat = stopword.remove(kalimat)

        jumlah_kata_baru = len(kalimat.split())
        if(jumlah_kata_awal == jumlah_kata_baru):
            kondisi = False
        else:
            jumlah_kata_awal = jumlah_kata_baru
    # 4. Tahap Stemming operation
    kalimat = stemmer.stem(kalimat)
    corpus.append((kategori, kalimat))

'''
Export data pada variabel corpus ke Excel
'''
dataset = DataFrame(corpus, columns=['kategori', 'berita'])
writer = ExcelWriter('text_preprocessing_result.xlsx')
dataset.to_excel(writer, sheet_name='data_teks', index=False)
writer.close()

'''
Gunakan kode python bagian ini, jika:
dataset hasil pra-pemrosesan teks diambil dari Excel
'''
corpus = pd.read_excel('text_preprocessing_result.xlsx', sheet_name='data_teks')

'''
Machine Learning
'''
# konversi List ke Dataframe
list_to_df = pd.DataFrame(corpus)
# menambahkan nama kolom pada dataframe
# eksekusi kode ini jika dataset tidak diambil dari Excel
list_to_df.columns = ['kategori', 'berita']

# create independent variable
X = list_to_df['berita']

# create dependen/target variable
y = list_to_df['kategori']

# split the train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# feature Extraction: Tf-IDF
vectorization = TfidfVectorizer()
x_train_tfidf = vectorization.fit_transform(X_train)
x_test_tf_idf = vectorization.transform(X_test)

# k-nearest neighbors
classifier = KNeighborsClassifier(n_neighbors=3, weights='distance')
classifier.fit(x_train_tfidf, y_train)

# prediction
y_pred = classifier.predict(x_test_tf_idf)

'''
Evaluasi Machine Learning
'''
# accuracy evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# classification report
print(classification_report(y_test, y_pred))

# confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    cmap='Blues',
    fmt='g',
    xticklabels=['Predicted 0', 'Predicted 1'],
    yticklabels=['Actual 0', 'Actual 1']
)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()