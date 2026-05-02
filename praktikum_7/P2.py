# importing the libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud 
from collections import Counter # Dipindahkan ke atas agar lebih rapi

import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- BAGIAN PERBAIKAN ---
# Mengunduh data yang diperlukan NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') # PERBAIKAN: Wajib diunduh untuk versi NLTK terbaru
# -------------------------

# import dataset
# Pastikan path file benar sesuai dengan lokasi di komputer Anda
data_sms = pd.read_csv("dataset/spam.csv", delimiter=',', encoding='latin1')

# drop columns
data_sms.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# removes duplicate rows
data_sms = data_sms.drop_duplicates(keep='first')

# rename the columns name
data_sms.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)

# convert the target variable
encoder = LabelEncoder()
data_sms['target'] = encoder.fit_transform(data_sms['target'])

'''
Data Pre-processing
'''
ps = PorterStemmer()

def transform_text(text):
    # Transform the text to lowercase
    text = text.lower()
    # tokenization
    text = nltk.word_tokenize(text)

    # removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # removing stop words and punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# membuat kolom baru 'transformed_text'
data_sms['transformed_text'] = data_sms['text'].apply(transform_text)

# Visualisasi Top 30 Words of Spam
spam_carpos = []
for sentence in data_sms[data_sms['target'] == 1]['transformed_text'].tolist():
    for word in sentence.split():
        spam_carpos.append(word)

# Menggunakan Counter untuk menghitung frekuensi kata
filter_df = pd.DataFrame(Counter(spam_carpos).most_common(30))

plt.figure(figsize=(10, 6))
sns.barplot(data = filter_df, x = filter_df[0], y = filter_df[1], palette = 'bright')
plt.xticks(rotation = 90)
plt.title("Top 30 Words in Spam Messages")
plt.show()

'''
Model Building
'''
tfid = TfidfVectorizer()

# create independent variables
X = tfid.fit_transform(data_sms['transformed_text']).toarray()
# create target variable
y = data_sms['target'].values

# split the train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# initialize the models
classifier = DecisionTreeClassifier(max_depth = 5, criterion = 'gini')

# train the Models
classifier.fit(X_train, y_train)

# prediction
y_pred = classifier.predict(X_test)

# Evaluasi
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))