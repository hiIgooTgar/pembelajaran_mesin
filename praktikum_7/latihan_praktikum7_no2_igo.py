# importing the libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud #importing WordCloud for text visualization
import nltk #importing NLTK for natural language processing
from nltk.corpus import stopwords
nltk.download('stopwords') # Downloading stopwords data
nltk.download('punkt') # Downloading tokenizer data
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import dataset
data_sms = pd.read_csv("dataset/spam.csv", delimiter=',', encoding='latin1')

# checking for null values
data_sms.isna().sum()
   
# check for any duplicated rows
data_sms.duplicated().sum()

# displays duplicate rows
duplicate_rows = data_sms[data_sms.duplicated()]

# drop columns
data_sms.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
data_sms.info()

# removes duplicate rows
data_sms = data_sms.drop_duplicates(keep='first')

# rename the columns name
data_sms.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)

# convert the target variable
encoder = LabelEncoder()
data_sms['target'] = encoder.fit_transform(data_sms['target'])

# komposisi value variabel target
data_sms['target'].value_counts()

# visualisasi data
plt.pie(data_sms['target'].value_counts(),labels=['ham', 'spam'], autopct="%0.2f%%")
plt.show()

'''
Data Pre-processing
'''
# creating an instance of the Porter Stemmer
ps = PorterStemmer()

# text preprocessing function
def transform_text(text):
# Transform the text to lowercase
    text = text.lower()
# tokenization using NLTK
    text = nltk.word_tokenize(text)

# removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
# removing stop words and punctuation
    text = y[:]
    y.clear()

# loop through the tokens and remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

# stemming using Porter Stemmer
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        # join the processed tokens back into a single string
    return " ".join(y)

# creating a New Column: 'transformed_text'
data_sms['transformed_text'] = data_sms['text'].apply(transform_text)

# word cloud for spam messages
wc = WordCloud(width = 500, height = 500, min_font_size = 10, background_color = 'white')
spam_wc = wc.generate(data_sms[data_sms['target'] == 1]['transformed_text'].str.cat(sep = " "))
plt.figure(figsize = (15,6))
plt.imshow(spam_wc)
plt.show()

# word cloud for not spam messages
ham_wc = wc.generate(data_sms[data_sms['target'] == 0]['transformed_text'].str.cat(sep = " "))
plt.figure(figsize = (15,6))
plt.imshow(ham_wc)
plt.show()

# find top 30 words of spam
spam_carpos = []
for sentence in data_sms[data_sms['target'] == 1]['transformed_text'].tolist():
    for word in sentence.split():
        spam_carpos.append(word)
        from collections import Counter
filter_df = pd.DataFrame(Counter(spam_carpos).most_common(30))
sns.barplot(data = filter_df, x = filter_df[0], y = filter_df[1], palette = 'bright')
plt.xticks(rotation = 90)
plt.show()

'''
Model Building
'''
cv = CountVectorizer()
tfid = TfidfVectorizer()

# create independent variables
X = tfid.fit_transform(data_sms['transformed_text']).toarray()

# create target/dependent variable
y = data_sms['target'].values

# split the train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# initialize the models
classifier = DecisionTreeClassifier(max_depth = 5, criterion = 'gini')

# train the Models
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
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['Predicted ham', 'Predicted spam'], yticklabels=['Actual ham', 'Actual spam'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()