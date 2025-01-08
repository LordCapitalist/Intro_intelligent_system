import openpyxl
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

nltk.download('punkt_tab')
nltk.download('stopwords')
stopword_set = set(stopwords.words('english'))

#stop_word_set = list(stopwords.words('english'))
#print(stopword_set)

def Import():
    file_path = ".\\Behandlet_enron_data.xlsx"
    df = pd.read_excel(file_path)
    #print(df)
    
    #print("Spam data:")
    #print(spam_data)
        
    #print("Not spam data:")
    #print(not_spam_data)
    return df

df = Import()
spam_data = df[df['label_num'] == 1]  
not_spam_data = df[df['label_num'] == 0]


def preprocess_text():
    corpus_spam = []
    corpus_not_spam = []
    for i in range(len(spam_data)):
        text = spam_data['text'].iloc[i].lower()
        text = text.translate(str.maketrans('', '' , string.punctuation)).split()
        text = [stemmer.stem(word) for word in text if word not in stopword_set]
        text = ' '.join(text)
        corpus_spam.append(text)
    
    for i in range(len(not_spam_data)):
        text = not_spam_data['text'].iloc[i].lower()
        text = text.translate(str.maketrans('', '' , string.punctuation)).split()
        text = [stemmer.stem(word) for word in text if word not in stopword_set]
        text = ' '.join(text)
        corpus_not_spam.append(text)
    
    #for i in range(len(df)):
        #text = df['text'].iloc[i].lower()
        #text = text.translate(str.maketrans('', '' , string.punctuation)).split()
        #text = [stemmer.stem(word) for word in text if word not in stopword_set]

        #tokens = word_tokenize(text.lower())
        #tokens = tokens.translate(str.maketrans('', '' , string.punctuation)).split()
        #tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        #text = ' '.join(text)
        #corpus.append(text)
    
    print(corpus_spam , corpus_not_spam)
    
preprocess_text()

#df['tokens'] = df['text'].apply(preprocess_text)
#spam_keywords = ["free", "win", "prize", "limited", "offer", "exclusive", "urgent", "money"]

#def recognize_spam(tokens):
    #spam_count = sum(1 for token in tokens if token in spam_keywords)
    #if spam_count > 0:
        #return "Spam"
    #else:
        #return "Not Spam"

#data['spam_recognition'] = data['tokens'].apply(recognize_spam)

#print("\nData with Spam Recognition:")
#print(data[['text', 'spam_recognition']])

