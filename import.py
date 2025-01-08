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

spam_data_model = spam_data.head(spam_data.shape[0] - 300)
spam_data_test = spam_data.tail(300)


not_spam_data_model = not_spam_data.head(not_spam_data.shape[0] - 300)
not_spam_data_test = not_spam_data.tail(300)




def preprocess_text_spam():
    corpus_spam = []
    
    for i in range(len(spam_data_model)):
        text = spam_data_model['text'].iloc[i].lower()
        text = text.translate(str.maketrans('', '' , string.punctuation)).split()
        text = [stemmer.stem(word) for word in text if word not in stopword_set]
        corpus_spam.append(text)

  
    #for i in range(len(df)):
        #text = df['text'].iloc[i].lower()
        #text = text.translate(str.maketrans('', '' , string.punctuation)).split()
        #text = [stemmer.stem(word) for word in text if word not in stopword_set]

        #tokens = word_tokenize(text.lower())
        #tokens = tokens.translate(str.maketrans('', '' , string.punctuation)).split()
        #tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        #text = ' '.join(text)
        #corpus.append(text)
    
    return corpus_spam
    
corpus_spam = preprocess_text_spam()



def preprocess_text_not_spam():
    corpus_not_spam = []
    
    for i in range(len(not_spam_data_model)):
        text = not_spam_data_model['text'].iloc[i].lower()
        text = text.translate(str.maketrans('', '' , string.punctuation)).split()
        text = [stemmer.stem(word) for word in text if word not in stopword_set]
        corpus_not_spam.append(text)
    
    return corpus_not_spam

corpus_not_spam = preprocess_text_not_spam()


dict_spam = {}
for i in corpus_spam:
    for j in i:
        dict_spam[j] = dict_spam.get(j, 0) + 1
    
dict_not_spam = {}
for i in corpus_not_spam:
    for j in i:
        dict_not_spam[j] = dict_not_spam.get(j, 0) + 1


dict_tf_spam = {}

for i in corpus_spam:
    for j in i:
        dict_tf_spam[j] = dict_tf_spam.get(j, 0) + 1 / len(corpus_spam)


dict_tf_not_spam = {}

for i in corpus_not_spam:
    for j in i:
        dict_tf_not_spam[j] = dict_tf_not_spam.get(j, 0) + 1 / len(corpus_not_spam)

print(dict_tf_not_spam)

sorted_dict_spam = dict(sorted(dict_tf_spam.items(), key=lambda item: item[1], reverse=True))

print("Words sorted by frequency in spam emails:")
for word, count in sorted_dict_spam.items():
    print(f"{word}: {count}")
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

