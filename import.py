import openpyxl
import string
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



stemmer = PorterStemmer()

nltk.download('punkt_tab')
nltk.download('stopwords')
stopword_set = set(stopwords.words('english'))


def Import():
    file_path = ".\\Behandlet_enron_data.xlsx"
    df = pd.read_excel(file_path)
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



def check_tf(word):
    if word in dict_tf_spam:
        tf_spam = dict_tf_spam[word]
    else:
        tf_spam = 0
        
    if word in dict_tf_not_spam:
        tf_not_spam = dict_tf_not_spam[word]
    else:
        tf_not_spam = 0
        
    if tf_spam > tf_not_spam:
        return tf_spam
    else:
        return -tf_not_spam

    


def check_mail(mail):
    mail = mail.lower()
    mail = mail.translate(str.maketrans('', '' , string.punctuation)).split()
    mail = [stemmer.stem(word) for word in mail if word not in stopword_set]
    sum = 0
    for i in mail:
        sum += check_tf(i)
    
    if sum > 0:
        return 1
    
    if sum < 0:
        return 0



 
def test():
    wrong = 0
    right = 0
    for i in range(len(spam_data_test)):
        text = spam_data_test['text'].iloc[i]
        mail = text

        if check_mail(mail) == 1:
            right += 1
        else:
            wrong += 1

    print(right/len(spam_data_test) *100 , "percentage of spam mails are classified correctly") 




test()
