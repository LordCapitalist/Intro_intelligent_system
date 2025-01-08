import openpyxl
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))
print(stop_words)

def Import():
    file_path = ".\\Behandlet_enron_data.xlsx"
    data = pd.read_excel(file_path)
    print(data)
    spam_data = data[data['label_num'] == 1]  
    not_spam_data = data[data['label_num'] == 0]
 
    print("Spam data:")
    print(spam_data)
        
    print("Not spam data:")
    print(not_spam_data)
    return data

data = Import()


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens

data['tokens'] = data['text'].apply(preprocess_text)
spam_keywords = ["free", "win", "prize", "limited", "offer", "exclusive", "urgent", "money"]

def recognize_spam(tokens):
    spam_count = sum(1 for token in tokens if token in spam_keywords)
    if spam_count > 0:
        return "Spam"
    else:
        return "Not Spam"

data['spam_recognition'] = data['tokens'].apply(recognize_spam)

print("\nData with Spam Recognition:")
print(data[['text', 'spam_recognition']])

