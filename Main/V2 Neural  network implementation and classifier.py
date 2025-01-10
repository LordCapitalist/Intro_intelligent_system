##import all neccessary datasets and functions
import openpyxl
import pandas as pd
import numpy as np 
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.porter import PorterStemmer
import re


## downloads stemmers and stopwords from nltk
stemmer = PorterStemmer()

nltk.download('punkt_tab')
nltk.download('stopwords')
stopword_set = set(stopwords.words('english'))

## get the email dataset
data_file_path = "Main\\Behandlet_enron_data.xlsx"
df = pd.read_excel(data_file_path)


## get the vectors for individual words from the glove file
glove_file_path ="Main\\Pre_trained_word_vectors\\glove.6b.50d.txt"


## applying the Glove vectorizer to make every word into a 50 dimensional vector so the neural network can use it.
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r' , encoding='utf-8') as file:
        for i, line in enumerate(file):
            try:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector

            except ValueError:
                embeddings[word] = np.zeros(50, dtype='float32')  # Assign zero vector for all error lines in the glove file
                continue

    return embeddings

glove_embeddings = load_glove_embeddings(glove_file_path)


## Stemming sentences to remove . , ^ () [] and other punctuations.
def preprocess_email(email):
    # Lowercase and remove non-alphanumeric characters
    email = email.lower()
    email = re.sub(r'[^a-z0-9\s]', '', email)
    return email.split()

email_text = ""
tokens = preprocess_email(email_text)

## get the word vector for each word in the email
def get_word_vector(word, embeddings, embedding_dim=50):
    return embeddings.get(word, np.zeros(embedding_dim))

## Map each token to its GloVe vector
word_vectors = [get_word_vector(token, glove_embeddings) for token in tokens]


## Placing the vectors into 1 vector so the whole sentence works inside the neural network
def aggregate_vectors(vectors):
    if len(vectors) == 0:
        return np.zeros(50)  # Return zero vector if no valid words
    return np.mean(vectors, axis=0)

email_vector = aggregate_vectors(word_vectors)

print(email_vector)

## build neural network




## Cross entropy loss function gives a value between 0 and 1 which is perfect for spam or not spam. It is also a classification loss function which is perfect for discrete values.

from torch import nn
from torch import optim

optimizer = optim.Adam(???, lr=0.01)

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(50, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 20)
        self.fc5 = nn.Linear(20, 1)

        self.relu = nn.ReLU()

    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
model = Network()

# i loop, kig på trainloader og testloader, kør gennem nogle epochs
# kig også på torch.save / torch.load
y_true = ...
y_pred = model(x)

optimizer.zero_grad()

loss = nn.MSELoss(y_true, y_pred) # måske byt om på de to?

loss.backward()

optimizer.step()