import pandas as pd  # For handling and manipulating data
import string  # For handling punctuation removal
import numpy as np  # For numerical computations
import nltk  # Natural language processing toolkit
from nltk.corpus import stopwords  # Stopwords for text preprocessing
from nltk.stem.porter import PorterStemmer  # Stemming functionality
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF vectorization of text
import torch
import torch.nn as nn
import torch.optim as optim
import ZModule

# Download necessary NLTK resources
nltk.download('punkt')  # Tokenizer
nltk.download('stopwords')  # Stopword list

# Initialize the Porter Stemmer and load English stopwords
stemmer = PorterStemmer()
stopword_set = set(stopwords.words('english'))

# Function to preprocess text data
def preprocess_text(corpus):
    processed_corpus = []
    for text in corpus:
        text = text.lower()  # Convert text to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        # Perform stemming and remove stopwords
        text = [stemmer.stem(word) for word in text.split() if word not in stopword_set]
        processed_corpus.append(" ".join(text))  # Combine words back into a single string
    return processed_corpus


# Function to import data from an Excel file
def import_data():
    file_path = ".\\Behandlet_enron_data.xlsx"
    df = pd.read_excel(file_path)
    return df

df = import_data()

spam_data = df[df['label_num'] == 1]
not_spam_data = df[df['label_num'] == 0]

corpus_spam = preprocess_text(spam_data['text'].tolist())
corpus_not_spam = preprocess_text(not_spam_data['text'].tolist())

corpus = corpus_spam + corpus_not_spam

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()



# Create the labels for the spam (1) and non-spam (0) classes
y = np.array([1] * len(corpus_spam) + [0] * len(corpus_not_spam))

# x = ["blah", "blah", "blah", "blah", "blah"]

# spam = 3, not_spam = 4
# y = [1, 1, 1, 0, 0, 0, 0]

# y = a*x+b
# Complex: y = relu(x) + sigmoid(a*x) + b


# Split data into training and testing sets (80% train, 20% test) # Prevent overfitting (no remember)
X_train = X[:int(0.8 * len(corpus))]
X_test = X[int(0.8 * len(corpus)):]
y_train = y[:int(0.8 * len(corpus))]
y_test = y[int(0.8 * len(corpus)):]



# Load model
input_dim = X_train.shape[1]
model = ZModule.ZModule(input_dim)
model.load_state_dict(torch.load("model.pth"))

X_test_tensor = torch.tensor(X_test, dtype=torch.float32) # X_test is in pandas format. Convert to Numpy
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

import random

# Function to test a random sample from the test dataset
def test_random_sample(model, X_test_tensor, y_test_tensor):
    # Set the model to evaluation mode (Test mode)
    model.eval()
    
    # Select a random index from the test dataset
    random_index = random.randint(0, X_test_tensor.shape[0] - 1)
    
    # Get the corresponding sample and its true label
    sample = X_test_tensor[random_index].unsqueeze(0)  # Add batch dimension
    true_label = y_test_tensor[random_index].item()
    
    # Pass the sample through the model to get the prediction
    with torch.no_grad(): # No train
        prediction = model(sample)
        predicted_label = torch.round(torch.sigmoid(prediction)).item()  # Convert to binary [[1]] => 1
    
    # Display the results
    print(f"Random Sample Index: {random_index}")
    print(f"True Label: {int(true_label)}")
    print(f"Predicted Label: {int(predicted_label)}")
    print(f"Model Output (raw): {prediction.item()}")

# Test a random sample
test_random_sample(model, X_test_tensor, y_test_tensor)
