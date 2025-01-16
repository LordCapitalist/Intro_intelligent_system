import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras import layers, models


# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize necessary NLTK tools
stemmer = PorterStemmer()
stopword_set = set(stopwords.words('english'))

# Preprocess the data
def preprocess_text(corpus):
    processed_corpus = []
    for text in corpus:
        text = text.lower()  # Convert to lower case
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = [stemmer.stem(word) for word in text.split() if word not in stopword_set]  # Stem and remove stopwords
        processed_corpus.append(" ".join(text))  # Join words back into a string
    return processed_corpus

# Import data (adjust the file path as needed)
def import_data():
    file_path = "./Behandlet_enron_data.xlsx"
    df = pd.read_excel(file_path)
    return df

# Split data into spam and non-spam
df = import_data()
spam_data = df[df['label_num'] == 1]
not_spam_data = df[df['label_num'] == 0]

# Preprocess the text data
corpus_spam = preprocess_text(spam_data['text'].tolist())
corpus_not_spam = preprocess_text(not_spam_data['text'].tolist())

# Combine both spam and non-spam corpora for vectorization
corpus = corpus_spam + corpus_not_spam

# Vectorize the text data using TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()  # Convert sparse matrix to dense array

# Create target labels for spam (1) and not spam (0)
y = [1] * len(corpus_spam) + [0] * len(corpus_not_spam)

# Split data into train/test sets (80% train, 20% test)
X_train = X[:int(0.8 * len(corpus))]
X_test = X[int(0.8 * len(corpus)):]

y_train = y[:int(0.8 * len(corpus))]
y_test = y[int(0.8 * len(corpus)):]

# Make sure data is in numpy array and of correct type (float32)
import numpy as np
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)

# Reshape the data if needed (for flat arrays)
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

# Check input shape
print(X_train.shape)
print(X_test.shape)

# Define and compile the neural network model
model = models.Sequential([
    layers.Dense(128, input_dim=X_train.shape[1], activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output for classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

