import pandas as pd
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

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
    file_path = "./Behandlet_enron_data.xlsx"  # Adjust this file path as needed
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
y = np.array([1] * len(corpus_spam) + [0] * len(corpus_not_spam))

# Split data into train/test sets (80% train, 20% test)
X_train = X[:int(0.8 * len(corpus))]
X_test = X[int(0.8 * len(corpus)):]

y_train = y[:int(0.8 * len(corpus))]
y_test = y[int(0.8 * len(corpus)):]

# Initialize weights (for a simple model)
input_dim = X_train.shape[1]  # Number of features (input dimensions)
output_dim = 1  # Binary output (0 or 1)
hidden_units = 64  # Number of hidden units

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_units) * 0.01  # Weights for the first layer (input_dim x hidden_units)
b1 = np.zeros((1, hidden_units))  # Bias for the first layer

W2 = np.random.randn(hidden_units, output_dim) * 0.01  # Weights for the second layer (hidden_units x output_dim)
b2 = np.zeros((1, output_dim))  # Bias for the second layer

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Loss function (binary cross-entropy) and its derivative
def binary_crossentropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_derivative(y_true, y_pred):
    return y_pred - y_true

# Training the model
epochs = 10
learning_rate = 0.01

for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(X_train, W1) + b1  # Linear transformation
    A1 = sigmoid(Z1)  # Activation function

    Z2 = np.dot(A1, W2) + b2  # Linear transformation for output
    A2 = sigmoid(Z2)  # Sigmoid activation for binary classification

    # Print shapes of variables during forward pass
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Z1 shape: {Z1.shape}, A1 shape: {A1.shape}")
    print(f"Z2 shape: {Z2.shape}, A2 shape: {A2.shape}")

    # Compute loss
    loss = binary_crossentropy(y_train, A2)

    # Backward pass (gradient descent)
    dA2 = binary_crossentropy_derivative(y_train, A2)  # Derivation of loss wrt output
    dZ2 = dA2 * sigmoid_derivative(A2)  # Gradient at the second layer

    # Print shapes during backward pass
    print(f"dZ2 shape: {dZ2.shape}")
    
    # Update dW2 computation
    dW2 = np.dot(A1.T, dZ2)  # (64, 1)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Gradient for the first layer
    dA1 = np.dot(dZ2, W2.T)  # (4136, 64)
    dZ1 = dA1 * sigmoid_derivative(A1)  # Gradient at the first layer

    dW1 = np.dot(X_train.T, dZ1)  # (64, 64)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Print the shapes of weight gradients
    print(f"dW2 shape: {dW2.shape}, db2 shape: {db2.shape}")
    print(f"dW1 shape: {dW1.shape}, db1 shape: {db1.shape}")

    # Update weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 1 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

# Evaluate the model
def predict(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2

# Predictions on test set
y_pred = predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)  # Convert probabilities to binary class

# Calculate accuracy
accuracy = np.mean(y_pred_class == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')







