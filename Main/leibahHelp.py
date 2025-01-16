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
import openpyxl

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

# Load the dataset
df = import_data()
# print(df.head())

# Separate the dataset into spam and non-spam based on the 'label_num' column
spam_data = df[df['label_num'] == 1]
not_spam_data = df[df['label_num'] == 0]

# Preprocess text for spam and non-spam messages
corpus_spam = preprocess_text(spam_data['text'].tolist())
corpus_not_spam = preprocess_text(not_spam_data['text'].tolist())
# print(corpus_spam)
# print(corpus_not_spam)

# Combine spam and non-spam preprocessed text into a single corpus
corpus = corpus_spam + corpus_not_spam

# Initialize TF-IDF vectorizer and transform the corpus
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()  # Convert text data into numerical format

# Create the labels for the spam (1) and non-spam (0) classes
y = np.array([1] * len(corpus_spam) + [0] * len(corpus_not_spam))

# Split data into training and testing sets (80% train, 20% test)
X_train = X[:int(0.8 * len(corpus))]
X_test = X[int(0.8 * len(corpus)):]
y_train = y[:int(0.8 * len(corpus))]
y_test = y[int(0.8 * len(corpus)):]

# Train the neural network
epochs = 25  # Number of training epochs
learning_rate = 0.01  # Learning rate for gradient descent
    
# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]  # Number of features from TF-IDF
model = ZModule.ZModule(input_dim)  # Initialize model with input dimension
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode

    # Forward pass
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for each epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Predict on test data
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    y_pred_class = (y_pred_test > 0.5).float()  # Convert probabilities to binary predictions

torch.save(model.state_dict(), "model.pth")
print("Saved model")

# Compute and display accuracy
accuracy = (y_pred_class == y_test_tensor).float().mean().item()
print(f"Test Accuracy: {accuracy * 100:.2f}%")
