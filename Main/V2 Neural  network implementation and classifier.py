import openpyxl
import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import torch
from torch import nn
from torch import optim
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Downloads and initializes NLTK resources
stemmer = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')
stopword_set = set(stopwords.words('english'))

# Load dataset
data_file_path = "Main\\Behandlet_enron_data.xlsx"
df = pd.read_excel(data_file_path)

# Preprocess emails
def preprocess_text(df):
    processed_emails = []
    for i in range(len(df)):
        text = df['text'].iloc[i].lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = text.split()
        text = [stemmer.stem(word) for word in text if word not in stopword_set]
        processed_emails.append(text)
    return processed_emails

processed_emails = preprocess_text(df)

# Load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove_file_path = "Main\\glove.6b.50d.txt"
glove_embeddings = load_glove_embeddings(glove_file_path)

# Get the word vector for a single word
def get_word_vector(word, embeddings, embedding_dim=50):
    return embeddings.get(word, np.zeros(embedding_dim))

# Map each email to its aggregated vector
def aggregate_email_vectors(processed_emails, embeddings, embedding_dim=50):
    email_vectors = []
    for email in processed_emails:
        word_vectors = [get_word_vector(word, embeddings, embedding_dim) for word in email]
        email_vector = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(embedding_dim)
        email_vectors.append(email_vector)
    return np.array(email_vectors)

email_vectors = aggregate_email_vectors(processed_emails, glove_embeddings)

# Output the vector for the first email
print("First email vector:", email_vectors[0])
print("Shape of email vectors:", email_vectors.shape)


## build neural network
labels = df['label_num'].values
print(labels)  


X_train, X_test, y_train, y_test = train_test_split(email_vectors, labels, test_size=0.2, random_state=42)

## Cross entropy loss function gives a value between 0 and 1 which is perfect for spam or not spam. It is also a classification loss function which is perfect for discrete values.

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # Long for classification
y_test = torch.tensor(y_test, dtype=torch.long)

# optimizer = optim.Adam(???, lr=0.01)

class EmailClassifier(nn.Module):
    def __init__(self):
        super(EmailClassifier,self).__init__()
        self.fc1 = nn.Linear(50, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 16)
        self.bn5 = nn.BatchNorm1d(16)
        self.fc6 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(0.5)
        
        
        
        
        
        

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        x = self.leaky_relu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        return x
            
model = EmailClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)



# Training parameters
num_epochs = 200
batch_size = 64

# Dataloader for batching
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(batch_X)  # Forward pass
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")



# Evaluation mode
model.eval()

with torch.no_grad():
    outputs = model(X_test)  # Forward pass
    predictions = torch.argmax(outputs, axis=1)  # Get predicted class
    accuracy = (predictions == y_test).float().mean()  # Calculate accuracy

print(f"Test Accuracy: {accuracy * 100:.2f}%")


torch.save(model, 'email_classifier_model.pth')




