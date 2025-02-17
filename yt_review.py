import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset (Assuming a CSV file with 'review' and 'sentiment' columns)
data = pd.read_csv('youtube_ad_reviews.csv')

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)

# Apply preprocessing
data['cleaned_review'] = data['review'].apply(preprocess_text)

# Convert text to numerical representation using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_review'])
y = data['sentiment']  # Assuming 'sentiment' contains 0 (negative) & 1 (positive)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save Model & Vectorizer
import joblib
joblib.dump(model, 'youtube_ad_review_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully!")
