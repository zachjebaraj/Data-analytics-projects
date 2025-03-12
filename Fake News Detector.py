import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import string

# Step 1: Load the Dataset
df = pd.read_csv("/Users/zach/Documents/news.csv")  # Ensure dataset includes 'text' and 'label' columns

# Step 2: Data Cleaning
def clean_text(text):
    text = text.lower()                           # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)          # Remove text in brackets
    text = re.sub(r'http\S+', '', text)           # Remove URLs
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub(r'\n', '', text)                # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)        # Remove words containing numbers
    return text

df['text'] = df['text'].apply(clean_text)

# Step 3: Feature Engineering
vectorizer = TfidfVectorizer(max_df=0.7, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Step 4: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# Step 6: Predict Function
def predict_news(news_text):
    cleaned_text = clean_text(news_text)
    transformed_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_text)
    return "Fake News" if prediction[0] == 1 else "Real News"

# Example Usage
example_text = "Breaking news! Scientists discover AI that writes perfect articles."
print("Prediction:", predict_news(example_text))
