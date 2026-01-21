import nltk
import re
import pandas as pd
import json
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download necessary NLTK data packages

def download_nltk_data():
    needed = [
        'punkt',
        'stopwords',
        'wordnet',
        'vader_lexicon',
        'omw-1.4'
    ]

    for package in needed:
        nltk.download(package)
print("All NLTK data packages have been downloaded.")


#Social Media Sentiment Analysis Script

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
REPEAT_RE = re.compile(r"(.)\1{2,}", re.DOTALL)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Preprocess data
def preprocess_text(text):
    text = text.lower()
    # Remove URLs, mentions, hashtags
    text = URL_RE.sub(' <URL> ', text)
    text = MENTION_RE.sub(' <MENTION> ', text)
    
    # Remove hashtags but keep the text
    text = REPEAT_RE.sub(r"\1\1", text)
    
    # Keep hashtags as words (strip # but keep the token)
    hastags = HASHTAG_RE.findall(text)
    for h in hashtags:
        text = text.replace(h, "hashtag_" + h[1:])  # Remove '#' but keep the word
    
    text = re.sub (r"[^a-z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Load dataset
def load_dataset(pos_path="positive_data.json", neg_path="negative_data.json") -> pd.DataFrame:
    with open(pos_path, "r", encoding="utf-8") as f:
        pos = json.load(f)
    with open(neg_path, "r", encoding="utf-8") as f:
        neg = json.load(f)
        
    data = pos + neg
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    if "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'sentence' and 'label' fields.")

    return df


# TRAINING AND EVALUATION
def train_and_evaluate(df: pd.DataFrame):
    df["processed_text"] = df["sentence"].astype(str).apply(preprocess_text)
    y = df["label"].map({"positive": 1, "negative": 0}).values
    
    #TF-IDF Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5)
    x = vectorizer.fit_transform(df["processed_text"])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    return model, vectorizer

#Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()
def predict_sentiment_ml(text: str, model, vectorizer):
    processed = preprocess_text(text)
    X = vectorizer.transform([processed])
    
    #predict prob
    
    proba = model.predict_proba(X)[0]
    pred = proba.argmax()
    confidence = float(proba[pred])
    
    label = 'positive' if pred == 1 else 'negative'
    return label, confidence

# Test VADER sentiment analysis on a sample text
sample_text = "I love this product! It works great and exceeds my expectations."
print(get_vader_sentiment(sample_text)) 

# Apply VADER sentiment analysis to the entire dataset
df['vader_sentiment'] = df['sentence'].apply(get_vader_sentiment)
print(df[['sentence', 'vader_sentiment']].head())
print("VADER Sentiment Distribution:")
print(df['vader_sentiment'].value_counts())


# Evaluate VADER sentiment analysis accuracy
df['vader_sentiment'] = df['sentence'].apply(get_vader_sentiment)

vader_labels = df['vader_sentiment'].map({'positive': 1, 'negative': 0, 'neutral': -1})
valid_indices = vader_labels != -1

print("VADER Sentiment Analysis Accuracy (excluding neutral):")
print(accuracy_score(y[valid_indices], vader_labels[valid_indices]))

def predict_sentiment(text):
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)
    return 'positive' if prediction[0] == 1 else 'negative'

test_sentences = [
    "I absolutely love this! Best purchase ever.",
    "This is terrible, I want my money back.",
    "It's okay, nothing special."
]
for text in test_sentences:
    print(f"Text: {text} => Predicted Sentiment: {predict_sentiment(text)}")
    
    
# Save the trained model and vectorizer
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
