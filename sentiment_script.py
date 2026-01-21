import nltk
import pandas as pd
import json
import random

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('all')
print("All NLTK data packages have been downloaded.")

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)

with open('sentiment_data.json', 'r') as f:
    positive_data = json.load(f)

with open('negative_sentiment_data.json', 'r') as f:
    negative_data = json.load(f)
    
data = positive_data + negative_data
random.shuffle(data)

print(f"Total records loaded: {len(data)}")