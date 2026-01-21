import nltk
import pandas as pd
import json
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('all')
print("All NLTK data packages have been downloaded.")

# Load sentiment data from JSON files
with open('positive_data.json', 'r') as f:
    positive_data = json.load(f)

with open('negative_data.json', 'r') as f:
    negative_data = json.load(f)
    
data = positive_data + negative_data
random.shuffle(data)

print(f"Total records loaded: {len(data)}")

# Create DataFrame
df = pd.DataFrame(data)
print(df['label'].value_counts())

df.head()
# Preprocess text data
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
# Apply preprocessing
df['processed_text'] = df['sentence'].apply(preprocess_text)
print(df[['sentence', 'processed_text']].head())

# Vectorization
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df['processed_text'])
y = df['label'].map({'positive': 1, 'negative': 0}).values

print(f"Feature matrix shape: {x.shape}")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()
def get_vader_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'
  
  
# Test VADER sentiment analysis on sample sentences  
sample_sentences = [
    "I love this product! It works great and exceeds my expectations.",
    "This is the worst service I have ever received.",
    "The movie was okay, not too bad but not great either."
]

sample_text = "I love this product! It works great and exceeds my expectations."
print(get_vader_sentiment(sample_text))

#for sentence in sample_sentences:
#    sentiment = get_vader_sentiment(sentence)
#    print(f"Sentence: {sentence}\nVADER Sentiment: {sentiment}\n") 

# Apply VADER sentiment analysis to the entire dataset
df['vader_sentiment'] = df['sentence'].apply(get_vader_sentiment)
print(df[['sentence', 'vader_sentiment']].head())
print("VADER Sentiment Distribution:")
print(df['vader_sentiment'].value_counts())
