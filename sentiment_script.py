import nltk
import re
import pandas as pd
import json
import random
import joblib

from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------- Download necessary NLTK data packages ----------
def download_nltk_data():
    needed = ["punkt", "stopwords", "wordnet", "vader_lexicon", "omw-1.4"]
    for package in needed:
        nltk.download(package)
    print("All required NLTK data packages are ready.")


# ---------- Social Media regex ----------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
REPEAT_RE = re.compile(r"(.)\1{2,}", re.DOTALL)  # sooo -> soo


# We'll initialize these AFTER downloading NLTK data
stop_words = None
lemmatizer = None


# ---------- Preprocess data ----------
def preprocess_text(text: str) -> str:
    global stop_words, lemmatizer

    text = str(text).lower()

    # Replace URLs and mentions with tokens
    text = URL_RE.sub(" <URL> ", text)
    text = MENTION_RE.sub(" <MENTION> ", text)

    # Reduce long repeated characters
    text = REPEAT_RE.sub(r"\1\1", text)

    # Convert hashtags into tokens (keep the word)
    hashtags = HASHTAG_RE.findall(text)
    for h in hashtags:
        text = text.replace(h, " hashtag_" + h[1:])

    # Keep letters/numbers/whitespace + special token chars < > _
    text = re.sub(r"[^a-z0-9_<>\s]", " ", text)

    tokens = word_tokenize(text)

    # Remove stopwords but keep special tokens
    cleaned = []
    for t in tokens:
        if t in {"<url>", "<mention>"}:
            cleaned.append(t)
        elif t not in stop_words:
            cleaned.append(lemmatizer.lemmatize(t))

    return " ".join(cleaned)


# ---------- Load dataset ----------
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


# ---------- Training and evaluation ----------
def train_and_evaluate(df: pd.DataFrame):
    df["processed_text"] = df["sentence"].astype(str).apply(preprocess_text)
    y = df["label"].map({"positive": 1, "negative": 0}).values

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(df["processed_text"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    return model, vectorizer


# ---------- VADER + ML prediction ----------
sia = SentimentIntensityAnalyzer()

def predict_sentiment_ml(text: str, model, vectorizer):
    processed = preprocess_text(text)
    X = vectorizer.transform([processed])

    proba = model.predict_proba(X)[0]
    pred = proba.argmax()
    confidence = float(proba[pred])

    label = "positive" if pred == 1 else "negative"
    return label, confidence

def predict_sentiment_vader(text: str):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive", float(compound)
    elif compound <= -0.05:
        return "negative", float(compound)
    else:
        return "neutral", float(compound)

def analyze_social_post(text: str, model, vectorizer):
    ml_label, ml_conf = predict_sentiment_ml(text, model, vectorizer)
    vader_label, vader_score = predict_sentiment_vader(text)

    if vader_label == "neutral":
        final = ml_label
        reason = "VADER neutral → using ML result."
    elif vader_label == ml_label:
        final = ml_label
        reason = "Both methods agree."
    else:
        final = ml_label
        reason = "Methods disagree → using ML result."

    return {
        "text": text,
        "ml": {"label": ml_label, "confidence": round(ml_conf, 3)},
        "vader": {"label": vader_label, "compound": round(vader_score, 3)},
        "final_label": final,
        "note": reason,
    }


# ---------- Save/Load ----------
def save_artifacts(model, vectorizer, model_path="sentiment_model.joblib", vectorizer_path="vectorizer.joblib"):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)


if __name__ == "__main__":
    download_nltk_data()

    # init globals AFTER NLTK data is present
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    df = load_dataset()
    model, vectorizer = train_and_evaluate(df)
    save_artifacts(model, vectorizer)

    examples = [
        "I absolutely love this!!! 🔥🔥 #blessed",
        "Worst service ever. never again. @company",
        "It’s okay tbh… nothing crazy",
    ]

    for t in examples:
        result = analyze_social_post(t, model, vectorizer)
        print(json.dumps(result, indent=2))
