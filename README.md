
---

# 📊 Social Media Sentiment Analysis (NLP)

A Python-based **sentiment analysis system** designed for **real-world social media text**.
It combines a **machine learning model (Naive Bayes + TF-IDF)** with **VADER sentiment analysis** to handle hashtags, mentions, informal language, and noisy input.

This project focuses on **practical NLP**, not just clean textbook data.

---

## 🚀 Features

* Social media–aware preprocessing

  * Hashtags (`#blessed`)
  * Mentions (`@brand`)
  * URL normalization
  * Repeated character handling (`soooo → soo`)
* Dual sentiment approach:

  * **ML classifier** trained on labeled data
  * **VADER** (rule-based, optimized for social text)
* Confidence scores for predictions
* Simple ensemble logic to combine ML + VADER
* Save/load trained models with `joblib`
* Easily extensible for analytics tools or APIs

---

## 🧠 How It Works

1. **Text preprocessing**
   Normalizes social media–specific patterns while preserving sentiment signals.

2. **Vectorization**
   Uses **TF-IDF (unigrams + bigrams)** to capture context like *“not good”*.

3. **Model training**
   Multinomial Naive Bayes with stratified train/test split.

4. **Prediction**

   * ML model predicts sentiment + confidence
   * VADER provides rule-based sentiment
   * Final label chosen via simple decision logic

---

## 🛠️ Tech Stack

* Python 3.10+
* NLTK
* scikit-learn
* pandas
* joblib

---

## 📂 Project Structure

```
Sentiment_Analyse/
│
├── sentiment_script.py        # Main training & analysis script
├── positive_data.json         # Positive samples
├── negative_data.json         # Negative samples
├── sentiment_model.joblib     # Saved model (generated)
├── vectorizer.joblib          # Saved vectorizer (generated)
├── README.md
```

---

## ▶️ Getting Started

### Install dependencies

```bash
pip install nltk scikit-learn pandas joblib
```

### Run the project

```bash
python sentiment_script.py
```

The script will:

* Download required NLTK resources (first run only)
* Train and evaluate the model
* Analyze example social media posts

---

## 🧪 Example Output

```json
{
  "text": "I absolutely love this!!! 🔥🔥 #blessed",
  "ml": {
    "label": "positive",
    "confidence": 0.93
  },
  "vader": {
    "label": "positive",
    "compound": 0.89
  },
  "final_label": "positive",
  "note": "Both methods agree."
}

---

## 👤 Author

**Dima**
---