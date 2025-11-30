# Fintech Review Analytics

## Overview
This project analyzes user reviews for Ethiopian fintech and banking apps to extract:
- **Sentiment** (positive, negative, neutral) of user reviews
- **Common themes** and key features mentioned by users
- Early insights into user experience and app performance

The analysis is performed using NLP techniques (TF-IDF, lemmatization, stopword removal) and sentiment analysis with VADER, combined with clustering to identify recurring themes.

---

## Features
- **Preprocessing**: Lowercasing, tokenization, stopword removal, lemmatization
- **Sentiment Analysis**: VADER-based sentiment scoring and labeling
- **Thematic Analysis**: TF-IDF vectorization and KMeans clustering
- **Visualizations**: 
  - Sentiment distribution per bank
  - Top themes per bank
- **CSV Output**: Combined results with sentiment scores and identified themes

---

## Installation

1. **Clone the repository**
```
git clone https://github.com/yourusername/fintech-review-analytics.git
cd fintech-review-analytics
```
2. **Create a virtual environment**
```
python -m venv venv
```
3. **Activate the virtual environment**
```
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```
4. **Install dependencies**
```
pip install -r requirements.txt
```
**Usage**
Run the main script to preprocess reviews, compute sentiment, and cluster themes:
```
python src/sentiment_theme_analysis.py
```
The processed results are saved to:
```
data/processed/reviews_with_sentiment_themes.csv
```
**Project Structure**
```
fintech-review-analytics/
├─ data/
│  ├─ raw/                  # Raw review CSV files
│  └─ processed/            # Preprocessed and analyzed review data
├─ src/
│  └─ sentiment_theme_analysis.py
├─ visualizations/          # Generated charts and plots
├─ requirements.txt         # Python dependencies
└─ README.md
```
**Data Requirements**

The raw review data should be placed in 
```
data/raw/reviews_raw.csv
```
Expected columns include: review_id, review_text, rating, review_date, user_name, bank_code, bank_name, etc.