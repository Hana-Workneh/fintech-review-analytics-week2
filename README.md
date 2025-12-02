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


**Database Schema**

Database: bank_reviews
Tables:

banks
Column	Type	Description
bank_id	SERIAL	Unique bank identifier
bank_name	TEXT	Name of the bank
app_name	TEXT	Name of the mobile app
reviews
Column	Type	Description
review_id	UUID	Unique identifier for the review
review_text	TEXT	Text of the user review
rating	INT	Rating given by the user (1–5)
review_date	DATE	Date of the review
user_name	TEXT	Name of the reviewer
thumbs_up	INT	Number of thumbs up / likes
reply_content	TEXT	Reply from the bank, if any
bank_code	TEXT	Bank code / short identifier
bank_name	TEXT	Name of the bank
app_id	TEXT	App identifier
source	TEXT	Source of review (Google Play / App Store)
processed_text	TEXT	Cleaned / preprocessed review text
sentiment_label	TEXT	Sentiment category (positive, neutral, negative)
sentiment_score	FLOAT	Sentiment score (0–1)
themes	TEXT	Identified themes / topics of the review
Task 3 – Data Ingestion

Goal: Insert processed review data into PostgreSQL.

Script: src/insert_to_postgres.py

CSV input: ./data/processed/reviews_with_sentiment_theme.csv

Actions:

Connect to PostgreSQL.

Read CSV data.

Insert data into the reviews table.

Result: reviews table now contains ~1,966 rows.

Sample Query:

SELECT bank_id, COUNT(*) FROM reviews GROUP BY bank_id;

**Task 4 – Insights & Recommendations**

Goal: Analyze reviews and derive actionable insights for each bank app.

Script: src/task4_insights.py

Analysis includes:

Average rating per bank

Sentiment distribution (positive/neutral/negative)

Top themes (drivers and pain points)


Visualizations: Saved in ./output/task4_plots/
Rating distribution per bank

Sentiment counts per bank

Top review themes

Example Output (CBE Mobile):

```
Bank: Commercial Bank of Ethiopia
Average Rating: 4.12
Sentiment Counts: positive: 235, neutral: 133, negative: 32
Top Themes: User Interface, Feature Requests, Transaction Performance
Suggested Improvements: Improve user interface, Improve feature requests, Improve transaction performance

```

