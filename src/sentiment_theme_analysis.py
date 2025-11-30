"""
Task 2: Sentiment & Thematic Analysis (Enhanced with TF-IDF + Theme Clustering)
Description:
- Compute sentiment scores for reviews
- Extract keywords and cluster them into 3-5 themes per bank
- Save results as CSV
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure necessary NLTK data is downloaded
download('punkt')
download('stopwords')
download('wordnet')

# ----------------------------
# Configuration
# ----------------------------
RAW_REVIEWS_PATH = "data/raw/reviews_raw.csv"
PROCESSED_PATH = "data/processed/reviews_with_sentiment_themes.csv"
NUM_THEMES = 4  # Number of clusters per bank for thematic analysis

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv(RAW_REVIEWS_PATH)

# ----------------------------
# Preprocessing Functions
# ----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Lowercase, tokenize, remove stopwords, lemmatize"""
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok.isalpha() and tok not in stop_words]
    return " ".join(tokens)

print("Preprocessing reviews...")
tqdm.pandas()
df['processed_text'] = df['review_text'].progress_apply(preprocess_text)

# ----------------------------
# Sentiment Analysis
# ----------------------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """Compute sentiment score and label using VADER"""
    vs = analyzer.polarity_scores(text)
    score = vs['compound']
    if score >= 0.05:
        label = 'positive'
    elif score <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return pd.Series([score, label])

print("Computing sentiment scores...")
df[['sentiment_score', 'sentiment_label']] = df['processed_text'].progress_apply(get_sentiment)

# ----------------------------
# Thematic Analysis via TF-IDF + KMeans
# ----------------------------
def extract_themes(bank_df, num_themes=NUM_THEMES):
    """TF-IDF vectorization + KMeans clustering to discover themes"""
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2))
    X = tfidf.fit_transform(bank_df['processed_text'])

    # Cluster reviews into themes
    km = KMeans(n_clusters=num_themes, random_state=42)
    bank_df['theme_cluster'] = km.fit_predict(X)

    # Map cluster to top keywords
    clusters = {}
    for i in range(num_themes):
        cluster_center = km.cluster_centers_[i]
        top_indices = cluster_center.argsort()[::-1][:10]
        keywords = [tfidf.get_feature_names_out()[idx] for idx in top_indices]
        clusters[i] = ", ".join(keywords)
    
    # Map theme labels to reviews
    bank_df['identified_theme'] = bank_df['theme_cluster'].map(clusters)
    return bank_df

print("Clustering themes per bank...")
df_list = []
for bank_code, bank_df in df.groupby('bank_code'):
    clustered = extract_themes(bank_df)
    df_list.append(clustered)

df_final = pd.concat(df_list)

# ----------------------------
# Save Results
# ----------------------------
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
df_final.to_csv(PROCESSED_PATH, index=False)
print(f"Sentiment and thematic analysis saved to: {PROCESSED_PATH}")

# ----------------------------
# Quick Overview
# ----------------------------
print("\nSample Results:")
for bank_code, bank_df in df_final.groupby('bank_code'):
    print(f"\nBank: {bank_code}")
    print(bank_df[['review_text','sentiment_label','sentiment_score','identified_theme']].head(3).to_string(index=False))


# -------------------------------  
# VISUALIZATIONS FOR TASK 2  
# -------------------------------  

# Set seaborn style
sns.set(style="whitegrid")

# 1️⃣ Sentiment Distribution per Bank
plt.figure(figsize=(10,6))
sns.countplot(data=df_final, x='bank_name', hue='sentiment_label', palette='Set2')
plt.title("Sentiment Distribution per Bank")
plt.xlabel("Bank")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

# 2️⃣ Top Themes per Bank
for bank in df_final['bank_name'].unique():
    bank_df = df_final[df_final['bank_name'] == bank]
    
    if 'identified_theme' in bank_df.columns and not bank_df.empty:
        # Split the comma-separated themes and flatten
        themes_list = bank_df['identified_theme'].dropna().str.split(', ')
        flat_themes = [item for sublist in themes_list for item in sublist]
        
        if flat_themes:
            # Count theme occurrences
            theme_counts = Counter(flat_themes)
            top_themes = theme_counts.most_common(5)  # top 5 themes
            themes, counts = zip(*top_themes)
            
            plt.figure(figsize=(8,4))
            sns.barplot(x=list(counts), y=list(themes), palette='viridis')
            plt.title(f"Top 5 Themes for {bank}")
            plt.xlabel("Frequency")
            plt.ylabel("Theme")
            plt.tight_layout()
            plt.show()
        else:
            print(f"No themes found for {bank}")
    else:
        print(f"No 'identified_theme' column or empty dataframe for {bank}")
