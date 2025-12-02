import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
df = pd.read_csv('./data/processed/reviews_with_sentiment_theme.csv')

# Use bank_name as key
banks = df['bank_name'].unique()

# Directory to save plots
import os
os.makedirs('./output/task4_plots', exist_ok=True)

# Task 4: Insights and Recommendations
for bank in banks:
    bank_reviews = df[df['bank_name'] == bank]
    
    # Average rating
    avg_rating = bank_reviews['rating'].mean()
    
    # Sentiment distribution
    sentiment_counts = bank_reviews['sentiment_label'].value_counts()
    
    # Top themes (drivers & pain points)
    top_themes = bank_reviews['themes'].value_counts().head(5)
    
    print(f"\nBank: {bank}")
    print(f"Average Rating: {avg_rating:.2f}")
    print("Sentiment Counts:\n", sentiment_counts)
    print("Top Themes (drivers/pain points):\n", top_themes)
    
    # Recommendations placeholder
    recommendations = []
    for theme in top_themes.index:
        if theme.lower() in ['user interface', 'feature requests', 'transaction performance']:
            recommendations.append(f"Improve {theme.lower()}")
    print("Suggested Improvements:", recommendations if recommendations else "No immediate suggestions")
    
    # Visualization 1: Rating Distribution
    plt.figure(figsize=(6,4))
    sns.histplot(bank_reviews['rating'], bins=5, kde=False)
    plt.title(f'{bank} - Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig(f'./output/task4_plots/{bank}_rating_dist.png')
    plt.close()
    
    # Visualization 2: Sentiment Counts
    plt.figure(figsize=(6,4))
    sns.countplot(x='sentiment_label', data=bank_reviews, order=sentiment_counts.index)
    plt.title(f'{bank} - Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig(f'./output/task4_plots/{bank}_sentiment.png')
    plt.close()
    
    # Visualization 3: Top Themes
    plt.figure(figsize=(6,4))
    sns.barplot(x=top_themes.values, y=top_themes.index)
    plt.title(f'{bank} - Top Themes')
    plt.xlabel('Count')
    plt.ylabel('Theme')
    plt.savefig(f'./output/task4_plots/{bank}_themes.png')
    plt.close()

print("\nAll plots saved in ./output/task4_plots/")
