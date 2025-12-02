import psycopg2
import pandas as pd
import uuid 

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="bank_reviews",
    user="postgres",
    password="Wor234wor,"  
)
cur = conn.cursor()

# Read CSV
df = pd.read_csv("data/processed/reviews_with_sentiment_themes.csv")

# Optional: map banks to bank_ids
cur.execute("SELECT bank_id, bank_name FROM banks")
bank_mapping = {name: bid for bid, name in cur.fetchall()}

for _, row in df.iterrows():
    bank_name = row['bank_code']  # adjust if your CSV column is different
    if bank_name not in bank_mapping:
        print(f"Skipping unknown bank: {bank_name}")
        continue
    
    bank_id = bank_mapping[bank_name]
    review_id = str(uuid.uuid4())  # generate unique review_id
    
    cur.execute("""
        INSERT INTO reviews (
            review_id, bank_id, review_text, rating, review_date,
            sentiment_label, sentiment_score, source
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        review_id,
        bank_id,
        row['review_text'],
        row['rating'],
        row['review_date'],
        row['sentiment_label'],
        row['sentiment_score'],
        row['source']
    ))

# Commit and close
conn.commit()
cur.close()
conn.close()

print("Data inserted successfully!")
