import pandas as pd
import re

def clean_text(text):
    """
    Membersihkan teks review dengan menghapus karakter spesial dan mengubah ke lowercase.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus angka & karakter spesial
    text = ' '.join(text.split())  # Hapus spasi berlebih
    return text

def categorize_rating(rating):
    """
    Mengkategorikan rating menjadi sentiment.
    """
    if rating >= 4:
        return 'Positive'
    elif rating >= 3:
        return 'Neutral'
    else:
        return 'Negative'

def clean_and_preprocess(df):
    """
    Membersihkan dan memproses data startup.
    """
    print("\n🧹 CLEANING & PREPROCESSING DATA")
    print("-" * 40)

    # 1. Handle missing values
    df['review_text'] = df['review_text'].fillna('')
    df = df[df['review_text'] != ''].copy()
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating']).copy()
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df = df.dropna(subset=['review_date']).copy()

    # 2. Remove duplicates
    df = df.drop_duplicates().copy()

    # 3. Text preprocessing
    df['review_text_clean'] = df['review_text'].apply(clean_text)
    df['review_length'] = df['review_text'].str.len()
    df['word_count'] = df['review_text'].str.split().str.len()

    # 4. Feature engineering dari tanggal
    df['year'] = df['review_date'].dt.year
    df['month'] = df['review_date'].dt.month
    df['day_of_week'] = df['review_date'].dt.day_name()
    df['quarter'] = df['review_date'].dt.quarter

    # 5. Kategorisasi rating
    df['rating_category'] = df['rating'].apply(categorize_rating)

    print(f"✓ Data setelah cleaning: {df.shape}")
    print(f"✓ Text preprocessing completed")
    print(f"✓ Feature engineering completed")

    return df