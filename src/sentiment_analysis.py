import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

def get_sentiment(text):
    """
    Menganalisis sentimen teks menggunakan TextBlob.
    """
    if pd.isna(text) or text == '':
        return 0, 'Neutral'

    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        return polarity, 'Positive'
    elif polarity < -0.1:
        return polarity, 'Negative'
    else:
        return polarity, 'Neutral'

def sentiment_analysis(df):
    """
    Melakukan analisis sentimen pada review startup.
    """
    print("\n🎭 SENTIMENT ANALYSIS")
    print("-" * 35)

    print("Analyzing sentiment...")
    df[['sentiment_score', 'sentiment_label']] = df['review_text_clean'].apply(
        lambda x: pd.Series(get_sentiment(x))
    )

    # Visualize sentiment analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Sentiment Distribution
    sentiment_counts = df['sentiment_label'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
    ax1.set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')

    # 2. Sentiment by App
    sentiment_app = pd.crosstab(df['app_name'], df['sentiment_label'], normalize='index') * 100
    sentiment_app.plot(kind='bar', stacked=True, ax=ax2, color=colors)
    ax2.set_title('Sentiment Distribution by App (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Application')
    ax2.set_ylabel('Percentage')
    ax2.legend(title='Sentiment')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Sentiment Score Distribution
    ax3.hist(df['sentiment_score'], bins=30, color='lightblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_title('Sentiment Score Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Sentiment Score')
    ax3.set_ylabel('Frequency')

    # 4. Sentiment by Sector
    sentiment_sector = pd.crosstab(df['sector'], df['sentiment_label'], normalize='index') * 100
    sentiment_sector.plot(kind='bar', ax=ax4, color=colors)
    ax4.set_title('Sentiment by Sector (%)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Sector')
    ax4.set_ylabel('Percentage')
    ax4.legend(title='Sentiment')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('output/sentiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print sentiment summary
    print("✓ Sentiment analysis completed")
    print(f"\nSentiment Summary:")
    print(df['sentiment_label'].value_counts())

    print(f"\nAverage Sentiment Score by App:")
    print(df.groupby('app_name')['sentiment_score'].mean().round(3))

    return df