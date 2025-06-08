import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def create_dashboard_summary(df):
    """
    Membuat dashboard summary untuk presentasi.
    """
    print("\n📊 DASHBOARD SUMMARY")
    print("-" * 32)

    fig = plt.figure(figsize=(20, 12))

    # 1. Overall KPIs
    plt.subplot(3, 4, 1)
    kpis = [
        ('Total Reviews', f"{len(df):,}"),
        ('Avg Rating', f"{df['rating'].mean():.2f}"),
        ('Apps Analyzed', f"{df['app_name'].nunique()}"),
        ('Positive Sentiment', f"{(df['sentiment_label'] == 'Positive').mean():.1%}")
    ]

    y_pos = range(len(kpis))
    plt.barh(y_pos, [1]*len(kpis), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.yticks(y_pos, [kpi[0] for kpi in kpis])

    for i, (label, value) in enumerate(kpis):
        plt.text(0.5, i, value, ha='center', va='center', fontweight='bold', color='white')

    plt.title('Key Performance Indicators', fontweight='bold')
    plt.xlim(0, 1)

    # 2. Rating Distribution by App
    plt.subplot(3, 4, 2)
    app_ratings = df.groupby('app_name')['rating'].mean().sort_values(ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(app_ratings)))
    app_ratings.plot(kind='barh', color=colors)
    plt.title('Average Rating by App', fontweight='bold')
    plt.xlabel('Rating')

    # 3. Sentiment Analysis
    plt.subplot(3, 4, 3)
    sentiment_counts = df['sentiment_label'].value_counts()
    colors = ['#ff6b6b', '#feca57', '#48dbfb']
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Sentiment Distribution', fontweight='bold')

    # 4. Reviews Over Time (if available)
    plt.subplot(3, 4, 4)
    if 'review_date' in df.columns and df['review_date'].notna().any():
        df['review_date'] = pd.to_datetime(df['review_date'])
        monthly_counts = df.groupby(df['review_date'].dt.to_period('M')).size()
        monthly_counts.plot(kind='line', marker='o', color='purple')
        plt.title('Review Volume Over Time', fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No Date Data\nAvailable', ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Review Timeline', fontweight='bold')

    # 5. Top Issues Word Cloud
    plt.subplot(3, 4, 5)
    negative_reviews = df[df['sentiment_label'] == 'Negative']['review_text_clean']
    if len(negative_reviews) > 0:
        all_negative_text = ' '.join(negative_reviews.dropna())
        wordcloud = WordCloud(width=500, height=300, background_color='white').generate(all_negative_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Common Issues in Negative Reviews', fontweight='bold')

    # 6. Rating vs Sentiment Correlation
    plt.subplot(3, 4, 6)
    scatter = plt.scatter(df['rating'], df['sentiment_score'], alpha=0.5, c=df['rating'], cmap='viridis')
    plt.xlabel('Rating')
    plt.ylabel('Sentiment Score')
    plt.title('Rating vs Sentiment', fontweight='bold')
    plt.colorbar(scatter)

    # 7. Sector Performance
    plt.subplot(3, 4, 7)
    sector_performance = df.groupby('sector')['rating'].mean().sort_values(ascending=False)
    sector_performance.plot(kind='bar', color='lightcoral')
    plt.title('Performance by Sector', fontweight='bold')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)

    # 8. Review Length Distribution
    plt.subplot(3, 4, 8)
    plt.hist(df['review_length'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.axvline(df['review_length'].mean(), color='red', linestyle='--',
               label=f'Mean: {df["review_length"].mean():.0f}')
    plt.title('Review Length Distribution', fontweight='bold')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.legend()

    plt.suptitle('STARTUP REVIEWS ANALYSIS DASHBOARD', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('output/dashboard_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Dashboard created and saved as 'output/dashboard_summary.png'")