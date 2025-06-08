import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_insights_and_recommendations(df):
    """
    Menghasilkan insights dan rekomendasi bisnis berdasarkan review startup.
    """
    print("\n💡 BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("-" * 52)

    # 1. App Performance Analysis
    print("📊 APP PERFORMANCE ANALYSIS")
    app_metrics = df.groupby('app_name').agg({
        'rating': ['count', 'mean', 'std'],
        'sentiment_score': 'mean',
        'review_length': 'mean'
    }).round(3)

    app_metrics.columns = ['review_count', 'avg_rating', 'rating_std', 'avg_sentiment', 'avg_review_length']
    app_metrics = app_metrics.reset_index()

    # Calculate performance score
    app_metrics['performance_score'] = (
        app_metrics['avg_rating'] * 0.4 +
        (app_metrics['avg_sentiment'] + 1) * 2.5 * 0.3 +  
        (app_metrics['review_count'] / app_metrics['review_count'].max()) * 5 * 0.3
    )

    app_metrics = app_metrics.sort_values('performance_score', ascending=False)

    print("🏆 Top Performing Apps:")
    print(app_metrics.head())

    # 2. Sector Analysis
    print("\n📈 SECTOR PERFORMANCE")
    sector_metrics = df.groupby('sector').agg({
        'rating': 'mean',
        'sentiment_score': 'mean',
        'app_name': 'nunique'
    }).round(3)

    sector_metrics.columns = ['avg_rating', 'avg_sentiment', 'num_apps']
    sector_metrics = sector_metrics.sort_values('avg_rating', ascending=False)

    print("🌟 Sector Performance:")
    print(sector_metrics)

    # 3. Common Issues Analysis
    print("\n⚠️ COMMON ISSUES IN NEGATIVE REVIEWS")
    negative_reviews = df[df['sentiment_label'] == 'Negative']['review_text_clean']

    if len(negative_reviews) > 0:
        all_negative_text = ' '.join(negative_reviews.dropna())

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_negative_text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Common Issues in Negative Reviews', fontsize=14, fontweight='bold')
        plt.savefig('output/common_issues_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 4. Business Recommendations
    recommendations = [
        "🎯 Fokus pada aplikasi dengan rating rendah untuk perbaikan layanan.",
        "⭐ Pelajari strategi aplikasi dengan rating tinggi untuk meningkatkan performa.",
        "⚠️ Perhatikan sentimen negatif yang tinggi, segera tingkatkan customer support.",
        "📈 Monitor tren sentimen dan rating untuk mengidentifikasi pola perubahan."
    ]

    print("\n✅ BUSINESS RECOMMENDATIONS")
    for rec in recommendations:
        print(f"   - {rec}")

    return {
        'app_metrics': app_metrics,
        'sector_metrics': sector_metrics,
        'recommendations': recommendations
    }