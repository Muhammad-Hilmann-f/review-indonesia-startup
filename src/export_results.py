import pandas as pd
import json

def export_results(df, insights, models):
    """
    Export hasil analisis ke berbagai format.
    """
    print("\n💾 EXPORTING RESULTS")
    print("-" * 31)

    try:
        # 1. Export processed dataset
        df.to_csv('output/processed_startup_reviews.csv', index=False)
        print("✅ Processed dataset exported to 'output/processed_startup_reviews.csv'")

        # 2. Export summary statistics
        summary_stats = {
            'total_reviews': len(df),
            'average_rating': df['rating'].mean(),
            'apps_analyzed': df['app_name'].nunique(),
            'sectors_analyzed': df['sector'].nunique(),
            'positive_sentiment_ratio': (df['sentiment_label'] == 'Positive').mean(),
            'negative_sentiment_ratio': (df['sentiment_label'] == 'Negative').mean(),
            'neutral_sentiment_ratio': (df['sentiment_label'] == 'Neutral').mean()
        }

        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv('output/analysis_summary.csv', index=False)
        print("✅ Summary statistics exported to 'output/analysis_summary.csv'")

        # 3. Export app performance metrics
        insights['app_metrics'].to_csv('output/app_performance_metrics.csv')
        print("✅ App performance metrics exported to 'output/app_performance_metrics.csv'")

        # 4. Export sector analysis
        insights['sector_metrics'].to_csv('output/sector_analysis.csv')
        print("✅ Sector analysis exported to 'output/sector_analysis.csv'")

        # 5. Export business recommendations as JSON
        with open('output/business_recommendations.json', 'w') as f:
            json.dump(insights['recommendations'], f, indent=4)
        print("✅ Business recommendations exported to 'output/business_recommendations.json'")

        # 6. Export trained models (dummy saving for now)
        model_data = {
            'sentiment_model': str(models['sentiment_model']),
            'rating_model': str(models['rating_model'])
        }

        with open('output/model_results.json', 'w') as f:
            json.dump(model_data, f, indent=4)

        print("✅ Model results exported to 'output/model_results.json'")

        print("\n📁 All results exported successfully!")

    except Exception as e:
        print(f"❌ Error exporting results: {str(e)}")