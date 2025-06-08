import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def customer_segmentation(df):
    """
    Melakukan segmentasi pelanggan berdasarkan perilaku review.
    """
    print("\n👥 CUSTOMER SEGMENTATION")
    print("-" * 38)

    # Aggregate data by reviewer (if available)
    if 'reviewer_name' in df.columns:
        user_features = df.groupby('reviewer_name').agg({
            'rating': ['count', 'mean'],
            'review_length': 'mean',
            'sentiment_score': 'mean'
        }).round(2)

        user_features.columns = ['review_frequency', 'avg_rating', 'avg_review_length', 'avg_sentiment']
        user_features = user_features.reset_index()
    else:
        # Alternative: segment by app performance
        user_features = df.groupby('app_name').agg({
            'rating': ['count', 'mean'],
            'review_length': 'mean',
            'sentiment_score': 'mean'
        }).round(2)

        user_features.columns = ['review_count', 'avg_rating', 'avg_review_length', 'avg_sentiment']
        user_features = user_features.reset_index()

    # Prepare features for clustering
    feature_cols = ['avg_rating', 'avg_review_length', 'avg_sentiment']
    X = user_features[feature_cols].fillna(0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means Clustering
    optimal_k = 3  # Bisa diubah sesuai Elbow Method
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    user_features['cluster'] = kmeans.fit_predict(X_scaled)

    # Visualize clustering
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Clustering visualization
    scatter = ax1.scatter(user_features['avg_rating'], user_features['avg_sentiment'],
                        c=user_features['cluster'], cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Average Rating')
    ax1.set_ylabel('Average Sentiment')
    ax1.set_title('Customer Segments', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax1)

    # 2. Cluster characteristics
    cluster_summary = user_features.groupby('cluster')[feature_cols].mean()
    cluster_summary.plot(kind='bar', ax=ax2)
    ax2.set_title('Cluster Characteristics', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cluster')
    ax2.tick_params(axis='x', rotation=0)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. Cluster distribution
    cluster_counts = user_features['cluster'].value_counts().sort_index()
    ax3.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index],
            autopct='%1.1f%%')
    ax3.set_title('Cluster Distribution', fontsize=14, fontweight='bold')

    # 4. Feature importance in clustering
    feature_importance = np.abs(kmeans.cluster_centers_).mean(axis=0)
    ax4.bar(feature_cols, feature_importance, color='lightcoral')
    ax4.set_title('Feature Importance in Clustering', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Importance')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('output/customer_segmentation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Interpret clusters
    print("✓ Customer segmentation completed")
    print(f"\nCluster Summary:")
    for i in range(optimal_k):
        cluster_data = user_features[user_features['cluster'] == i]
        print(f"\nCluster {i} ({len(cluster_data)} members):")
        print(f"  - Avg Rating: {cluster_data['avg_rating'].mean():.2f}")
        print(f"  - Avg Sentiment: {cluster_data['avg_sentiment'].mean():.3f}")
        print(f"  - Avg Review Length: {cluster_data['avg_review_length'].mean():.0f}")

    return user_features