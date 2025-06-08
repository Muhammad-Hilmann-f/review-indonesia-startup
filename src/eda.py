import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def exploratory_data_analysis(df):
    """
    Melakukan Exploratory Data Analysis (EDA) untuk melihat pola data.
    """
    print("\n📊 EXPLORATORY DATA ANALYSIS")
    print("-" * 42)

    # Setup plot style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Distribusi Rating per App
    plt.subplot(2, 3, 1)
    rating_by_app = df.groupby('app_name')['rating'].mean().sort_values(ascending=False)
    rating_by_app.plot(kind='bar', color='skyblue')
    plt.title('Average Rating by App', fontsize=14, fontweight='bold')
    plt.xlabel('Application')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)

    # 2. Distribusi Rating Categories
    plt.subplot(2, 3, 2)
    rating_dist = df['rating_category'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.pie(rating_dist.values, labels=rating_dist.index, autopct='%1.1f%%', colors=colors)
    plt.title('Distribution of Rating Categories', fontsize=14, fontweight='bold')

    # 3. Reviews per Sector
    plt.subplot(2, 3, 3)
    sector_counts = df['sector'].value_counts()
    sector_counts.plot(kind='bar', color='lightcoral')
    plt.title('Number of Reviews by Sector', fontsize=14, fontweight='bold')
    plt.xlabel('Sector')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45)

    # 4. Rating Distribution
    plt.subplot(2, 3, 4)
    plt.hist(df['rating'], bins=5, color='lightgreen', alpha=0.7, edgecolor='black')
    plt.title('Rating Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')

    # 5. Review Length Distribution
    plt.subplot(2, 3, 5)
    plt.hist(df['review_length'], bins=30, color='orange', alpha=0.7)
    plt.title('Review Length Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Review Length (characters)')
    plt.ylabel('Frequency')

    # 6. Heatmap Correlation
    plt.subplot(2, 3, 6)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('output/eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n📈 SUMMARY STATISTICS:")
    print("-" * 25)
    print(f"Total Reviews: {len(df):,}")
    print(f"Average Rating: {df['rating'].mean():.2f}")
    print(f"Apps Analyzed: {df['app_name'].nunique()}")
    print(f"Sectors: {df['sector'].nunique()}")

    if 'review_date' in df.columns:
        print(f"Date Range: {df['review_date'].min()} to {df['review_date'].max()}")

    print(f"\nTop Apps by Review Count:")
    print(df['app_name'].value_counts().head())

    print(f"\nAverage Rating by Sector:")
    print(df.groupby('sector')['rating'].mean().round(2))