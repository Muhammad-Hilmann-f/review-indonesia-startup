from export_results import export_results
from dashboard import create_dashboard_summary
from insights import generate_insights_and_recommendations
from predictive_modeling import predictive_modeling
from customer_segmentation import customer_segmentation
from sentiment_analysis import sentiment_analysis
from eda import exploratory_data_analysis
from preprocessing import clean_and_preprocess
from data_loader import load_and_combine_data

folder_path = r"F:\Tugas-Kuliah\data sience\dataset"

# Load data
combined_df = load_and_combine_data(folder_path)

# Preprocessing data
if combined_df is not None:
    processed_df = clean_and_preprocess(combined_df)

    # Exploratory Data Analysis
    exploratory_data_analysis(processed_df)

    # Sentiment Analysis
    processed_df = sentiment_analysis(processed_df)

    # Customer Segmentation
    segmentation_results = customer_segmentation(processed_df)

    # Predictive Modeling
    models = predictive_modeling(processed_df)

    # Business Insights & Recommendations
    insights = generate_insights_and_recommendations(processed_df)

    # Create Dashboard
    create_dashboard_summary(processed_df)

    # Export Results
    export_results(processed_df, insights, models)